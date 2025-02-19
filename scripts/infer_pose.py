# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch
#from packnet_sfm.geometry.pose_utils import euler2mat  #from pytorch3d import transforms
import json
import functools

from glob import glob
from cv2 import imwrite

from packnet_sfm.models.model_wrapper_valeo import ModelWrapper
from packnet_sfm.datasets.augmentations_valeo_fisheye import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image_valeo import load_convert_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor

poses = dict()


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the amount of files to process in folder')
    parser.add_argument('--offset', type=int, default=None,
                        help='Start at offset for files to process in folder')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args

def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)

@torch.no_grad()
def infer_and_save_pose(input_file_refs, input_file, model_wrapper, image_shape, half, save):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file_refs : list(str)
        Reference image file paths
    input_file : str
        Image file for pose estimation
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    base_name = os.path.basename(input_file)
    base_name_ref0 = os.path.basename(input_file_refs[0])

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    def process_image(filename):
        image = load_convert_image(filename)
        # Resize and to tensor
        image = resize_image(image, image_shape)
        image = to_tensor(image).unsqueeze(0)

        # Send image to GPU if available
        if torch.cuda.is_available():
            image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        return image

    image_ref = [process_image(input_file_ref) for input_file_ref in input_file_refs]
    image = process_image(input_file)

    # Depth inference (returns predicted inverse depth)
    pose_tensor = model_wrapper.pose(image, image_ref)[0][0]  # take the pose from 1st to 2nd image
    print(model_wrapper.pose(image, image_ref))
    rot_matrix = euler_angles_to_matrix(pose_tensor[3:], convention="ZYX")
    translation = pose_tensor[:3]

    poses[base_name+base_name_ref0] = (rot_matrix, translation)


def main(args):
    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if os.path.isdir(args.input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} files'.format(len(files)))
    else:
        raise RuntimeError("Input needs directory, not file")

    if not os.path.isdir(args.output):
        root, file_name = os.path.split(args.output)
        os.makedirs(root, exist_ok=True)
    else:
        raise RuntimeError("Output needs to be a file")

    # Process each file
    list_of_files = list(zip(files[rank():-2:world_size()],
                             files[rank() + 1:-1:world_size()],
                             files[rank() + 2::world_size()]))
    if args.offset:
        list_of_files = list_of_files[args.offset:]
    if args.limit:
        list_of_files = list_of_files[:args.limit]
    for fn1, fn2, fn3 in list_of_files:
        print(fn1)
        print(fn3)
        print(fn2)
        infer_and_save_pose([fn1, fn3], fn2, model_wrapper, image_shape, args.half, args.save)
        print(fn1)
        print(fn2)
        infer_and_save_pose([fn1], fn2, model_wrapper, image_shape, args.half, args.save)
        print(fn3)
        print(fn2)
        infer_and_save_pose([fn3], fn2, model_wrapper, image_shape, args.half, args.save)

    position = np.zeros(3)
    orientation = np.eye(3)
    for key in sorted(poses.keys()):
        rot_matrix, translation = poses[key]
        orientation = orientation.dot(rot_matrix.tolist())
        position += orientation.dot(translation.tolist())
        poses[key] = {"rot": rot_matrix.tolist(),
                      "trans": translation.tolist(),
                      "pose": [*orientation[0], position[0],
                               *orientation[1], position[1],
                               *orientation[2], position[2],
                               0, 0, 0, 1]}

    json.dump(poses, open(args.output, "w"), sort_keys=True)
    print(f"Written pose of {len(list_of_files)} images to {args.output}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
