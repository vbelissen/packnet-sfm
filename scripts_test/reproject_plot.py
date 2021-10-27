# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from glob import glob
from cv2 import imwrite

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.geometry.camera_fisheye_valeo import CameraFisheye
from packnet_sfm.datasets.kitti_based_valeo_dataset_utils import \
    read_raw_calib_files_camera_valeo, transform_from_rot_trans
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.losses.multiview_photometric_loss import SSIM

import torch.nn.functional as funct
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
import cv2

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)

def parse_args():
    parser = argparse.ArgumentParser(description='Recalibration tool, for a specific sequence from the Valeo dataset')
    parser.add_argument('--checkpoint',             type=str,                                                   help='Checkpoint file (.ckpt)')
    parser.add_argument('--input_folder',           type=str,              default=None,                        help='Input base folder')
    parser.add_argument('--input_imgs',             type=str, nargs='+',   default=None,                        help='Input images')
    parser.add_argument('--every_n_files',          type=int,            default=1,                           help='Step in files if folders are used')
    parser.add_argument('--image_shape',            type=int, nargs='+',   default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half',                   action="store_true",                                        help='Use half precision (fp16)')
    parser.add_argument('--save_folder',            type=str,              default='/home/vbelissen/Downloads', help='Where to save pictures')
    parser.add_argument('--rot_values',             type=float, nargs='+', default=None,                        help='List of rotation values')
    parser.add_argument('--trans_values',           type=float, nargs='+', default=None,                        help='List of translation values')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt')
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (args.input_folder is None and args.input_imgs is not None) or (args.input_folder is not None and args.input_imgs is None), \
        'You need to provide either a list of input base folders for images or a list of input images, one for each camera'
    return args

def get_base_folder(image_file):
    """The base folder"""
    return '/'.join(image_file.split('/')[:-6])

def get_camera_name(image_file):
    """Returns 'cam_i', i between 0 and 4"""
    return image_file.split('/')[-2]

def get_sequence_name(image_file):
    """Returns a sequence name like '20180227_185324'."""
    return image_file.split('/')[-3]

def get_split_type(image_file):
    """Returns 'train', 'test' or 'test_sync'."""
    return image_file.split('/')[-4]

def get_path_to_theta_lut(image_file):
    """Get the current folder from image_file."""
    return os.path.join(get_base_folder(image_file),
                        'calibrations_theta_lut',
                        'fisheye',
                        get_split_type(image_file),
                        get_sequence_name(image_file),
                        get_sequence_name(image_file) + '_' + get_camera_name(image_file) + '_1280_800.npy')

def get_path_to_ego_mask(image_file):
    """Get the current folder from image_file."""
    return os.path.join(get_base_folder(image_file),
                        'semantic_masks',
                        'fisheye',
                        get_split_type(image_file),
                        get_sequence_name(image_file),
                        get_sequence_name(image_file) + '_' + get_camera_name(image_file) + '.npy')

def get_intrinsics(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary."""
    cam = get_camera_name(image_file)
    #intr = calib_data[cam]['intrinsics']
    base_intr = calib_data[cam]['base_intrinsics']
    intr = calib_data[cam]['intrinsics']
    poly_coeffs = np.array([float(intr['c1']),
                            float(intr['c2']),
                            float(intr['c3']),
                            float(intr['c4'])])
    principal_point = np.array([float(base_intr['cx_offset_px']),
                                float(base_intr['cy_offset_px'])])
    scale_factors = np.array([1., float(intr['pixel_aspect_ratio'])])
    return poly_coeffs, principal_point, scale_factors

def get_extrinsics_pose_matrix(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']

    t = np.array([float(extr['pos_x_m']), float(extr['pos_y_m']), float(extr['pos_z_m'])])

    x_rad  = np.pi / 180. * float(extr['rot_x_deg'])
    z1_rad = np.pi / 180. * float(extr['rot_z1_deg'])
    z2_rad = np.pi / 180. * float(extr['rot_z2_deg'])
    x_rad += np.pi  # gcam
    cosx  = np.cos(x_rad)
    sinx  = np.sin(x_rad)
    cosz1 = np.cos(z1_rad)
    sinz1 = np.sin(z1_rad)
    cosz2 = np.cos(z2_rad)
    sinz2 = np.sin(z2_rad)

    Rx  = np.array([[     1,     0,    0],
                    [     0,  cosx, sinx],
                    [     0, -sinx, cosx]])
    Rz1 = np.array([[ cosz1, sinz1,    0],
                    [-sinz1, cosz1,    0],
                    [     0,     0,    1]])
    Rz2 = np.array([[cosz2, -sinz2,    0],
                    [sinz2,  cosz2,    0],
                    [    0,      0,    1]])

    R = np.matmul(Rz2, np.matmul(Rx, Rz1))
    T_other_convention = -np.dot(R,t)
    pose_matrix = transform_from_rot_trans(R, T_other_convention).astype(np.float32)

    return pose_matrix

def transform_from_rot_trans_torch(R, t):
    """
    Transformation matrix from rotation matrix and translation vector.

    Parameters
    ----------
    R : np.array [3,3]
        Rotation matrix
    t : np.array [3]
        translation vector

    Returns
    -------
    matrix : np.array [4,4]
        Transformation matrix
    """
    R = R.view(3, 3)
    t = t.view(3, 1)
    return torch.cat([torch.cat([R, t], dim=1), torch.tensor([0, 0, 0, 1]).view(1, 4).float().cuda()], dim=0)

def get_extrinsics_pose_matrix_extra_trans_rot_torch(image_file, calib_data, extra_xyz_m, extra_xyz_deg):
    """Get intrinsics from the calib_data dictionary."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']

    t = torch.from_numpy(np.array([float(extr['pos_x_m']),
                                   float(extr['pos_y_m']),
                                   float(extr['pos_z_m'])])).float().cuda() + extra_xyz_m

    x_rad  = np.pi / 180. * (float(extr['rot_x_deg'])  + extra_xyz_deg[0])
    z1_rad = np.pi / 180. * (float(extr['rot_z1_deg']) + extra_xyz_deg[1])
    z2_rad = np.pi / 180. * (float(extr['rot_z2_deg']) + extra_xyz_deg[2])
    x_rad += np.pi  # gcam
    cosx  = torch.cos(x_rad)
    sinx  = torch.sin(x_rad)
    cosz1 = torch.cos(z1_rad)
    sinz1 = torch.sin(z1_rad)
    cosz2 = torch.cos(z2_rad)
    sinz2 = torch.sin(z2_rad)

    Rx = torch.zeros((3, 3), dtype=cosx.dtype).cuda()
    Rx[0,0] = 1
    Rx[1, 1] = cosx
    Rx[2, 2] = cosx
    Rx[1, 2] = sinx
    Rx[2, 1] = -sinx
    Rz1 = torch.zeros((3, 3), dtype=cosx.dtype).cuda()
    Rz1[0, 0] = cosz1
    Rz1[1, 1] = cosz1
    Rz1[0, 1] = sinz1
    Rz1[1, 0] = -sinz1
    Rz1[2, 2] = 1
    Rz2 = torch.zeros((3, 3), dtype=cosx.dtype).cuda()
    Rz2[0, 0] = cosz2
    Rz2[1, 1] = cosz2
    Rz2[0, 1] = -sinz2
    Rz2[1, 0] = sinz2
    Rz2[2, 2] = 1

    R = Rz2 @ (Rx @ Rz1)
    T_other_convention = -R @ t
    pose_matrix = transform_from_rot_trans_torch(R, T_other_convention)

    return pose_matrix

def reproject_plot_values(input_files, model_wrapper, image_shape, half):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : list (number of cameras) of lists (number of files) of str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    N_files = len(input_files[0])
    N_cams = 4
    image_area = image_shape[0] * image_shape[1]

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    calib_data = {}
    for i_cam in range(N_cams):
        base_folder_str = get_base_folder(input_files[i_cam][0])
        split_type_str  = get_split_type(input_files[i_cam][0])
        seq_name_str    = get_sequence_name(input_files[i_cam][0])
        camera_str      = get_camera_name(input_files[i_cam][0])
        calib_data[camera_str] = read_raw_calib_files_camera_valeo(base_folder_str, split_type_str, seq_name_str, camera_str)

    cams = []
    not_masked = []

    for i_cam in range(N_cams):
        path_to_theta_lut = get_path_to_theta_lut(input_files[i_cam][0])
        path_to_ego_mask = get_path_to_ego_mask(input_files[i_cam][0])
        poly_coeffs, principal_point, scale_factors = get_intrinsics(input_files[i_cam][0], calib_data)

        poly_coeffs = torch.from_numpy(poly_coeffs).unsqueeze(0)
        principal_point = torch.from_numpy(principal_point).unsqueeze(0)
        scale_factors = torch.from_numpy(scale_factors).unsqueeze(0)
        pose_matrix = torch.from_numpy(get_extrinsics_pose_matrix(input_files[i_cam][0], calib_data)).unsqueeze(0)
        pose_tensor = Pose(pose_matrix)

        cams.append(CameraFisheye(path_to_theta_lut=[path_to_theta_lut],
                                  path_to_ego_mask=[path_to_ego_mask],
                                  poly_coeffs=poly_coeffs.float(),
                                  principal_point=principal_point.float(),
                                  scale_factors=scale_factors.float(),
                                  Tcw=pose_tensor))
        if torch.cuda.is_available():
            cams[i_cam] = cams[i_cam].to('cuda:{}'.format(rank()), dtype=dtype)

        ego_mask = np.load(path_to_ego_mask)
        not_masked.append(torch.from_numpy(ego_mask.astype(float)).cuda().float())

    extra_trans_m = [torch.tensor(args.trans_values[3*i:3*(i+1)]).cuda() for i in range(4)]
    extra_rot_deg = [torch.tensor(args.rot_values[3*i:3*(i+1)]).cuda() for i in range(4)]

    count = 0


    for i_file in range(N_files):

        base_0, ext_0 = os.path.splitext(os.path.basename(input_files[0][i_file]))
        print(base_0)

        images          = []
        pred_inv_depths = []
        pred_depths     = []
        for i_cam in range(N_cams):
            images.append(load_image(input_files[i_cam][i_file]).convert('RGB'))
            images[i_cam] = resize_image(images[i_cam], image_shape)
            images[i_cam] = to_tensor(images[i_cam]).unsqueeze(0)
            if torch.cuda.is_available():
                images[i_cam] = images[i_cam].to('cuda:{}'.format(rank()), dtype=dtype)
            with torch.no_grad():
                pred_inv_depths.append(model_wrapper.depth(images[i_cam]))
                pred_depths.append(inv2depth(pred_inv_depths[i_cam]))

        def reconstruct_2imgs(i_cam1, i_cam2, extra_trans_list, extra_rot_list, original):

            for i, i_cam in enumerate([i_cam1, i_cam2]):
                pose_matrix = get_extrinsics_pose_matrix_extra_trans_rot_torch(input_files[i_cam][i_file], calib_data, extra_trans_list[i], extra_rot_list[i]).unsqueeze(0)
                pose_tensor = Pose(pose_matrix).to('cuda:{}'.format(rank()), dtype=dtype)
                CameraFisheye.Twc.fget.cache_clear()
                cams[i_cam].Tcw = pose_tensor

            world_points1 = cams[i_cam1].reconstruct(pred_depths[i_cam1], frame='w')
            world_points2 = cams[i_cam2].reconstruct(pred_depths[i_cam2], frame='w')

            ref_coords1to2 = cams[i_cam2].project(world_points1, frame='w')
            ref_coords2to1 = cams[i_cam1].project(world_points2, frame='w')

            reconstructedImg2to1 = funct.grid_sample(images[i_cam2]*not_masked[i_cam2], ref_coords1to2, mode='bilinear', padding_mode='zeros', align_corners=True)
            reconstructedImg1to2 = funct.grid_sample(images[i_cam1]*not_masked[i_cam1], ref_coords2to1, mode='bilinear', padding_mode='zeros', align_corners=True)


            cv2.imwrite(args.save_folder + '/cam_' + str(i_cam1) + '_file_' + str(i_file) + '_orig.png', (images[i_cam1][0].permute(1, 2, 0))[:,:,[2,1,0]].detach().cpu().numpy() * 255)
            cv2.imwrite(args.save_folder + '/cam_' + str(i_cam2) + '_file_' + str(i_file) + '_orig.png', (images[i_cam2][0].permute(1, 2, 0))[:,:,[2,1,0]].detach().cpu().numpy() * 255)
            cv2.imwrite(args.save_folder + '/cam_' + str(i_cam1) + '_file_' + str(i_file) + '_recons_from_' + str(i_cam2) + '_' + original +'.png', ((reconstructedImg2to1*not_masked[i_cam1])[0].permute(1, 2, 0))[:,:,[2,1,0]].detach().cpu().numpy() * 255)
            cv2.imwrite(args.save_folder + '/cam_' + str(i_cam2) + '_file_' + str(i_file) + '_recons_from_' + str(i_cam1) + '_' + original +'.png', ((reconstructedImg1to2*not_masked[i_cam2])[0].permute(1, 2, 0))[:,:,[2,1,0]].detach().cpu().numpy() * 255)

        for i in range(4):
            reconstruct_2imgs(i, (i + 1) % 4, [torch.zeros(3).cuda(), torch.zeros(3).cuda()], [torch.zeros(3).cuda(), torch.zeros(3).cuda()], 'original_calib')
            reconstruct_2imgs(i, (i + 1) % 4, [extra_trans_m[i], extra_trans_m[(i + 1) % 4]], [extra_rot_deg[i],   extra_rot_deg[(i + 1) % 4]], 'modified_calib')

        count += 1


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

    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if args.input_folder is None:
        files = [[args.input_imgs[i]] for i in range(4)]
    else:
        files = [[] for _ in range(4)]
        for i in range(4):
            for ext in ['png', 'jpg']:
                files[i] = glob((os.path.join(args.input_folder, 'cam_' + str(i) + '/', '*.{}'.format(ext))))
            files[i].sort()
            files[i] = files[i][::args.every_n_files]
            print0('Found {} files'.format(len(files[i])))

    n_files = len(files[0])
    # Process each file
    reproject_plot_values(files, model_wrapper, image_shape, args.half)

if __name__ == '__main__':
    args = parse_args()
    main(args)
