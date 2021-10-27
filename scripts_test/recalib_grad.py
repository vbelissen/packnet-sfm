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
#from packnet_sfm.datasets.kitti_based_valeo_dataset_fisheye_singleView import KITTIBasedValeoDatasetFisheye_singleView
from packnet_sfm.datasets.kitti_based_valeo_dataset_fisheye_singleView import *
from packnet_sfm.geometry.camera_fisheye_valeo import CameraFisheye
from packnet_sfm.datasets.kitti_based_valeo_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, read_raw_calib_files_camera_valeo, transform_from_rot_trans
from packnet_sfm.geometry.pose import Pose

import torch.nn.functional as funct
import torch.nn as nn





import open3d as o3d
import matplotlib.pyplot as plt
import time
from matplotlib.cm import get_cmap
from scipy.optimize import minimize
import cv2


def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim




def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM 3D visualization of point clouds maps from images')
    parser.add_argument('--checkpoints', nargs='+', type=str, help='Checkpoint files (.ckpt), one for each camera')
    parser.add_argument('--input_folders', nargs='+', type=str, help='Input base folders', default=None)
    parser.add_argument('--input_imgs', nargs='+', type=str, help='Input images', default=None)
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    args = parser.parse_args()
    checkpoints = args.checkpoints
    N = len(checkpoints)
    for i in range(N):
        assert checkpoints[i].endswith('.ckpt')
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert args.input_folders is None and args.input_imgs is not None or args.input_folders is not None and args.input_imgs is None, \
        'You need to provide either a list of input base folders for images or a list of input images, one for each .ckpt'
    if args.input_folders is None:
        assert len(args.input_imgs) == N, 'You need to provide a list of input images, one for each .ckpt'
    if args.input_imgs is None:
        assert len(args.input_folders) == N, 'You need to provide a list of input folders, one for each .ckpt'
    return args, N

def get_next_file(idx, file):
    """Get next file given next idx and current file."""
    base, ext = os.path.splitext(os.path.basename(file))
    base_splitted = base.split('_')
    base_number = base_splitted[-1]
    return os.path.join(os.path.dirname(file), '_'.join(base_splitted[:-1]) + '_' + str(idx).zfill(len(base_number)) + ext)

def get_base_folder(image_file):
    """The base folder"""
    return '/'.join(image_file.split('/')[:-6])

def get_frame_index_int(image_file):
    """Returns an int-type index of the image file"""
    return int(image_file.split('_')[-1].split('.')[0])

def get_camera_name(image_file):
    """Returns 'cam_i', i between 0 and 4"""
    return image_file.split('/')[-2]

def get_sequence_name(image_file):
    """Returns a sequence name like '20180227_185324'."""
    return image_file.split('/')[-3]

def get_split_type(image_file):
    """Returns 'train', 'test' or 'test_sync'."""
    return image_file.split('/')[-4]

def get_images_type(image_file):
    """Returns 'images_multiview' or 'images_multiview_frontOnly."""
    return image_file.split('/')[-5]

def get_current_folder(image_file):
    """Get the current folder from image_file."""
    return os.path.dirname(image_file)

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

def get_depth_file(image_file):
    """Get the corresponding depth file from an image file."""
    base, ext = os.path.splitext(os.path.basename(image_file))
    return os.path.join(get_base_folder(image_file),
                        'depth_maps',
                        'fisheye',
                        get_split_type(image_file),
                        get_sequence_name(image_file),
                        get_camera_name(image_file).replace('cam', 'velodyne'),
                        base.replace('cam', 'velodyne') + '.npz')

def get_extrinsics_pose_matrix(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']

    t = np.array([float(extr['pos_x_m']), float(extr['pos_y_m']), float(extr['pos_z_m'])])

    x_rad  = np.pi / 180. * float(extr['rot_x_deg'])
    z1_rad = np.pi / 180. * float(extr['rot_z1_deg'])
    z2_rad = np.pi / 180. * float(extr['rot_z2_deg'])
    x_rad += np.pi  # gcam
    #z1_rad += np.pi  # gcam
    #z2_rad += np.pi  # gcam
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
    #pose_matrix = invert_pose_numpy(pose_matrix)
    return pose_matrix

def get_extrinsics_pose_matrix_extra_rot(image_file, calib_data, extra_x_deg, extra_y_deg, extra_z_deg):
    """Get intrinsics from the calib_data dictionary."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']

    t = np.array([float(extr['pos_x_m']), float(extr['pos_y_m']), float(extr['pos_z_m'])])

    x_rad  = np.pi / 180. * (float(extr['rot_x_deg'])  + extra_x_deg)
    z1_rad = np.pi / 180. * (float(extr['rot_z1_deg']) + extra_y_deg)
    z2_rad = np.pi / 180. * (float(extr['rot_z2_deg']) + extra_z_deg)
    x_rad += np.pi  # gcam
    #z1_rad += np.pi  # gcam
    #z2_rad += np.pi  # gcam
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
    return torch.cat([torch.cat([R, t], dim=1), torch.tensor([0, 0, 0, 1]).view(1, 4).float()], dim=0)

def get_extrinsics_pose_matrix_extra_rot_torch(image_file, calib_data, extra_xyz_deg):
    """Get intrinsics from the calib_data dictionary."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']

    t = torch.from_numpy(np.array([float(extr['pos_x_m']), float(extr['pos_y_m']), float(extr['pos_z_m'])])).float()

    x_rad  = np.pi / 180. * (float(extr['rot_x_deg'])  + extra_xyz_deg[0])
    z1_rad = np.pi / 180. * (float(extr['rot_z1_deg']) + extra_xyz_deg[1])
    z2_rad = np.pi / 180. * (float(extr['rot_z2_deg']) + extra_xyz_deg[2])
    x_rad += np.pi  # gcam
    #z1_rad += np.pi  # gcam
    #z2_rad += np.pi  # gcam
    cosx  = torch.cos(x_rad)
    sinx  = torch.sin(x_rad)
    cosz1 = torch.cos(z1_rad)
    sinz1 = torch.sin(z1_rad)
    cosz2 = torch.cos(z2_rad)
    sinz2 = torch.sin(z2_rad)

    # Rx  = torch.tensor([[     1,     0,    0],
    #                     [     0,  cosx, sinx],
    #                     [     0, -sinx, cosx]], requires_grad=True)
    # Rz1 = torch.tensor([[ cosz1, sinz1,    0],
    #                     [-sinz1, cosz1,    0],
    #                     [     0,     0,    1]], requires_grad=True)
    # Rz2 = torch.tensor([[cosz2, -sinz2,    0],
    #                     [sinz2,  cosz2,    0],
    #                     [    0,      0,    1]], requires_grad=True)
    Rx = torch.zeros((3, 3), dtype=cosx.dtype)
    Rx[0,0] = 1
    Rx[1, 1] = cosx
    Rx[2, 2] = cosx
    Rx[1, 2] = sinx
    Rx[2, 1] = -sinx
    Rz1 = torch.zeros((3, 3), dtype=cosx.dtype)
    Rz1[0, 0] = cosz1
    Rz1[1, 1] = cosz1
    Rz1[0, 1] = sinz1
    Rz1[1, 0] = -sinz1
    Rz1[2, 2] = 1
    Rz2 = torch.zeros((3, 3), dtype=cosx.dtype)
    Rz2[0, 0] = cosz2
    Rz2[1, 1] = cosz2
    Rz2[0, 1] = -sinz2
    Rz2[1, 0] = sinz2
    Rz2[2, 2] = 1

    R = Rz2 @ (Rx @ Rz1)

    T_other_convention = -R @ t
    pose_matrix = transform_from_rot_trans_torch(R, T_other_convention)

    return pose_matrix

@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half, save):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
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
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]

    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=inv2depth(pred_inv_depth))
    else:
        # Prepare RGB image
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        imwrite(output_file, image[:, :, ::-1])

#@torch.no_grad()
def infer_optimal_calib(input_files, model_wrappers, image_shape, half):
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

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    for i_file in range(N_files):

        cams = []
        not_masked = []

        camera_names = []
        for i_cam in range(N_cams):
            camera_names.append(get_camera_name(input_files[i_cam][i_file]))

        for i_cam in range(N_cams):
            base_folder_str = get_base_folder(input_files[i_cam][i_file])
            split_type_str = get_split_type(input_files[i_cam][i_file])
            seq_name_str = get_sequence_name(input_files[i_cam][i_file])
            camera_str = get_camera_name(input_files[i_cam][i_file])

            calib_data = {}
            calib_data[camera_str] = read_raw_calib_files_camera_valeo(base_folder_str, split_type_str, seq_name_str, camera_str)

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

        base_0, ext_0 = os.path.splitext(os.path.basename(input_files[0][i_file]))
        print(base_0)

        images = []
        images_numpy = []
        pred_inv_depths = []
        pred_depths = []
        for i_cam in range(N_cams):
            images.append(load_image(input_files[i_cam][i_file]).convert('RGB'))
            images[i_cam] = resize_image(images[i_cam], image_shape)
            images[i_cam] = to_tensor(images[i_cam]).unsqueeze(0)
            if torch.cuda.is_available():
                images[i_cam] = images[i_cam].to('cuda:{}'.format(rank()), dtype=dtype)
            images_numpy.append(images[i_cam][0].cpu().numpy())
            #images_numpy[i_cam] = images_numpy[i_cam][not_masked[i_cam]]
            with torch.no_grad():
                pred_inv_depths.append(model_wrappers[0].depth(images[i_cam]))
                pred_depths.append(inv2depth(pred_inv_depths[i_cam]))
            #mask_colors_blue = np.sum(np.abs(colors_tmp - np.array([0.6, 0.8, 1])), axis=1) < 0.6  # bleu ciel


        def photo_loss_2imgs(i_cam1, i_cam2, rot_vect_list, save_pictures, rot_str):

            # Computes the photometric loss between 2 images of adjacent cameras
            # It reconstructs each image from the adjacent one, using calibration data,
            # depth prediction model and correction angles alpha, beta, gamma

            for i, i_cam in enumerate([i_cam1, i_cam2]):
                base_folder_str = get_base_folder(input_files[i_cam][i_file])
                split_type_str = get_split_type(input_files[i_cam][i_file])
                seq_name_str = get_sequence_name(input_files[i_cam][i_file])
                camera_str = get_camera_name(input_files[i_cam][i_file])
                calib_data = {}
                calib_data[camera_str] = read_raw_calib_files_camera_valeo(base_folder_str, split_type_str, seq_name_str, camera_str)
                pose_matrix = get_extrinsics_pose_matrix_extra_rot_torch(input_files[i_cam][i_file], calib_data, rot_vect_list[i]).unsqueeze(0)
                pose_tensor = Pose(pose_matrix).to('cuda:{}'.format(rank()), dtype=dtype)
                CameraFisheye.Twc.fget.cache_clear()
                cams[i_cam].Tcw = pose_tensor

            world_points1 = cams[i_cam1].reconstruct(pred_depths[i_cam1], frame='w')
            world_points2 = cams[i_cam2].reconstruct(pred_depths[i_cam2], frame='w')

            #depth1_wrt_cam2 = torch.norm(cams[i_cam2].Tcw @ world_points1, dim=1, keepdim=True)
            #depth2_wrt_cam1 = torch.norm(cams[i_cam1].Tcw @ world_points2, dim=1, keepdim=True)

            ref_coords1to2 = cams[i_cam2].project(world_points1, frame='w')
            ref_coords2to1 = cams[i_cam1].project(world_points2, frame='w')

            reconstructedImg2to1 = funct.grid_sample(images[i_cam2]*not_masked[i_cam2], ref_coords1to2, mode='bilinear', padding_mode='zeros', align_corners=True)
            reconstructedImg1to2 = funct.grid_sample(images[i_cam1]*not_masked[i_cam1], ref_coords2to1, mode='bilinear', padding_mode='zeros', align_corners=True)

            #print(reconstructedImg2to1)
            # z = reconstructedImg2to1.sum()
            # z.backward()

            if save_pictures:
                cv2.imwrite('/home/vbelissen/Downloads/cam_' + str(i_cam1) + '_orig.png',
                            torch.transpose((images[i_cam1][0, :, :, :]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy() * 255)
                cv2.imwrite('/home/vbelissen/Downloads/cam_' + str(i_cam2) + '_orig.png',
                            torch.transpose((images[i_cam2][0, :, :, :]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy() * 255)
                cv2.imwrite('/home/vbelissen/Downloads/cam_' + str(i_cam1) + '_recons_from_' + str(i_cam2) + '_rot' + rot_str + '.png',
                            torch.transpose(((reconstructedImg2to1*not_masked[i_cam1])[0, :, :, :]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy() * 255)
                cv2.imwrite('/home/vbelissen/Downloads/cam_' + str(i_cam2) + '_recons_from_' + str(i_cam1) + '_rot' + rot_str + '.png',
                            torch.transpose(((reconstructedImg1to2*not_masked[i_cam2])[0, :, :, :]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy() * 255)

            #reconstructedDepth2to1 = funct.grid_sample(pred_depths[i_cam2], ref_coords1to2, mode='bilinear', padding_mode='zeros', align_corners=True)
            #reconstructedDepth1to2 = funct.grid_sample(pred_depths[i_cam1], ref_coords2to1, mode='bilinear', padding_mode='zeros', align_corners=True)

            #reconstructedEgo2to1 = funct.grid_sample(not_masked[i_cam2], ref_coords1to2, mode='bilinear', padding_mode='zeros', align_corners=True)
            #reconstructedEgo1to2 = funct.grid_sample(not_masked[i_cam1], ref_coords2to1, mode='bilinear', padding_mode='zeros', align_corners=True)

            l1_loss_1 = torch.abs(images[i_cam1]*not_masked[i_cam1] - reconstructedImg2to1*not_masked[i_cam1])
            l1_loss_2 = torch.abs(images[i_cam2]*not_masked[i_cam2] - reconstructedImg1to2*not_masked[i_cam2])

            # SSIM loss
            ssim_loss_weight = 0.85
            ssim_loss_1 = SSIM(images[i_cam1]*not_masked[i_cam1], reconstructedImg2to1*not_masked[i_cam1], C1=1e-4, C2=9e-4, kernel_size=3)
            ssim_loss_2 = SSIM(images[i_cam2]*not_masked[i_cam2], reconstructedImg1to2*not_masked[i_cam2], C1=1e-4, C2=9e-4, kernel_size=3)

            ssim_loss_1 = torch.clamp((1. - ssim_loss_1) / 2., 0., 1.)
            ssim_loss_2 = torch.clamp((1. - ssim_loss_2) / 2., 0., 1.)

            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss_1 = ssim_loss_weight * ssim_loss_1.mean(1, True) + (1 - ssim_loss_weight) * l1_loss_1.mean(1, True)
            photometric_loss_2 = ssim_loss_weight * ssim_loss_2.mean(1, True) + (1 - ssim_loss_weight) * l1_loss_2.mean(1, True)

            mask1 = photometric_loss_1 != 0
            s1 = mask1.sum()
            loss_1 = (photometric_loss_1 * mask1).sum() / s1 if s1 > 0 else 0

            mask2 = photometric_loss_2 != 0
            s2 = mask2.sum()
            loss_2 = (photometric_loss_2 * mask2).sum() / s2 if s2 > 0 else 0

            #valid1 = ...
            #valid2 = ...

            return loss_1 + loss_2


        extra_rot_deg = torch.autograd.Variable(torch.zeros(12), requires_grad=True)
        save_pictures=False

        n_epochs = 200
        loss_tab = np.zeros(n_epochs)

        for epoch in range(n_epochs):

            loss = sum([photo_loss_2imgs(
                i,
                (i+1)%4,
                [extra_rot_deg[3 * i:3 * (i + 1)],
                 extra_rot_deg[3 * ((i+1)%4):3 * (((i+1)%4) + 1)]],
                save_pictures,
                '_' + '_'.join([str(int(100 * extra_rot_deg[i_rot])/100) for i_rot in range(12)])
            ) for i in range(4)])
            print(loss)

            loss.backward()
            print(extra_rot_deg.grad)
            loss_tab[epoch] = loss.item()

            with torch.no_grad():
                extra_rot_deg.sub_(extra_rot_deg.grad*2.0)
                extra_rot_deg.grad.zero_()

        plt.plot(loss_tab)
        plt.show()


def main(args, N):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    configs = []
    state_dicts = []
    for i in range(N):
        config, state_dict = parse_test_file(args.checkpoints[i])
        configs.append(config)
        state_dicts.append(state_dict)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = configs[0].datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(configs[0].debug)


    model_wrappers = []
    for i in range(N):
        # Initialize model wrapper from checkpoint arguments
        if i==0:
            model_wrappers.append(ModelWrapper(configs[i], load_datasets=False))
            # Restore monodepth_model state
            model_wrappers[i].load_state_dict(state_dicts[i])
        else:
            model_wrappers.append(0)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        for i in range(N):
            if i == 0:
                model_wrappers[i] = model_wrappers[i].to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    for i in range(N):
        if i == 0:
            model_wrappers[i].eval()

    if args.input_folders is None:
        files = [[args.input_imgs[i]] for i in range(N)]
    else:
        files = [[] for i in range(N)]
        for i in range(N):
            for ext in ['png', 'jpg']:
                files[i] = glob.glob((os.path.join(args.input_folders[i], '*.{}'.format(ext))))
            files[i].sort()
            print0('Found {} files'.format(len(files[i])))

    n_files = len(files[0])
    # Process each file
    infer_optimal_calib(files, model_wrappers, image_shape, args.half)


if __name__ == '__main__':
    args, N = parse_args()
    main(args, N)
