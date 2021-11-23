# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from glob import glob
import sys

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import inv2depth, depth2inv
from packnet_sfm.geometry.camera_multifocal_valeo import CameraMultifocal
from packnet_sfm.datasets.kitti_based_valeo_dataset_utils import \
    read_raw_calib_files_camera_valeo_with_suffix, transform_from_rot_trans
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.losses.multiview_photometric_loss import SSIM

import torch.nn.functional as funct
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import cv2

# Pairs of adjacent cameras
# If 4 cameras are included, adjacent pairs are front-right, right-rear, rear-left, left-front.
# If 5 cameras are included, must add pairs with long range (LR):
#       LR-left, LR-front, LR-right (LR is not adjacent to the rear camera)
CAMERA_CONTEXT_PAIRS = {
    4: [[0, 1], [1, 2], [2, 3], [3, 0]],
    5: [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 3], [4, 1]]
}

def parse_args():
    parser = argparse.ArgumentParser(description='Recalibration tool, for a specific sequence from the Valeo dataset')
    parser.add_argument('--checkpoints',            type=str, nargs='+',                                      help='Checkpoint files (.ckpt)')
    parser.add_argument('--input_folder',           type=str,            default=None,                        help='Input base folder')
    parser.add_argument('--input_imgs',             type=str, nargs='+', default=None,                        help='Input images')
    parser.add_argument('--image_shape',            type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--n_epochs',               type=int,            default=1,                           help='Number of epochs')
    parser.add_argument('--every_n_files',          type=int,            default=1,                           help='Step in files if folders are used')
    parser.add_argument('--lr',                     type=float,          default=0.05,                        help='Learning rate')
    parser.add_argument('--scheduler_step_size',    type=int,            default=20,                          help='How many epochs before reducing lr')
    parser.add_argument('--scheduler_gamma',        type=float,          default=1.0,                         help='Factor for lr reduction (<=1)')
    parser.add_argument('--regul_weight_trans',     type=float,          default=5.0,                         help='Regularization weight for position correction')
    parser.add_argument('--regul_weight_rot',       type=float,          default=0.001,                       help='Regularization weight for rotation correction')
    parser.add_argument('--regul_weight_overlap',   type=float,          default=0.01,                        help='Regularization weight for the overlap between cameras')
    parser.add_argument('--save_pictures',          action='store_true', default=False)
    parser.add_argument('--save_plots',             action='store_true', default=False)
    parser.add_argument('--save_rot_tab',           action='store_true', default=False)
    parser.add_argument('--show_plots',             action='store_true', default=False)
    parser.add_argument('--save_folder',            type=str,            default='/home/vbelissen/Downloads', help='Where to save pictures')
    parser.add_argument('--frozen_cams_trans',      type=int, nargs='+', default=None,                        help='List of frozen cameras in translation')
    parser.add_argument('--frozen_cams_rot',        type=int, nargs='+', default=None,                        help='List of frozen cameras in rotation')
    parser.add_argument('--calibrations_suffix',    type=str,            default='',                          help='If you want another calibration folder')
    parser.add_argument('--depth_suffix',           type=str,            default='',                          help='If you want another folder for depth maps (according to calibration)')
    parser.add_argument('--use_lidar',              action='store_true', default=False)
    parser.add_argument('--lidar_weight',           type=float,          default=0.01,                        help='Weight in lidar loss')

    args = parser.parse_args()
    checkpoints = args.checkpoints
    N = len(checkpoints)
    for i in range(N):
        assert checkpoints[i].endswith('.ckpt')
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (args.input_folder is None and args.input_imgs is not None) or (args.input_folder is not None and args.input_imgs is None), \
        'You need to provide either a list of input base folders for images or a list of input images, one for each camera'
    assert N == 4 or N == 5, 'You should have 4 or 5 cameras in the setup'
    return args, N

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

def get_camera_type(image_file, calib_data):
    """Returns the camera type."""
    cam = get_camera_name(image_file)
    camera_type = calib_data[cam]['type']
    assert camera_type == 'fisheye' or camera_type == 'perspective', \
        'Only fisheye and perspective cameras supported'
    return camera_type

def get_camera_type_int(camera_type):
    """Returns an int based on the camera type."""
    if camera_type == 'fisheye':
        return 0
    elif camera_type == 'perspective':
        return 1
    else:
        return 2

def get_intrinsics_fisheye(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary (fisheye cam)."""
    cam = get_camera_name(image_file)
    #intr = calib_data[cam]['intrinsics']
    base_intr = calib_data[cam]['base_intrinsics']
    intr = calib_data[cam]['intrinsics']
    poly_coeffs = np.array([float(intr['c1']),
                            float(intr['c2']),
                            float(intr['c3']),
                            float(intr['c4'])],dtype='float32')
    principal_point = np.array([float(base_intr['cx_offset_px']),
                                float(base_intr['cy_offset_px'])],dtype='float32')
    scale_factors = np.array([1., float(intr['pixel_aspect_ratio'])],dtype='float32')
    return poly_coeffs, principal_point, scale_factors

def get_null_intrinsics_fisheye():
    """Get null intrinsics (fisheye cam)."""
    return np.zeros(4,dtype='float32'), np.zeros(2,dtype='float32'), np.zeros(2,dtype='float32')

def get_intrinsics_distorted(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary (distorted perspective cam)."""
    cam = get_camera_name(image_file)
    base_intr = calib_data[cam]['base_intrinsics']
    intr = calib_data[cam]['intrinsics']
    cx, cy = float(base_intr['cx_px']), float(base_intr['cy_px'])
    fx, fy = float(intr['f_x_px']), float(intr['f_y_px'])
    k1, k2, k3 = float(intr['dist_k1']), float(intr['dist_k2']), float(intr['dist_k3'])
    p1, p2 = float(intr['dist_p1']), float(intr['dist_p2'])
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]],dtype='float32')
    return K, np.array([k1, k2, k3],dtype='float32'), np.array([p1, p2],dtype='float32')

def get_null_intrinsics_distorted():
    """Get null intrinsics (distorted perspective cam)."""
    return np.zeros((3, 3),dtype='float32'), np.zeros(3,dtype='float32'), np.zeros(2,dtype='float32')

def get_full_intrinsics(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary (fisheye or distorted perspective cam)."""
    camera_type = get_camera_type(image_file, calib_data)
    if camera_type == 'fisheye':
        poly_coeffs, principal_point, scale_factors = get_intrinsics_fisheye(image_file, calib_data)
        K, k, p = get_null_intrinsics_distorted()
    elif camera_type == 'perspective':
        poly_coeffs, principal_point, scale_factors = get_null_intrinsics_fisheye()
        K, k, p = get_intrinsics_distorted(image_file, calib_data)
    else:
        sys.exit('Wrong camera type')
    return poly_coeffs, principal_point, scale_factors, K, k, p

def get_depth_file(image_file, depth_suffix):
    """
    Get the corresponding depth file from an image file.

    Parameters
    ----------
    image_file: string
        The image filename
    depth_suffix: string
        Can be empty ('') or like '_1' if you want to use another depth map folder
        (typically for recalibrated depth maps)
    """
    base, ext = os.path.splitext(os.path.basename(image_file))
    return os.path.join(get_base_folder(image_file),
                        'depth_maps' + depth_suffix,
                        'fisheye',
                        get_split_type(image_file),
                        get_sequence_name(image_file),
                        get_camera_name(image_file).replace('cam', 'velodyne'),
                        base.replace('cam', 'velodyne') + '.npz')

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

def get_extrinsics_pose_matrix(image_file, calib_data):
    """Get extrinsics from the calib_data dictionary (fisheye or distorted perspective cam)."""
    camera_type = get_camera_type(image_file, calib_data)
    if camera_type == 'fisheye':
        return get_extrinsics_pose_matrix_fisheye(image_file, calib_data)
    elif camera_type == 'perspective':
        return get_extrinsics_pose_matrix_distorted(image_file, calib_data)
    else:
        sys.exit('Wrong camera type')

def get_extrinsics_pose_matrix_fisheye(image_file, calib_data):
    """Get extrinsics from the calib_data dictionary (fisheye cam)."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']
    t = np.array([float(extr['pos_x_m']), float(extr['pos_y_m']), float(extr['pos_z_m'])])
    x_rad  = np.pi / 180. * float(extr['rot_x_deg'])
    z1_rad = np.pi / 180. * float(extr['rot_z1_deg'])
    z2_rad = np.pi / 180. * float(extr['rot_z2_deg'])
    x_rad += np.pi  # gcam
    cosx, sinx = np.cos(x_rad), np.sin(x_rad)
    cosz1, sinz1 = np.cos(z1_rad), np.sin(z1_rad)
    cosz2, sinz2 = np.cos(z2_rad), np.sin(z2_rad)

    Rx  = np.array([[     1,     0,    0],
                    [     0,  cosx, sinx],
                    [     0, -sinx, cosx]])
    Rz1 = np.array([[ cosz1, sinz1,    0],
                    [-sinz1, cosz1,    0],
                    [    0,      0,    1]])
    Rz2 = np.array([[cosz2, -sinz2,    0],
                    [sinz2,  cosz2,    0],
                    [    0,      0,    1]])

    R = np.matmul(Rz2, np.matmul(Rx, Rz1))
    T_other_convention = -np.dot(R,t)
    pose_matrix = transform_from_rot_trans(R, T_other_convention).astype(np.float32)

    return pose_matrix

def get_extrinsics_pose_matrix_distorted(image_file, calib_data):
    """Get extrinsics from the calib_data dictionary (distorted perspective cam)."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']
    T_other_convention = np.array([float(extr['t_x_m']), float(extr['t_y_m']), float(extr['t_z_m'])])
    R = np.array(extr['R'])
    pose_matrix = transform_from_rot_trans(R, T_other_convention).astype(np.float32)

    return pose_matrix

def get_extrinsics_pose_matrix_extra_trans_rot_torch(image_file, calib_data, extra_xyz_m, extra_xyz_deg):
    """Get extrinsics from the calib_data dictionary, with extra translation and rotation."""
    cam = get_camera_name(image_file)
    extr = calib_data[cam]['extrinsics']
    camera_type = get_camera_type(image_file, calib_data)
    # May be subject to modifications:
    # At the time of coding,
    #       fisheye cams are encoded with 3 rotation values and the position of COP
    #       perspective cams are encoded with a rotation matrix and the position of the origin in the cam reference
    if camera_type == 'perspective':
        T_other_convention = torch.from_numpy(np.array([float(extr['t_x_m']),
                                                        float(extr['t_y_m']),
                                                        float(extr['t_z_m'])])).float().cuda() + extra_xyz_m
        R_ini = torch.from_numpy(np.array(extr['R'])).float().cuda()
        x_rad = np.pi / 180. * extra_xyz_deg[0]
        z1_rad = np.pi / 180. * extra_xyz_deg[1]
        z2_rad = np.pi / 180. * extra_xyz_deg[2]
    elif camera_type == 'fisheye':
        t = torch.from_numpy(np.array([float(extr['pos_x_m']),
                                       float(extr['pos_y_m']),
                                       float(extr['pos_z_m'])])).float().cuda() + extra_xyz_m
        x_rad = np.pi / 180. * (float(extr['rot_x_deg']) + extra_xyz_deg[0])
        z1_rad = np.pi / 180. * (float(extr['rot_z1_deg']) + extra_xyz_deg[1])
        z2_rad = np.pi / 180. * (float(extr['rot_z2_deg']) + extra_xyz_deg[2])
    else:
        sys.exit('Wrong camera type')

    cosx = torch.cos(x_rad)
    sinx = torch.sin(x_rad)
    cosz1 = torch.cos(z1_rad)
    sinz1 = torch.sin(z1_rad)
    cosz2 = torch.cos(z2_rad)
    sinz2 = torch.sin(z2_rad)

    Rx  = torch.zeros((3, 3), dtype=cosx.dtype).cuda()
    Rz1 = torch.zeros((3, 3), dtype=cosx.dtype).cuda()
    Rz2 = torch.zeros((3, 3), dtype=cosx.dtype).cuda()

    Rx[0, 0],  Rx[1, 1],  Rx[2, 2],  Rx[1, 2],  Rx[2, 1]  =     1,  cosx,  cosx,    sinx, -sinx
    Rz1[0, 0], Rz1[1, 1], Rz1[0, 1], Rz1[1, 0], Rz1[2, 2] = cosz1, cosz1,  sinz1, -sinz1,     1
    Rz2[0, 0], Rz2[1, 1], Rz2[0, 1], Rz2[1, 0], Rz2[2, 2] = cosz2, cosz2, -sinz2,  sinz2,     1

    if camera_type == 'fisheye':
        R = Rz2 @ (Rx @ Rz1)
        T_other_convention = -R @ t
    elif camera_type == 'perspective':
        R = (Rz2 @ (Rx @ Rz1)) @ R_ini
    pose_matrix = transform_from_rot_trans_torch(R, T_other_convention)

    return pose_matrix

def l1_lidar_loss(inv_pred, inv_gt):
    mask = (inv_gt > 0.).detach()
    return nn.L1Loss(inv_pred[mask], inv_gt[mask])

def infer_optimal_calib(input_files, model_wrappers, image_shape):
    """
    Process a list of input files to infer correction in extrinsic calibration.
    Files should all correspond to the same car.
    Number of cameras is assumed to be 4 or 5.

    Parameters
    ----------
    input_file : list (number of cameras) of lists (number of files) of str
        Image file
    model_wrappers : nn.Module
        Model wrappers used for inference
    image_shape : Image shape
        Input image shape
    """
    N_files = len(input_files[0])
    N_cams = len(input_files)
    image_area = image_shape[0] * image_shape[1]

    camera_context_pairs = CAMERA_CONTEXT_PAIRS[N_cams]

    calib_data = {}
    for i_cam in range(N_cams):
        base_folder_str = get_base_folder(input_files[i_cam][0])
        split_type_str  = get_split_type(input_files[i_cam][0])
        seq_name_str    = get_sequence_name(input_files[i_cam][0])
        camera_str      = get_camera_name(input_files[i_cam][0])
        calib_data[camera_str] = read_raw_calib_files_camera_valeo_with_suffix(base_folder_str,
                                                                               split_type_str,
                                                                               seq_name_str,
                                                                               camera_str,
                                                                               args.calibrations_suffix)

    cams = []
    cams_untouched=[]
    not_masked = []

    # Assume all images are from the same sequence (thus same cameras)
    for i_cam in range(N_cams):
        path_to_ego_mask = get_path_to_ego_mask(input_files[i_cam][0])
        poly_coeffs, principal_point, scale_factors, K, k, p = get_full_intrinsics(input_files[i_cam][0], calib_data)

        poly_coeffs = torch.from_numpy(poly_coeffs).unsqueeze(0)
        principal_point = torch.from_numpy(principal_point).unsqueeze(0)
        scale_factors = torch.from_numpy(scale_factors).unsqueeze(0)
        K = torch.from_numpy(K).unsqueeze(0)
        k = torch.from_numpy(k).unsqueeze(0)
        p = torch.from_numpy(p).unsqueeze(0)
        pose_matrix = torch.from_numpy(get_extrinsics_pose_matrix(input_files[i_cam][0], calib_data)).unsqueeze(0)
        pose_tensor = Pose(pose_matrix)
        camera_type = get_camera_type(input_files[i_cam][0], calib_data)
        camera_type_int = torch.tensor([get_camera_type_int(camera_type)])

        cams.append(CameraMultifocal(poly_coeffs=poly_coeffs.float(),
                                     principal_point=principal_point.float(),
                                     scale_factors=scale_factors.float(),
                                     K=K.float(),
                                     k1=k[:, 0].float(),
                                     k2=k[:, 1].float(),
                                     k3=k[:, 2].float(),
                                     p1=p[:, 0].float(),
                                     p2=p[:, 1].float(),
                                     camera_type=camera_type_int,
                                     Tcw=pose_tensor))

        cams_untouched.append(CameraMultifocal(poly_coeffs=poly_coeffs.float(),
                                               principal_point=principal_point.float(),
                                               scale_factors=scale_factors.float(),
                                               K=K.float(),
                                               k1=k[:, 0].float(),
                                               k2=k[:, 1].float(),
                                               k3=k[:, 2].float(),
                                               p1=p[:, 0].float(),
                                               p2=p[:, 1].float(),
                                               camera_type=camera_type_int,
                                               Tcw=pose_tensor))
        if torch.cuda.is_available():
            cams[i_cam] = cams[i_cam].to('cuda:{}'.format(rank()))
            cams_untouched[i_cam] = cams_untouched[i_cam].to('cuda:{}'.format(rank()))

        ego_mask = np.load(path_to_ego_mask)
        not_masked.append(torch.from_numpy(ego_mask.astype(float)).cuda().float())

    # Learning variables
    extra_trans_m = [torch.autograd.Variable(torch.zeros(3).cuda(), requires_grad=True) for _ in range(N_cams)]
    extra_rot_deg = [torch.autograd.Variable(torch.zeros(3).cuda(), requires_grad=True) for _ in range(N_cams)]

    # Constraints: translation
    frozen_cams_trans = args.frozen_cams_trans
    if frozen_cams_trans is not None:
        for i_cam in frozen_cams_trans:
            extra_trans_m[i_cam].requires_grad = False
    # Constraints: rotation
    frozen_cams_rot = args.frozen_cams_rot
    if frozen_cams_rot is not None:
        for i_cam in frozen_cams_rot:
            extra_rot_deg[i_cam].requires_grad = False

    # Parameters from argument parser
    save_pictures = args.save_pictures
    n_epochs = args.n_epochs
    learning_rate = args.lr
    step_size = args.scheduler_step_size
    gamma = args.scheduler_gamma

    # Table of loss
    loss_tab = np.zeros(n_epochs)

    # Table of extra rotation values
    extra_rot_values_tab = np.zeros((N_cams * 3, N_files * n_epochs))

    # Optimizer
    optimizer = optim.Adam(extra_trans_m + extra_rot_deg, lr=learning_rate)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Regularization weights
    regul_weight_trans = torch.tensor(args.regul_weight_trans).cuda()
    regul_weight_rot = torch.tensor(args.regul_weight_rot).cuda()
    regul_weight_overlap = torch.tensor(args.regul_weight_overlap).cuda()

    # Loop on the number of epochs
    count = 0
    for epoch in range(n_epochs):
        print('Epoch ' + str(epoch) + '/' + str(n_epochs))

        # Initialize loss
        loss_sum = 0

        # Loop on the number of files
        for i_file in range(N_files):

            # Filename for camera 0
            base_0, ext_0 = os.path.splitext(os.path.basename(input_files[0][i_file]))
            print(base_0)

            # Initialize list of tensors: images, predicted inverse depths and predicted depths
            images, pred_inv_depths, pred_depths = [], [], []
            input_depth_files, has_gt_depth, gt_depth, gt_inv_depth = [], [], [], []
            nb_gt_depths = 0

            # Reset camera poses Twc
            CameraMultifocal.Twc.fget.cache_clear()

            # Loop on cams and predict depth
            for i_cam in range(N_cams):
                images.append(load_image(input_files[i_cam][i_file]).convert('RGB'))
                images[i_cam] = resize_image(images[i_cam], image_shape)
                images[i_cam] = to_tensor(images[i_cam]).unsqueeze(0)
                if torch.cuda.is_available():
                    images[i_cam] = images[i_cam].to('cuda:{}'.format(rank()))
                with torch.no_grad():
                    pred_inv_depths.append(model_wrappers[i_cam].depth(images[i_cam]))
                    pred_depths.append(inv2depth(pred_inv_depths[i_cam]))

                if args.use_lidar:
                    input_depth_files.append(get_depth_file(input_files[i_cam][i_file], args.depth_suffix))
                    has_gt_depth.append(os.path.exists(input_depth_files[i_cam]))
                    if has_gt_depth[i_cam]:
                        nb_gt_depths += 1
                        gt_depth.append(np.load(input_depth_files[i_cam])['velodyne_depth'].astype(np.float32))
                        gt_depth[i_cam] = torch.from_numpy(gt_depth[i_cam]).unsqueeze(0).unsqueeze(0)
                        gt_inv_depth.append(depth2inv(gt_depth[i_cam]))
                        if torch.cuda.is_available():
                            gt_depth[i_cam] = gt_depth[i_cam].to('cuda:{}'.format(rank()))
                            gt_inv_depth[i_cam] = gt_inv_depth[i_cam].to('cuda:{}'.format(rank()))
                    else:
                        gt_depth.append(0)
                        gt_inv_depth.append(0)

                pose_matrix = get_extrinsics_pose_matrix_extra_trans_rot_torch(input_files[i_cam][i_file],
                                                                               calib_data,
                                                                               extra_trans_m[i_cam],
                                                                               extra_rot_deg[i_cam]).unsqueeze(0)
                pose_tensor = Pose(pose_matrix).to('cuda:{}'.format(rank()))
                cams[i_cam].Tcw = pose_tensor

            # Define a loss function between 2 images
            def photo_loss_2imgs(i_cam1, i_cam2, extra_trans_list, extra_rot_list, save_pictures):
                # Computes the photometric loss between 2 images of adjacent cameras
                # It reconstructs each image from the adjacent one, applying correction in rotation and translation

                # # Apply correction on two cams
                # for i, i_cam in enumerate([i_cam1, i_cam2]):
                #     pose_matrix = get_extrinsics_pose_matrix_extra_trans_rot_torch(input_files[i_cam][i_file],
                #                                                                    calib_data,
                #                                                                    extra_trans_list[i],
                #                                                                    extra_rot_list[i]).unsqueeze(0)
                #     pose_tensor = Pose(pose_matrix).to('cuda:{}'.format(rank()))
                #     CameraMultifocal.Twc.fget.cache_clear()
                #     cams[i_cam].Tcw = pose_tensor

                # Reconstruct 3D points for each cam
                world_points1 = cams[i_cam1].reconstruct(pred_depths[i_cam1], frame='w')
                world_points2 = cams[i_cam2].reconstruct(pred_depths[i_cam2], frame='w')

                # Get coordinates of projected points on other cam
                ref_coords1to2 = cams[i_cam2].project(world_points1, frame='w')
                ref_coords2to1 = cams[i_cam1].project(world_points2, frame='w')

                # Reconstruct each image from the adjacent camera
                reconstructedImg2to1 = funct.grid_sample(images[i_cam2] * not_masked[i_cam2],
                                                         ref_coords1to2,
                                                         mode='bilinear', padding_mode='zeros', align_corners=True)
                reconstructedImg1to2 = funct.grid_sample(images[i_cam1] * not_masked[i_cam1],
                                                         ref_coords2to1,
                                                         mode='bilinear', padding_mode='zeros', align_corners=True)
                # Save pictures if requested
                if save_pictures:
                    # Save original files if first epoch
                    if epoch == 0:
                        cv2.imwrite(args.save_folder + '/cam_' + str(i_cam1) + '_file_' + str(i_file) + '_orig.png',
                                    (images[i_cam1][0].permute(1, 2, 0))[:, :, [2, 1, 0]].detach().cpu().numpy() * 255)
                        cv2.imwrite(args.save_folder + '/cam_' + str(i_cam2) + '_file_' + str(i_file) + '_orig.png',
                                    (images[i_cam2][0].permute(1, 2, 0))[:, :, [2, 1, 0]].detach().cpu().numpy() * 255)
                    # Save reconstructed images
                    cv2.imwrite(args.save_folder + '/epoch_' + str(epoch) + '_file_' + str(i_file) + '_cam_' + str(i_cam1) + '_recons_from_' + str(i_cam2) + '.png',
                                ((reconstructedImg2to1 * not_masked[i_cam1])[0].permute(1, 2, 0))[:, :, [2, 1, 0]].detach().cpu().numpy() * 255)
                    cv2.imwrite(args.save_folder + '/epoch_' + str(epoch) + '_file_' + str(i_file) + '_cam_' + str(i_cam2) + '_recons_from_' + str(i_cam1) + '.png',
                                ((reconstructedImg1to2 * not_masked[i_cam2])[0].permute(1, 2, 0))[:, :, [2, 1, 0]].detach().cpu().numpy() * 255)

                # L1 loss
                l1_loss_1 = torch.abs(images[i_cam1] * not_masked[i_cam1] - reconstructedImg2to1 * not_masked[i_cam1])
                l1_loss_2 = torch.abs(images[i_cam2] * not_masked[i_cam2] - reconstructedImg1to2 * not_masked[i_cam2])

                # SSIM loss
                ssim_loss_weight = 0.85
                ssim_loss_1 = SSIM(images[i_cam1] * not_masked[i_cam1],
                                   reconstructedImg2to1 * not_masked[i_cam1],
                                   C1=1e-4, C2=9e-4, kernel_size=3)
                ssim_loss_2 = SSIM(images[i_cam2] * not_masked[i_cam2],
                                   reconstructedImg1to2 * not_masked[i_cam2],
                                   C1=1e-4, C2=9e-4, kernel_size=3)

                ssim_loss_1 = torch.clamp((1. - ssim_loss_1) / 2., 0., 1.)
                ssim_loss_2 = torch.clamp((1. - ssim_loss_2) / 2., 0., 1.)

                # Photometric loss: alpha * ssim + (1 - alpha) * l1
                photometric_loss_1 = ssim_loss_weight * ssim_loss_1.mean(1, True) + (1 - ssim_loss_weight) * l1_loss_1.mean(1, True)
                photometric_loss_2 = ssim_loss_weight * ssim_loss_2.mean(1, True) + (1 - ssim_loss_weight) * l1_loss_2.mean(1, True)

                # Compute the number of valid pixels
                mask1 = (reconstructedImg2to1 * not_masked[i_cam1]).sum(axis=1, keepdim=True) != 0
                s1 = mask1.sum().float()
                mask2 = (reconstructedImg1to2 * not_masked[i_cam2]).sum(axis=1, keepdim=True) != 0
                s2 = mask2.sum().float()

                # Compute the photometric losses weighed by the number of valid pixels
                loss_1 = (photometric_loss_1 * mask1).sum() / s1 if s1 > 0 else 0
                loss_2 = (photometric_loss_2 * mask2).sum() / s2 if s2 > 0 else 0

                # The final loss can be regularized to encourage a similar overlap between images
                if s1 > 0 and s2 > 0:
                    return loss_1 + loss_2 + regul_weight_overlap * image_area * (1 / s1 + 1 / s2)
                else:
                    return 0.

            def lidar_loss(i_cam1, extra_trans, extra_rot):
                if args.use_lidar and has_gt_depth[i_cam1]:
                    # pose_matrix_oldCalib = get_extrinsics_pose_matrix_extra_trans_rot_torch(input_files[i_cam1][i_file],
                    #                                                                         calib_data,
                    #                                                                         torch.zeros(3).cuda(),
                    #                                                                         torch.zeros(3).cuda()).unsqueeze(0)
                    # pose_tensor_oldCalib = Pose(pose_matrix_oldCalib).to('cuda:{}'.format(rank()))
                    # cam_old = CameraMultifocal(cams[i_cam1])
                    # cam_old.Tcw = pose_tensor_oldCalib
                    world_points_gt_oldCalib = cams_untouched[i_cam1].reconstruct(gt_depth[i_cam1], frame='w')

                    # pose_matrix_newCalib = get_extrinsics_pose_matrix_extra_trans_rot_torch(input_files[i_cam1][i_file],
                    #                                                                         calib_data,
                    #                                                                         extra_trans,
                    #                                                                         extra_rot).unsqueeze(0)
                    # pose_tensor_newCalib = Pose(pose_matrix_newCalib).to('cuda:{}'.format(rank()))
                    # CameraMultifocal.Twc.fget.cache_clear()
                    # cams[i_cam1].Tcw = pose_tensor_newCalib

                    # Get coordinates of projected points on new cam
                    ref_coords = cams[i_cam1].project(world_points_gt_oldCalib, frame='w')

                    # Reconstruct projected lidar from the new camera
                    reprojected_gt_inv_depth = funct.grid_sample(gt_inv_depth[i_cam1], ref_coords,
                                                             mode='nearest', padding_mode='zeros', align_corners=True)

                    print(pred_inv_depths[i_cam1].shape)
                    print(reprojected_gt_inv_depth.shape)
                    
                    return l1_lidar_loss(pred_inv_depths[i_cam1], reprojected_gt_inv_depth)
                else:
                    return 0.

            if nb_gt_depths > 0:
                final_lidar_weight = (N_cams / nb_gt_depths) * args.lidar_weight
            else:
                final_lidar_weight = 0.

            # The final loss consists of summing the photometric loss of all pairs of adjacent cameras
            # and is regularized to prevent weights from exploding
            loss = sum([photo_loss_2imgs(p[0], p[1],
                                         [extra_trans_m[p[0]], extra_trans_m[p[1]]],
                                         [extra_rot_deg[p[0]], extra_rot_deg[p[1]]],
                                         save_pictures)
                        for p in camera_context_pairs]) \
                   + regul_weight_rot * sum([(extra_rot_deg[i] ** 2).sum() for i in range(N_cams)]) \
                   + regul_weight_trans * sum([(extra_trans_m[i] ** 2).sum() for i in range(N_cams)]) \
                   + final_lidar_weight * sum([lidar_loss(i, extra_trans_m[i], extra_rot_deg[i]) for i in range(N_cams)])

            # Optimization steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save correction values and print loss
            with torch.no_grad():
                loss_sum += loss.item()
                for i_cam in range(N_cams):
                    for j in range(3):
                        extra_rot_values_tab[3 * i_cam + j, count] = extra_rot_deg[i_cam][j].item()
                print('Loss:')
                print(loss)

            count += 1

        # Update scheduler
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        scheduler.step()
        with torch.no_grad():
            print('End of epoch')
            print(extra_trans_m)
            print(extra_rot_deg)

        loss_tab[epoch] = loss_sum/N_files

    # Plot/save loss if requested
    plt.figure()
    plt.plot(loss_tab)
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig(os.path.join(args.save_folder, get_sequence_name(input_files[0][0]) + '_loss.png'))

    # Plot/save correction values if requested
    plt.figure()
    for j in range(N_cams * 3):
        plt.plot(extra_rot_values_tab[j])
    if args.show_plots:
        plt.show()
    if args.save_plots:
        plt.savefig(os.path.join(args.save_folder, get_sequence_name(input_files[0][0]) + '_extra_rot.png'))

    # Save correction values table if requested
    if args.save_rot_tab:
        np.save(os.path.join(args.save_folder, get_sequence_name(input_files[0][0]) + '_rot_tab.npy'),
                extra_rot_values_tab)

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
        model_wrappers.append(ModelWrapper(configs[i], load_datasets=False))
        # Restore monodepth_model state
        model_wrappers[i].load_state_dict(state_dicts[i])

    # Send model to GPU if available
    if torch.cuda.is_available():
        for i in range(N):
            model_wrappers[i] = model_wrappers[i].to('cuda:{}'.format(rank()))

    # Set to eval mode
    for i in range(N):
        model_wrappers[i].eval()

    if args.input_folder is None:
        files = [[args.input_imgs[i]] for i in range(N)]
    else:
        files = [[] for _ in range(N)]
        for i in range(N):
            for ext in ['png', 'jpg']:
                files[i] = glob((os.path.join(args.input_folder, 'cam_' + str(i) + '/', '*.{}'.format(ext))))
            files[i].sort()
            files[i] = files[i][::args.every_n_files]
            print0('Found {} files'.format(len(files[i])))

    n_files = len(files[0])
    # Process each file
    infer_optimal_calib(files, model_wrappers, image_shape)

if __name__ == '__main__':
    args, N = parse_args()
    main(args, N)
