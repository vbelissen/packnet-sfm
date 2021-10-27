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



import open3d as o3d

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM 3D visualization of point clouds maps from images')
    parser.add_argument('--checkpoint1', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--checkpoint2', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input1', type=str, help='Input file or folder')
    parser.add_argument('--input2', type=str, help='Input file or folder')
    parser.add_argument('--hasGTdepth1', type=int, choices=[0, 1], default=0)
    parser.add_argument('--hasGTdepth2', type=int, choices=[0, 1], default=0)
    parser.add_argument('--output1', type=str, help='Output file or folder')
    parser.add_argument('--output2', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    args = parser.parse_args()
    assert args.checkpoint1.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.checkpoint2.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input1) and is_image(args.output1)) or \
           (not is_image(args.input1) and not is_image(args.input1)), \
        'Input and output must both be images or folders'
    assert (is_image(args.input2) and is_image(args.output2)) or \
           (not is_image(args.input2) and not is_image(args.input2)), \
        'Input and output must both be images or folders'
    return args

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

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

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

@torch.no_grad()
def infer_plot_and_save_3D_pcl(input_file1, input_file2,
                               output_file1, output_file2,
                               model_wrapper1, model_wrapper2,
                               hasGTdepth1, hasGTdepth2,
                               image_shape, half, save):
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
    if not is_image(output_file1):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file1, exist_ok=True)
        output_file1 = os.path.join(output_file1, os.path.basename(input_file1))
    if not is_image(output_file2):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file2, exist_ok=True)
        output_file2 = os.path.join(output_file2, os.path.basename(input_file2))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image1 = load_image(input_file1).convert('RGB')
    image2 = load_image(input_file2).convert('RGB')
    # Resize and to tensor
    image1 = resize_image(image1, image_shape)
    image2 = resize_image(image2, image_shape)
    image1 = to_tensor(image1).unsqueeze(0)
    image2 = to_tensor(image2).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image1 = image1.to('cuda:{}'.format(rank()), dtype=dtype)
        image2 = image2.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth1 = model_wrapper1.depth(image1)
    pred_inv_depth2 = model_wrapper2.depth(image2)
    pred_depth1 = inv2depth(pred_inv_depth1)
    pred_depth2 = inv2depth(pred_inv_depth2)

    base_folder_str1 = get_base_folder(input_file1)
    split_type_str1  = get_split_type(input_file1)
    seq_name_str1    = get_sequence_name(input_file1)
    camera_str1      = get_camera_name(input_file1)

    base_folder_str2 = get_base_folder(input_file2)
    split_type_str2 = get_split_type(input_file2)
    seq_name_str2 = get_sequence_name(input_file2)
    camera_str2 = get_camera_name(input_file2)

    calib_data1 = {}
    calib_data2 = {}
    calib_data1[camera_str1] = read_raw_calib_files_camera_valeo(base_folder_str1, split_type_str1, seq_name_str1, camera_str1)
    calib_data2[camera_str2] = read_raw_calib_files_camera_valeo(base_folder_str2, split_type_str2, seq_name_str2, camera_str2)

    path_to_theta_lut1 = get_path_to_theta_lut(input_file1)
    path_to_ego_mask1  = get_path_to_ego_mask(input_file1)
    poly_coeffs1, principal_point1, scale_factors1 = get_intrinsics(input_file1, calib_data1)
    path_to_theta_lut2 = get_path_to_theta_lut(input_file2)
    path_to_ego_mask2  = get_path_to_ego_mask(input_file2)
    poly_coeffs2, principal_point2, scale_factors2 = get_intrinsics(input_file2, calib_data2)

    poly_coeffs1 = torch.from_numpy(poly_coeffs1).unsqueeze(0)
    principal_point1 = torch.from_numpy(principal_point1).unsqueeze(0)
    scale_factors1 = torch.from_numpy(scale_factors1).unsqueeze(0)
    poly_coeffs2 = torch.from_numpy(poly_coeffs2).unsqueeze(0)
    principal_point2 = torch.from_numpy(principal_point2).unsqueeze(0)
    scale_factors2 = torch.from_numpy(scale_factors2).unsqueeze(0)

    pose_matrix1 = torch.from_numpy(get_extrinsics_pose_matrix(input_file1, calib_data1)).unsqueeze(0)
    pose_matrix2 = torch.from_numpy(get_extrinsics_pose_matrix(input_file2, calib_data2)).unsqueeze(0)
    pose_tensor1 = Pose(pose_matrix1)
    pose_tensor2 = Pose(pose_matrix2)

    ego_mask1 = np.load(path_to_ego_mask1)
    ego_mask2 = np.load(path_to_ego_mask2)
    not_masked1 = ego_mask1.astype(bool).reshape(-1)
    not_masked2 = ego_mask2.astype(bool).reshape(-1)

    cam1 = CameraFisheye(path_to_theta_lut=[path_to_theta_lut1],
                          path_to_ego_mask=[path_to_ego_mask1],
                          poly_coeffs=poly_coeffs1.float(),
                          principal_point=principal_point1.float(),
                          scale_factors=scale_factors1.float(),
                          Tcw=pose_tensor1)
    cam2 = CameraFisheye(path_to_theta_lut=[path_to_theta_lut2],
                          path_to_ego_mask=[path_to_ego_mask2],
                          poly_coeffs=poly_coeffs2.float(),
                          principal_point=principal_point2.float(),
                          scale_factors=scale_factors2.float(),
                          Tcw=pose_tensor2)
    if torch.cuda.is_available():
        cam1 = cam1.to('cuda:{}'.format(rank()), dtype=dtype)
        cam2 = cam2.to('cuda:{}'.format(rank()), dtype=dtype)

    world_points1 = cam1.reconstruct(pred_depth1, frame='w')
    world_points1 = world_points1[0].cpu().numpy()
    world_points1 = world_points1.reshape((3,-1)).transpose()
    world_points2 = cam2.reconstruct(pred_depth2, frame='w')
    world_points2 = world_points2[0].cpu().numpy()
    world_points2 = world_points2.reshape((3,-1)).transpose()

    if hasGTdepth1:
        gt_depth_file1 = get_depth_file(input_file1)
        gt_depth1 = np.load(gt_depth_file1)['velodyne_depth'].astype(np.float32)
        gt_depth1 = torch.from_numpy(gt_depth1).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            gt_depth1 = gt_depth1.to('cuda:{}'.format(rank()), dtype=dtype)

        gt_depth_3d1 = cam1.reconstruct(gt_depth1, frame='w')
        gt_depth_3d1 = gt_depth_3d1[0].cpu().numpy()
        gt_depth_3d1 = gt_depth_3d1.reshape((3,-1)).transpose()
    if hasGTdepth2:
        gt_depth_file2 = get_depth_file(input_file2)
        gt_depth2 = np.load(gt_depth_file2)['velodyne_depth'].astype(np.float32)
        gt_depth2 = torch.from_numpy(gt_depth2).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            gt_depth2 = gt_depth2.to('cuda:{}'.format(rank()), dtype=dtype)

        gt_depth_3d2 = cam2.reconstruct(gt_depth2, frame='w')
        gt_depth_3d2 = gt_depth_3d2[0].cpu().numpy()
        gt_depth_3d2 = gt_depth_3d2.reshape((3, -1)).transpose()

    world_points1 = world_points1[not_masked1]
    world_points2 = world_points2[not_masked2]
    if hasGTdepth1:
        gt_depth_3d1 = gt_depth_3d1[not_masked1]
    if hasGTdepth2:
        gt_depth_3d2 = gt_depth_3d2[not_masked1]


    pcl1 = o3d.geometry.PointCloud()
    pcl1.points = o3d.utility.Vector3dVector(world_points1)
    img_numpy1 = image1[0].cpu().numpy()
    img_numpy1 = img_numpy1.reshape((3,-1)).transpose()
    pcl2 = o3d.geometry.PointCloud()
    pcl2.points = o3d.utility.Vector3dVector(world_points2)
    img_numpy2 = image2[0].cpu().numpy()
    img_numpy2 = img_numpy2.reshape((3,-1)).transpose()

    img_numpy1 = img_numpy1[not_masked1]
    pcl1.colors = o3d.utility.Vector3dVector(img_numpy1)
    img_numpy2 = img_numpy2[not_masked2]
    pcl2.colors = o3d.utility.Vector3dVector(img_numpy2)
    #pcl.paint_uniform_color([1.0, 0.0, 0])

    #print("Radius oulier removal")
    #cl, ind = pcl.remove_radius_outlier(nb_points=10, radius=0.5)
    #display_inlier_outlier(pcl, ind)

    remove_outliers = True
    if remove_outliers:
        cl1, ind1 = pcl1.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.3)
        #display_inlier_outlier(pcl, ind)
        inlier_cloud1 = pcl1.select_by_index(ind1)
        #inlier_cloud.paint_uniform_color([0.0, 1.0, 0])
        outlier_cloud1 = pcl1.select_by_index(ind1, invert=True)
        outlier_cloud1.paint_uniform_color([0.0, 0.0, 1.0])
        cl2, ind2 = pcl2.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.3)
        #display_inlier_outlier(pcl, ind)
        inlier_cloud2 = pcl2.select_by_index(ind2)
        #inlier_cloud.paint_uniform_color([0.0, 1.0, 0])
        outlier_cloud2 = pcl2.select_by_index(ind2, invert=True)
        outlier_cloud2.paint_uniform_color([0.0, 0.0, 1.0])

    if hasGTdepth1:
        pcl_gt1 = o3d.geometry.PointCloud()
        pcl_gt1.points = o3d.utility.Vector3dVector(gt_depth_3d1)
        pcl_gt1.paint_uniform_color([1.0, 0.0, 0])
    if hasGTdepth2:
        pcl_gt2 = o3d.geometry.PointCloud()
        pcl_gt2.points = o3d.utility.Vector3dVector(gt_depth_3d2)
        pcl_gt2.paint_uniform_color([1.0, 0.0, 0])

    if remove_outliers:
        toPlot = [inlier_cloud1, inlier_cloud2]
        if hasGTdepth1:
            toPlot.append(pcl_gt1)
        if hasGTdepth2:
            toPlot.append(pcl_gt2)
        toPlotClear = list(toPlot)
        toPlot.append(outlier_cloud1)
        toPlot.append(outlier_cloud2)
        o3d.visualization.draw_geometries(toPlot)
        o3d.visualization.draw_geometries(toPlotClear)

    toPlot = [pcl1, pcl2]
    if hasGTdepth1:
        toPlot.append(pcl_gt1)
    if hasGTdepth2:
        toPlot.append(pcl_gt2)
    o3d.visualization.draw_geometries(toPlot)

    rgb1 = image1[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    rgb2 = image2[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    # Prepare inverse depth
    viz_pred_inv_depth1 = viz_inv_depth(pred_inv_depth1[0]) * 255
    viz_pred_inv_depth2 = viz_inv_depth(pred_inv_depth2[0]) * 255
    # Concatenate both vertically
    image1 = np.concatenate([rgb1, viz_pred_inv_depth1], 0)
    image2 = np.concatenate([rgb2, viz_pred_inv_depth2], 0)
    # Save visualization
    print('Saving {} to {}'.format(
        pcolor(input_file1, 'cyan', attrs=['bold']),
        pcolor(output_file1, 'magenta', attrs=['bold'])))
    imwrite(output_file1, image1[:, :, ::-1])
    print('Saving {} to {}'.format(
        pcolor(input_file2, 'cyan', attrs=['bold']),
        pcolor(output_file2, 'magenta', attrs=['bold'])))
    imwrite(output_file2, image2[:, :, ::-1])



def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config1, state_dict1 = parse_test_file(args.checkpoint1)
    config2, state_dict2 = parse_test_file(args.checkpoint2)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config1.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config1.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper1 = ModelWrapper(config1, load_datasets=False)
    model_wrapper2 = ModelWrapper(config2, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper1.load_state_dict(state_dict1)
    model_wrapper2.load_state_dict(state_dict2)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper1 = model_wrapper1.to('cuda:{}'.format(rank()), dtype=dtype)
        model_wrapper2 = model_wrapper2.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper1.eval()
    model_wrapper2.eval()

    if os.path.isdir(args.input1):
        # If input file is a folder, search for image files
        files1 = []
        for ext in ['png', 'jpg']:
            files1.extend(glob((os.path.join(args.input1, '*.{}'.format(ext)))))
        files1.sort()
        print0('Found {} files'.format(len(files1)))
    else:
        # Otherwise, use it as is
        files1 = [args.input1]
    if os.path.isdir(args.input2):
        # If input file is a folder, search for image files
        files2 = []
        for ext in ['png', 'jpg']:
            files2.extend(glob((os.path.join(args.input2, '*.{}'.format(ext)))))
        files2.sort()
        print0('Found {} files'.format(len(files2)))
    else:
        # Otherwise, use it as is
        files2 = [args.input2]

    n_files = len(files1)
    # Process each file
    for fn1, fn2 in zip(files1[rank()::world_size()], files2[rank()::world_size()]):
        infer_plot_and_save_3D_pcl(fn1, fn2,
                                   args.output1, args.output2,
                                   model_wrapper1, model_wrapper2,
                                   bool(int(args.hasGTdepth1)), bool(int(args.hasGTdepth2)),
                                   image_shape, args.half, args.save)


if __name__ == '__main__':
    args = parse_args()
    main(args)
