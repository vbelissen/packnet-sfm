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
import matplotlib.pyplot as plt
import time
from matplotlib.cm import get_cmap


lookat_vector = np.array([-6.3432556344086555, 0.72397009410040813, 1.6189638309453105])
front_vector = np.array([-0.99318640673281822, -0.097484566091692121, 0.063855468482092601])
up_vector = np.array([0.070547183600666891, -0.06681235377561065, 0.9952684081537887])
zoom_float = 0.02

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM 3D visualization of point clouds maps from images')
    parser.add_argument('--checkpoints', nargs='+', type=str, help='Checkpoint files (.ckpt), one for each camera')
    parser.add_argument('--input_folders', nargs='+', type=str, help='Input base folders', default=None)
    parser.add_argument('--input_imgs', nargs='+', type=str, help='Input images', default=None)
    parser.add_argument('--output', type=str, help='Output folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    parser.add_argument('--stop', type=int, default=0,
                        help='If you want to stop for checking')
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
def infer_plot_and_save_3D_pcl(input_files, output_folder, model_wrappers, image_shape, half, save, stop):
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
    N_cams = len(input_files)
    N_files = len(input_files[0])

    camera_names = []
    for i_cam in range(N_cams):
        camera_names.append(get_camera_name(input_files[i_cam][0]))

    cams = []
    not_masked = []

    cams_x = []
    cams_y = []

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1000, -1000, -1), max_bound=(1000, 1000, 5))

    # let's assume all images are from the same sequence (thus same cameras)
    for i_cam in range(N_cams):
        base_folder_str = get_base_folder(input_files[i_cam][0])
        split_type_str  = get_split_type(input_files[i_cam][0])
        seq_name_str    = get_sequence_name(input_files[i_cam][0])
        camera_str      = get_camera_name(input_files[i_cam][0])

        calib_data = {}
        calib_data[camera_str] = read_raw_calib_files_camera_valeo(base_folder_str, split_type_str, seq_name_str, camera_str)

        cams_x.append(float(calib_data[camera_str]['extrinsics']['pos_x_m']))
        cams_y.append(float(calib_data[camera_str]['extrinsics']['pos_y_m']))

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
        not_masked.append(ego_mask.astype(bool).reshape(-1))


    # create output dirs for each cam
    seq_name = get_sequence_name(input_files[0][0])
    for i_cam in range(N_cams):
        os.makedirs(os.path.join(output_folder, seq_name, 'depth', camera_names[i_cam]), exist_ok=True)
        os.makedirs(os.path.join(output_folder, seq_name, 'rgb', camera_names[i_cam]), exist_ok=True)


    i_file=0
    values = np.arange(5,15.0,0.5)
    N_values = values.size
    nb_points = np.zeros((N_values, 4))
    n = 0
    for K in values:
        print(K)

        base_0, ext_0 = os.path.splitext(os.path.basename(input_files[0][i_file]))
        print(base_0)

        images = []
        images_numpy = []
        pred_inv_depths = []
        pred_depths = []
        world_points = []
        input_depth_files = []
        has_gt_depth = []
        gt_depth = []
        gt_depth_3d = []
        pcl_full = []
        pcl_only_inliers = []
        pcl_only_outliers = []
        pcl_gt = []
        rgb = []
        viz_pred_inv_depths = []
        for i_cam in range(N_cams):
            images.append(load_image(input_files[i_cam][i_file]).convert('RGB'))
            images[i_cam] = resize_image(images[i_cam], image_shape)
            images[i_cam] = to_tensor(images[i_cam]).unsqueeze(0)
            if torch.cuda.is_available():
                images[i_cam] = images[i_cam].to('cuda:{}'.format(rank()), dtype=dtype)
            images_numpy.append(images[i_cam][0].cpu().numpy())
            images_numpy[i_cam] = images_numpy[i_cam].reshape((3, -1)).transpose()
            images_numpy[i_cam] = images_numpy[i_cam][not_masked[i_cam]]
            pred_inv_depths.append(model_wrappers[i_cam].depth(images[i_cam]))
            pred_depths.append(inv2depth(pred_inv_depths[i_cam]))
            world_points.append(cams[i_cam].reconstruct(K*pred_depths[i_cam], frame='w'))
            world_points[i_cam] = world_points[i_cam][0].cpu().numpy()
            world_points[i_cam] = world_points[i_cam].reshape((3, -1)).transpose()
            world_points[i_cam] = world_points[i_cam][not_masked[i_cam]]
            cam_name = camera_names[i_cam]
            cam_int = cam_name.split('_')[-1]
            input_depth_files.append(get_depth_file(input_files[i_cam][i_file]))
            has_gt_depth.append(os.path.exists(input_depth_files[i_cam]))
            if has_gt_depth[i_cam]:
                gt_depth.append(np.load(input_depth_files[i_cam])['velodyne_depth'].astype(np.float32))
                gt_depth[i_cam] = torch.from_numpy(gt_depth[i_cam]).unsqueeze(0).unsqueeze(0)
                if torch.cuda.is_available():
                    gt_depth[i_cam] = gt_depth[i_cam].to('cuda:{}'.format(rank()), dtype=dtype)
                gt_depth_3d.append(cams[i_cam].reconstruct(gt_depth[i_cam], frame='w'))
                gt_depth_3d[i_cam] = gt_depth_3d[i_cam][0].cpu().numpy()
                gt_depth_3d[i_cam] = gt_depth_3d[i_cam].reshape((3, -1)).transpose()
                #gt_depth_3d[i_cam] = gt_depth_3d[i_cam][not_masked[i_cam]]
            else:
                gt_depth.append(0)
                gt_depth_3d.append(0)
            pcl_full.append(o3d.geometry.PointCloud())
            pcl_full[i_cam].points = o3d.utility.Vector3dVector(world_points[i_cam])
            pcl_full[i_cam].colors = o3d.utility.Vector3dVector(images_numpy[i_cam])


            pcl = pcl_full[i_cam]#.select_by_index(ind)
            points_tmp = np.asarray(pcl.points)
            colors_tmp = np.asarray(pcl.colors)
            # remove points that are above
            mask_height = points_tmp[:, 2] > 2.5# * (abs(points_tmp[:, 0]) < 10) * (abs(points_tmp[:, 1]) < 3)
            mask_colors_blue = np.sum(np.abs(colors_tmp - np.array([0.6, 0.8, 1])), axis=1) < 0.6  # bleu ciel
            mask_colors_green = np.sum(np.abs(colors_tmp - np.array([0.2, 1, 0.4])), axis=1) < 0.8
            mask_colors_green2 = np.sum(np.abs(colors_tmp - np.array([0, 0.5, 0.15])), axis=1) < 0.2
            mask = 1-mask_height*mask_colors_blue
            mask2 = 1-mask_height*mask_colors_green
            mask3 = 1- mask_height*mask_colors_green2
            #maskY = points_tmp[:, 1] > 1
            mask = mask*mask2*mask3#*maskY
            pcl = pcl.select_by_index(np.where(mask)[0])
            cl, ind = pcl.remove_statistical_outlier(nb_neighbors=7, std_ratio=1.4)
            pcl = pcl.select_by_index(ind)
            pcl_only_inliers.append(pcl)#pcl_full[i_cam].select_by_index(ind)[mask])
            #pcl_only_outliers.append(pcl_full[i_cam].select_by_index(ind, invert=True))
            #pcl_only_outliers[i_cam].paint_uniform_color([0.0, 0.0, 1.0])
            if has_gt_depth[i_cam]:
                pcl_gt.append(o3d.geometry.PointCloud())
                pcl_gt[i_cam].points = o3d.utility.Vector3dVector(gt_depth_3d[i_cam])
                gt_inv_depth = 1 / (np.linalg.norm(gt_depth_3d[i_cam], axis=1) + 1e-6)
                cm = get_cmap('plasma')
                normalizer = .35#np.percentile(gt_inv_depth, 95)
                gt_inv_depth /= (normalizer + 1e-6)
                #print(cm(np.clip(gt_inv_depth, 0., 1.0)).shape)  # [:, :3]

                pcl_gt[i_cam].colors = o3d.utility.Vector3dVector(cm(np.clip(gt_inv_depth, 0., 1.0))[:, :3])
            else:
                pcl_gt.append(0)
        #o3d.visualization.draw_geometries(pcl_full + [e for i, e in enumerate(pcl_gt) if e != 0])
        color_cam = False
        if color_cam:
            for i_cam in range(4):
                if i_cam == 0:
                    pcl_only_inliers[i_cam].paint_uniform_color([1.0, 0.0, 0.0])#.colors = o3d.utility.Vector3dVector(images_numpy[i_cam])
                elif i_cam == 1:
                    pcl_only_inliers[i_cam].paint_uniform_color([0.0, 1.0, 0.0])
                elif i_cam == 2:
                    pcl_only_inliers[i_cam].paint_uniform_color([0.0, 0.0, 1.0])
                elif i_cam == 3:
                    pcl_only_inliers[i_cam].paint_uniform_color([0.3, 0.4, 0.3])
        dist_total = np.zeros(4)
        for i_cam in range(4):
            pcl1 = pcl_only_inliers[i_cam % 4].uniform_down_sample(10)
            pcl2 = pcl_only_inliers[(i_cam+1) % 4].uniform_down_sample(10)
            points_tmp1 = np.asarray(pcl1.points)
            points_tmp2 = np.asarray(pcl2.points)
            if i_cam == 3 or i_cam == 0: # comparaison entre 3 et 0 ou entre 0 et 1
                maskX1 = points_tmp1[:, 0] > cams_x[0]
                maskX2 = points_tmp2[:, 0] > cams_x[0]
            if i_cam == 1 or i_cam == 2: # comparaison entre 3 et 0 ou entre 0 et 1
                maskX1 = points_tmp1[:, 0] < cams_x[2]
                maskX2 = points_tmp2[:, 0] < cams_x[2]
            if i_cam == 2 or i_cam == 3: # comparaison entre 3 et 0 ou entre 0 et 1
                maskY1 = points_tmp1[:, 1] > cams_y[3]
                maskY2 = points_tmp2[:, 1] > cams_y[3]
            if i_cam == 0 or i_cam == 1: # comparaison entre 3 et 0 ou entre 0 et 1
                maskY1 = points_tmp1[:, 1] < cams_y[1]
                maskY2 = points_tmp2[:, 1] < cams_y[1]
            mask_far1 = np.sqrt(np.square(points_tmp1[:, 0] - (cams_x[0] + cams_x[2])/2) + np.square(points_tmp1[:, 1] - (cams_y[1] + cams_y[3])/2)) < 5
            mask_far2 = np.sqrt(np.square(points_tmp2[:, 0] - (cams_x[0] + cams_x[2]) / 2) + np.square(points_tmp2[:, 1] - (cams_y[1] + cams_y[3]) / 2)) < 5
            pcl1 = pcl1.select_by_index(np.where(maskX1 * maskY1 * mask_far1)[0])
            pcl2 = pcl2.select_by_index(np.where(maskX2 * maskY2 * mask_far2)[0])
            #o3d.visualization.draw_geometries([pcl1, pcl2])
            dists = pcl1.compute_point_cloud_distance(pcl2)
            nb_points[n, i_cam] = np.sum(np.asarray(dists)<0.5)
            print(nb_points[n, i_cam])
            dists = np.mean(np.asarray(dists))
            dist_total[i_cam] = dists
        print(np.mean(dist_total))

        # vis_full = o3d.visualization.Visualizer()
        # vis_full.create_window(visible = True, window_name = 'full'+str(i_file))
        # for i_cam in range(N_cams):
        #     vis_full.add_geometry(pcl_full[i_cam])
        # for i, e in enumerate(pcl_gt):
        #     if e != 0:
        #         vis_full.add_geometry(e)
        # ctr = vis_full.get_view_control()
        # ctr.set_lookat(lookat_vector)
        # ctr.set_front(front_vector)
        # ctr.set_up(up_vector)
        # ctr.set_zoom(zoom_float)
        # #vis_full.run()
        # vis_full.destroy_window()
        i_cam2=0
        suff=''
        vis_only_inliers = o3d.visualization.Visualizer()
        vis_only_inliers.create_window(visible = True, window_name = 'inliers'+str(i_file))
        for i_cam in range(N_cams):
            vis_only_inliers.add_geometry(pcl_only_inliers[i_cam])
        for i, e in enumerate(pcl_gt):
            if e != 0:
                vis_only_inliers.add_geometry(e)
        ctr = vis_only_inliers.get_view_control()
        ctr.set_lookat(lookat_vector)
        ctr.set_front(front_vector)
        ctr.set_up(up_vector)
        ctr.set_zoom(zoom_float)
        param = o3d.io.read_pinhole_camera_parameters('/home/vbelissen/Downloads/test/cameras_jsons/test'+str(i_cam2+1)+suff+'.json')
        ctr.convert_from_pinhole_camera_parameters(param)
        opt = vis_only_inliers.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        #vis_only_inliers.update_geometry('inliers0')
        vis_only_inliers.poll_events()
        vis_only_inliers.update_renderer()
        if stop:
            vis_only_inliers.run()
        #param = vis_only_inliers.get_view_control().convert_to_pinhole_camera_parameters()
        #o3d.io.write_pinhole_camera_parameters('/home/vbelissen/Downloads/test.json', param)
        vis_only_inliers.destroy_window()
        del ctr
        del vis_only_inliers
        del opt
        n += 1

    plt.plot(values, nb_points)
    plt.show()
    plt.close()
    plt.plot(values, np.sum(nb_points,axis=1))
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
        model_wrappers.append(ModelWrapper(configs[i], load_datasets=False))
        # Restore monodepth_model state
        model_wrappers[i].load_state_dict(state_dicts[i])

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        for i in range(N):
            model_wrappers[i] = model_wrappers[i].to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    for i in range(N):
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
    infer_plot_and_save_3D_pcl(files, args.output, model_wrappers, image_shape, args.half, args.save, bool(int(args.stop)))


if __name__ == '__main__':
    args, N = parse_args()
    main(args, N)
