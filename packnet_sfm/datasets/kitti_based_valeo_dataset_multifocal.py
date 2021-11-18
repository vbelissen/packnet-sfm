# Copyright 2020 Toyota Research Institute.  All rights reserved.

import glob
import numpy as np
import os
import sys
from PIL import Image, ImageFile


from torch.utils.data import Dataset

from packnet_sfm.datasets.kitti_based_valeo_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, read_raw_calib_files_camera_valeo, transform_from_rot_trans
from packnet_sfm.utils.image_valeo import load_convert_image
from packnet_sfm.geometry.pose_utils import invert_pose_numpy

########################################################################################################################

# Cameras from the stero pair (left is the origin)

PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_convert_image(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

########################################################################################################################
#### DATASET
########################################################################################################################

class KITTIBasedValeoDatasetMultifocal(Dataset):
    """
    KITTI-based Valeo dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    file_list : str
        Split file, with paths to the images to be used
    train : bool
        True if the dataset will be used for training
    data_transform : Function
        Transformations applied to the sample
    depth_type : str
        Which depth type to load
    with_pose : bool
        True if returning ground-truth pose
    back_context : int
        Number of backward frames to consider as context
    forward_context : int
        Number of forward frames to consider as context
    strides : tuple
        List of context strides
    with_geometric_context : bool
        True if surrounding camera views are used in context
    """
    def __init__(self, root_dir, file_list_and_geometric_context, train=True,
                 data_transform=None, depth_type=None, with_pose=False, with_geometric_context=False,
                 with_spatiotemp_context=False,
                 back_context=0, forward_context=0, strides=(1,), cameras=None):
        # Assertions
        backward_context = back_context
        assert backward_context >= 0 and forward_context >= 0, 'Invalid contexts'

        self.backward_context = backward_context
        self.backward_context_paths = []
        self.forward_context = forward_context
        self.forward_context_paths = []

        self.with_context = (backward_context != 0 or forward_context != 0)
        self.split = file_list_and_geometric_context.split('/')[-1].split('.')[0]

        self.train = train
        self.root_dir = root_dir
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.with_pose = with_pose

        self.with_geometric_context = with_geometric_context
        self.with_spatiotemp_context = with_spatiotemp_context

        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        self.cameras = cameras
        self.num_cameras = len(cameras)

        self.max_geometric_context = 3 # maximum number of overlapping cameras

        assert self.num_cameras == 1
        assert self.cameras == ['mixed']

        with open(file_list_and_geometric_context, "r") as f:
            data_geometric_context = f.readlines()

        if self.with_geometric_context:
            self.paths_geometric_context  = []
            if self.with_spatiotemp_context:
                self.backward_context_paths_geometric_context = []
                self.forward_context_paths_geometric_context = []

        self.paths = []
        # Get file list from data
        for i, fname in enumerate(data_geometric_context):
            list_paths = fname.split()[0].split(',')
            Np = len(list_paths)
            path = os.path.join(root_dir, list_paths[0])
            if self.with_geometric_context:
                paths_geometric_context_i = [os.path.join(root_dir, list_paths[p]) for p in range(1, Np)]
            if not self.with_depth:
                self.paths.append(path)
                if self.with_geometric_context:
                    self.paths_geometric_context.append(paths_geometric_context_i)
            else:
                # Check if the depth file exists
                depth = self._get_depth_file(path)
                if depth is not None and os.path.exists(depth):
                    self.paths.append(path)
                    if self.with_geometric_context:
                        self.paths_geometric_context.append(paths_geometric_context_i)

        # If using context, filter file list
        if self.with_context:
            paths_with_context = []
            if self.with_geometric_context:
                paths_with_context_geometric_context = []
                if self.with_spatiotemp_context:
                    paths_with_context_backward_context_paths_geometric_context = []
                    paths_with_context_forward_context_paths_geometric_context = []
            for stride in strides:
                for idx, file in enumerate(self.paths):
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(file, backward_context, forward_context, stride)
                    if backward_context_idxs is not None and forward_context_idxs is not None:
                        paths_with_context.append(self.paths[idx])
                        self.forward_context_paths.append(forward_context_idxs)
                        self.backward_context_paths.append(backward_context_idxs[::-1])
                        if self.with_geometric_context:
                            if self.with_spatiotemp_context:
                                paths_with_context_geometric_context.append(self.paths_geometric_context[idx])
                                paths_with_context_backward_context_paths_geometric_context_tmp = []
                                paths_with_context_forward_context_paths_geometric_context_tmp = []
                                for path_geometric_context in self.paths_geometric_context[idx]:
                                    backward_context_idxs_geometric_context, forward_context_idxs_geometric_context = \
                                        self._get_sample_context(
                                            path_geometric_context, backward_context, forward_context, stride
                                        )
                                    backward_context_paths_tmp, _ = self._get_context_files(
                                        path_geometric_context, backward_context_idxs_geometric_context
                                    )
                                    forward_context_paths_tmp, _ = self._get_context_files(
                                        path_geometric_context, forward_context_idxs_geometric_context
                                    )
                                    paths_with_context_backward_context_paths_geometric_context_tmp.append(
                                        backward_context_paths_tmp
                                    )
                                    paths_with_context_forward_context_paths_geometric_context_tmp.append(
                                        forward_context_paths_tmp
                                    )
                                paths_with_context_backward_context_paths_geometric_context.append(
                                    paths_with_context_backward_context_paths_geometric_context_tmp
                                )
                                paths_with_context_forward_context_paths_geometric_context.append(
                                    paths_with_context_forward_context_paths_geometric_context_tmp
                                )
            self.paths = paths_with_context
            if self.with_geometric_context:
                self.paths_geometric_context = paths_with_context_geometric_context
                if self.with_spatiotemp_context:
                    self.backward_context_paths_geometric_context = paths_with_context_backward_context_paths_geometric_context
                    self.forward_context_paths_geometric_context = paths_with_context_forward_context_paths_geometric_context

########################################################################################################################

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        base_splitted = base.split('_')
        base_number = base_splitted[-1]
        return os.path.join(os.path.dirname(file),
                            '_'.join(base_splitted[:-1]) + '_' + str(idx).zfill(len(base_number)) + ext)

    @staticmethod
    def _get_base_folder(image_file):
        """The base folder"""
        return '/'.join(image_file.split('/')[:-6])

    @staticmethod
    def _get_frame_index_int(image_file):
        """Returns an int-type index of the image file"""
        return int(image_file.split('_')[-1].split('.')[0])

    @staticmethod
    def _get_camera_name(image_file):
        """Returns 'cam_i', i between 0 and 4"""
        return image_file.split('/')[-2]

    @staticmethod
    def _get_sequence_name(image_file):
        """Returns a sequence name like '20180227_185324'."""
        return image_file.split('/')[-3]

    @staticmethod
    def _get_split_type(image_file):
        """Returns 'train', 'test' or 'test_sync'."""
        return image_file.split('/')[-4]

    @staticmethod
    def _get_images_type(image_file):
        """Returns 'images_multiview' or 'images_multiview_frontOnly."""
        return image_file.split('/')[-5]

    @staticmethod
    def _get_current_folder(image_file):
        """Get the current folder from image_file."""
        return os.path.dirname(image_file)

    def _get_camera_type(self, image_file, calib_data):
        cam = self._get_camera_name(image_file)
        camera_type = calib_data[cam]['type']
        assert camera_type == 'fisheye' or camera_type == 'perspective', \
            'Only fisheye and perspective cameras supported'
        return camera_type

    def _get_camera_type_int(self, camera_type):
        if camera_type == 'fisheye':
            return 0
        elif camera_type == 'perspective':
            return 1
        else:
            return 2

    def _get_intrinsics_fisheye(self, image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        cam = self._get_camera_name(image_file)
        base_intr = calib_data[cam]['base_intrinsics']
        intr = calib_data[cam]['intrinsics']
        poly_coeffs = np.array([float(intr['c1']),
                                float(intr['c2']),
                                float(intr['c3']),
                                float(intr['c4'])], dtype='float32')
        principal_point = np.array([float(base_intr['cx_offset_px']),
                                    float(base_intr['cy_offset_px'])], dtype='float32')
        scale_factors = np.array([1., float(intr['pixel_aspect_ratio'])], dtype='float32')
        return poly_coeffs, principal_point, scale_factors

    def _get_null_intrinsics_fisheye(self):
        return np.zeros(4, dtype='float32'), np.zeros(2, dtype='float32'), np.zeros(2, dtype='float32')

    def _get_intrinsics_distorted(self, image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        cam = self._get_camera_name(image_file)
        base_intr = calib_data[cam]['base_intrinsics']
        intr = calib_data[cam]['intrinsics']
        cx, cy = float(base_intr['cx_px']), float(base_intr['cy_px'])
        fx, fy = float(intr['f_x_px']), float(intr['f_y_px'])
        k1, k2, k3 = float(intr['dist_k1']), float(intr['dist_k2']), float(intr['dist_k3'])
        p1, p2 = float(intr['dist_p1']), float(intr['dist_p2'])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype='float32')
        return K, np.array([k1, k2, k3], dtype='float32'), np.array([p1, p2], dtype='float32')

    def _get_null_intrinsics_distorted(self):
        return np.zeros((3, 3), dtype='float32'), np.zeros(3, dtype='float32'), np.zeros(2, dtype='float32')

    def _get_full_intrinsics(self, image_file, calib_data):
        camera_type = self._get_camera_type(image_file, calib_data)
        if camera_type == 'fisheye':
            poly_coeffs, principal_point, scale_factors = self._get_intrinsics_fisheye(image_file, calib_data)
            K, k, p = self._get_null_intrinsics_distorted()
        elif camera_type == 'perspective':
            poly_coeffs, principal_point, scale_factors = self._get_null_intrinsics_fisheye()
            K, k, p = self._get_intrinsics_distorted(image_file, calib_data)
        else:
            sys.exit('Wrong camera type')
        return poly_coeffs, principal_point, scale_factors, K, k, p

    def _get_extrinsics_pose_matrix(self, image_file, calib_data):
        camera_type = self._get_camera_type(image_file, calib_data)
        if camera_type == 'fisheye':
            return self._get_extrinsics_pose_matrix_fisheye(image_file, calib_data)
        elif camera_type == 'perspective':
            return self._get_extrinsics_pose_matrix_distorted(image_file, calib_data)
        else:
            sys.exit('Wrong camera type')

    def _get_extrinsics_pose_matrix_fisheye(self, image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        cam = self._get_camera_name(image_file)
        if image_file in self.pose_cache:
            return self.pose_cache[image_file]

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

        self.pose_cache[image_file] = pose_matrix
        return pose_matrix

    def _get_extrinsics_pose_matrix_distorted(self, image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        cam = self._get_camera_name(image_file)
        if image_file in self.pose_cache:
            return self.pose_cache[image_file]

        extr = calib_data[cam]['extrinsics']

        T_other_convention = np.array([float(extr['t_x_m']), float(extr['t_y_m']), float(extr['t_z_m'])])
        R = np.array(extr['R'])
        pose_matrix = transform_from_rot_trans(R, T_other_convention).astype(np.float32)

        self.pose_cache[image_file] = pose_matrix
        return pose_matrix

    def _read_raw_calib_files(self, base_folder, split_type, seq_name, cameras):
        """Read raw calibration files from folder."""
        data = {}
        camera = cameras[0]
        data[camera] = read_raw_calib_files_camera_valeo(base_folder, split_type, seq_name, camera)
        return data

    def _get_path_to_theta_lut(self, image_file):
        """Get the current folder from image_file."""
        return os.path.join(self._get_base_folder(image_file),
                            'calibrations_theta_lut',
                            'fisheye',
                            self._get_split_type(image_file),
                            self._get_sequence_name(image_file),
                            self._get_sequence_name(image_file) + '_' + self._get_camera_name(image_file) + '_1280_800.npy')

    def _get_path_to_ego_mask(self, image_file):
        """Get the current folder from image_file."""
        return os.path.join(self._get_base_folder(image_file),
                            'semantic_masks',
                            'fisheye',
                            self._get_split_type(image_file),
                            self._get_sequence_name(image_file),
                            self._get_sequence_name(image_file) + '_' + self._get_camera_name(image_file) + '.npy')
########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if self.depth_type in ['velodyne']:
            return read_npz_depth(depth_file, self.depth_type)
        elif self.depth_type in ['groundtruth']:
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    def _get_depth_file(self, image_file):
        """Get the corresponding depth file from an image file."""
        base, ext = os.path.splitext(os.path.basename(image_file))
        return os.path.join(self._get_base_folder(image_file),
                            'depth_maps',
                            'fisheye',
                            self._get_split_type(image_file),
                            self._get_sequence_name(image_file),
                            self._get_camera_name(image_file).replace('cam', 'velodyne'),
                            base.replace('cam', 'velodyne') + '.npz')

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = self._get_frame_index_int(sample_name)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_index = self._cache[parent_folder]
        else:
            max_index = int(os.path.splitext(os.path.basename(sorted(glob.glob(os.path.join(parent_folder, '*' + ext)))[-1]))[0].split('_')[-1])
            self._cache[parent_folder] = max_index

        # We slightly modify the search of images wrt packnet:
        # the first context image is searched at a distance of stride, then one by one
        if backward_context > 0:
            back_distance = stride + (backward_context - 1) * 1
        else:
            back_distance = 0
        if forward_context > 0:
            forw_distance = stride + (forward_context - 1) * 1
        else:
            forw_distance = 0
        # Check bounds
        if (f_idx - back_distance) < 0 or (
                f_idx + forw_distance * stride) > max_index:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        first_back_image = True
        while len(backward_context_idxs) < backward_context and c_idx >= 0:
            if first_back_image:
                c_idx -= stride
                first_back_image = False
            else:
                c_idx -= 1
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        first_forw_image = True
        while len(forward_context_idxs) < forward_context and c_idx <= max_index:
            if first_forw_image:
                c_idx += stride
                first_forw_image = False
            else:
                c_idx += 1
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx > max_index:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        if self.with_depth:
            depth_context_paths = [self._get_depth_file(f) for f in image_context_paths]
            return image_context_paths, depth_context_paths
        else:
            return image_context_paths, None


########################################################################################################################

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        sample = {
            'idx': idx,
            'filename': '%s_%010d' % (self.split, idx),
            'rgb': load_convert_image(self.paths[idx]),
        }

        # Add intrinsics
        #parent_folder = self._get_parent_folder(self.paths[idx])
        base_folder_str = self._get_base_folder(self.paths[idx])
        split_type_str = self._get_split_type(self.paths[idx])
        seq_name_str = self._get_sequence_name(self.paths[idx])
        camera_str = self._get_camera_name(self.paths[idx])
        calib_identifier = base_folder_str + split_type_str + seq_name_str + camera_str
        #current_folder = self._get_current_folder(self.paths[idx])
        if calib_identifier in self.calibration_cache:
            c_data = self.calibration_cache[calib_identifier]
        else:
            c_data = self._read_raw_calib_files(base_folder_str, split_type_str, seq_name_str, [camera_str])
            self.calibration_cache[calib_identifier] = c_data

        camera_type = self._get_camera_type(self.paths[idx], c_data)
        camera_type_int = self._get_camera_type_int(camera_type)
        poly_coeffs, principal_point, scale_factors, K, k, p = self._get_full_intrinsics(self.paths[idx], c_data)

        sample.update({
            'camera_type': camera_type_int,
            'intrinsics_poly_coeffs': poly_coeffs,
            'intrinsics_principal_point': principal_point,
            'intrinsics_scale_factors': scale_factors,
            'intrinsics_K': K,
            'intrinsics_k': k,
            'intrinsics_p': p,
            'path_to_ego_mask': self._get_path_to_ego_mask(self.paths[idx]),
        })

        # sample.update({
        #     'path_to_theta_lut': self._get_path_to_theta_lut(self.paths[idx]),
        # })

        if self.with_geometric_context:
            sample.update({
                'pose_matrix': self._get_extrinsics_pose_matrix(self.paths[idx], c_data),
            })

        if self.with_geometric_context and self.with_spatiotemp_context:
            sample.update({
                'with_spatiotemp_context': 1,
            })
        # # Add pose information if requested
        # if self.with_pose:
        #     sample.update({
        #         'pose': self._get_pose(self.paths[idx]),
        #     })

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': self._read_depth(self._get_depth_file(self.paths[idx])),
            })

        # Add context information if requested
        if self.with_context:
            # Add context images

            # 1. TEMPORAL CONTEXT
            image_context_paths_backward, _ = \
                self._get_context_files(self.paths[idx], self.backward_context_paths[idx])
            image_context_paths_forward, _ = \
                self._get_context_files(self.paths[idx], self.forward_context_paths[idx])

            image_temporal_context_paths = image_context_paths_backward + image_context_paths_forward
            n_temporal_context = len(image_temporal_context_paths)
            image_temporal_context = [load_convert_image(f) for f in image_temporal_context_paths]

            sample.update({
                'rgb_temporal_context': image_temporal_context,
            })

            # 2. GEOMETRIC CONTEXT
            if self.with_geometric_context:
                image_context_paths_geometric_context = self.paths_geometric_context[idx]
                n_geometric_context = len(image_context_paths_geometric_context)
                base_folder_str_geometric_context = [
                    self._get_base_folder(context_path) for context_path in image_context_paths_geometric_context
                ]
                split_type_str_geometric_context = [
                    self._get_split_type(context_path) for context_path in image_context_paths_geometric_context
                ]
                seq_name_str_geometric_context = [
                    self._get_sequence_name(context_path) for context_path in image_context_paths_geometric_context
                ]
                camera_str_geometric_context = [
                    self._get_camera_name(context_path) for context_path in image_context_paths_geometric_context
                ]
                calib_identifier_geometric_context = [
                    base_folder_str + split_type_str + seq_name_str + camera_str
                    for base_folder_str, split_type_str, seq_name_str, camera_str
                    in zip(base_folder_str_geometric_context, 
                           split_type_str_geometric_context, 
                           seq_name_str_geometric_context, 
                           camera_str_geometric_context)
                ]
                c_data_geometric_context = []
                for i_context in range(n_geometric_context):
                    if calib_identifier_geometric_context[i_context] in self.calibration_cache:
                        c_data_geometric_context.append(
                            self.calibration_cache[calib_identifier_geometric_context[i_context]]
                        )
                    else:
                        c_data_tmp = self._read_raw_calib_files(base_folder_str_geometric_context[i_context],
                                                                split_type_str_geometric_context[i_context],
                                                                seq_name_str_geometric_context[i_context],
                                                                [camera_str_geometric_context[i_context]])
                        c_data_geometric_context.append(c_data_tmp)
                        self.calibration_cache[calib_identifier_geometric_context[i_context]] = c_data_tmp
                camera_type_geometric_context = [
                    self._get_camera_type(image_context_paths_geometric_context[i_context], 
                                          c_data_geometric_context[i_context])
                    for i_context in range(n_geometric_context)
                ]
                camera_type_geometric_context_int = [
                    self._get_camera_type_int(camera_type_geometric_context[i_context])
                    for i_context in range(n_geometric_context)
                ]
                poly_coeffs_geometric_context = []
                principal_point_geometric_context = []
                scale_factors_geometric_context = []
                K_geometric_context = []
                k_geometric_context = []
                p_geometric_context = []
                for i_context in range(n_geometric_context):
                    poly_coeffs_tmp, principal_point_tmp, scale_factors_tmp, K_tmp, k_tmp, p_tmp = \
                        self._get_full_intrinsics(image_context_paths_geometric_context[i_context], 
                                                  c_data_geometric_context[i_context])
                    poly_coeffs_geometric_context.append(poly_coeffs_tmp)
                    principal_point_geometric_context.append(principal_point_tmp)
                    scale_factors_geometric_context.append(scale_factors_tmp)
                    K_geometric_context.append(K_tmp)
                    k_geometric_context.append(k_tmp)
                    p_geometric_context.append(p_tmp)
                path_to_ego_mask_geometric_context = [
                    self._get_path_to_ego_mask(context_path) 
                    for context_path in image_context_paths_geometric_context
                ]
                absolute_pose_matrix_geometric_context = [
                    self._get_extrinsics_pose_matrix(image_context_paths_geometric_context[i_context], 
                                                     c_data_geometric_context[i_context])
                    for i_context in range(n_geometric_context)
                ]
                relative_pose_matrix_geometric_context = [
                    (absolute_context_pose @ invert_pose_numpy(sample['pose_matrix'])).astype(np.float32)
                    for absolute_context_pose in absolute_pose_matrix_geometric_context
                ]

                image_geometric_context = [load_convert_image(f) for f in image_context_paths_geometric_context]

                # must fill with dummy values
                for i_context in range(n_geometric_context, self.max_geometric_context):
                    image_geometric_context.append(Image.new('RGB', (1280, 800)))
                    camera_type_geometric_context_int.append(2)
                    K_tmp, k_tmp, p_tmp = self._get_null_intrinsics_distorted()
                    poly_coeffs_tmp, principal_point_tmp, scale_factors_tmp = self._get_null_intrinsics_fisheye()
                    poly_coeffs_geometric_context.append(poly_coeffs_tmp)
                    principal_point_geometric_context.append(principal_point_tmp)
                    scale_factors_geometric_context.append(scale_factors_tmp)
                    K_geometric_context.append(K_tmp)
                    k_geometric_context.append(k_tmp)
                    p_geometric_context.append(p_tmp)
                    path_to_ego_mask_geometric_context.append('')
                    relative_pose_matrix_geometric_context.append(np.eye(4).astype(np.float32))
                    absolute_pose_matrix_geometric_context.append(np.eye(4).astype(np.float32))

                camera_type_geometric_context_int = np.array(camera_type_geometric_context_int)

                sample.update({
                    'rgb_geometric_context': image_geometric_context,
                    'camera_type_geometric_context': camera_type_geometric_context_int,
                    'intrinsics_poly_coeffs_geometric_context': poly_coeffs_geometric_context,
                    'intrinsics_principal_point_geometric_context': principal_point_geometric_context,
                    'intrinsics_scale_factors_geometric_context': scale_factors_geometric_context,
                    'intrinsics_K_geometric_context': K_geometric_context,
                    'intrinsics_k_geometric_context': k_geometric_context,
                    'intrinsics_p_geometric_context': p_geometric_context,
                    'path_to_ego_mask_geometric_context': path_to_ego_mask_geometric_context,
                    'pose_matrix_geometric_context': relative_pose_matrix_geometric_context,
                    'pose_matrix_geometric_context_absolute': absolute_pose_matrix_geometric_context,
                })
            else:
                sample.update({
                    'rgb_geometric_context': [],
                    'camera_type_geometric_context': [],
                    'intrinsics_poly_coeffs_geometric_context': [],
                    'intrinsics_principal_point_geometric_context': [],
                    'intrinsics_scale_factors_geometric_context': [],
                    'intrinsics_K_geometric_context': [],
                    'intrinsics_k_geometric_context': [],
                    'intrinsics_p_geometric_context': [],
                    'path_to_ego_mask_geometric_context': [],
                    'pose_matrix_geometric_context': [],
                    'pose_matrix_geometric_context_absolute': [],

                    'rgb_geometric_context_temporal_context': [],
                })

            # 3. GEOMETRIC-TEMPORAL CONTEXT
            if self.with_geometric_context and self.with_spatiotemp_context:
                # Backward
                image_context_paths_geometric_context_backward_nested = \
                    self.backward_context_paths_geometric_context[idx]
                # Forward
                image_context_paths_geometric_context_forward_nested = \
                    self.forward_context_paths_geometric_context[idx]
                image_geometric_context_temporal_context_paths_nested = [
                    b + f for b, f in zip(image_context_paths_geometric_context_backward_nested,
                                          image_context_paths_geometric_context_forward_nested)
                ]
                image_geometric_context_temporal_context_paths = [
                    item for sublist in image_geometric_context_temporal_context_paths_nested for item in sublist
                ]
                n_spatiotemp_context = len(image_geometric_context_temporal_context_paths)
                image_geometric_context_temporal_context = [
                    load_convert_image(f) for f in image_geometric_context_temporal_context_paths
                ]
                # must fill with dummy values
                for i_context in range(n_geometric_context, self.max_geometric_context):
                    for j in range(n_temporal_context):
                        image_geometric_context_temporal_context.append(Image.new('RGB', (1280, 800)))

                sample.update({
                    'rgb_geometric_context_temporal_context': image_geometric_context_temporal_context,
                })
            else:
                sample.update({
                    'rgb_geometric_context_temporal_context':  [],
                })

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        # Return sample
        return sample

########################################################################################################################
