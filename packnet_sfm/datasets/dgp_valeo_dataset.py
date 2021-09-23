# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import torch
import numpy as np

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.camera import Camera, generate_depth_map
from dgp.utils.geometry import Pose
from packnet_sfm.geometry.pose_utils import invert_pose_numpy

from packnet_sfm.utils.misc import make_list
from packnet_sfm.utils.types import is_tensor, is_numpy, is_list, is_str


cam_left_dict = {
    '1': '5',
    '5': '7',
    '6': '1',
    '7': '9',
    '8': '6',
    '9': '8',
}
cam_right_dict = {
    '1': '6',
    '5': '1',
    '6': '8',
    '7': '5',
    '8': '9',
    '9': '7',
}

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        return sample[0]

    # Otherwise, stack sample
    stacked_sample = {}
    for key in sample[0]:
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx']:#['idx', 'dataset_idx', 'sensor_name', 'filename']:
            stacked_sample[key] = sample[0][key]
        else:
            # Stack torch tensors
            if is_str(sample[0][key]):
                stacked_sample[key] = [s[key] for s in sample]
            elif is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            # Stack numpy arrays
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            # Stack list
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                if is_str(sample[0][key][0]):
                    for i in range(len(sample)):
                        stacked_sample[key].append(sample[i][key])
                # Stack list of torch tensors
                if is_tensor(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            torch.stack([s[key][i] for s in sample], 0))
                # Stack list of numpy arrays
                if is_numpy(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            np.stack([s[key][i] for s in sample], 0))
    # Return stacked sample
    return stacked_sample

########################################################################################################################
#### DATASET
########################################################################################################################

class DGPvaleoDataset:
    """
    DGP dataset class

    Parameters
    ----------
    path : str
        Path to the dataset
    split : str {'train', 'val', 'test'}
        Which dataset split to use
    cameras : list of str
        Which cameras to get information from
    depth_type : str
        Which lidar will be used to generate ground-truth information
    with_pose : bool
        If enabled pose estimates are also returned
    with_semantic : bool
        If enabled semantic estimates are also returned
    back_context : int
        Size of the backward context
    forward_context : int
        Size of the forward context
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, path, split,
                 cameras=None,
                 depth_type=None,
                 with_pose=False,
                 with_semantic=False,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 with_geometric_context=False,
                 ):
        self.path = path
        self.split = split
        self.dataset_idx = 0

        self.bwd = back_context
        self.fwd = forward_context
        self.has_context = back_context + forward_context > 0
        self.with_geometric_context = with_geometric_context

        self.num_cameras = len(cameras)
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_semantic = with_semantic

        # arrange cameras alphabetically
        cameras = sorted(cameras)
        cameras_left = list(cameras)
        cameras_right = list(cameras)
        for i_cam in range(self.num_cameras):
            replaced = False
            for k in cam_left_dict:
                if not replaced and k in cameras_left[i_cam]:
                    cameras_left[i_cam] = cameras_left[i_cam].replace(k, cam_left_dict[k])
                    replaced = True
            replaced = False
            for k in cam_right_dict:
                if not replaced and k in cameras_right[i_cam]:
                    cameras_right[i_cam] = cameras_right[i_cam].replace(k, cam_right_dict[k])
                    replaced = True

        print(cameras)
        print(cameras_left)
        print(cameras_right)

        # arrange cameras left and right and extract sorting indices
        self.cameras_left_sort_idxs  = list(np.argsort(cameras_left))
        self.cameras_right_sort_idxs = list(np.argsort(cameras_right))

        cameras_left_sorted  = sorted(cameras_left)
        cameras_right_sorted = sorted(cameras_right)

        self.dataset = SynchronizedSceneDataset(path,
            split=split,
            datum_names=cameras,
            backward_context=back_context,
            forward_context=forward_context,
            requested_annotations=None,
            only_annotated_datums=False,
        )

        if self.with_geometric_context:
            self.dataset_left = SynchronizedSceneDataset(path,
                split=split,
                datum_names=cameras_left_sorted,
                backward_context=back_context,
                forward_context=forward_context,
                requested_annotations=None,
                only_annotated_datums=False,
            )

            self.dataset_right = SynchronizedSceneDataset(path,
                split=split,
                datum_names=cameras_right_sorted,
                backward_context=back_context,
                forward_context=forward_context,
                requested_annotations=None,
                only_annotated_datums=False,
            )

    @staticmethod
    def _get_base_folder(image_file):
        """The base folder"""
        return '/'.join(image_file.split('/')[:-4])

    @staticmethod
    def _get_sequence_name(image_file):
        """Returns a sequence name like '20180227_185324'."""
        return image_file.split('/')[-4]

    @staticmethod
    def _get_camera_name(image_file):
        """Returns 'cam_i', i between 0 and 4"""
        return image_file.split('/')[-2]

    def _get_path_to_ego_mask(self, image_file):
        """Get the current folder from image_file."""
        return os.path.join(self._get_base_folder(image_file),
                            self._get_sequence_name(image_file),
                            'semantic_masks',
                            self._get_camera_name(image_file) + '.npy')

    def generate_depth_map(self, sample_idx, datum_idx, filename):
        """
        Generates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        sample_idx : int
            sample index
        datum_idx : int
            Datum index
        filename :
            Filename used for loading / saving

        Returns
        -------
        depth : np.array [H, W]
            Depth map for that datum in that sample
        """
        # Generate depth filename
        filename = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(self.depth_type)))
        # Load and return if exists
        if os.path.exists(filename):
            return np.load(filename, allow_pickle=True)['depth']
        # Otherwise, create, save and return
        else:
            # Get pointcloud
            scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
            pc_datum_idx_in_sample = self.dataset.get_datum_index_for_datum_name(
                scene_idx, sample_idx_in_scene, self.depth_type)
            pc_datum_data = self.dataset.get_point_cloud_from_datum(
                scene_idx, sample_idx_in_scene, pc_datum_idx_in_sample)
            # Create camera
            camera_rgb = self.get_current('rgb', datum_idx)
            camera_pose = self.get_current('pose', datum_idx)
            camera_intrinsics = self.get_current('intrinsics', datum_idx)
            camera = Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())
            # Generate depth map
            world_points = pc_datum_data['pose'] * pc_datum_data['point_cloud']
            depth = generate_depth_map(camera, world_points, camera_rgb.size[::-1])
            # Save depth map
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
            # Return depth map
            return depth

    def get_current(self, key, sensor_idx):
        """Return current timestep of a key from a sensor"""
        return self.sample_dgp[self.bwd][sensor_idx][key]

    def get_current_left(self, key, sensor_idx):
        """Return current timestep of a key from a sensor"""
        return self.sample_dgp_left[self.bwd][sensor_idx][key]

    def get_current_right(self, key, sensor_idx):
        """Return current timestep of a key from a sensor"""
        return self.sample_dgp_right[self.bwd][sensor_idx][key]

    def get_backward(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(0, self.bwd)]

    def get_backward_left(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd == 0 else \
            [self.sample_dgp_left[i][sensor_idx][key] \
             for i in range(0, self.bwd)]

    def get_backward_right(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd == 0 else \
            [self.sample_dgp_right[i][sensor_idx][key] \
             for i in range(0, self.bwd)]

    def get_forward(self, key, sensor_idx):
        """Return forward timestep of a key from a sensor"""
        return [] if self.fwd == 0 else \
            [self.sample_dgp[i][sensor_idx][key] \
             for i in range(self.bwd + 1, self.bwd + self.fwd + 1)]

    def get_forward_left(self, key, sensor_idx):
        """Return forward timestep of a key from a sensor"""
        return [] if self.fwd == 0 else \
            [self.sample_dgp_left[i][sensor_idx][key] \
             for i in range(self.bwd + 1, self.bwd + self.fwd + 1)]

    def get_forward_right(self, key, sensor_idx):
        """Return forward timestep of a key from a sensor"""
        return [] if self.fwd == 0 else \
            [self.sample_dgp_right[i][sensor_idx][key] \
             for i in range(self.bwd + 1, self.bwd + self.fwd + 1)]

    def get_context(self, key, sensor_idx):
        """Get both backward and forward contexts"""
        return self.get_backward(key, sensor_idx) + self.get_forward(key, sensor_idx)

    def get_context_left(self, key, sensor_idx):
        """Get both backward and forward contexts"""
        return self.get_backward_left(key, sensor_idx) + self.get_forward_left(key, sensor_idx)

    def get_context_right(self, key, sensor_idx):
        """Get both backward and forward contexts"""
        return self.get_backward_right(key, sensor_idx) + self.get_forward_right(key, sensor_idx)

    def get_filename(self, sample_idx, datum_idx):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : int
            Sample index
        datum_idx : int
            Datum index

        Returns
        -------
        filename : str
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.get_scene_directory(scene_idx)
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene, datum_indices[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def get_filename_left(self, sample_idx, datum_idx):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : int
            Sample index
        datum_idx : int
            Datum index

        Returns
        -------
        filename : str
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset_left.dataset_item_index[sample_idx]
        scene_dir = self.dataset_left.get_scene_directory(scene_idx)
        filename = self.dataset_left.get_datum(
            scene_idx, sample_idx_in_scene, datum_indices[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def get_filename_right(self, sample_idx, datum_idx):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : int
            Sample index
        datum_idx : int
            Datum index

        Returns
        -------
        filename : str
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset_right.dataset_item_index[sample_idx]
        scene_dir = self.dataset_right.get_scene_directory(scene_idx)
        filename = self.dataset_right.get_datum(
            scene_idx, sample_idx_in_scene, datum_indices[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def get_camera_idx_left(self, camera_idx):
        return self.cameras_left_sort_idxs[camera_idx]

    def get_camera_idx_right(self, camera_idx):
        return self.cameras_right_sort_idxs[camera_idx]

    def __len__(self):
        """Length of dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a dataset sample"""
        # Get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]
        if self.with_geometric_context:
            self.sample_dgp_left = self.dataset_left[idx]
            self.sample_dgp_left = [make_list(sample) for sample in self.sample_dgp_left]
            self.sample_dgp_right = self.dataset_right[idx]
            self.sample_dgp_right = [make_list(sample) for sample in self.sample_dgp_right]

        # print('self.sample_dgp :')
        # print(self.sample_dgp)
        # print('self.sample_dgp_left :')
        # print(self.sample_dgp_left)
        # print('self.sample_dgp_right :')
        # print(self.sample_dgp_right)


        # Loop over all cameras
        sample = []
        for i in range(self.num_cameras):
            i_left = self.get_camera_idx_left(i)
            i_right = self.get_camera_idx_right(i)

            # print(self.get_current('datum_name', i))
            # print(self.get_filename(idx, i))
            # print(self.get_current('intrinsics', i))
            # print(self.with_depth)
            data = {
                'idx': idx,
                'dataset_idx': self.dataset_idx,
                'sensor_name': self.get_current('datum_name', i),
                #
                'filename': self.get_filename(idx, i),
                'splitname': '%s_%010d' % (self.split, idx),
                #
                'rgb': self.get_current('rgb', i),
                'intrinsics': self.get_current('intrinsics', i),
                'extrinsics': self.get_current('extrinsics', i).matrix,
                'path_to_ego_mask': os.path.join(os.path.dirname(self.path), self._get_path_to_ego_mask(self.get_filename(idx, i))),
            }

            # If depth is returned
            if self.with_depth:
                data.update({
                    'depth': self.generate_depth_map(idx, i, data['filename'])
                })

            # If pose is returned
            if self.with_pose:
                data.update({
                    'pose': self.get_current('pose', i).matrix,
                })

            if self.has_context:
                orig_extrinsics = Pose.from_matrix(data['extrinsics'])
                data.update({
                    'rgb_context': self.get_context('rgb', i),
                    'intrinsics_context': self.get_context('intrinsics', i),
                    'extrinsics_context':
                        [(extrinsics.inverse() * orig_extrinsics).matrix
                         for extrinsics in self.get_context('extrinsics', i)],

                })
                data.update({
                    'path_to_ego_mask_context': [os.path.join(os.path.dirname(self.path), self._get_path_to_ego_mask(self.get_filename(idx, i)))
                                                 for _ in range(len(data['rgb_context']))],
                })
                data.update({
                    'context_type': [],
                })
                for _ in range(self.bwd):
                    data['context_type'].append('backward')

                for _ in range(self.fwd):
                    data['context_type'].append('forward')

                # If context pose is returned
                if self.with_pose:
                    # Get original values to calculate relative motion
                    orig_pose = Pose.from_matrix(data['pose'])
                    data.update({
                        'pose_context':
                            [(orig_pose.inverse() * pose).matrix
                             for pose in self.get_context('pose', i)],
                    })

            if self.with_geometric_context:
                orig_extrinsics       = data['extrinsics']

                orig_extrinsics_left  = self.get_current_left('extrinsics', i_left).matrix
                orig_extrinsics_right = self.get_current_right('extrinsics', i_right).matrix

                data['rgb_context'].append(self.get_current_left('rgb', i_left))
                data['rgb_context'].append(self.get_current_right('rgb', i_right))

                data['intrinsics_context'].append(self.get_current_left('intrinsics', i_left))
                data['intrinsics_context'].append(self.get_current_right('intrinsics', i_right))

                data['extrinsics_context'].append((invert_pose_numpy(orig_extrinsics_left) @ orig_extrinsics))
                data['extrinsics_context'].append((invert_pose_numpy(orig_extrinsics_right) @ orig_extrinsics))

                #data['extrinsics_context'].append((orig_extrinsics.inverse() * orig_extrinsics_left).matrix)
                #data['extrinsics_context'].append((orig_extrinsics.inverse() * orig_extrinsics_right).matrix)

                data['path_to_ego_mask_context'].append(os.path.join(os.path.dirname(self.path),
                                                                     self._get_path_to_ego_mask(self.get_filename_left(idx, i_left))))
                data['path_to_ego_mask_context'].append(os.path.join(os.path.dirname(self.path),
                                                                     self._get_path_to_ego_mask(self.get_filename_right(idx, i_right))))

                data['context_type'].append('left')
                data['context_type'].append('right')

                data.update({
                    'sensor_name_left': self.get_current_left('datum_name', i_left),
                    'sensor_name_right': self.get_current_right('datum_name', i_right),
                    #
                    'filename_left': self.get_filename_left(idx, i_left),
                    'filename_right': self.get_filename_right(idx, i_right),
                    #
                    #'rgb_left': self.get_current_left('rgb', i),
                    #'rgb_right': self.get_current_right('rgb', i),
                    #'intrinsics_left': self.get_current_left('intrinsics', i),
                    #'intrinsics_right': self.get_current_right('intrinsics', i),
                    #'extrinsics_left': self.get_current_left('extrinsics', i).matrix,
                    #'extrinsics_right': self.get_current_right('extrinsics', i).matrix,
                    #'path_to_ego_mask_left': self._get_path_to_ego_mask(self.get_filename_left(idx, i)),
                    #'path_to_ego_mask_right': self._get_path_to_ego_mask(self.get_filename_right(idx, i)),
                })

                # data.update({
                #     'extrinsics_context_left':
                #         [(orig_extrinsics_left.inverse() * extrinsics_left).matrix
                #          for extrinsics_left in self.get_context_left('extrinsics', i)],
                #     'extrinsics_context_right':
                #         [(orig_extrinsics_right.inverse() * extrinsics_right).matrix
                #          for extrinsics_right in self.get_context_right('extrinsics', i)],
                #     'intrinsics_context_left': self.get_context_left('intrinsics', i),
                #     'intrinsics_context_right': self.get_context_right('intrinsics', i),
                # })

            sample.append(data)

        # Apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]

        # Return sample (stacked if necessary)
        return stack_sample(sample)

########################################################################################################################

