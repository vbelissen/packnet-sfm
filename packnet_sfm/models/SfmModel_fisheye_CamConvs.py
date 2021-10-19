# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random
import torch.nn as nn
from packnet_sfm.utils.image import interpolate_scales
from packnet_sfm.utils.image_valeo import flip_model_and_cam_features
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.utils.misc import make_list

import torch
import numpy as np
from packnet_sfm.utils.image_valeo import image_grid, centered_2d_grid, meshgrid

class SfmModel_fisheye_CamConvs(nn.Module):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 upsample_depth_maps=False, **kwargs):
        super().__init__()
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps
        self._logs = {}
        self._losses = {}

        self._network_requirements = {
                'depth_net': True,  # Depth network required
                'pose_net': True,   # Pose network required
            }
        self._train_requirements = {
                'gt_depth': False,  # No ground-truth depth required
                'gt_pose': False,   # No ground-truth pose required
            }

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def losses(self):
        """Return metrics."""
        return self._losses

    def add_loss(self, key, val):
        """Add a new loss to the dictionary and detaches it."""
        self._losses[key] = val.detach()

    @property
    def network_requirements(self):
        """
        Networks required to run the model

        Returns
        -------
        requirements : dict
            depth_net : bool
                Whether a depth network is required by the model
            pose_net : bool
                Whether a depth network is required by the model
        """
        return self._network_requirements

    @property
    def train_requirements(self):
        """
        Information required by the model at training stage

        Returns
        -------
        requirements : dict
            gt_depth : bool
                Whether ground truth depth is required by the model at training time
            gt_pose : bool
                Whether ground truth pose is required by the model at training time
        """
        return self._train_requirements

    def add_depth_net(self, depth_net):
        """Add a depth network to the model"""
        self.depth_net = depth_net

    def add_pose_net(self, pose_net):
        """Add a pose network to the model"""
        self.pose_net = pose_net

    def compute_inv_depths(self, image):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flip_lr = random.random() < self.flip_lr_prob if self.training else False
        inv_depths = make_list(flip_model(self.depth_net, image, flip_lr))
        # If upsampling depth maps
        if self.upsample_depth_maps:
            inv_depths = interpolate_scales(
                inv_depths, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return inv_depths

    def compute_inv_depths_with_cam(self, image, cam_features):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flip_lr = random.random() < self.flip_lr_prob if self.training else False
        inv_depths = make_list(flip_model_and_cam_features(self.depth_net, image, cam_features, flip_lr))
        # If upsampling depth maps
        if self.upsample_depth_maps:
            inv_depths = interpolate_scales(
                inv_depths, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return inv_depths

    def compute_poses(self, image, contexts):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(image, contexts)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def compute_poses_with_cam(self, image, cam_features, contexts, context_cam_features):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(image, cam_features, contexts, context_cam_features)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def forward(self, batch, return_logs=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored

        Returns
        -------
        output : dict
            Dictionary containing predicted inverse depth maps and poses
        """
        # Generate inverse depth predictions

        B, _, _, _ = batch['rgb'].shape
        device = batch['rgb'].get_device()
        H = 800
        W = 1280
        theta_tensor = torch.zeros(B, 1, H, W).float().to(device)
        for b in range(B):
            theta_tensor[b, 0] = torch.from_numpy(np.load(batch['path_to_theta_lut'][b]))
        yi, xi = centered_2d_grid(B, H, W, batch['rgb'].dtype, device,
                                  batch['intrinsics_principal_point'],
                                  batch['intrinsics_scale_factors'])
        target_cam_conv_features = torch.cat([theta_tensor, xi.float(), yi.float()], 1)

        inv_depths = self.compute_inv_depths_with_cam(batch['rgb'], target_cam_conv_features)
        # Generate pose predictions if available
        pose = None
        if 'rgb_context' in batch and self.pose_net is not None:
            n_context = len(batch['rgb_context'])
            ref_theta_tensor = []
            ref_yi, ref_xi = [], []
            ref_cam_conv_features = []
            for n in range(n_context):
                ref_theta_tensor.append(torch.zeros(B, 1, H, W).float().to(device))
                for b in range(B):
                    ref_theta_tensor[n][b, 0] = torch.from_numpy(np.load(batch['path_to_theta_lut_context'][n][b]))
                ref_yi_n, ref_xi_n = centered_2d_grid(B, H, W, batch['rgb'].dtype, device,
                                                      batch['intrinsics_principal_point_context'][n],
                                                      batch['intrinsics_scale_factors_context'][n])
                ref_xi.append(ref_xi_n.float())
                ref_yi.append(ref_yi_n.float())
                ref_cam_conv_features.append(torch.cat([ref_theta_tensor[n], ref_xi[n], ref_yi[n]], 1))

            pose = self.compute_poses_with_cam(batch['rgb'], 
                                               target_cam_conv_features, 
                                               batch['rgb_context'], 
                                               ref_cam_conv_features)
        # Return output dictionary
        return {
            'inv_depths': inv_depths,
            'poses': pose,
        }
