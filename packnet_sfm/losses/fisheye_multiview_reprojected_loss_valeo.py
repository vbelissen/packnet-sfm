# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import numpy as np

from packnet_sfm.utils.image import match_scales
from packnet_sfm.geometry.camera_fisheye_valeo import CameraFisheye
from packnet_sfm.geometry.camera_fisheye_valeo_utils import view_synthesis
from packnet_sfm.utils.depth import calc_smoothness, inv2depth
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling

########################################################################################################################


class ReprojectedLoss(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scalesto consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_scales=4,
                 progressive_scaling=0.0, mask_ego=True, **kwargs):
        super().__init__()
        self.n = num_scales
        self.progressive_scaling = progressive_scaling
        self.mask_ego = mask_ego
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

        # Asserts
########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

########################################################################################################################

    def warp_target_pixels(self, gt_depths, depths,
                           path_to_theta_lut,     path_to_ego_mask,     poly_coeffs,     principal_point,     scale_factors,
                           ref_path_to_theta_lut, ref_path_to_ego_mask, ref_poly_coeffs, ref_principal_point, ref_scale_factors,
                           pose):

        B, _, H, W = depths[0].shape
        device     = depths[0].get_device()

        # Generate cameras for all scales
        cams, ref_cams = [], []
        target_pixels_gt_warped = []
        target_pixels_warped    = []
        for i in range(self.n):
            _, _, DH, DW = depths[i].shape
            scale_factor = DW / float(W)
            cam = CameraFisheye(path_to_theta_lut=path_to_theta_lut,
                                      path_to_ego_mask=path_to_ego_mask,
                                      poly_coeffs=poly_coeffs.float(),
                                      principal_point=principal_point.float(),
                                      scale_factors=scale_factors.float()).scaled(scale_factor).to(device)
            ref_cam = CameraFisheye(path_to_theta_lut=ref_path_to_theta_lut,
                                          path_to_ego_mask=ref_path_to_ego_mask,
                                          poly_coeffs=ref_poly_coeffs.float(),
                                          principal_point=ref_principal_point.float(),
                                          scale_factors=ref_scale_factors.float(), Tcw=pose).scaled(scale_factor).to(device)

            world_points    = cam.reconstruct(depths[i],    frame='w')
            world_points_gt = cam.reconstruct(gt_depths[i], frame='w')
            # Project world points onto reference camera
            ref_coords    = ref_cam.project(world_points,    frame='w') # [B, DH, DW, 2]
            ref_coords_gt = ref_cam.project(world_points_gt, frame='w') # [B, DH, DW, 2]

            target_pixels_gt_warped.append(ref_coords_gt.transpose(2,3).transpose(1,2)) # [B, 2, DH, DW]
            target_pixels_warped.append(ref_coords.transpose(2,3).transpose(1,2))

        return target_pixels_gt_warped, target_pixels_warped

########################################################################################################################


########################################################################################

    def forward(self, gt_depth, depths,
                path_to_theta_lut,     path_to_ego_mask,     poly_coeffs,     principal_point,     scale_factors,
                ref_path_to_theta_lut, ref_path_to_ego_mask, ref_poly_coeffs, ref_principal_point, ref_scale_factors, # ALL LISTS !!!
                poses, return_logs=False, progress=0.0):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        context : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        poses : list of Pose
            Camera transformation between original and context
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Loop over all reference images
        reprojected_losses = [[] for _ in range(self.n)]
        gt_depths = match_scales(gt_depth, depths, self.n)
        if self.mask_ego:
            device = gt_depth.get_device()
            B = len(path_to_ego_mask)
            ego_mask_tensor     = torch.zeros(B, 1, 800, 1280)
            for b in range(B):
                ego_mask_tensor[b, 0]     = torch.from_numpy(np.load(path_to_ego_mask[b])).float()
            ego_mask_tensors     = []  # = torch.zeros(B, 1, 800, 1280)
            for i in range(self.n):
                B, C, H, W = gt_depths[i].shape
                if W < 1280:
                    inv_scale_factor = int(1280 / W)
                    ego_mask_tensors.append(-nn.MaxPool2d(inv_scale_factor, inv_scale_factor)(-ego_mask_tensor).to(device))
                else:
                    ego_mask_tensors.append(ego_mask_tensor.to(device))

            gt_depths = [a * b for a, b in zip(gt_depths, ego_mask_tensors)]

        gt_depths_mask = [(gt_depths[i] > 0.).detach() > 0 for i in range(self.n)]

        for j, pose in enumerate(poses):
            # Calculate warped images
            target_pixels_gt_warped, target_pixels_warped \
                = self.warp_target_pixels(gt_depths, depths,
                                          path_to_theta_lut,        path_to_ego_mask,        poly_coeffs,        principal_point,        scale_factors,
                                          ref_path_to_theta_lut[j], ref_path_to_ego_mask[j], ref_poly_coeffs[j], ref_principal_point[j], ref_scale_factors[j],
                                          pose)

            for i in range(self.n):
                X_gt = target_pixels_gt_warped[i][:, 0, :, :].unsqueeze(1)[gt_depths_mask[i]]
                Y_gt = target_pixels_gt_warped[i][:, 1, :, :].unsqueeze(1)[gt_depths_mask[i]]

                X = target_pixels_warped[i][:, 0, :, :].unsqueeze(1)[gt_depths_mask[i]]
                Y = target_pixels_warped[i][:, 1, :, :].unsqueeze(1)[gt_depths_mask[i]]

                inside_of_bounds_mask = torch.logical_not(((X_gt > 1) + (X_gt < -1) + (Y_gt > 1) + (Y_gt < -1) + (X > 1) + (X < -1) + (Y > 1) + (Y < -1))).detach()

                X_gt = X_gt[inside_of_bounds_mask]
                Y_gt = Y_gt[inside_of_bounds_mask]
                X    = X[inside_of_bounds_mask]
                Y    = Y[inside_of_bounds_mask]

                pixels_gt = torch.stack([X_gt, Y_gt]).view(2, -1).transpose(0,1)
                pixels    = torch.stack([   X,    Y]).view(2, -1).transpose(0, 1)

                reprojected_loss = torch.sqrt(torch.mean((pixels_gt - pixels) ** 2))
                reprojected_losses[i].append(reprojected_loss)

        loss = sum([sum([l.mean() for l in reprojected_losses[i]]) / len(reprojected_losses[i]) for i in range(self.n)]) / self.n
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################
