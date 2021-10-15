# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as funct


from packnet_sfm.utils.image import match_scales
from packnet_sfm.geometry.camera_fisheye_valeo import CameraFisheye
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_distorted_valeo import CameraDistorted
from packnet_sfm.geometry.camera_multifocal_valeo import CameraMultifocal
from packnet_sfm.geometry.camera_utils import view_synthesis
from packnet_sfm.utils.depth import calc_smoothness, inv2depth, depth2inv
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling
from packnet_sfm.utils.types import is_list
from packnet_sfm.utils.image import interpolate_image

import numpy as np
import time
import cv2

torch.autograd.set_detect_anomaly(True)

########################################################################################################################

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

########################################################################################################################

class MultiViewPhotometricLoss(LossBase):
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
    def __init__(self, num_scales=4, ssim_loss_weight=0.85, occ_reg_weight=0.1, smooth_loss_weight=0.1,
                 C1=1e-4, C2=9e-4, photometric_reduce_op='mean', disp_norm=True, clip_loss=0.5, mask_ego=True,
                 progressive_scaling=0.0, padding_mode='zeros',
                 automask_loss=False, **kwargs):
        super().__init__()
        self.n = num_scales
        self.progressive_scaling = progressive_scaling
        self.ssim_loss_weight = ssim_loss_weight
        self.occ_reg_weight = occ_reg_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss
        self.mask_ego = mask_ego
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

########################################################################################################################

    def warp_ref_image(self,
                       inv_depths,
                       camera_type,
                       intrinsics_poly_coeffs,
                       intrinsics_principal_point,
                       intrinsics_scale_factors,
                       intrinsics_K,
                       intrinsics_k,
                       intrinsics_p,
                       ref_image,
                       ref_pose,
                       ref_ego_mask_tensors,
                       ref_camera_type,
                       ref_intrinsics_poly_coeffs,
                       ref_intrinsics_principal_point,
                       ref_intrinsics_scale_factors,
                       ref_intrinsics_K,
                       ref_intrinsics_k,
                       ref_intrinsics_p):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(
                CameraMultifocal(intrinsics_poly_coeffs,
                                 intrinsics_principal_point,
                                 intrinsics_scale_factors,
                                 intrinsics_K,
                                 intrinsics_k[:, 0],
                                 intrinsics_k[:, 1],
                                 intrinsics_k[:, 2],
                                 intrinsics_p[:, 0],
                                 intrinsics_p[:, 1],
                                 camera_type,
                                 Tcw=None).scaled(scale_factor).to(device)
            )
            ref_cams.append(
                CameraMultifocal(ref_intrinsics_poly_coeffs,
                                 ref_intrinsics_principal_point,
                                 ref_intrinsics_scale_factors,
                                 ref_intrinsics_K,
                                 ref_intrinsics_k[:, 0],
                                 ref_intrinsics_k[:, 1],
                                 ref_intrinsics_k[:, 2],
                                 ref_intrinsics_p[:, 0],
                                 ref_intrinsics_p[:, 1],
                                 ref_camera_type,
                                 Tcw=ref_pose).scaled(scale_factor).to(device)
            )
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [
            view_synthesis(ref_images[i],
                           depths[i],
                           ref_cams[i],
                           cams[i],
                           padding_mode=self.padding_mode)
            for i in range(self.n)
        ]

        ref_tensors_warped = [
            view_synthesis(ref_ego_mask_tensors[i],
                           depths[i],
                           ref_cams[i],
                           cams[i],
                           padding_mode=self.padding_mode,
                           mode='nearest')
            for i in range(self.n)
        ]
        # Return warped reference image
        return ref_warped, ref_tensors_warped

    # def warp_ref_image(self,
    #                    inv_depths,
    #                    camera_type,
    #                    intrinsics_poly_coeffs,
    #                    intrinsics_principal_point,
    #                    intrinsics_scale_factors,
    #                    intrinsics_K,
    #                    intrinsics_k,
    #                    intrinsics_p,
    #                    ref_image,
    #                    ref_pose,
    #                    ref_ego_mask_tensors,
    #                    ref_camera_type,
    #                    ref_intrinsics_poly_coeffs,
    #                    ref_intrinsics_principal_point,
    #                    ref_intrinsics_scale_factors,
    #                    ref_intrinsics_K,
    #                    ref_intrinsics_k,
    #                    ref_intrinsics_p):
    #     """
    #     Warps a reference image to produce a reconstruction of the original one.
    #
    #     Parameters
    #     ----------
    #     inv_depths : torch.Tensor [B,1,H,W]
    #         Inverse depth map of the original image
    #     ref_image : torch.Tensor [B,3,H,W]
    #         Reference RGB image
    #     K : torch.Tensor [B,3,3]
    #         Original camera intrinsics
    #     ref_K : torch.Tensor [B,3,3]
    #         Reference camera intrinsics
    #     pose : Pose
    #         Original -> Reference camera transformation
    #
    #     Returns
    #     -------
    #     ref_warped : torch.Tensor [B,3,H,W]
    #         Warped reference image (reconstructing the original one)
    #     """
    #     B, C, H, W = ref_image.shape
    #     device = ref_image.get_device()
    #     # Generate cameras for all scales
    #     # View synthesis
    #     depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
    #     ref_images = match_scales(ref_image, inv_depths, self.n)
    #     cams, ref_cams = [[] for i in range(self.n)], [[] for i in range(self.n)]
    #     world_points = []
    #     ref_coords = []
    #     for i in range(self.n):
    #         _, _, DH, DW = inv_depths[i].shape
    #         scale_factor = DW / float(W)
    #         world_points.append(torch.zeros(B, 3, DH, DW).to(device))
    #         ref_coords.append(torch.zeros(B, DH, DW, 2).to(device))
    #         for b in range(B):
    #             if camera_type[b] == 'fisheye':
    #                 cams[i].append(
    #                     CameraFisheye(path_to_theta_lut='',
    #                                   path_to_ego_mask='',
    #                                   poly_coeffs=intrinsics_poly_coeffs[b].unsqueeze(0).float(),
    #                                   principal_point=intrinsics_principal_point[b].unsqueeze(0).float(),
    #                                   scale_factors=intrinsics_scale_factors[b].unsqueeze(0).float())
    #                         .scaled(scale_factor).to(device)
    #                 )
    #             elif camera_type[b] == 'perspective':
    #                 cams[i].append(
    #                     CameraDistorted(K=intrinsics_K[b].unsqueeze(0).float(),
    #                                     k1=intrinsics_k[b, 0].unsqueeze(0),
    #                                     k2=intrinsics_k[b, 1].unsqueeze(0),
    #                                     k3=intrinsics_k[b, 2].unsqueeze(0),
    #                                     p1=intrinsics_p[b, 0].unsqueeze(0),
    #                                     p2=intrinsics_p[b, 1].unsqueeze(0))
    #                         .scaled(scale_factor).to(device)
    #                 )
    #             if ref_camera_type[b] == 'fisheye':
    #                 ref_cams[i].append(
    #                     CameraFisheye(path_to_theta_lut='',
    #                                   path_to_ego_mask='',
    #                                   poly_coeffs=ref_intrinsics_poly_coeffs[b].unsqueeze(0).float(),
    #                                   principal_point=ref_intrinsics_principal_point[b].unsqueeze(0).float(),
    #                                   scale_factors=ref_intrinsics_scale_factors[b].unsqueeze(0).float(),
    #                                   Tcw=Pose(ref_pose.mat[b].unsqueeze(0)))
    #                         .scaled(scale_factor).to(device)
    #                 )
    #             elif ref_camera_type[b] == 'perspective':
    #                 ref_cams[i].append(
    #                     CameraDistorted(K=ref_intrinsics_K[b].unsqueeze(0).float(),
    #                                     k1=ref_intrinsics_k[b, 0].unsqueeze(0),
    #                                     k2=ref_intrinsics_k[b, 1].unsqueeze(0),
    #                                     k3=ref_intrinsics_k[b, 2].unsqueeze(0),
    #                                     p1=ref_intrinsics_p[b, 0].unsqueeze(0),
    #                                     p2=ref_intrinsics_p[b, 1].unsqueeze(0),
    #                                     Tcw=Pose(ref_pose.mat[b].unsqueeze(0)))
    #                         .scaled(scale_factor).to(device)
    #                 )
    #
    #             if ref_camera_type[b] != 'dummy':
    #                 world_points[i][b] = cams[i][b].reconstruct(depths[i][b].unsqueeze(0), frame='w')
    #                 ref_coords[i][b] = ref_cams[i][b].project(world_points[i][b].unsqueeze(0), frame='w')
    #
    #     ref_warped = [
    #         funct.grid_sample(ref_images[i],
    #                           ref_coords[i],
    #                           mode='bilinear',
    #                           padding_mode=self.padding_mode,
    #                           align_corners=True)
    #         for i in range(self.n)
    #     ]
    #     ref_ego_mask_tensors_warped = [
    #         funct.grid_sample(ref_ego_mask_tensors[i],
    #                           ref_coords[i],
    #                           mode = 'nearest',
    #                           padding_mode=self.padding_mode, align_corners=None)
    #         for i in range(self.n)
    #     ]
    #
    #     return ref_warped, ref_ego_mask_tensors_warped

########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                return torch.cat(losses, 1).min(1, True)[0].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        self.add_metric('photometric_loss', photometric_loss)
        return photometric_loss

    def nonzero_reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                # each image gets the same weight in the loss,
                # regardless of the number of masked pixels
                nonzero_mean = []
                nonzero_losses = 0
                for l in losses:
                    mask = l!=0
                    s = mask.sum()
                    if s > 0:
                        nonzero_mean.append((l*mask).sum()/s)
                        nonzero_losses += 1
                if nonzero_losses > 0:
                    return sum(nonzero_mean) / nonzero_losses
                else:
                    return 0
            if self.photometric_reduce_op == 'weightedMean':
                # each image gets the a different weight in the loss,
                # depending on the number of masked pixels
                nonzero_mean = []
                nonzero_pixels = 0
                for l in losses:
                    mask = l!=0
                    s = mask.sum()
                    if s > 0:
                        nonzero_mean.append((l*mask).sum())
                        nonzero_pixels += s
                if nonzero_pixels > 0:
                    return sum(nonzero_mean) / nonzero_pixels
                else:
                    return 0
            elif self.photometric_reduce_op == 'min':
                C = torch.cat(losses,1)
                zero_pixels = (C.max(1,True)[0] == 0)
                C[C == 0] = 10000
                min_pixels = C.min(1, True)[0]
                min_pixels[zero_pixels] = 0
                return min_pixels.mean()
                # mask = min_pixels!=0
                # s = mask.sum()
                # if s > 0:
                #     return (min_pixels*mask).sum()/s
                # else:
                #     return 0
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        self.add_metric('photometric_loss', photometric_loss)
        return photometric_loss

########################################################################################################################

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)
        return smoothness_loss

########################################################################################################################

    def forward(self,
                image,
                ref_images_temporal_context,
                ref_images_geometric_context,
                ref_images_geometric_context_temporal_context,
                inv_depths,
                poses_temporal_context,
                poses_geometric_context,
                poses_geometric_context_temporal_context,
                camera_type,
                intrinsics_poly_coeffs,
                intrinsics_principal_point,
                intrinsics_scale_factors,
                intrinsics_K,
                intrinsics_k,
                intrinsics_p,
                path_to_ego_mask,
                camera_type_geometric_context,
                intrinsics_poly_coeffs_geometric_context,
                intrinsics_principal_point_geometric_context,
                intrinsics_scale_factors_geometric_context,
                intrinsics_K_geometric_context,
                intrinsics_k_geometric_context,
                intrinsics_p_geometric_context,
                path_to_ego_mask_geometric_context,
                return_logs=False, progress=0.0):
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
        photometric_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)

        n_temporal_context = len(ref_images_temporal_context)
        n_geometric_context = len(ref_images_geometric_context)
        assert len(ref_images_geometric_context_temporal_context) == n_temporal_context * n_geometric_context
        B = len(path_to_ego_mask)

        device = image.get_device()

        # getting ego masks for target and source cameras
        # fullsize mask
        H_full = 800
        W_full = 1280
        ego_mask_tensor = torch.ones(B, 1, H_full, W_full).to(device)
        ref_ego_mask_tensor_geometric_context = []
        for i_geometric_context in range(n_geometric_context):
            ref_ego_mask_tensor_geometric_context.append(torch.ones(B, 1, H_full, W_full).to(device))
        for b in range(B):
            ego_mask_tensor[b, 0] = torch.from_numpy(np.load(path_to_ego_mask[b])).float()
            for i_geometric_context in range(n_geometric_context):
                if camera_type_geometric_context[b, i_geometric_context] != 2:
                    ref_ego_mask_tensor_geometric_context[i_geometric_context][b, 0] = \
                        torch.from_numpy(np.load(path_to_ego_mask_geometric_context[i_geometric_context][b])).float()

        # resized masks
        ego_mask_tensors = []
        ref_ego_mask_tensors_geometric_context = []
        for i_geometric_context in range(n_geometric_context):
            ref_ego_mask_tensors_geometric_context.append([])
        for i in range(self.n):
            _, _, H, W = images[i].shape
            if W < W_full:
                ego_mask_tensors.append(
                    interpolate_image(ego_mask_tensor, shape=(B, 1, H, W), mode='nearest', align_corners=None)
                )
                for i_geometric_context in range(n_geometric_context):
                    ref_ego_mask_tensors_geometric_context[i_geometric_context].append(
                        interpolate_image(ref_ego_mask_tensor_geometric_context[i_geometric_context],
                                          shape=(B, 1, H, W),
                                          mode='nearest',
                                          align_corners=None)
                    )
            else:
                ego_mask_tensors.append(ego_mask_tensor)
                for i_geometric_context in range(n_geometric_context):
                    ref_ego_mask_tensors_geometric_context[i_geometric_context].append(
                        ref_ego_mask_tensor_geometric_context[i_geometric_context]
                    )

        # temporal context
        for j, (ref_image, pose) in enumerate(zip(ref_images_temporal_context, poses_temporal_context)):
            # Calculate warped images
            ref_warped, ref_ego_mask_tensors_warped = \
                self.warp_ref_image(inv_depths,
                                    camera_type,
                                    intrinsics_poly_coeffs,
                                    intrinsics_principal_point,
                                    intrinsics_scale_factors,
                                    intrinsics_K,
                                    intrinsics_k,
                                    intrinsics_p,
                                    ref_image,
                                    pose,
                                    ego_mask_tensors,
                                    camera_type,
                                    intrinsics_poly_coeffs,
                                    intrinsics_principal_point,
                                    intrinsics_scale_factors,
                                    intrinsics_K,
                                    intrinsics_k,
                                    intrinsics_p)
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i] * ego_mask_tensors[i] * ref_ego_mask_tensors_warped[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i] * ego_mask_tensors[i] * ego_mask_tensors[i])

        # geometric context
        for j, (ref_image, pose) in enumerate(zip(ref_images_geometric_context, poses_geometric_context)):
            # Calculate warped images
            ref_warped, ref_ego_mask_tensors_warped = \
                self.warp_ref_image(inv_depths,
                                    camera_type,
                                    intrinsics_poly_coeffs,
                                    intrinsics_principal_point,
                                    intrinsics_scale_factors,
                                    intrinsics_K,
                                    intrinsics_k,
                                    intrinsics_p,
                                    ref_image,
                                    Pose(pose),
                                    ref_ego_mask_tensors_geometric_context[j],
                                    camera_type_geometric_context[:, j],
                                    intrinsics_poly_coeffs_geometric_context[j],
                                    intrinsics_principal_point_geometric_context[j],
                                    intrinsics_scale_factors_geometric_context[j],
                                    intrinsics_K_geometric_context[j],
                                    intrinsics_k_geometric_context[j],
                                    intrinsics_p_geometric_context[j])
            print(j)
            print(camera_type_geometric_context[:, j])
            print(ref_image)
            print(torch.isnan(ref_warped[0]).sum())
            print(torch.isnan(ref_ego_mask_tensors_warped[0]).sum())
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, images)
            print(torch.isnan(photometric_loss[0]).sum())
            for i in range(self.n):
                print(i)
                photometric_losses[i].append(photometric_loss[i] * ego_mask_tensors[i] * ref_ego_mask_tensors_warped[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i] * ego_mask_tensors[i] * ref_ego_mask_tensors_geometric_context[j][i])

        # # geometric-temporal context
        # for j, (ref_image, pose) in enumerate(zip(ref_images_geometric_context_temporal_context, poses_geometric_context_temporal_context)):
        #     # Calculate warped images
        #     j_geometric = j // n_temporal_context
        #     ref_warped, ref_ego_mask_tensors_warped = \
        #         self.warp_ref_image(inv_depths,
        #                             camera_type,
        #                             intrinsics_poly_coeffs,
        #                             intrinsics_principal_point,
        #                             intrinsics_scale_factors,
        #                             intrinsics_K,
        #                             intrinsics_k,
        #                             intrinsics_p,
        #                             ref_image,
        #                             pose, # ATTENTION A CORRIGER (changement de repere !)
        #                             ref_ego_mask_tensors_geometric_context[j_geometric],
        #                             camera_type_geometric_context[j_geometric],
        #                             intrinsics_poly_coeffs_geometric_context[j_geometric],
        #                             intrinsics_principal_point_geometric_context[j_geometric],
        #                             intrinsics_scale_factors_geometric_context[j_geometric],
        #                             intrinsics_K_geometric_context[j_geometric],
        #                             intrinsics_k_geometric_context[j_geometric],
        #                             intrinsics_p_geometric_context[j_geometric])
        #     # Calculate and store image loss
        #     photometric_loss = self.calc_photometric_loss(ref_warped, images)
        #     for i in range(self.n):
        #         for b in range(B):
        #             print(dummy_camera_geometric_context)
        #             print(ref_images_geometric_context)
        #             print(j)
        #             print(j_geometric)
        #             print(dummy_camera_geometric_context[b, j_geometric])
        #             print(photometric_loss[i][b, 0, :, :])
        #             print(ref_ego_mask_tensors_warped[i][b, 0, :, :])
        #             if dummy_camera_geometric_context[b, j_geometric] == 1:
        #                 photometric_loss[i][b, 0, :, :] = 0.0
        #                 ref_ego_mask_tensors_warped[i][b, 0, :, :] = 0.0
        #         photometric_losses[i].append(photometric_loss[i] * ego_mask_tensors[i] * ref_ego_mask_tensors_warped[i])
        #     # If using automask
        #     if self.automask_loss:
        #         # Calculate and store unwarped image loss
        #         ref_images = match_scales(ref_image, inv_depths, self.n)
        #         unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
        #         for i in range(self.n):
        #             photometric_losses[i].append(unwarped_image_loss[i] * ego_mask_tensors[i] * ref_ego_mask_tensors_geometric_context[j_geometric][i])

        # Calculate reduced photometric loss
        loss = self.nonzero_reduce_photometric_loss(photometric_losses)
        # Include smoothness loss if requested
        if self.smooth_loss_weight > 0.0:
            loss += self.calc_smoothness_loss([a * b for a, b in zip(inv_depths, ego_mask_tensors)],
                                              [a * b for a, b in zip(images,     ego_mask_tensors)])
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################
