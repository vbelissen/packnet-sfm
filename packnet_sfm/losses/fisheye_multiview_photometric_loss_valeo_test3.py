# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import numpy as np

from packnet_sfm.utils.image import match_scales
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_fisheye_valeo import CameraFisheye
from packnet_sfm.geometry.camera_fisheye_valeo_utils import view_synthesis
from packnet_sfm.utils.depth import calc_smoothness, inv2depth
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling

import cv2
import time

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
                 C1=1e-4, C2=9e-4, photometric_reduce_op='mean', disp_norm=True, clip_loss=0.5,
                 progressive_scaling=0.0, padding_mode='zeros',
                 automask_loss=False, mask_ego=True, warp_ego_tensor=False, **kwargs):
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
        self.warp_ego_tensor = warp_ego_tensor
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

    def warp_ref_image(self, inv_depths, ref_image,
                       path_to_theta_lut,     path_to_ego_mask,     poly_coeffs,      principal_point,      scale_factors,
                       ref_path_to_theta_lut, ref_path_to_ego_mask, ref_poly_coeffs,  ref_principal_point,  ref_scale_factors,
                       same_timestamp_as_origin,
                       pose_matrix_context,
                       pose):
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
        for b in range(B):
            if same_timestamp_as_origin[b]:
                pose.mat[b, :, :] = pose_matrix_context[b, :, :]
        # pose_matrix = torch.zeros(B, 4, 4)
        # for b in range(B):
        #     if not same_timestamp_as_origin[b]:
        #         pose_matrix[b, :, :] = pose.mat[b, :, :]
        #     else:
        #         pose_matrix[b, :, :] = pose_matrix_context[b, :, :]
        #pose_matrix = Pose(pose_matrix)
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(CameraFisheye(path_to_theta_lut=path_to_theta_lut,
                                      path_to_ego_mask=path_to_ego_mask,
                                      poly_coeffs=poly_coeffs.float(),
                                      principal_point=principal_point.float(),
                                      scale_factors=scale_factors.float()).scaled(scale_factor).to(device))
            ref_cams.append(CameraFisheye(path_to_theta_lut=ref_path_to_theta_lut,
                                          path_to_ego_mask=ref_path_to_ego_mask,
                                          poly_coeffs=ref_poly_coeffs.float(),
                                          principal_point=ref_principal_point.float(),
                                          scale_factors=ref_scale_factors.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [view_synthesis(
            ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode) for i in range(self.n)]
        # Return warped reference image
        return ref_warped

    def warp_ref_image_tensor(self, inv_depths, ref_image, ref_tensor,
                       path_to_theta_lut,     path_to_ego_mask,     poly_coeffs,      principal_point,      scale_factors,
                       ref_path_to_theta_lut, ref_path_to_ego_mask, ref_poly_coeffs,  ref_principal_point,  ref_scale_factors,
                       same_timestamp_as_origin,
                       pose_matrix_context,
                       pose):
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
        for b in range(B):
            if same_timestamp_as_origin[b]:
                pose.mat[b, :, :] = pose_matrix_context[b, :, :]
        # pose_matrix = torch.zeros(B, 4, 4)
        # for b in range(B):
        #     if not same_timestamp_as_origin[b]:
        #         pose_matrix[b, :, :] = pose.mat[b, :, :]
        #     else:
        #         pose_matrix[b, :, :] = pose_matrix_context[b, :, :]
        #pose_matrix = Pose(pose_matrix)
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(CameraFisheye(path_to_theta_lut=path_to_theta_lut,
                                      path_to_ego_mask=path_to_ego_mask,
                                      poly_coeffs=poly_coeffs.float(),
                                      principal_point=principal_point.float(),
                                      scale_factors=scale_factors.float()).scaled(scale_factor).to(device))
            ref_cams.append(CameraFisheye(path_to_theta_lut=ref_path_to_theta_lut,
                                          path_to_ego_mask=ref_path_to_ego_mask,
                                          poly_coeffs=ref_poly_coeffs.float(),
                                          principal_point=ref_principal_point.float(),
                                          scale_factors=ref_scale_factors.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [view_synthesis(
            ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode, mode='bilinear') for i in range(self.n)]
        ref_tensors = match_scales(ref_tensor, inv_depths, self.n, mode='nearest', align_corners=None)
        ref_tensors_warped = [view_synthesis(
            ref_tensors[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode, mode='nearest', align_corners=None) for i in range(self.n)]
        #print(ref_tensors[0][:, :, ::40, ::40])
        #print(ref_tensors_warped[0][:, :, ::40, ::40])
        # Return warped reference image
        return ref_warped, ref_tensors_warped

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

    def forward(self, image, context, inv_depths,
                path_to_theta_lut,     path_to_ego_mask,     poly_coeffs,     principal_point,     scale_factors,
                ref_path_to_theta_lut, ref_path_to_ego_mask, ref_poly_coeffs, ref_principal_point, ref_scale_factors, # ALL LISTS !!!
                same_timestep_as_origin,
                pose_matrix_context,
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
        photometric_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)

        n_context = len(context)

        if self.mask_ego:
            device = image.get_device()

            B = len(path_to_ego_mask)

            ego_mask_tensor     = torch.zeros(B, 1, 800, 1280).to(device)
            ref_ego_mask_tensor = []#[torch.zeros(B, 1, 800, 1280).to(device)] * n_context
            for i_context in range(n_context):
                ref_ego_mask_tensor.append(torch.zeros(B, 1, 800, 1280).to(device))
            for b in range(B):
                ego_mask_tensor[b, 0]     = torch.from_numpy(np.load(path_to_ego_mask[b])).float()
                for i_context in range(n_context):
                    ref_ego_mask_tensor[i_context][b, 0] = torch.from_numpy(np.load(ref_path_to_ego_mask[i_context][b])).float()

            ego_mask_tensors     = []  # = torch.zeros(B, 1, 800, 1280)
            ref_ego_mask_tensors = []#[[]] *  n_context  # = torch.zeros(B, 1, 800, 1280)
            for i_context in range(n_context):
                ref_ego_mask_tensors.append([])
            for i in range(self.n):
                B, C, H, W = images[i].shape
                if W < 1280:
                    inv_scale_factor = int(1280 / W)
                    ego_mask_tensors.append(-nn.MaxPool2d(inv_scale_factor, inv_scale_factor)(-ego_mask_tensor))
                    for i_context in range(n_context):
                        ref_ego_mask_tensors[i_context].append(-nn.MaxPool2d(inv_scale_factor, inv_scale_factor)(-ref_ego_mask_tensor[i_context]))
                else:
                    ego_mask_tensors.append(ego_mask_tensor)
                    for i_context in range(n_context):
                        ref_ego_mask_tensors[i_context].append(ref_ego_mask_tensor[i_context])
                # ego_mask_tensor = ego_mask_tensor.to(device)

            for i_context in range(n_context):
                B, C, H, W = context[i_context].shape
                if W < 1280:
                    inv_scale_factor = int(1280 / W)
                    ref_ego_mask_tensor[i_context] = -nn.MaxPool2d(inv_scale_factor, inv_scale_factor)(-ref_ego_mask_tensor[i_context])

        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            # Calculate warped images
            if self.mask_ego:
                if self.warp_ego_tensor:
                    ref_warped, ref_ego_mask_tensors_warped \
                        = self.warp_ref_image_tensor(inv_depths, ref_image, ref_ego_mask_tensor[j],
                                                     path_to_theta_lut,        path_to_ego_mask,        poly_coeffs,        principal_point,        scale_factors,
                                                     ref_path_to_theta_lut[j], ref_path_to_ego_mask[j], ref_poly_coeffs[j], ref_principal_point[j], ref_scale_factors[j],
                                                     same_timestep_as_origin[j],
                                                     pose_matrix_context[j],
                                                     pose)
                    photometric_loss = self.calc_photometric_loss(
                        [a * b * c for a, b, c in zip(ref_warped, ego_mask_tensors, ref_ego_mask_tensors_warped)],
                        [a * b     for a, b    in zip(images,     ego_mask_tensors)])

                else:
                    ref_warped = self.warp_ref_image(inv_depths, ref_image * ref_ego_mask_tensor[j],
                                                     path_to_theta_lut,        path_to_ego_mask,        poly_coeffs,        principal_point,        scale_factors,
                                                     ref_path_to_theta_lut[j], ref_path_to_ego_mask[j], ref_poly_coeffs[j], ref_principal_point[j], ref_scale_factors[j],
                                                     same_timestep_as_origin[j],
                                                     pose_matrix_context[j],
                                                     pose)
                    photometric_loss = self.calc_photometric_loss([a * b for a, b in zip(ref_warped, ego_mask_tensors)],
                                                                  [a * b for a, b in zip(images,     ego_mask_tensors)])
            else:
                ref_warped = self.warp_ref_image(inv_depths, ref_image,
                                                 path_to_theta_lut,        path_to_ego_mask,        poly_coeffs,        principal_point,        scale_factors,
                                                 ref_path_to_theta_lut[j], ref_path_to_ego_mask[j], ref_poly_coeffs[j], ref_principal_point[j], ref_scale_factors[j],
                                                 same_timestep_as_origin[j],
                                                 pose_matrix_context[j],
                                                 pose)
                photometric_loss = self.calc_photometric_loss(ref_warped, images)
            # tt = str(int(time.time() % 10000))
            # for i in range(self.n):
            #     B, C, H, W = images[i].shape
            #     print(photometric_loss[i][:,:,::20,::20])
            #     for b in range(B):
            #         orig_PIL_0   = torch.transpose((ref_image[b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
            #         orig_PIL   = torch.transpose((ref_image[b,:,:,:]* ref_ego_mask_tensor[j][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
            #         warped_PIL_0 = torch.transpose((ref_warped[i][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
            #         warped_PIL = torch.transpose((ref_warped[i][b,:,:,:] * ego_mask_tensors[i][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
            #         target_PIL_0 = torch.transpose((images[i][b, :, :, :]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
            #         target_PIL = torch.transpose((images[i][b, :, :, :] * ego_mask_tensors[i][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
            #
            #         cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_orig_PIL_0.png',   orig_PIL_0*255)
            #         cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_orig_PIL.png',   orig_PIL*255)
            #         cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_warped_PIL_0.png', warped_PIL_0 * 255)
            #         cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_warped_PIL.png', warped_PIL * 255)
            #         cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_target_PIL_0.png', target_PIL_0 * 255)
            #         cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_target_PIL.png', target_PIL * 255)
            #
            # print(photometric_loss[0].shape)

            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                if self.mask_ego:
                    unwarped_image_loss = self.calc_photometric_loss([a * b for a, b in zip(ref_images, ref_ego_mask_tensors[j])],
                                                                     [a * b for a, b in zip(images,     ego_mask_tensors)])
                else:
                    unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i])
        # Calculate reduced photometric loss
        loss = self.reduce_photometric_loss(photometric_losses)
        # Include smoothness loss if requested
        if self.smooth_loss_weight > 0.0:
            if self.mask_ego:
                loss += self.calc_smoothness_loss([a * b for a, b in zip(inv_depths, ego_mask_tensors)],
                                                  [a * b for a, b in zip(images,     ego_mask_tensors)])
            else:
                loss += self.calc_smoothness_loss(inv_depths, images)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################
