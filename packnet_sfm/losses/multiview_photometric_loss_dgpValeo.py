# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_utils import view_synthesis
from packnet_sfm.utils.depth import calc_smoothness, inv2depth, depth2inv
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling
from packnet_sfm.utils.types import is_list
from packnet_sfm.utils.image import interpolate_image

import numpy as np
import time
import cv2

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

    def warp_ref_image(self, inv_depths, ref_image, ref_tensor, K, ref_K, pose, ref_extrinsics, ref_context_type):
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
            if ref_context_type[b] == 'left' or ref_context_type[b] == 'right':
                pose.mat[b, :, :] = ref_extrinsics[b, :, :]
                pose.mat[b,:3,3]=0
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = [view_synthesis(
            ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode) for i in range(self.n)]
        # Return warped reference image

        ref_tensors = match_scales(ref_tensor, inv_depths, self.n, mode='nearest', align_corners=None)
        ref_tensors_warped = [view_synthesis(
            ref_tensors[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode, mode='nearest') for i in range(self.n)]

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

    def forward(self, image, context,
                inv_depths, poses,
                path_to_ego_mask, path_to_ego_mask_context,
                K,                ref_K,
                extrinsics,       ref_extrinsics,
                context_type,
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

        inv_depths2 = []
        for k in range(len(inv_depths)):

            Btmp, C, H, W = inv_depths[k].shape
            depths2 = torch.zeros_like(inv_depths[k])
            for i in range(H):
                for j in range(W):
                    depths2[0, 0, i, j] = (2/H)**4 * (2/W)**4 * i**2 * (H - i)**2 * j**2 * (W - j)**2 * 20

            inv_depths2.append(depth2inv(depths2))

        if is_list(context_type[0][0]):
            n_context = len(context_type[0])
        else:
            n_context = len(context_type)

        #n_context = len(context)
        device = image.get_device()
        B = len(path_to_ego_mask)
        if is_list(path_to_ego_mask[0]):
            H_full, W_full = np.load(path_to_ego_mask[0][0]).shape
        else:
            H_full, W_full = np.load(path_to_ego_mask[0]).shape

        # getting ego masks for target and source cameras
        # fullsize mask
        ego_mask_tensor = torch.ones(B, 1, H_full, W_full).to(device)
        ref_ego_mask_tensor = []
        for i_context in range(n_context):
            ref_ego_mask_tensor.append(torch.ones(B, 1, H_full, W_full).to(device))
        for b in range(B):
            if self.mask_ego:
                if is_list(path_to_ego_mask[b]):
                    ego_mask_tensor[b, 0] = torch.from_numpy(np.load(path_to_ego_mask[b][0])).float()
                else:
                    ego_mask_tensor[b, 0] = torch.from_numpy(np.load(path_to_ego_mask[b])).float()
                for i_context in range(n_context):
                    if is_list(path_to_ego_mask_context[0][0]):
                        paths_context_ego = [p[i_context][0] for p in path_to_ego_mask_context]
                    else:
                        paths_context_ego = path_to_ego_mask_context[i_context]
                    ref_ego_mask_tensor[i_context][b, 0] = torch.from_numpy(np.load(paths_context_ego[b])).float()
        # resized masks
        ego_mask_tensors = []
        ref_ego_mask_tensors = []
        for i_context in range(n_context):
            ref_ego_mask_tensors.append([])
        for i in range(self.n):
            Btmp, C, H, W = images[i].shape
            if W < W_full:
                #inv_scale_factor = int(W_full / W)
                #print(W_full / W)
                #ego_mask_tensors.append(-nn.MaxPool2d(inv_scale_factor, inv_scale_factor)(-ego_mask_tensor))
                ego_mask_tensors.append(interpolate_image(ego_mask_tensor,
                                                          shape=(Btmp, 1, H, W),
                                                          mode='nearest',
                                                          align_corners=None))
                for i_context in range(n_context):
                    #ref_ego_mask_tensors[i_context].append(-nn.MaxPool2d(inv_scale_factor, inv_scale_factor)(-ref_ego_mask_tensor[i_context]))
                    ref_ego_mask_tensors[i_context].append(interpolate_image(ref_ego_mask_tensor[i_context],
                                                          shape=(Btmp, 1, H, W),
                                                          mode='nearest',
                                                          align_corners=None))
            else:
                ego_mask_tensors.append(ego_mask_tensor)
                for i_context in range(n_context):
                    ref_ego_mask_tensors[i_context].append(ref_ego_mask_tensor[i_context])
        for i_context in range(n_context):
            _, C, H, W = context[i_context].shape
            if W < W_full:
                inv_scale_factor = int(W_full / W)
                #ref_ego_mask_tensor[i_context] = -nn.MaxPool2d(inv_scale_factor, inv_scale_factor)(-ref_ego_mask_tensor[i_context])
                ref_ego_mask_tensor[i_context] = interpolate_image(ref_ego_mask_tensor[i_context],
                                                                         shape=(Btmp, 1, H, W),
                                                                         mode='nearest',
                                                                         align_corners=None)

        print(ref_extrinsics)
        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            print(ref_extrinsics[j])
            ref_context_type = [c[j][0] for c in context_type] if is_list(context_type[0][0]) else context_type[j]
            print(ref_context_type)
            print(pose.mat)
            # Calculate warped images
            ref_warped, ref_ego_mask_tensors_warped = self.warp_ref_image(inv_depths2,
                                                                          ref_image,
                                                                          ref_ego_mask_tensor[j],
                                                                          K, ref_K[:, j, :, :],
                                                                          pose,
                                                                          ref_extrinsics[j],
                                                                          ref_context_type)
            print(pose.mat)
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, images)

            tt = str(int(time.time() % 10000))
            for i in range(self.n):
                B, C, H, W = images[i].shape
                for b in range(B):
                    orig_PIL_0   = torch.transpose((ref_image[b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
                    orig_PIL     = torch.transpose((ref_image[b,:,:,:]* ref_ego_mask_tensor[j][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
                    warped_PIL_0 = torch.transpose((ref_warped[i][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
                    warped_PIL   = torch.transpose((ref_warped[i][b,:,:,:] * ego_mask_tensors[i][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
                    target_PIL_0 = torch.transpose((images[i][b, :, :, :]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()
                    target_PIL   = torch.transpose((images[i][b, :, :, :] * ego_mask_tensors[i][b,:,:,:]).unsqueeze(0).unsqueeze(4), 1, 4).squeeze().detach().cpu().numpy()

                    cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_orig_PIL_0.png',   orig_PIL_0*255)
                    cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_orig_PIL.png',   orig_PIL*255)
                    cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_warped_PIL_0.png', warped_PIL_0 * 255)
                    cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_warped_PIL.png', warped_PIL * 255)
                    cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_target_PIL_0.png', target_PIL_0 * 255)
                    cv2.imwrite('/home/users/vbelissen/test' + '_' + str(j) + '_' + tt + '_' + str(b) + '_' + str(i) + '_target_PIL.png', target_PIL * 255)


            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i] * ego_mask_tensors[i] * ref_ego_mask_tensors_warped[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i] * ego_mask_tensors[i] * ref_ego_mask_tensors[j][i])
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
