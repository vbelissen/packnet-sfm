# Copyright 2020 Toyota Research Institute.  All rights reserved.
import torch

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.fisheye_multiview_photometric_loss import MultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs


class SelfSupModel_fisheye(SfmModel):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, mask_occlusion=False, mask_disocclusion=False, mask_spatial_context=False, mask_temporal_context=False,
                 depth_consistency_weight=0.2, **kwargs):
        self.mask_occlusion = mask_occlusion
        self.mask_disocclusion = mask_disocclusion
        self.mask_spatial_context = mask_spatial_context
        self.mask_temporal_context = mask_temporal_context
        self.depth_consistency_weight = depth_consistency_weight
        self.use_ref_depth = ((self.mask_occlusion or self.mask_disocclusion) and (self.mask_spatial_context or self.mask_temporal_context)) or (self.depth_consistency_weight > 0)
        # Initializes SfmModel
        super().__init__(**kwargs)
        # Initializes the photometric loss

        self._photometric_loss = MultiViewPhotometricLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
        }

    def self_supervised_loss(self, image, ref_images, inv_depths, ref_inv_depths, poses,
                             path_to_theta_lut,         path_to_ego_mask,         poly_coeffs,         principal_point,         scale_factors,
                             path_to_theta_lut_context, path_to_ego_mask_context, poly_coeffs_context, principal_point_context, scale_factors_context,
                             same_timestep_as_origin,
                             pose_matrix_context,
                             return_logs=False, progress=0.0):
        """
        Calculates the self-supervised photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._photometric_loss(
            image, ref_images, inv_depths, ref_inv_depths,
            path_to_theta_lut,         path_to_ego_mask,         poly_coeffs,         principal_point,         scale_factors,
            path_to_theta_lut_context, path_to_ego_mask_context, poly_coeffs_context, principal_point_context, scale_factors_context,
            same_timestep_as_origin,
            pose_matrix_context,
            poses, return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        # Calculate predicted depth and pose output
        output = super().forward(batch, return_logs=return_logs)
        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            with torch.no_grad():
                context_inv_depths = []
                n_context = len(batch['rgb_context_original'])
                for i_context in range(n_context):
                    if self.use_ref_depth:
                        context = {'rgb': batch['rgb_context_original'][i_context]}
                        output_context = super().forward(context, return_logs=return_logs)
                        context_inv_depths.append(output_context['inv_depths'])
                    else:
                        context_inv_depths.append(0)
            # Otherwise, calculate self-supervised loss
            self_sup_output = self.self_supervised_loss(
                batch['rgb_original'], batch['rgb_context_original'],
                output['inv_depths'], context_inv_depths, output['poses'],
                batch['path_to_theta_lut'],
                batch['path_to_ego_mask'],
                batch['intrinsics_poly_coeffs'],
                batch['intrinsics_principal_point'],
                batch['intrinsics_scale_factors'],
                batch['path_to_theta_lut_context'],
                batch['path_to_ego_mask_context'],
                batch['intrinsics_poly_coeffs_context'],
                batch['intrinsics_principal_point_context'],
                batch['intrinsics_scale_factors_context'],
                batch['same_timestep_as_origin_context'],
                batch['pose_matrix_context'],
                return_logs=return_logs, progress=progress)
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }
