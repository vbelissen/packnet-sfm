# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.fisheye_multiview_photometric_loss_valeo_test3 import MultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs


class SelfSupModel_fisheye_valeo_testPose3(SfmModel):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
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

    def self_supervised_loss(self, image, ref_images, inv_depths, poses,
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
            image, ref_images, inv_depths,
            path_to_theta_lut,         path_to_ego_mask,         poly_coeffs,         principal_point,         scale_factors,
            path_to_theta_lut_context, path_to_ego_mask_context, poly_coeffs_context, principal_point_context, scale_factors_context,
            same_timestep_as_origin,
            pose_matrix_context,
            poses, return_logs=return_logs, progress=progress)

    def forward(self, batch, mask_ego=True, return_logs=False, progress=0.0):
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
            # Otherwise, calculate self-supervised loss
            self_sup_output = self.self_supervised_loss(
                batch['rgb_original'], batch['rgb_context_original'],
                output['inv_depths'], output['poses'],
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
