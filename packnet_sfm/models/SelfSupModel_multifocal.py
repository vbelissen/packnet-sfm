# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SfmModel_multifocal import SfmModel_multifocal
from packnet_sfm.losses.multiview_photometric_loss_multifocal import MultiViewPhotometricLoss
from packnet_sfm.losses.pose_consistency_loss import PoseConsistencyLoss

from packnet_sfm.models.model_utils import merge_outputs


class SelfSupModel_multifocal(SfmModel_multifocal):
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
        self._pose_consistency_loss = PoseConsistencyLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
        }

    def self_supervised_loss(self,
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
            return_logs=return_logs, progress=progress
        )

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
                batch['rgb_original'],
                batch['rgb_temporal_context_original'],
                batch['rgb_geometric_context_original'],
                batch['rgb_geometric_context_temporal_context_original'],
                output['inv_depths'],
                output['poses_temporal_context'],
                batch['pose_matrix_geometric_context'],
                output['poses_geometric_context_temporal_context'],
                batch['camera_type'],
                batch['intrinsics_poly_coeffs'],
                batch['intrinsics_principal_point'],
                batch['intrinsics_scale_factors'],
                batch['intrinsics_K'],
                batch['intrinsics_k'],
                batch['intrinsics_p'],
                batch['path_to_ego_mask'],
                batch['camera_type_geometric_context'],
                batch['intrinsics_poly_coeffs_geometric_context'],
                batch['intrinsics_principal_point_geometric_context'],
                batch['intrinsics_scale_factors_geometric_context'],
                batch['intrinsics_K_geometric_context'],
                batch['intrinsics_k_geometric_context'],
                batch['intrinsics_p_geometric_context'],
                batch['path_to_ego_mask_geometric_context'],
                return_logs=return_logs, progress=progress)

            pose_consistency_loss = self._pose_consistency_loss(output['poses_temporal_context'],
                                                                output['poses_geometric_context_temporal_context'],
                                                                batch['camera_type_geometric_context'])
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'] + pose_consistency_loss['loss'],
                **merge_outputs(output, self_sup_output),
            }
