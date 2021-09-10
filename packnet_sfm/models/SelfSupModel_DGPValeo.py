# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.multiview_photometric_loss_dgpValeo import MultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs


class SelfSupModel_DGPValeo(SfmModel):
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

    def self_supervised_loss(self,
                             image, ref_images,
                             inv_depths, poses,
                             path_to_ego_mask, path_to_ego_mask_context,
                             intrinsics, ref_intrinsics,
                             extrinsics, ref_extrinsics,
                             context_type,
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
            image, ref_images,
            inv_depths, poses,
            path_to_ego_mask, path_to_ego_mask_context,
            intrinsics, ref_intrinsics,
            extrinsics, ref_extrinsics,
            context_type,
            return_logs=return_logs, progress=progress)

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
        # print(batch['rgb_original'].shape)
        # print(len(batch['rgb_context_original']))
        # print(batch['rgb_context_original'][0].shape)
        # print(output['inv_depths'].shape)
        # print(output['poses'].shape)
        # print(batch['path_to_ego_mask'].shape)
        # print(batch['path_to_ego_mask_context'].shape)
        # print(batch['intrinsics'].shape)
        # print(batch['intrinsics_context'].shape)
        # print(batch['extrinsics'].shape)
        # print(batch['extrinsics_context'].shape)


        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            # Otherwise, calculate self-supervised loss
            self_sup_output = self.self_supervised_loss(
                batch['rgb_original'],  batch['rgb_context_original'],
                output['inv_depths'], output['poses'],
                batch['path_to_ego_mask'], batch['path_to_ego_mask_context'],
                batch['intrinsics'], batch['intrinsics_context'],
                batch['extrinsics'], batch['extrinsics_context'],
                batch['context_type'],
                return_logs=return_logs, progress=progress)
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }
