# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.GenericSfmModel import GenericSfmModel
from packnet_sfm.losses.generic_multiview_photometric_loss_cpu import GenericMultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs
import numpy as np


class GenericSelfSupModel_cpu(GenericSfmModel):
    """
    Model that inherits a depth and pose network from GenericSfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    kwargs : dict
        Extra parameters
    """

    def __init__(self, depth_net=None, pose_net=None, **kwargs):
        # Initializes GenericSfmModel
        super().__init__(depth_net, pose_net, **kwargs)
        # Initializes the photometric loss
        self._photometric_loss = GenericMultiViewPhotometricLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
        }

    @property
    def requires_depth_net(self):
        return True

    @property
    def requires_pose_net(self):
        return True

    @property
    def requires_gt_depth(self):
        return False

    @property
    def requires_gt_pose(self):
        return False

    def self_supervised_loss(self, image, ref_images, inv_depths, ray_surface, poses,
                             intrinsics, return_logs=False, progress=0.0):
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
            image, ref_images, inv_depths, ray_surface, intrinsics, intrinsics, poses,
            return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=True, force_flip=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : boolf
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
                output['inv_depths'], output['ray_surface'], output['poses'], batch['intrinsics'],
                return_logs=return_logs, progress=progress)
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }
