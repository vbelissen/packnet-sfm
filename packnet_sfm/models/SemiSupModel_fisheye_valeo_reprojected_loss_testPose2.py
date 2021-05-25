# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch

from packnet_sfm.models.SelfSupModel_fisheye_valeo_testPose2 import SfmModel, SelfSupModel_fisheye_valeo_testPose2
from packnet_sfm.losses.supervised_loss_valeo import SupervisedLoss
from packnet_sfm.losses.fisheye_multiview_reprojected_loss_valeo import ReprojectedLoss
from packnet_sfm.models.model_utils import merge_outputs
from packnet_sfm.utils.depth import depth2inv, inv2depth


class SemiSupModel_fisheye_valeo_reprojected_loss(SelfSupModel_fisheye_valeo_testPose2):
    """
    Model that inherits a depth and pose networks, plus the self-supervised loss from
    SelfSupModel and includes a supervised loss for semi-supervision.

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_loss_weight=0.9, reprojected_loss_weight=10000., **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # If supervision weight is 0.0, use SelfSupModel directly
        assert 0. < supervised_loss_weight <= 1., "Model requires (0, 1] supervision"
        # Store weight and initializes supervised loss
        self.supervised_loss_weight = supervised_loss_weight
        self._supervised_loss = SupervisedLoss(**kwargs)

        # Pose network is only required if there is self-supervision
        self._network_requirements['pose_net'] = self.supervised_loss_weight < 1
        # GT depth is only required if there is supervision
        self._train_requirements['gt_depth'] = self.supervised_loss_weight > 0

        self._reprojected_loss = ReprojectedLoss(**kwargs)
        self.reprojected_loss_weight = reprojected_loss_weight# 10000.

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._supervised_loss.logs
        }

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        path_to_ego_mask,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._supervised_loss(
            inv_depths, gt_inv_depths,
            path_to_ego_mask,
            return_logs=return_logs, progress=progress)

    def reprojected_loss(self, gt_depth, depths, poses,
                         path_to_theta_lut,         path_to_ego_mask,         poly_coeffs,         principal_point,         scale_factors,
                         path_to_theta_lut_context, path_to_ego_mask_context, poly_coeffs_context, principal_point_context, scale_factors_context,
                         return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._reprojected_loss(
            gt_depth, depths,
            path_to_theta_lut,         path_to_ego_mask,         poly_coeffs,         principal_point,         scale_factors,
            path_to_theta_lut_context, path_to_ego_mask_context, poly_coeffs_context, principal_point_context, scale_factors_context,
            poses,
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
        if not self.training:
            # If not training, no need for self-supervised loss
            return SfmModel.forward(self, batch)
        else:
            if self.supervised_loss_weight == 1.:
                # If no self-supervision, no need to calculate loss
                self_sup_output = SfmModel.forward(self, batch)
                loss = torch.tensor([0.]).type_as(batch['rgb'])
            else:
                # Otherwise, calculate and weight self-supervised loss
                self_sup_output = SelfSupModel_fisheye_valeo_testPose2.forward(self, batch)
                loss = (1.0 - self.supervised_loss_weight) * self_sup_output['loss']

            reproj_output =  self.reprojected_loss(
                batch['depth'], inv2depth(self_sup_output['inv_depths']),
                self_sup_output['poses'],
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
                return_logs=return_logs, progress=progress)


            # Calculate and weight supervised loss
            loss += self.reprojected_loss_weight * reproj_output['loss']
            # Merge and return outputs
            return {
                'loss': loss,
                **merge_outputs(self_sup_output, reproj_output),
            }
