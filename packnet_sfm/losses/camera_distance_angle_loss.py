# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.losses.loss_base import LossBase


class CameraDistanceAngleLoss(LossBase):
    """
    Velocity loss for pose translation.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred_pose, gt_pose_context_cameras, same_timestep_as_origin, **kwargs):
        """
        Calculates velocity loss.

        Parameters
        ----------
        pred_pose : list of Pose
            Predicted pose transformation between origin and reference
        gt_pose_context : list of Pose
            Ground-truth pose transformation between origin and reference

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        pred_trans = [pose.mat[:, :3, -1].norm(dim=-1) for pose in pred_pose]
        gt_trans = [pose[:, :3, -1].norm(dim=-1) for pose in gt_pose_context_cameras]

        # trace(R) = 1 + 2.cos(theta) <=> cos(theta) = (tr(R) - 1)/2
        pred_rot_cos = [(pose.mat[:, :3, :3].diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)-1)/2 for pose in pred_pose]
        gt_rot_cos = [(pose[:, :3, :3].diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)-1)/2 for pose in gt_pose_context_cameras]
        # Calculate velocity supervision loss
        loss1 = sum([((pred - gt) * same).abs().mean()
                    for pred, gt, same in zip(pred_trans, gt_trans, same_timestep_as_origin)]) / len(gt_trans)
        loss2 = sum([((pred - gt) * same).abs().mean()
                    for pred, gt, same in zip(pred_rot_cos, gt_rot_cos, same_timestep_as_origin)]) / len(gt_rot_cos)
        loss = loss1+loss2
        self.add_metric('camera_distance_angle_loss', loss)
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }
