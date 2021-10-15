# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from packnet_sfm.utils.image import match_scales
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling


torch.autograd.set_detect_anomaly(True)
########################################################################################################################

def mat2euler(M):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-M[:, 0, 1], M[:, 0, 0])
       y = asin(M[:, 0, 2])
       x = atan2(-M[:, 1, 2], M[:, 2, 2])
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)

    device = M.get_device()

    B, _, _ = M.shape
    cy = torch.sqrt(M[:, 2, 2] * M[:, 2, 2] + M[:, 1, 2] * M[:, 1, 2])
    x = torch.zeros(B).to(device)
    y = torch.zeros(B).to(device)
    z = torch.zeros(B).to(device)

    mask_cy = (cy > 1e-4)

    z[mask_cy] = torch.atan2(-M[mask_cy, 0, 1], M[mask_cy, 0, 0])  # atan2(cos(y)*sin(z), cos(y)*cos(z))
    y[mask_cy] = torch.atan2(M[mask_cy, 0, 2], cy[mask_cy])  # atan2(sin(y), cy)
    x[mask_cy] = torch.atan2(-M[mask_cy, 1, 2], M[mask_cy, 2, 2])  # atan2(cos(y)*sin(x), cos(x)*cos(y))

    z[~mask_cy] = torch.atan2(M[~mask_cy, 1, 0], M[~mask_cy, 1, 1])
    y[~mask_cy] = torch.atan2(M[~mask_cy, 0, 2], cy[~mask_cy])  # atan2(sin(y), cy)

    # if cy > 1e-4:  # cos(y) not close to zero, standard form
    #     z = torch.atan2(-M[:, 0, 1], M[:, 0, 0])  # atan2(cos(y)*sin(z), cos(y)*cos(z))
    #     y = torch.atan2(M[:, 0, 2], cy)  # atan2(sin(y), cy)
    #     x = torch.atan2(-M[:, 1, 2], M[:, 2, 2])  # atan2(cos(y)*sin(x), cos(x)*cos(y))
    # else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
    #     # so M[:, 1, 0] -> sin(z), M[:, 1, 1] -> cos(z) and
    #     z = torch.atan2(M[:, 1, 0], M[:, 1, 1])
    #     y = torch.atan2(M[:, 0, 2], cy)  # atan2(sin(y), cy)
    #     x = torch.zeros(B)
    return torch.stack([x, y, z],dim=1) # z, y, x

class PoseConsistencyLoss(LossBase):
    """
    Pose Consistency loss for spatio-temporal contexts.
    """
    def __init__(self, pose_consistency_translation_loss_weight=0.1, pose_consistency_rotation_loss_weight=0.1, **kwargs):
        super().__init__()
        self.pose_consistency_translation_loss_weight = pose_consistency_translation_loss_weight
        self.pose_consistency_rotation_loss_weight = pose_consistency_rotation_loss_weight

    ########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {

        }

########################################################################################################################

    def calculate_loss(self, pose1, pose2, camera_type):
        """
        Calculates the pose consistency loss.

        Parameters
        ----------


        Returns
        -------

        """
        trans_loss = (pose1.mat[:, :3, 3] - pose2.mat[:, :3, 3]).norm(dim=-1)
        rot_loss = (mat2euler(pose1.mat[:, :3, :3]) - mat2euler(pose2.mat[:, :3, :3])).norm(dim=-1)

        mask = (camera_type < 2)
        trans_loss_final = trans_loss[mask].mean()
        rot_loss_final = rot_loss[mask].mean()

        return self.pose_consistency_translation_loss_weight * trans_loss_final \
               + self.pose_consistency_rotation_loss_weight * rot_loss_final

    def forward(self,
                poses_temporal_context,
                poses_geometric_context_temporal_context,
                camera_type_geometric_context,
                **kwargs):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        n_t = len(poses_temporal_context)
        n_g = len(poses_geometric_context_temporal_context) // n_t
        losses = []
        for i_g in range(n_g):
            for i_t in range(n_t):
                losses.append(self.calculate_loss(poses_temporal_context[i_t],
                                                  poses_geometric_context_temporal_context[i_g * n_t + i_t],
                                                  camera_type_geometric_context[:, i_g]))
        loss = sum(losses) / len(losses)

        self.add_metric('pose_consistency_loss', loss)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }