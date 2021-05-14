# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SelfSupModel_fisheye_valeo import SelfSupModel_fisheye_valeo
from packnet_sfm.losses.camera_distance_loss import CameraDistanceLoss


class CameraDistanceSupModelFisheye(SelfSupModel_fisheye_valeo):
    """
    Self-supervised model with additional velocity supervision loss.

    Parameters
    ----------
    velocity_loss_weight : float
        Weight for velocity supervision
    kwargs : dict
        Extra parameters
    """
    def __init__(self, velocity_loss_weight=0.1, **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        # Stores velocity supervision loss weight
        self._camera_distance_loss = CameraDistanceLoss(**kwargs)
        self._camera_distance_loss_weight = velocity_loss_weight

        # GT pose is required
        #self._train_requirements['gt_pose'] = True

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
        output = super().forward(batch, return_logs, progress)
        if self.training:
            # Update self-supervised loss with velocity supervision
            camera_distance_loss = self._camera_distance_loss(output['poses'],
                                                              batch['pose_matrix_context'],
                                                              batch['same_timestep_as_origin_context'])
            output['loss'] += self._camera_distance_loss_weight * camera_distance_loss['loss']
        return output
