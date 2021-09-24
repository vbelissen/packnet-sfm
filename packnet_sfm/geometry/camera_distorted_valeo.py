# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_utils import scale_intrinsics
from packnet_sfm.utils.image import image_grid

import numpy as np

########################################################################################################################

class CameraDistorted(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self, K, k1=0, k2=0, k3=0, p1=0, p2=0, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        K : torch.Tensor [3,3]
            Camera intrinsics
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.K = K
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.Tcw = Pose.identity(len(K)) if Tcw is None else Tcw

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.K)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.K = self.K.to(*args, **kwargs)
        self.k1 = self.k1.to(*args, **kwargs)
        self.k2 = self.k2.to(*args, **kwargs)
        self.k3 = self.k3.to(*args, **kwargs)
        self.p1 = self.p1.to(*args, **kwargs)
        self.p2 = self.p2.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

########################################################################################################################

    @property
    def fx(self):
        """Focal length in x"""
        return self.K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[:, 1, 2]

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

    @property
    @lru_cache()
    def Kinv(self):
        """Inverse intrinsics (for lifting)"""
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv

########################################################################################################################

    def scaled(self, x_scale, y_scale=None):
        """
        Returns a scaled version of the camera (changing intrinsics)

        Parameters
        ----------
        x_scale : float
            Resize scale in x
        y_scale : float
            Resize scale in y. If None, use the same as x_scale

        Returns
        -------
        camera : Camera
            Scaled version of the current cmaera
        """
        # If single value is provided, use for both dimensions
        if y_scale is None:
            y_scale = x_scale
        # If no scaling is necessary, return same camera
        if x_scale == 1. and y_scale == 1.:
            return self
        # Scale intrinsics and return new camera with same Pose
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return CameraDistorted(K, k1=self.k1, k2=self.k2, k3=self.k3, p1=self.p1, p2=self.p2, Tcw=self.Tcw)

########################################################################################################################

    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        B, C, H, W = depth.shape
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        device = depth.get_device()

        # Estimate the outward undistored rays in the camera frame
        Xnorm_u = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)

        version = 'v1'
        N = 5

        if version == 'v1':
            x = Xnorm_u[:, 0, :, :].view(B, 1, H, W)
            y = Xnorm_u[:, 1, :, :].view(B, 1, H, W)

            x_src = torch.clone(x)
            y_src = torch.clone(y)

            for _ in range(N):
                r2 = x.pow(2) + y.pow(2)
                r4 = r2.pow(2)
                r6 = r2 * r4

                rad_dist = 1 / (1 + self.k1.view(B,1,1,1) * r2 + self.k2.view(B,1,1,1) * r4 + self.k3.view(B,1,1,1) * r6)
                tang_dist_x = 2 * self.p1.view(B,1,1,1) * x * y + self.p2.view(B,1,1,1) * (r2 + 2 * x.pow(2))
                tang_dist_y = 2 * self.p2.view(B,1,1,1) * x * y + self.p1.view(B,1,1,1) * (r2 + 2 * y.pow(2))

                x = (x_src - tang_dist_x) * rad_dist
                y = (y_src - tang_dist_y) * rad_dist

            # Distorted rays
            Xnorm_d = torch.stack([x.squeeze(1), y.squeeze(1), torch.ones_like(x).squeeze(1)], dim=1)

        elif version == 'v2':
            x_list = []
            y_list = []

            x_list.append(Xnorm_u[:, 0, :, :].view(B, 1, H, W))
            y_list.append(Xnorm_u[:, 1, :, :].view(B, 1, H, W))

            for n in range(N):
                r2 = x_list[-1].pow(2) + y_list[-1].pow(2)
                r4 = r2.pow(2)
                r6 = r2 * r4

                rad_dist = 1 / (1 + self.k1.view(B,1,1,1) * r2 + self.k2.view(B,1,1,1) * r4 + self.k3.view(B,1,1,1) * r6)
                tang_dist_x = 2 * self.p1.view(B,1,1,1) * x_list[-1] * y_list[-1] + self.p2.view(B,1,1,1) * (r2 + 2 * x_list[-1].pow(2))
                tang_dist_y = 2 * self.p2.view(B,1,1,1) * x_list[-1] * y_list[-1] + self.p1.view(B,1,1,1) * (r2 + 2 * y_list[-1].pow(2))

                x_list.append((x_list[0] - tang_dist_x) * rad_dist)
                y_list.append((y_list[0] - tang_dist_y) * rad_dist)

            # Distorted rays
            Xnorm_d = torch.stack([x_list[-1].squeeze(1), y_list[-1].squeeze(1), torch.ones_like(x_list[-1]).squeeze(1)], dim=1)

        elif version == 'v3':
            Xnorm_d = torch.zeros(B, 3, H, W)
            for b in range(B):
                Xnorm_d[b] = torch.from_numpy(np.load(self.path_to_theta_lut[b]))

        Xnorm_d = Xnorm_d.to(device)

        # ATTENTION RENORMALISER Xnorm_d

        #Xnorm_d /= norm(Xnorm_d)
        # Scale rays to metric depth
        Xc = ((Xnorm_d / torch.sqrt((Xnorm_d[:, 0, :, :].pow(2)
                                    + Xnorm_d[:, 1, :, :].pow(2)
                                    + Xnorm_d[:, 2, :, :].pow(2)).clamp(min=1e-5)).view(B, 1, H, W)) * depth).float()

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return self.Twc @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project(self, X, frame='w'):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            Xc = X.view(B, 3, -1)
        elif frame == 'w':
            Xc = (self.Tcw @ X).view(B, 3, -1)
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        # Z-Normalize points
        Z  = Xc[:, 2].clamp(min=1e-5)
        Xn = Xc[:, 0] / Z
        Yn = Xc[:, 1] / Z

        r2 = Xn.pow(2) + Yn.pow(2)
        r4 = r2.pow(2)
        r6 = r2 * r4

        # Distorted normalized points
        rad_dist = (1 + self.k1.view(B,1) * r2 + self.k2.view(B,1) * r4 + self.k3.view(B,1) * r6)
        Xd = Xn * rad_dist + 2 * self.p1.view(B,1) * Xn * Yn + self.p2.view(B,1) * (r2 + 2 * Xn.pow(2))
        Yd = Yn * rad_dist + 2 * self.p2.view(B,1) * Xn * Yn + self.p1.view(B,1) * (r2 + 2 * Yn.pow(2))

        # Final projection
        print(self.K)
        print(self.K.shape)
        print(self.fx.shape)
        print(Xd.shape)
        print(self.cx.shape)
        u = self.fx * Xd + self.cx
        v = self.fy * Yd + self.cy

        # normalized coordinates
        uNorm = 2 * u / (W - 1) - 1.
        vNorm = 2 * v / (H - 1) - 1.

        # Return pixel coordinates
        return torch.stack([uNorm, vNorm], dim=-1).view(B, H, W, 2)
