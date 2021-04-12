# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn
import math

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_fisheye_valeo_utils import scale_intrinsics, get_roots_table_tensor
from packnet_sfm.utils.image import image_grid

########################################################################################################################

class CameraFisheye(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self,
                 poly_coeffs,
                 principal_point=torch.Tensor([0., 0.]),
                 scale_factors=torch.Tensor([1., 1.]),
                 Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        intrinsics : dictionary (keys : ax, ay, cx, cy, c1, c2, c3, c5)
            Camera intrinsics
            poly_coeffs [c1, c2, c3, c4]
            principal_point [cx, cy]
            scale_factors [ax, ay]
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.poly_coeffs = poly_coeffs
        self.principal_point = principal_point
        self.scale_factors = scale_factors
        #self.K = K
        self.Tcw = Pose.identity(3) if Tcw is None else Tcw#Pose.identity(len(K)) if Tcw is None else Tcw

    # def __len__(self):
    #     """Batch size of the camera intrinsics"""
    #     return len(self.K)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.poly_coeffs = self.poly_coeffs.to(*args, **kwargs)
        self.principal_point = self.principal_point.to(*args, **kwargs)
        self.scale_factors = self.scale_factors.to(*args, **kwargs)
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
        return CameraFisheye(K, Tcw=self.Tcw)

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

        theta_tensor = get_roots_table_tensor(self.poly_coeffs, self.principal_point, self.scale_factors, H, W)
        flat_theta_tensor = theta_tensor.flatten()

        # Estimate the outward rays in the camera frame
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
        # Scale rays to metric depth
        Xc = xnorm * depth

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

        # World to camera:
        if frame == 'c':
            Xc = X.view(B, 3, -1) # [B, 3, HW]
        elif frame == 'w':
            Xc = (self.Tcw @ X).view(B, 3, -1) # [B, 3, HW]
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        # Project 3D points onto the camera image plane
        X = Xc[:, 0] # [B, HW]
        Y = Xc[:, 1] # [B, HW]
        Z = Xc[:, 2] # [B, HW]
        phi = torch.atan2(Y, X) # [B, HW]
        rc = torch.sqrt(torch.pow(X, 2) + torch.pow(Y, 2)) # [B, HW]
        theta_1 = math.pi / 2 - torch.atan2(Z, rc) # [B, HW]
        theta_2 = torch.pow(theta_1, 2) # [B, HW]
        theta_3 = torch.pow(theta_1, 3) # [B, HW]
        theta_4 = torch.pow(theta_1, 4) # [B, HW]
        rho = self.poly_coeffs[0] * theta_1 \
              + self.poly_coeffs[1] * theta_2 \
              + self.poly_coeffs[2] * theta_3 \
              + self.poly_coeffs[3] * theta_4 # [B, HW]
        u = rho * torch.cos(phi) * self.scale_factors[0] + self.principal_point[0] # [B, HW]
        v = rho * torch.sin(phi) * self.scale_factors[1] + self.principal_point[1] # [B, HW]

        # Normalize points
        Xnorm = 2 * u / (W - 1) - 1.
        Ynorm = 2 * v / (H - 1) - 1.

        # Clamp out-of-bounds pixels
        # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
        # Xnorm[Xmask] = 2.
        # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
        # Ynorm[Ymask] = 2.

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
