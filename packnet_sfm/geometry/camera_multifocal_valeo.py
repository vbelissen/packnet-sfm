# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import lru_cache
import torch
import torch.nn as nn
import math
import numpy as np
import os

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_fisheye_valeo_utils import scale_intrinsics_fisheye
from packnet_sfm.geometry.camera_utils import scale_intrinsics
from packnet_sfm.utils.image_valeo import image_grid, centered_2d_grid, meshgrid


########################################################################################################################

class CameraMultifocal(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self,
                 poly_coeffs, principal_point, scale_factors,
                 K, k1, k2, k3, p1, p2,
                 camera_type, #int Tensor ; 0 is fisheye, 1 is distorted, 2 is other
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
        self.K = K
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.camera_type = camera_type
        self.Tcw = Pose.identity(len(camera_type)) if Tcw is None else Tcw

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.camera_type)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.poly_coeffs = self.poly_coeffs.to(*args, **kwargs)
        self.principal_point = self.principal_point.to(*args, **kwargs)
        self.scale_factors = self.scale_factors.to(*args, **kwargs)
        self.K = self.K.to(*args, **kwargs)
        self.k1 = self.k1.to(*args, **kwargs)
        self.k2 = self.k2.to(*args, **kwargs)
        self.k3 = self.k3.to(*args, **kwargs)
        self.p1 = self.p1.to(*args, **kwargs)
        self.p2 = self.p2.to(*args, **kwargs)
        self.camera_type = self.camera_type.to(*args, **kwargs)
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

    def mask_batch_fisheye(self):
        return self.camera_type == 0

    def mask_batch_distorted(self):
        return self.camera_type == 1

    # def idx_batch_fisheye(self):
    #     return torch.where(self.mask_batch_fisheye)[0]
    #
    # def idx_batch_distorted(self):
    #     return torch.where(self.mask_batch_distorted)[0]

    def n_batch_fisheye(self):
        return (self.mask_batch_fisheye()).sum()

    def n_batch_distorted(self):
        return (self.mask_batch_distorted()).sum()

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
        if y_scale is not None:
            assert y_scale == x_scale
        # If no scaling is necessary, return same camera
        if x_scale == 1.:
            return self
        # Scale intrinsics and return new camera with same Pose
        poly_coeffs, principal_point = scale_intrinsics_fisheye(self.poly_coeffs.clone(), self.principal_point.clone(), x_scale)
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return CameraMultifocal(poly_coeffs=poly_coeffs, principal_point=principal_point, scale_factors=self.scale_factors,
                                K=K, k1=self.k1, k2=self.k2, k3=self.k3, p1=self.p1, p2=self.p2,
                                Tcw=self.Tcw)

########################################################################################################################

    def reconstruct(self, depth, frame='w'):

        B, C, H, W = depth.shape
        device = depth.get_device()

        points = torch.zeros(B, 3, H, W).float().to(device)

        if self.n_batch_fisheye() > 0:
            mask_fisheye = self.mask_batch_fisheye()
            points[mask_fisheye] = self.reconstruct_fisheye(depth, mask_fisheye, frame)

        if self.n_batch_distorted() > 0:
            mask_distorted = self.mask_batch_distorted()
            points[mask_distorted] = self.reconstruct_distorted(depth, mask_distorted, frame)

        return points

    def project(self, X, frame='w'):

        B, C, H, W = X.shape
        device = X.get_device()

        coords = torch.zeros(B, H, W, 2).float().to(device)

        if self.n_batch_fisheye() > 0:
            mask_fisheye = self.mask_batch_fisheye()
            coords[mask_fisheye] = self.reconstruct_fisheye(X, mask_fisheye, frame)

        if self.n_batch_distorted() > 0:
            mask_distorted = self.mask_batch_distorted()
            coords[mask_distorted] = self.reconstruct_distorted(X, mask_distorted, frame)

        return coords

    def reconstruct_fisheye(self, depth, mask, frame='w'):
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
        B, C, H, W = depth[mask].shape
        device = depth.get_device()
        print(depth.dtype)
        assert C == 1

        xi, yi = meshgrid(B, H, W, depth.dtype, depth.device, normalized=False)
        print(xi.dtype)

        xi = ((xi - (W - 1) / 2 - self.principal_point[mask, 0].unsqueeze(1).unsqueeze(2).repeat([1, H, W]))
              * self.scale_factors[mask, 0].unsqueeze(1).unsqueeze(2).repeat([1, H, W])).unsqueeze(1)
        yi = ((yi - (H - 1) / 2 - self.principal_point[mask, 1].unsqueeze(1).unsqueeze(2).repeat([1, H, W]))
              * self.scale_factors[mask, 1].unsqueeze(1).unsqueeze(2).repeat([1, H, W])).unsqueeze(1)

        N = 12
        theta_tensor = (torch.zeros(B, 1, H, W)).to(device)
        ri = torch.sqrt(xi.pow(2) + yi.pow(2))
        for _ in range(N):
            t1 = theta_tensor
            t2 = theta_tensor * t1
            t3 = theta_tensor * t2
            t4 = theta_tensor * t3
            theta_tensor = t1 + .5 * (ri - (self.poly_coeffs[mask, 0].view(B, 1, 1, 1) * t1
                                            + self.poly_coeffs[mask, 1].view(B, 1, 1, 1) * t2
                                            + self.poly_coeffs[mask, 2].view(B, 1, 1, 1) * t3
                                            + self.poly_coeffs[mask, 3].view(B, 1, 1, 1) * t4)) \
                                   / (self.poly_coeffs[mask, 0].view(B, 1, 1, 1)
                                      + 2 * self.poly_coeffs[mask, 1].view(B, 1, 1, 1) * t1
                                      + 3 * self.poly_coeffs[mask, 2].view(B, 1, 1, 1) * t2
                                      + 4 * self.poly_coeffs[mask, 3].view(B, 1, 1, 1) * t3)
            # l'astuce pour que ça marche a été de multiplier la mise à jour par 0.5 (au lieu de 1 selon Newton...)

        #get_roots_table_tensor(self.poly_coeffs, self.principal_point, self.scale_factors, H, W).to(device)

        rc = depth[mask] * torch.sin(theta_tensor)

        #yi, xi = centered_2d_grid(B, H, W, depth.dtype, depth.device, self.principal_point, self.scale_factors)


        phi = torch.atan2(yi, xi).to(device)

        xc = rc * torch.cos(phi)
        yc = rc * torch.sin(phi)
        zc = depth[mask] * torch.cos(theta_tensor)
        #print(zc[0, 0, :, 127])

        # mask = (depth == 0).detach()
        # xc[mask] = 0.
        # yc[mask] = 0.
        # zc[mask] = 0.

        Xc = torch.cat([xc, yc, zc], dim=1)

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            print(Xc.dtype)
            print(yi.dtype)
            print(ri.dtype)
            print(rc.dtype)
            print(phi.dtype)
            print(theta_tensor.dtype)
            print(zc.dtype)
            print(self.Twc.mat[mask].dtype)
            return Pose(self.Twc.mat[mask]) @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project_fisheye(self, X, mask, frame='w'):
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
        B, C, H, W = X[mask].shape
        assert C == 3

        # World to camera:
        if frame == 'c':
            Xc = X.view(B, 3, -1) # [B, 3, HW]
        elif frame == 'w':
            Xc = (Pose(self.Tcw.mat[mask]) @ X).view(B, 3, -1) # [B, 3, HW]
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        c1 = self.poly_coeffs[mask, 0].unsqueeze(1)
        c2 = self.poly_coeffs[mask, 1].unsqueeze(1)
        c3 = self.poly_coeffs[mask, 2].unsqueeze(1)
        c4 = self.poly_coeffs[mask, 3].unsqueeze(1)

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

        rho = c1 * theta_1 + c2 * theta_2 + c3 * theta_3 + c4 * theta_4 # [B, HW]
        rho = rho * ((X != 0) | (Y != 0) | (Z != 0))
        u = rho * torch.cos(phi) / self.scale_factors[mask, 0].unsqueeze(1) + self.principal_point[mask, 0].unsqueeze(1) # [B, HW]
        v = rho * torch.sin(phi) / self.scale_factors[mask, 1].unsqueeze(1) + self.principal_point[mask, 1].unsqueeze(1) # [B, HW]

        # Normalize points
        Xnorm = 2 * u / (W - 1)# - 1.
        Ynorm = 2 * v / (H - 1)# - 1.

        # Clamp out-of-bounds pixels
        # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
        # Xnorm[Xmask] = 2.
        # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
        # Ynorm[Ymask] = 2.

        mask_out = (theta_1 * 180 * 2 / np.pi > 190.0).detach()
        Xnorm[mask_out] = 2.
        Ynorm[mask_out] = 2.

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)

    def reconstruct_distorted(self, depth, mask, frame='w'):
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
        B, C, H, W = depth[mask].shape
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        device = depth.get_device()

        # Estimate the outward undistored rays in the camera frame
        Xnorm_u = (self.Kinv[mask].bmm(flat_grid)).view(B, 3, H, W)

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

                rad_dist = 1 / (1 + self.k1[mask].view(B,1,1,1) * r2 + self.k2[mask].view(B,1,1,1) * r4 + self.k3[mask].view(B,1,1,1) * r6)
                tang_dist_x = 2 * self.p1[mask].view(B,1,1,1) * x * y + self.p2[mask].view(B,1,1,1) * (r2 + 2 * x.pow(2))
                tang_dist_y = 2 * self.p2[mask].view(B,1,1,1) * x * y + self.p1[mask].view(B,1,1,1) * (r2 + 2 * y.pow(2))

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

                rad_dist = 1 / (1 + self.k1[mask].view(B,1,1,1) * r2 + self.k2[mask].view(B,1,1,1) * r4 + self.k3[mask].view(B,1,1,1) * r6)
                tang_dist_x = 2 * self.p1[mask].view(B,1,1,1) * x_list[-1] * y_list[-1] + self.p2[mask].view(B,1,1,1) * (r2 + 2 * x_list[-1].pow(2))
                tang_dist_y = 2 * self.p2[mask].view(B,1,1,1) * x_list[-1] * y_list[-1] + self.p1[mask].view(B,1,1,1) * (r2 + 2 * y_list[-1].pow(2))

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
                                    + Xnorm_d[:, 2, :, :].pow(2)).clamp(min=1e-5)).view(B, 1, H, W)) * depth[mask]).float()

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return Pose(self.Twc.mat[mask]) @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

    def project_distorted(self, X, mask, frame='w'):
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
        B, C, H, W = X[mask].shape
        assert C == 3

        # Project 3D points onto the camera image plane
        if frame == 'c':
            Xc = X.view(B, 3, -1)
        elif frame == 'w':
            Xc = (Pose(self.Tcw.mat[mask]) @ X).view(B, 3, -1)
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
        rad_dist = (1 + self.k1[mask].view(B,1) * r2 + self.k2[mask].view(B,1) * r4 + self.k3[mask].view(B,1) * r6)
        Xd = Xn * rad_dist + 2 * self.p1[mask].view(B,1) * Xn * Yn + self.p2[mask].view(B,1) * (r2 + 2 * Xn.pow(2))
        Yd = Yn * rad_dist + 2 * self.p2[mask].view(B,1) * Xn * Yn + self.p1[mask].view(B,1) * (r2 + 2 * Yn.pow(2))

        # Final projection
        u = self.fx[mask].view(B,1) * Xd + self.cx[mask].view(B,1)
        v = self.fy[mask].view(B,1) * Yd + self.cy[mask].view(B,1)

        # normalized coordinates
        uNorm = 2 * u / (W - 1) - 1.
        vNorm = 2 * v / (H - 1) - 1.

        # Return pixel coordinates
        return torch.stack([uNorm, vNorm], dim=-1).view(B, H, W, 2).float()

