# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import math
import numpy as np
import os
from tqdm import tqdm

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_fisheye_valeo_utils import scale_intrinsics_fisheye, get_roots_table_tensor
from packnet_sfm.utils.image_valeo import image_grid, centered_2d_grid, meshgrid

from scipy.spatial.transform import Rotation as R

from PIL import Image
import os
import re
import argparse

import cv2
import open3d as o3d

main_folder = '/home/vbelissen/test_data/valeo_data_ready2train/data/dataset_valeo_cea_2017_2018/'
seq_idx     = '20170320_144339'
img_idx     = '00011702'

path_to_theta_lut = [main_folder + 'images/fisheye/train/' + seq_idx + '/cam_0/theta_tensor_1280_800.npy']
poly_coeffs       = torch.Tensor([282.85, -27.8671, 114.318, -36.6703]).unsqueeze(0)
principal_point   = torch.Tensor([0.046296, -7.33178]).unsqueeze(0)
scale_factors     = torch.Tensor([1., 1./1.00173]).unsqueeze(0)
Tcw               = Pose.identity(len(poly_coeffs))
Twc               = Tcw.inverse()

r = R.from_quat([1, 0, 0, 0])

depth_map_valeo = np.zeros((1, 1, 800, 1280))
depth_map_valeo[0, 0, :, :] = \
    np.load(main_folder + 'depth_maps/fisheye/train/' + seq_idx + '/velodyne_0/' + seq_idx + '_velodyne_0_' + img_idx + '.npz')['velodyne_depth']
depth_map_valeo = depth_map_valeo.astype('float32')

depth_map_valeo_tensor = torch.from_numpy(depth_map_valeo)

def reconstruct(depth, frame='w'):
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


    theta_tensor = torch.zeros(B, 1, H, W)
    for b in range(B):
        theta_tensor[b, 0] = torch.from_numpy(np.load(path_to_theta_lut[b]))
    theta_tensor = theta_tensor
    #get_roots_table_tensor(self.poly_coeffs, self.principal_point, self.scale_factors, H, W).to(device)

    rc = depth * torch.sin(theta_tensor)

    # yi, xi = centered_2d_grid(B, H, W, depth.dtype, depth.device, self.principal_point, self.scale_factors)

    xi, yi = meshgrid(B, H, W, depth.dtype, depth.device, normalized=False)

    xi = ((xi - (W - 1) / 2 - principal_point[:, 0].unsqueeze(1).unsqueeze(2).repeat([1, H, W]))
          / scale_factors[:, 0].unsqueeze(1).unsqueeze(2).repeat([1, H, W])
          ).unsqueeze(1)
    yi = ((yi - (H - 1) / 2 - principal_point[:, 1].unsqueeze(1).unsqueeze(2).repeat([1, H, W]))
          / scale_factors[:, 1].unsqueeze(1).unsqueeze(2).repeat([1, H, W])
          ).unsqueeze(1)
    phi = torch.atan2(yi, xi)

    xc = rc * torch.cos(phi)
    yc = rc * torch.sin(phi)
    zc = depth * torch.cos(theta_tensor)
    #print(zc[0, 0, :, 127])

    Xc = torch.cat([xc, yc, zc], dim=1)

    print(xc.shape)
    print(yc.shape)
    print(zc.shape)
    print(Xc.shape)
    print(Twc.shape)

    # If in camera frame of reference
    if frame == 'c':
        return Xc
    # If in world frame of reference
    elif frame == 'w':
        return Twc @ Xc
    # If none of the above
    else:
        raise ValueError('Unknown reference frame {}'.format(frame))

def project(X, frame='w'):
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
        Xc = (Tcw @ X).view(B, 3, -1) # [B, 3, HW]
    else:
        raise ValueError('Unknown reference frame {}'.format(frame))

    c1 = poly_coeffs[:, 0].unsqueeze(1)
    c2 = poly_coeffs[:, 1].unsqueeze(1)
    c3 = poly_coeffs[:, 2].unsqueeze(1)
    c4 = poly_coeffs[:, 3].unsqueeze(1)

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
    u = rho * torch.cos(phi) * scale_factors[:, 0].unsqueeze(1) + principal_point[:, 0].unsqueeze(1) # [B, HW]
    v = rho * torch.sin(phi) * scale_factors[:, 1].unsqueeze(1) + principal_point[:, 1].unsqueeze(1) # [B, HW]

    # Normalize points
    Xnorm = 2 * u / (W - 1)# - 1.
    Ynorm = 2 * v / (H - 1)# - 1.

    # Clamp out-of-bounds pixels
    # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
    # Xnorm[Xmask] = 2.
    # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
    # Ynorm[Ymask] = 2.

    # Return pixel coordinates
    return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)

print(depth_map_valeo.shape)
max_depth = np.max(depth_map_valeo[np.nonzero(depth_map_valeo)])
cv2.imshow("depth_map_valeo", cv2.applyColorMap((depth_map_valeo[0, 0]/max_depth*255).astype(np.uint8), cv2.COLORMAP_HSV))
cv2.waitKey()

valeo_point_cloud = reconstruct(depth_map_valeo_tensor, frame='w')
print(valeo_point_cloud.shape)
valeo_point_cloud_flattened = valeo_point_cloud.view(1, 3, -1).transpose(1, 2).squeeze()



pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(valeo_point_cloud_flattened)
pcl.paint_uniform_color([1.0, 0.0, 0])

depth_map_valeo_reprojected = project(valeo_point_cloud, frame='w')
depth_map_valeo_reprojected_numpy = depth_map_valeo_reprojected.numpy()
print(depth_map_valeo_reprojected_numpy.shape)

new_depth_map = np.zeros((800, 1280))
for h in tqdm(range(800)):
    for w in range(1280):
        u = int((1280-1)/2*(1+depth_map_valeo_reprojected_numpy[0, h, w, 0]))
        v = int((800-1)/2*(1+depth_map_valeo_reprojected_numpy[0, h, w, 1]))
        cop = np.zeros(3)
        #cop[0:2] = principal_point.numpy()[0, :]
        #cop = np.array([ 3.691   , -0.      ,  0.474331]).astype('float32')
        d = np.linalg.norm(valeo_point_cloud[0, :, h, w] - cop)
        if 0 <= v < 800 and 0 <= u < 1280:
            new_depth_map[v, u] = d
        else:
            print('error')

max_depth_new = np.max(new_depth_map[np.nonzero(new_depth_map)])
cv2.imshow("new_depth_map", cv2.applyColorMap((new_depth_map/max_depth_new*255).astype(np.uint8), cv2.COLORMAP_HSV))
cv2.waitKey()

# pcl_reconstruct = o3d.geometry.PointCloud()
# pcl_reconstruct.points = o3d.utility.Vector3dVector(point3d)
# pcl_reconstruct.paint_uniform_color([0.0, 0.0, 1.0])

new_depth_map_tensor = np.zeros((1, 1, 800, 1280))
new_depth_map_tensor[0, 0] = new_depth_map
new_depth_map_tensor = new_depth_map_tensor.astype('float32')
new_depth_map_tensor = torch.from_numpy(new_depth_map_tensor)

valeo_point_cloud_2 = reconstruct(new_depth_map_tensor, frame='w')
print(valeo_point_cloud_2.shape)
valeo_point_cloud_2_flattened = valeo_point_cloud_2.view(1, 3, -1).transpose(1, 2).squeeze()

pcl_2 = o3d.geometry.PointCloud()
pcl_2.points = o3d.utility.Vector3dVector(valeo_point_cloud_2_flattened)
pcl_2.paint_uniform_color([0.0, 1.0, 0])

o3d.visualization.draw_geometries([pcl, pcl_2])#, pcl_reconstruct])

