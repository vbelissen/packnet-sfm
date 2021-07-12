# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as funct
from scipy import optimize
from functools import lru_cache
from packnet_sfm.utils.image_valeo import centered_2d_grid


########################################################################################################################


def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Construct a [3,3] camera intrinsics from pinhole parameters"""
    return torch.tensor([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=dtype, device=device)


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


def scale_intrinsics_fisheye(poly_coeffs, principal_point, x_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    return poly_coeffs * x_scale, principal_point * x_scale

def scale_path_to_theta_lut(path_to_theta_lut, x_scale):
    L = len(path_to_theta_lut)
    path_to_theta_lut_clone = []
    #path_to_theta_lut_clone = path_to_theta_lut
    for l in range(L):
        splitted_path = path_to_theta_lut[l].split('_')
        w_res_str = splitted_path[-2]
        splitted_path[-2] = str(int(x_scale * int(w_res_str)))
        h_res_str_with_ext = splitted_path[-1]
        h_res_str_splitted = h_res_str_with_ext.split('.')
        h_res_new = str(int(x_scale * int(h_res_str_splitted[0])))
        splitted_path[-1] = '.'.join([h_res_new, h_res_str_splitted[1]])
        path_to_theta_lut_clone.append('_'.join(splitted_path))
    return path_to_theta_lut_clone

@lru_cache()
def get_roots_table_tensor(poly_coeffs, principal_point, scale_factors, H, W):
    theta_tensor = torch.zeros(H, W)

    c1 = poly_coeffs.squeeze()[0].cpu()
    c2 = poly_coeffs.squeeze()[0].cpu()
    c3 = poly_coeffs.squeeze()[0].cpu()
    c4 = poly_coeffs.squeeze()[0].cpu()

    yi, xi = centered_2d_grid(H, W, principal_point, scale_factors)

    yi = yi.cpu()
    xi = xi.cpu()

    # print(xi)
    # print(yi)

    def fun_rho_jac(theta):
        return c1 + 2 * c2 * theta + 3 * c3 * theta ** 2 + 4 * c4 * theta ** 3

    i = 0
    for u in range(0, W):
        for v in range(0, H):
            def fun_rho(theta):
                return c1 * theta + c2 * theta ** 2 + c3 * theta ** 3 + c4 * theta ** 4 - (xi[v, u] ** 2 + yi[v, u] ** 2) ** .5

            theta_tensor[v, u] = (optimize.root(fun_rho, [0], jac=fun_rho_jac, method='hybr').x)[0]
            print(i/(H*W))
            i += 1
    return theta_tensor


########################################################################################################################


def view_synthesis(ref_image, depth, ref_cam, cam,
                   mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')
    # View-synthesis given the projected reference points
    return funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=align_corners)


########################################################################################################################

def view_depth_synthesis(ref_image, depth, ref_depth, ref_cam, cam,
                   mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')
    # View-synthesis given the projected reference points
    ref_world_points = ref_cam.reconstruct(ref_depth, frame='w')
    ref_world_points_in_target_coords = cam.Tcw @ ref_world_points
    ref_depth_synthesis_in_target_coords = torch.norm(ref_world_points_in_target_coords, dim=1, keepdim=True)
    return funct.grid_sample(ref_image,                            ref_coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners), \
           funct.grid_sample(ref_depth_synthesis_in_target_coords, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def view_depth_synthesis2(ref_image, depth, ref_depth, ref_cam, cam,
                   mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    depth_wrt_ref_cam = torch.norm(ref_cam.Tcw @ world_points, dim=1, keepdim=True)
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')
    # View-synthesis given the projected reference points
    return funct.grid_sample(ref_image, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners), \
           depth_wrt_ref_cam, \
           funct.grid_sample(ref_depth, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


########################################################################################################################


def view_synthesis_generic(ref_image, depth, ref_cam, cam,
                           mode='bilinear', padding_mode='zeros', progress=0.0):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, progress=progress, frame='w')
    # View-synthesis given the projected reference points
    return funct.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)

########################################################################################################################
