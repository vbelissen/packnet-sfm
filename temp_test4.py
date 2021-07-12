import packnet_sfm.geometry.camera_fisheye_valeo
from packnet_sfm.geometry.camera_fisheye_valeo import CameraFisheye
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as funct
from PIL import Image
import cv2
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.datasets.kitti_based_valeo_dataset_utils import pose_from_oxts_packet, read_calib_file, read_raw_calib_files_camera_valeo, transform_from_rot_trans
from packnet_sfm.geometry.pose_utils import invert_pose_numpy
import time
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth, depth2inv
from cv2 import imwrite



simulated_depth=torch.zeros(1,1,800,1280)
for i in range(800):
    for j in range(1280):
        simulated_depth[0,0,i,j] = i*(799-i)*j*(1279-j)*i*(799-i)*j*(1279-j)/6.5270e+10/6.5270e+10*50
simulated_depth = simulated_depth.to(torch.device('cuda'))

simulated_depth = torch.from_numpy(np.load('/home/data/vbelissen/20170320_163113_cam_0_00006300.npz')['depth']).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)


img_front = Image.open('/home/data/vbelissen/valeo_multiview/images_multiview/fisheye/test/20170320_163113/cam_0/20170320_163113_cam_0_00006300.jpg').convert('RGB')
img_front_next = Image.open('/home/data/vbelissen/valeo_multiview/images_multiview/fisheye/test/20170320_163113/cam_0/20170320_163113_cam_0_00006301.jpg').convert('RGB')

img_left  = Image.open('/home/data/vbelissen/valeo_multiview/images_multiview/fisheye/test/20170320_163113/cam_3/20170320_163113_cam_3_00006286.jpg').convert('RGB')
img_right = Image.open('/home/data/vbelissen/valeo_multiview/images_multiview/fisheye/test/20170320_163113/cam_1/20170320_163113_cam_1_00006288.jpg').convert('RGB')

front_img_torch = torch.transpose(torch.from_numpy(np.array(img_front)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)
front_next_img_torch = torch.transpose(torch.from_numpy(np.array(img_front_next)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)
left_img_torch  = torch.transpose(torch.from_numpy(np.array(img_left)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)
right_img_torch = torch.transpose(torch.from_numpy(np.array(img_right)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)

path_to_ego_mask_front = '/home/data/vbelissen/valeo_multiview/semantic_masks/fisheye/test/20170320_163113/20170320_163113_cam_0.npy'
ego_mask_front = np.load(path_to_ego_mask_front)
not_masked_front = torch.from_numpy(ego_mask_front.astype(bool)).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

path_to_ego_mask_left = '/home/data/vbelissen/valeo_multiview/semantic_masks/fisheye/test/20170320_163113/20170320_163113_cam_3.npy'
ego_mask_left = np.load(path_to_ego_mask_left)
not_masked_left = torch.from_numpy(ego_mask_left.astype(bool)).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

path_to_ego_mask_right = '/home/data/vbelissen/valeo_multiview/semantic_masks/fisheye/test/20170320_163113/20170320_163113_cam_1.npy'
ego_mask_right = np.load(path_to_ego_mask_right)
not_masked_right = torch.from_numpy(ego_mask_front.astype(bool)).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)

simulated_depth[~not_masked_front] = 0

t_front = np.array([3.691,     0, 0.474331])
t_left  = np.array([2.011, 0.928, 0.835443])
t_right  = np.array([2.011, -0.928, 0.836435])

rx_front_rad  = 60.471   * np.pi/180
rx_front_mod  = rx_front_rad + np.pi
rz1_front_rad = 267.6573 * np.pi/180
rz2_front_rad = -1.9011  * np.pi/180

rx_left_rad   = 52.6295  * np.pi/180
rx_left_mod   = rx_left_rad + np.pi
rz1_left_rad  = 355.598  * np.pi/180
rz2_left_rad  = -1.36427 * np.pi/180

rx_right_rad   = 52.2021  * np.pi/180
rx_right_mod   = rx_right_rad + np.pi
rz1_right_rad  = 176.26216  * np.pi/180
rz2_right_rad  = 0.691546 * np.pi/180

cx_front  = np.cos(rx_front_mod)
sx_front  = np.sin(rx_front_mod)
cz1_front = np.cos(rz1_front_rad)
sz1_front = np.sin(rz1_front_rad)
cz2_front = np.cos(rz2_front_rad)
sz2_front = np.sin(rz2_front_rad)

cx_left   = np.cos(rx_left_mod)
sx_left   = np.sin(rx_left_mod)
cz1_left  = np.cos(rz1_left_rad)
sz1_left  = np.sin(rz1_left_rad)
cz2_left  = np.cos(rz2_left_rad)
sz2_left  = np.sin(rz2_left_rad)

cx_right   = np.cos(rx_right_mod)
sx_right   = np.sin(rx_right_mod)
cz1_right  = np.cos(rz1_right_rad)
sz1_right  = np.sin(rz1_right_rad)
cz2_right  = np.cos(rz2_right_rad)
sz2_right  = np.sin(rz2_right_rad)


Rx_front  = np.array([[        1,          0, 0], [         0,  cx_front, sx_front], [0, -sx_front, cx_front]])
Rz2_front = np.array([[cz2_front, -sz2_front, 0], [ sz2_front, cz2_front,        0], [0,         0,        1]])
Rz1_front = np.array([[cz1_front,  sz1_front, 0], [-sz1_front, cz1_front,        0], [0,         0,        1]])

Rx_left   = np.array([[        1,          0, 0], [         0,   cx_left,  sx_left], [0,  -sx_left,  cx_left]])
Rz1_left  = np.array([[ cz1_left,   sz1_left, 0], [ -sz1_left,  cz1_left,        0], [0,         0,        1]])
Rz2_left  = np.array([[ cz2_left,  -sz2_left, 0], [  sz2_left,  cz2_left,        0], [0,         0,        1]])

Rx_right   = np.array([[        1,          0, 0], [        0,  cx_right,  sx_right], [0, -sx_right, cx_right]])
Rz1_right  = np.array([[cz1_right,  sz1_right, 0], [-sz1_right, cz1_right,        0], [0,        0,        1]])
Rz2_right  = np.array([[cz2_right, -sz2_right, 0], [ sz2_right, cz2_right,        0], [0,        0,        1]])

R_front   = np.matmul(Rz2_front, np.matmul(Rx_front, Rz1_front))
R_left    = np.matmul(Rz2_left,  np.matmul(Rx_left,  Rz1_left))
R_right   = np.matmul(Rz2_right, np.matmul(Rx_right, Rz1_right))

T_other_front  = -np.dot(R_front, t_front)
T_other_left   = -np.dot(R_left,  t_left)
T_other_right  = -np.dot(R_right, t_right)

pose_matrix_front = transform_from_rot_trans(R_front, T_other_front).astype(np.float32)
pose_matrix_left  = transform_from_rot_trans(R_left,  T_other_left).astype(np.float32)
pose_matrix_right = transform_from_rot_trans(R_right, T_other_right).astype(np.float32)

pose_matrix_left  = pose_matrix_left @ invert_pose_numpy(pose_matrix_front)
pose_matrix_right = pose_matrix_right @ invert_pose_numpy(pose_matrix_front)
print(pose_matrix_left)
print(pose_matrix_right)

pose_matrix_front_torch = torch.zeros(1,4,4)
pose_matrix_left_torch  = torch.zeros(1,4,4)
pose_matrix_right_torch = torch.zeros(1,4,4)

pose_matrix_front_torch[0,:,:] = torch.from_numpy(pose_matrix_front)
pose_matrix_left_torch[0,:,:]  = torch.from_numpy(pose_matrix_left)
pose_matrix_right_torch[0,:,:] = torch.from_numpy(pose_matrix_right)


cam_front = CameraFisheye(path_to_theta_lut=['/home/data/vbelissen/valeo_multiview/calibrations_theta_lut/fisheye/test/20170320_163113/20170320_163113_cam_0_1280_800.npy'],
                          path_to_ego_mask=[''],
                          poly_coeffs=torch.from_numpy(np.array([[282.85,-27.8671,114.318,-36.6703]])).float(),
                          principal_point=torch.from_numpy(np.array([[0.046296,-7.33178]])).float(),
                          scale_factors=torch.from_numpy(np.array([[1.0,1.0]])).float())

cam_left = CameraFisheye(path_to_theta_lut=['/home/data/vbelissen/valeo_multiview/calibrations_theta_lut/fisheye/test/20170320_163113/20170320_163113_cam_0_1280_800.npy'],
                         path_to_ego_mask=[''], poly_coeffs=torch.from_numpy(np.array([[282.85,-27.8671,114.318,-36.6703]])).float(), 
                         principal_point=torch.from_numpy(np.array([[0.046296,-7.33178]])).float(),
                         scale_factors=torch.from_numpy(np.array([[1.0,1.0]])).float(),
                         Tcw=Pose(pose_matrix_left_torch))


cam_front = cam_front.to(torch.device('cuda'))
cam_left  = cam_left.to(torch.device('cuda'))

world_points = cam_front.reconstruct(simulated_depth,frame='w')

ref_coords_left = cam_left.project(world_points, frame='w')
# ref_coords_left[0,:,:,0][~not_masked_left[0,0,:,:]] = 2
# ref_coords_left[0,:,:,1][~not_masked_left[0,0,:,:]] = 2

left_img_torch[0, :, ~not_masked_left[0,0,:,:]] = 0

warped_front_left = funct.grid_sample(left_img_torch, ref_coords_left, mode='bilinear', padding_mode='zeros', align_corners=True)



warped_front_left_PIL = torch.transpose(warped_front_left.unsqueeze(4),1,4).squeeze().cpu().numpy()

tt = str(int(time.time()%10000))

cv2.imwrite('/home/users/vbelissen/test'+ tt +'_left.png',warped_front_left_PIL[:, :, ::-1])



cam_front = CameraFisheye(path_to_theta_lut=['/home/data/vbelissen/valeo_multiview/calibrations_theta_lut/fisheye/test/20170320_163113/20170320_163113_cam_0_1280_800.npy'],
                          path_to_ego_mask=[''],
                          poly_coeffs=torch.from_numpy(np.array([[282.85,-27.8671,114.318,-36.6703]])).float(),
                          principal_point=torch.from_numpy(np.array([[0.046296,-7.33178]])).float(),
                          scale_factors=torch.from_numpy(np.array([[1.0,1.0]])).float())

cam_right = CameraFisheye(path_to_theta_lut=['/home/data/vbelissen/valeo_multiview/calibrations_theta_lut/fisheye/test/20170320_163113/20170320_163113_cam_0_1280_800.npy'],
                         path_to_ego_mask=[''], poly_coeffs=torch.from_numpy(np.array([[282.85,-27.8671,114.318,-36.6703]])).float(), 
                         principal_point=torch.from_numpy(np.array([[0.046296,-7.33178]])).float(),
                         scale_factors=torch.from_numpy(np.array([[1.0,1.0]])).float(),
                         Tcw=Pose(pose_matrix_right_torch))


cam_front = cam_front.to(torch.device('cuda'))
cam_right = cam_right.to(torch.device('cuda'))

world_points = cam_front.reconstruct(simulated_depth,frame='w')

ref_coords_right = cam_right.project(world_points, frame='w')
# ref_coords_right[0,:,:,0][~not_masked_right[0,0,:,:]] = 2
# ref_coords_right[0,:,:,1][~not_masked_right[0,0,:,:]] = 2

right_img_torch[0, :, ~not_masked_right[0,0,:,:]] = 0

warped_front_right = funct.grid_sample(right_img_torch, ref_coords_right, mode='bilinear', padding_mode='zeros', align_corners=True)



warped_front_right_PIL = torch.transpose(warped_front_right.unsqueeze(4),1,4).squeeze().cpu().numpy()
cv2.imwrite('/home/users/vbelissen/test'+ tt +'_right.png',warped_front_right_PIL[:, :, ::-1])



threshold = 1.0
warped_front_left_black =     torch.sum(warped_front_left, axis=1) <= threshold
warped_front_right_black =     torch.sum(warped_front_right, axis=1) <= threshold
warped_front_left_not_black = torch.sum(warped_front_left, axis=1) > threshold
warped_front_right_not_black = torch.sum(warped_front_right, axis=1) > threshold
warped_front_left_black =     warped_front_left_black.unsqueeze(1).repeat(1, 3, 1, 1)
warped_front_right_black =     warped_front_right_black.unsqueeze(1).repeat(1, 3, 1, 1)
warped_front_left_not_black = warped_front_left_not_black.unsqueeze(1).repeat(1, 3, 1, 1)
warped_front_right_not_black = warped_front_right_not_black.unsqueeze(1).repeat(1, 3, 1, 1)

threshold_2 = 150.0
warped_front_left_black_2 =     torch.sum(warped_front_left, axis=1) <= threshold_2
warped_front_right_black_2 =     torch.sum(warped_front_right, axis=1) <= threshold_2
warped_front_left_not_black_2 = torch.sum(warped_front_left, axis=1) > threshold_2
warped_front_right_not_black_2 = torch.sum(warped_front_right, axis=1) > threshold_2
warped_front_left_black_2 =     warped_front_left_black_2.unsqueeze(1).repeat(1, 3, 1, 1)
warped_front_right_black_2 =     warped_front_right_black_2.unsqueeze(1).repeat(1, 3, 1, 1)
warped_front_left_not_black_2 = warped_front_left_not_black_2.unsqueeze(1).repeat(1, 3, 1, 1)
warped_front_right_not_black_2 = warped_front_right_not_black_2.unsqueeze(1).repeat(1, 3, 1, 1)

ref_warped = warped_front_left_not_black * warped_front_right_black     * warped_front_left \
            + warped_front_left_black     * warped_front_right_not_black * warped_front_right \
            + warped_front_left_not_black * warped_front_right_not_black * (warped_front_left_not_black_2 * warped_front_right_black_2     * warped_front_left
                                                                     + warped_front_left_black_2     * warped_front_right_not_black_2 * warped_front_right
                                                                     + warped_front_left_not_black_2 * warped_front_right_not_black_2 * (warped_front_left + warped_front_right) / 2)

ref_warped_PIL = torch.transpose(ref_warped.unsqueeze(4),1,4).squeeze().cpu().numpy()
cv2.imwrite('/home/users/vbelissen/test'+ tt +'_ref_warped.png',ref_warped_PIL[:, :, ::-1])

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

def SSIM1(x, y, kernel_size=3):
    """
    Calculates the SSIM (Structural SIMilarity) loss

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    kernel_size : int
        Convolutional parameter

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM loss
    """
    ssim_value = SSIM(x, y, C1=1e-4 , C2=9e-4 , kernel_size=kernel_size)
    return torch.clamp((1. - ssim_value) / 2., 0., 1.)

def calc_photometric_loss(t_est, images):
    """
    Calculates the photometric loss (L1 + SSIM)
    Parameters
    ----------
    t_est : list of torch.Tensor [B,3,H,W]
        List of warped reference images in multiple scales
    images : list of torch.Tensor [B,3,H,W]
        List of original images in multiple scales

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Photometric loss
    """
    # L1 loss
    n = len(t_est)
    clip_loss = 0.0
    ssim_loss_weight = 0.85

    l1_loss = [torch.abs(t_est[i] - images[i])
               for i in range(n)]
    # SSIM loss
    if ssim_loss_weight > 0.0:
        ssim_loss = [SSIM1(t_est[i], images[i], kernel_size=3)
                     for i in range(n)]
        # Weighted Sum: alpha * ssim + (1 - alpha) * l1
        photometric_loss = [ssim_loss_weight * ssim_loss[i].mean(1, True) +
                            (1 - ssim_loss_weight) * l1_loss[i].mean(1, True)
                            for i in range(n)]
    else:
        photometric_loss = l1_loss
    # Clip loss
    if clip_loss > 0.0:
        for i in range(n):
            mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
            photometric_loss[i] = torch.clamp(
                photometric_loss[i], max=float(mean + clip_loss * std))
    # Return total photometric loss
    return photometric_loss

loss0 = calc_photometric_loss([warped_front_left], [front_img_torch])
loss0b = calc_photometric_loss([warped_front_right], [front_img_torch])
loss1 = calc_photometric_loss([ref_warped], [front_img_torch])
loss2 = calc_photometric_loss([front_next_img_torch], [front_img_torch])
print(loss0[0][0,0,::100,::100])
print(loss0b[0][0,0,::100,::100])
print(loss1[0][0,0,::100,::100])
print(loss2[0][0,0,::100,::100])

simulated_depth_right = torch.from_numpy(np.load('/home/data/vbelissen/20170320_163113_cam_1_00006288.npz')['depth']).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)
simulated_depth_right[~not_masked_right] = 0
world_points_right = cam_right.reconstruct(simulated_depth_right,frame='w')

ref_coords_front = cam_front.project(world_points_right, frame='w')

front_img_torch[0, :, ~not_masked_front[0,0,:,:]] = 0
warped_right_front = funct.grid_sample(front_img_torch, ref_coords_front, mode='bilinear', padding_mode='zeros', align_corners=True)

warped_right_front_PIL = torch.transpose(warped_right_front.unsqueeze(4),1,4).squeeze().cpu().numpy()
cv2.imwrite('/home/users/vbelissen/test'+ tt +'_right_front.png',warped_right_front_PIL[:, :, ::-1])


simulated_depth_right_to_front = funct.grid_sample(simulated_depth_right, ref_coords_right, mode='bilinear', padding_mode='zeros', align_corners=True)

viz_pred_inv_depth_front = viz_inv_depth(depth2inv(simulated_depth)[0], normalizer=1.0) * 255
viz_pred_inv_depth_right = viz_inv_depth(depth2inv(simulated_depth_right)[0], normalizer=1.0) * 255
viz_pred_inv_depth_right_to_front = viz_inv_depth(depth2inv(simulated_depth_right_to_front)[0], normalizer=1.0) * 255

world_points_right_in_front_coords = cam_front.Tcw @ world_points_right
simulated_depth_right_in_front_coords = torch.norm(world_points_right_in_front_coords, dim=1, keepdim=True)
simulated_depth_right_in_front_coords[~not_masked_right] = 0
simulated_depth_right_to_front_in_front_coords = funct.grid_sample(simulated_depth_right_in_front_coords, ref_coords_right, mode='bilinear', padding_mode='zeros', align_corners=True)
viz_pred_inv_depth_right_to_front_in_front_coords = viz_inv_depth(depth2inv(simulated_depth_right_to_front_in_front_coords)[0], normalizer=1.0) * 255

imwrite('/home/users/vbelissen/test'+ tt +'_depth_front.png', viz_pred_inv_depth_front[:, :, ::-1])
imwrite('/home/users/vbelissen/test'+ tt +'_depth_right_to_front.png', viz_pred_inv_depth_right_to_front[:, :, ::-1])
imwrite('/home/users/vbelissen/test'+ tt +'_depth_right_to_front_in_front_coords.png', viz_pred_inv_depth_right_to_front_in_front_coords[:, :, ::-1])

for threshold in [1.0, 1.1, 1.5, 2.0]:
    mask = threshold * simulated_depth_right_to_front_in_front_coords < simulated_depth
    mask[simulated_depth_right_to_front_in_front_coords == 0] = 0
    imwrite('/home/users/vbelissen/test' + tt + '_mask_right_in_front_coords_' + str(threshold) + '.png', mask[0,0,:,:].detach().cpu().numpy() * 255)


simulated_depth_left = torch.from_numpy(np.load('/home/data/vbelissen/20170320_163113_cam_3_00006286.npz')['depth']).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)
simulated_depth_left[~not_masked_left] = 0
world_points_left = cam_left.reconstruct(simulated_depth_left,frame='w')

ref_coords_front = cam_front.project(world_points_left, frame='w')

front_img_torch[0, :, ~not_masked_front[0,0,:,:]] = 0
warped_left_front = funct.grid_sample(front_img_torch, ref_coords_front, mode='bilinear', padding_mode='zeros', align_corners=True)

warped_left_front_PIL = torch.transpose(warped_left_front.unsqueeze(4),1,4).squeeze().cpu().numpy()
cv2.imwrite('/home/users/vbelissen/test'+ tt +'_left_front.png',warped_left_front_PIL[:, :, ::-1])

simulated_depth_left_to_front = funct.grid_sample(simulated_depth_left, ref_coords_left, mode='bilinear', padding_mode='zeros', align_corners=True)

viz_pred_inv_depth_left = viz_inv_depth(depth2inv(simulated_depth_left)[0], normalizer=1.0) * 255
viz_pred_inv_depth_left_to_front = viz_inv_depth(depth2inv(simulated_depth_left_to_front)[0], normalizer=1.0) * 255

world_points_left_in_front_coords = cam_front.Tcw @ world_points_left
simulated_depth_left_in_front_coords = torch.norm(world_points_left_in_front_coords, dim=1, keepdim=True)
simulated_depth_left_in_front_coords[~not_masked_left] = 0
simulated_depth_left_to_front_in_front_coords = funct.grid_sample(simulated_depth_left_in_front_coords, ref_coords_left, mode='bilinear', padding_mode='zeros', align_corners=True)
viz_pred_inv_depth_left_to_front_in_front_coords = viz_inv_depth(depth2inv(simulated_depth_left_to_front_in_front_coords)[0], normalizer=1.0) * 255



imwrite('/home/users/vbelissen/test'+ tt +'_depth_left_to_front.png', viz_pred_inv_depth_left_to_front[:, :, ::-1])
imwrite('/home/users/vbelissen/test'+ tt +'_depth_left_to_front_in_front_coords.png', viz_pred_inv_depth_left_to_front_in_front_coords[:, :, ::-1])


for threshold in [1.0, 1.1, 1.5, 2.0]:
    mask = threshold * simulated_depth_left_to_front_in_front_coords < simulated_depth
    mask[simulated_depth_left_to_front_in_front_coords == 0] = 0
    imwrite('/home/users/vbelissen/test' + tt + '_mask_left_in_front_coords_' + str(threshold) + '.png', mask[0,0,:,:].detach().cpu().numpy() * 255)







simulated_depth_right = torch.from_numpy(np.load('/home/data/vbelissen/20170320_163113_cam_1_00006288.npz')['depth']).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)
simulated_depth_right[~not_masked_right] = 0
world_points_right = cam_right.reconstruct(simulated_depth_right,frame='w')
world_points_right_in_front_coords = cam_front.Tcw @ world_points_right
world_points_right_to_front_in_front_coords = funct.grid_sample(world_points_right_in_front_coords, ref_coords_right, mode='bilinear', padding_mode='zeros', align_corners=True)

rel_distances = torch.norm(world_points_right_to_front_in_front_coords - world_points, dim=1, keepdim=True)
abs_distances1 = torch.norm(world_points, dim=1, keepdim=True)
abs_distances2 = torch.norm(world_points_right_to_front_in_front_coords, dim=1, keepdim=True)

not_masked_right_warped = funct.grid_sample(not_masked_right, ref_coords_right, mode='bilinear', padding_mode='zeros', align_corners=True)



for threshold in [0.25, 0.75, 1.25]:
    mask1 = (rel_distances < abs_distances1 * threshold)
    mask2 = (rel_distances < abs_distances2 * threshold)
    mask = (~(mask1*mask2))*not_masked_right_warped
    imwrite('/home/users/vbelissen/test' + tt + '_mask_3d_right_in_front_coords_' + str(threshold) + '.png', mask[0,0,:,:].detach().cpu().numpy() * 255)


simulated_depth_left = torch.from_numpy(np.load('/home/data/vbelissen/20170320_163113_cam_3_00006286.npz')['depth']).to(torch.device('cuda')).unsqueeze(0).unsqueeze(0)
simulated_depth_left[~not_masked_left] = 0
world_points_left = cam_left.reconstruct(simulated_depth_left,frame='w')
world_points_left_in_front_coords = cam_front.Tcw @ world_points_left
world_points_left_to_front_in_front_coords = funct.grid_sample(world_points_left_in_front_coords, ref_coords_left, mode='bilinear', padding_mode='zeros', align_corners=True)

rel_distances = torch.norm(world_points_left_to_front_in_front_coords * not_masked_left - world_points * not_masked_left, dim=1, keepdim=True)
abs_distances1 = torch.norm(world_points, dim=1, keepdim=True)
abs_distances2 = torch.norm(world_points_left_to_front_in_front_coords, dim=1, keepdim=True)

not_masked_left_warped = funct.grid_sample(not_masked_left, ref_coords_left, mode='bilinear', padding_mode='zeros', align_corners=True)


for threshold in [0.25, 0.75, 1.25]:
    mask1 = (rel_distances < abs_distances1 * threshold)
    mask2 = (rel_distances < abs_distances2 * threshold)
    mask = (~(mask1*mask2))*not_masked_left_warped
    imwrite('/home/users/vbelissen/test' + tt + '_mask_3d_left_in_front_coords_' + str(threshold) + '.png', mask[0,0,:,:].detach().cpu().numpy() * 255)

