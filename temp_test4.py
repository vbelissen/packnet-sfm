import packnet_sfm.geometry.camera_fisheye_valeo
from packnet_sfm.geometry.camera_fisheye_valeo import CameraFisheye
import torch
import numpy as np
import torch.nn.functional as funct
from PIL import Image
import cv2
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.datasets.kitti_based_valeo_dataset_utils import pose_from_oxts_packet, read_calib_file, read_raw_calib_files_camera_valeo, transform_from_rot_trans
from packnet_sfm.geometry.pose_utils import invert_pose_numpy
import time

simulated_depth=torch.zeros(1,1,800,1280)
for i in range(800):
    for j in range(1280):
        simulated_depth[0,0,i,j] = i*(799-i)*j*(1279-j)*i*(799-i)*j*(1279-j)/6.5270e+10/6.5270e+10*50
simulated_depth = simulated_depth.to(torch.device('cuda'))

img_front = Image.open('/home/data/vbelissen/valeo_multiview/images_multiview/fisheye/test/20170320_163113/cam_0/20170320_163113_cam_0_00006300.jpg').convert('RGB')
img_left  = Image.open('/home/data/vbelissen/valeo_multiview/images_multiview/fisheye/test/20170320_163113/cam_3/20170320_163113_cam_3_00006286.jpg').convert('RGB')
img_right = Image.open('/home/data/vbelissen/valeo_multiview/images_multiview/fisheye/test/20170320_163113/cam_1/20170320_163113_cam_1_00006288.jpg').convert('RGB')

front_img_torch = torch.transpose(torch.from_numpy(np.array(img_front)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)
left_img_torch  = torch.transpose(torch.from_numpy(np.array(img_left)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)
right_img_torch = torch.transpose(torch.from_numpy(np.array(img_right)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)


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

warped_front_left = funct.grid_sample(left_img_torch, ref_coords_left, mode='bilinear', padding_mode='zeros', align_corners=True)

warped_front_left_PIL = torch.transpose(warped_front_left.unsqueeze(4),1,4).squeeze().cpu().numpy()

tt = str(int(time.time()%10000))

cv2.imwrite('/home/users/vbelissen/test'+ tt +'_left.png',warped_front_left_PIL)



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

warped_front_right = funct.grid_sample(right_img_torch, ref_coords_right, mode='bilinear', padding_mode='zeros', align_corners=True)

warped_front_right_PIL = torch.transpose(warped_front_right.unsqueeze(4),1,4).squeeze().cpu().numpy()
cv2.imwrite('/home/users/vbelissen/test'+ tt +'_right.png',warped_front_right_PIL)



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
cv2.imwrite('/home/users/vbelissen/test'+ tt +'_ref_warped.png',ref_warped_PIL)

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

loss = calc_photometric_loss([ref_warped], [front_img_torch])
print(loss)