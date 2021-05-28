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
left_img_torch = torch.transpose(torch.from_numpy(np.array(img_left)).float().unsqueeze(0).to(torch.device('cuda')),0,3).squeeze(3).unsqueeze(0)
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

