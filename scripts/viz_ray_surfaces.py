import numpy as np
import open3d as o3d

# 1. Pure pinhole
H = 1280
W = 800
cx = 558.14
cy = 464.74
fx = 1477.64
fy = 1482.52

n_steps = 20
step_x = H // n_steps
step_y = W // n_steps

print_pinhole_z = True
print_pinhole_norm = True

print_pinhole_distorted_z = True
print_pinhole_distorted_norm = True

print_fisheye_norm = True

I1 = np.zeros((H,W))
I2 = np.zeros((H,W))
for i in range(H):
    if i%step_x == 0 or i == H-1:
        I1[i, :] = 1
for i in range(W) or i == W-1:
    if i%step_y == 0:
        I2[:, i] = 1
I = I1 * I2

I = I.ravel()

indices = np.where(I)[0]
N = indices.size

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

Kinv = np.linalg.inv(K)

u = np.linspace(0, W - 1, int(W))
v = np.linspace(0, H - 1, int(H))
xu, yu = np.meshgrid(u, v)
grid = np.vstack([np.ravel(xu),np.ravel(yu),np.ones(H*W)])

rays_z = Kinv @ grid
rays_norm =  rays_z / np.linalg.norm(rays_z, axis=0)

#indices = np.arange(H*W)
#np.random.shuffle(indices)

rays_z = rays_z[:, indices]
rays_norm = rays_norm[:, indices]

x_shift = 1.5
y_shift = 1
rays_norm = rays_norm + np.transpose(np.tile(np.array([0,x_shift,0]), (N, 1)))

rays_list = []
rays_list_lines = []
colors_list = []

rays_list.append([0,0,0])

count = 0
for i in range(0, N):
    count += 1
    rays_list.append(list(rays_z[:, i]))
    if print_pinhole_z:
        rays_list_lines.append([0, count])
        colors_list.append([0, 0.4, 0.1])

count += 1
current = count
rays_list.append([0,x_shift,0])

for i in range(0, N):
    count += 1
    rays_list.append(list(rays_norm[:, i]))
    if print_pinhole_norm:
        rays_list_lines.append([current, count])
        colors_list.append([0, 0.9, 0.1])


# 2. Pinhole + distortion
k1 = -0.436544
k2 = 0.0834011
k3 = 0.169849
p1 = -0.00188862
p2 = 0.00896865

u = np.linspace(0, W - 1, W)
v = np.linspace(0, H - 1, H)
xu, yu = np.meshgrid(u, v)
zu = np.ones((H, W))
Xu = np.stack([xu, yu, zu], axis=0).reshape(3, -1)
Xp = np.dot(np.linalg.inv(K), Xu).reshape(3, H, W)
M = 5
x = Xp[0, :, :]
y = Xp[1, :, :]
x_src = np.array(x)
y_src = np.array(y)
for _ in range(M):
    r2 = np.square(x) + np.square(y)
    r4 = np.square(r2)
    r6 = r2 * r4
    rad_dist = 1 / (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)  # (1 + k4*r2 + k5*r4 + k6*r6) / (1 + k1*r2 + k2*r4 + k3*r6)
    tang_dist_x = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    tang_dist_y = 2 * p2 * x * y + p1 * (r2 + 2 * y ** 2)
    x = (x_src - tang_dist_x) * rad_dist
    y = (y_src - tang_dist_y) * rad_dist

print(np.max(r2))
print(np.min(rad_dist))
print(np.max(rad_dist))

rays_z = np.stack([x, y, np.ones((H, W))], axis=0).reshape(3,-1)
rays_norm =  rays_z / np.linalg.norm(rays_z, axis=0)

rays_z = rays_z[:, indices]
rays_norm = rays_norm[:, indices]

rays_z = rays_z + np.transpose(np.tile(np.array([y_shift,0,0]), (N, 1)))
count += 1
current = count
rays_list.append([y_shift,0,0])

for i in range(0, N):
    count += 1
    rays_list.append(list(rays_z[:, i]))
    if print_pinhole_distorted_z:
        rays_list_lines.append([current, count])
        colors_list.append([0, 0.1, 0.4])

rays_norm = rays_norm + np.transpose(np.tile(np.array([y_shift,x_shift,0]), (N, 1)))
count += 1
current = count
rays_list.append([y_shift,x_shift,0])

for i in range(0, N):
    count += 1
    rays_list.append(list(rays_norm[:, i]))
    if print_pinhole_distorted_norm:
        rays_list_lines.append([current, count])
        colors_list.append([0, 0.1, 0.9])




# 3. Fisheye

cx_offset_px = 0.046296
cy_offset_px = -7.33178

c1 = 282.85
c2 = -27.8671
c3 = 114.318
c4 = -36.6703
pixel_aspect_ratio_float = 1.00173
def fun_rho(theta):
    return c1 * theta + c2 * theta ** 2 + c3 * theta ** 3 + c4 * theta ** 4
def fun_rho_jac(theta):
    return c1 + 2 * c2 * theta + 3 * c3 * theta ** 2 + 4 * c4 * theta ** 3

u = np.linspace(0, W-1, W)
v = np.linspace(0, H-1, H)

xu, yu = np.meshgrid(u, v)
xi = (xu - (W - 1) / 2 - cx_offset_px)
yi = (yu - (H - 1) / 2 - cy_offset_px) * pixel_aspect_ratio_float

M = 12
theta_lut = np.zeros((H, W))
for i in range(M):
    theta_lut = theta_lut + .5*((xi**2 + yi**2)**.5 - fun_rho(theta_lut)) / fun_rho_jac(theta_lut)

rc = np.sin(theta_lut)
phi = np.arctan2(yi, xi)
xc = rc * np.cos(phi)
yc = rc * np.sin(phi)
zc = np.cos(theta_lut)

rays_norm = np.vstack([xc.ravel(), yc.ravel(), zc.ravel()])
rays_norm = rays_norm[:, indices]

rays_norm = rays_norm + np.transpose(np.tile(np.array([3*y_shift,x_shift,0]), (N, 1)))
count += 1
current = count
rays_list.append([3*y_shift,x_shift,0])

for i in range(0, N):
    count += 1
    rays_list.append(list(rays_norm[:, i]))
    if print_fisheye_norm:
        rays_list_lines.append([current, count])
        colors_list.append([1, 0, 0])

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(rays_list)
line_set.lines = o3d.utility.Vector2iVector(rays_list_lines)
line_set.colors = o3d.utility.Vector3dVector(colors_list)
o3d.visualization.draw_geometries([line_set])

#o3d.io.write_line_set("ray_surfaces.ply", line_set)

#l = o3d.io.read_line_set("ray_surfaces.ply")
#o3d.visualization.draw_geometries([l])