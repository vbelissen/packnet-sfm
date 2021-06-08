import json
import numpy as np
from scipy.spatial.transform import Rotation as R

with open('/home/vbelissen/Downloads/test/cameras_jsons/test1.json') as json_base_file:
    json_base_data = json.load(json_base_file)

N = 30
t = np.array([0,1,1])

count = 0

for i in range(N):
    json_data = dict(json_base_data)
    ext_matrix = json_data['extrinsic']
    ext_matrix = np.array(ext_matrix).reshape((4,4)).transpose()
    #ext_matrix_t = ext_matrix[:3,3]
    #ext_matrix[2, 3] *= 5

    int_matrix = json_data['intrinsic']['intrinsic_matrix']
    int_matrix = np.array(int_matrix).reshape((3,3)).transpose()

    r = R.from_euler('zyx', [0,360/N*i,0], degrees=True)
    print(r.as_matrix())

    r4 = np.eye(4)
    r4[:3,:3] = r.as_matrix()
    new_ext = r4 @ ext_matrix

    new_ext = list(new_ext.transpose().flatten())

    json_data['extrinsic'] = new_ext
    with open('/home/vbelissen/Downloads/test/cameras_jsons/sequence/test1_' + str(count) + '.json', 'w') as outfile:
        json.dump(json_data, outfile)

    count += 1

for i in range(N):
    json_data = dict(json_base_data)
    ext_matrix = json_data['extrinsic']
    ext_matrix = np.array(ext_matrix).reshape((4, 4)).transpose()
    # ext_matrix_t = ext_matrix[:3,3]
    # ext_matrix[2, 3] *= 5

    int_matrix = json_data['intrinsic']['intrinsic_matrix']
    int_matrix = np.array(int_matrix).reshape((3, 3)).transpose()


    new_ext = ext_matrix
    new_ext[0, 3] += np.sin(i / 10)
    # new_ext[:3,3] = t#-np.dot(r.as_matrix(),t)#ext_matrix_t

    new_ext = list(new_ext.transpose().flatten())

    json_data['extrinsic'] = new_ext
    with open('/home/vbelissen/Downloads/test/cameras_jsons/sequence/test1_' + str(count) + '.json', 'w') as outfile:
        json.dump(json_data, outfile)

    count += 1

for i in range(N):
    json_data = dict(json_base_data)
    ext_matrix = json_data['extrinsic']
    ext_matrix = np.array(ext_matrix).reshape((4, 4)).transpose()
    # ext_matrix_t = ext_matrix[:3,3]
    # ext_matrix[2, 3] *= 5

    int_matrix = json_data['intrinsic']['intrinsic_matrix']
    int_matrix = np.array(int_matrix).reshape((3, 3)).transpose()


    new_ext = ext_matrix
    new_ext[1, 3] += np.sin(i / 10)
    # new_ext[:3,3] = t#-np.dot(r.as_matrix(),t)#ext_matrix_t

    new_ext = list(new_ext.transpose().flatten())

    json_data['extrinsic'] = new_ext
    with open('/home/vbelissen/Downloads/test/cameras_jsons/sequence/test1_' + str(count) + '.json', 'w') as outfile:
        json.dump(json_data, outfile)

    count += 1

for i in range(N):
    json_data = dict(json_base_data)
    ext_matrix = json_data['extrinsic']
    ext_matrix = np.array(ext_matrix).reshape((4, 4)).transpose()
    # ext_matrix_t = ext_matrix[:3,3]
    # ext_matrix[2, 3] *= 5


    new_ext = ext_matrix
    new_ext[2, 3] += np.sin(i / 10)
    # new_ext[:3,3] = t#-np.dot(r.as_matrix(),t)#ext_matrix_t

    new_ext = list(new_ext.transpose().flatten())

    json_data['extrinsic'] = new_ext
    with open('/home/vbelissen/Downloads/test/cameras_jsons/sequence/test1_' + str(count) + '.json', 'w') as outfile:
        json.dump(json_data, outfile)

    count += 1