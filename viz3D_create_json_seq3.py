import json
import numpy as np
from scipy.spatial.transform import Rotation as R

with open('/home/vbelissen/Downloads/test/cameras_jsons/test3.json') as json_base_file:
    json_base_data = json.load(json_base_file)

with open('/home/vbelissen/Downloads/test/cameras_jsons/test342.json') as json_base_file_persp:
    json_base_data_persp = json.load(json_base_file_persp)

N = 90

count = 0

json_data = dict(json_base_data)
json_data_persp = dict(json_base_data_persp)

ext_matrix = json_data['extrinsic']
ext_matrix = np.array(ext_matrix).reshape((4,4)).transpose()

ext_matrix_persp = json_data_persp['extrinsic']
ext_matrix_persp = np.array(ext_matrix_persp).reshape((4,4)).transpose()

for i in range(N):
    json_data = dict(json_base_data)

    new_ext = (ext_matrix * (N - 1 - i) + ext_matrix_persp * i) / (N - 1)
    new_ext = list(new_ext.transpose().flatten())

    json_data['extrinsic'] = new_ext
    with open('/home/vbelissen/Downloads/test/cameras_jsons/sequence/test1_' + str(count)+ 'v342' + '.json', 'w') as outfile:
        json.dump(json_data, outfile)

    count += 1

