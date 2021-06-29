#!/bin/bash
source activate packnet-sfm-open3d
#sequences=("20180306_101220" "20180306_103209" "20180306_110204" "20180711_145425_1" "20180716_192137" "20180725_170732")
sequences=("20180725_170732")
for sequence in ${sequences[@]}; do
    echo $sequence
    python3 viz3D_Ncams_seq4.py --checkpoints /home/vbelissen/Downloads/test/config50.ckpt /home/vbelissen/Downloads/test/config51.ckpt /home/vbelissen/Downloads/test/config51.ckpt /home/vbelissen/Downloads/test/config51.ckpt --input_folders /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/${sequence}/cam_0/ /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/${sequence}/cam_1/ /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/${sequence}/cam_2/ /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/${sequence}/cam_3/ --output /home/vbelissen/Downloads/test/results/
done
#scp vbelissen@rox.intra.cea.fr:/media/BDD3/valeo/dataset_valeo_cea_2017_2018/pcls/fisheye/train

