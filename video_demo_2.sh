#!/bin/bash
source activate packnet-sfm-open3d
sequences=("20180306_103209" "20180306_110204" "20180711_145425_1" "20180716_192137")
cd ~/Downloads/test/results/
for sequence in ${sequences[@]}; do
    echo $sequence
    cd $sequence
    ffmpeg -r 25 -f image2 -i 1_%05d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p -s 1280x742 ${sequence}_1.mp4
    ffmpeg -r 25 -f image2 -i 0_%05d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p -s 1280x742 ${sequence}_0.mp4
    ffmpeg -r 25 -f image2 -i 1_34_%05d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p -s 1280x742 ${sequence}_1_34.mp4
    ffmpeg -r 25 -f image2 -i 0_34_%05d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p -s 1280x742 ${sequence}_0_34.mp4
    cd ..    
done
#scp vbelissen@rox.intra.cea.fr:/media/BDD3/valeo/dataset_valeo_cea_2017_2018/pcls/fisheye/train

