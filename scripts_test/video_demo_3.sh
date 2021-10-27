#!/bin/bash
sequences=("20180306_101220" "20180306_103209" "20180306_110204" "20180711_145425_1" "20180716_192137")
cd ~/Downloads/test/results/
for sequence in ${sequences[@]}; do
    echo $sequence
    cd $sequence
    ffmpeg -f concat -i mylist.txt -codec copy ${sequence}.mp4
    ffmpeg -f concat -i mylist_34.txt -codec copy ${sequence}_34.mp4
    cd ..    
done
#scp vbelissen@rox.intra.cea.fr:/media/BDD3/valeo/dataset_valeo_cea_2017_2018/pcls/fisheye/train

