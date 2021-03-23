#!/bin/bash
depth_dir=$1
out_img_list=$2
rm $out_img_list
ls "$1" > _temp_depth.txt
sort _temp_depth.txt > _temp_depth_sorted.txt
sed  's/\.depth\.exr//g' _temp_depth_sorted.txt > $out_img_list
