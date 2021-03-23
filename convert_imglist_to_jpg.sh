#!/bin/bash
in_img_dir=$1
in_img_list=$2
out_img_dir=$3
out_img_list=$4

mkdir -p "$out_img_dir"
rm $out_img_list
while IFS= read -r f; do
    convert "${in_img_dir}/${f}" "${out_img_dir}/${f%.*}.jpg"
    echo ${f%.*}.jpg >> "$out_img_list"
done < $in_img_list

