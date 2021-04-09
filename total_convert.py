import sys
import os
from argparse import ArgumentParser
import subprocess
import shutil

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--bundler_file", type=str, required=True)
    parser.add_argument("--in_png_image_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--aux_kapture_dir", type=str, required=True)
    parser.add_argument("--aux_colmap_dir", type=str, required=True)
    parser.add_argument("--out_imglist_txt", type=str, default="_temp_imglist.txt")
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument("--depth_th_3dpt", type=float, default=0.1, help='Threshold to clean-up depth maps, in meters. 0.1, 0.2 are good values')
    parser.add_argument("--inl_ratio_th", type=float, default=0.1, help='Threshold for pairs generation: if less then it, pair is ignored')
    
    parser.add_argument("--depth_resize_to", type=int, default=200, help='Resize depth_map to this size for speedup. 300 is a good compromise.')
    parser.add_argument("--dilation", type=float, default=0, help='Some magic number %) Default is fine')
    parser.add_argument("--seed", type=int, default=1234, help='Random seed for triplets generation')
    
    args = parser.parse_args()
    comment=r'''
    I am writing this scripts mostly for myself, but they should be easy to modify to anyone else trying to use it
    Basic setup: we have a 3D reconstruction model from Reality Capture (RC). 
    I am doing the reconstruction from video. Video is converted to the directory with png images by CR itself.
    On top of it we have to export:
     - the camera registration together with point cloud in a Bundler v.0.3 format - into file called smth.out
     - depth values - the CR does it in the .exr format
    Our final goal is to get the sequence in the IMC benchmark format https://github.com/ubc-vision/image-matching-benchmark
    Originally IMC sequences were created with help of COLMAP https://colmap.github.io/, but CR is orders of magnitude faster.
    We will be using for it kapture format converter https://github.com/naver/kapture/'''
    print (comment)
    print ()
    print ("*******************************************************")
    # Step 1
    # Create an image_list from the depth maps
    run_str = f'./create_image_list.sh "{args.depth_dir}" img_list_temp_png.txt'
    print ("Step 1: We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 1 done")
    print ()
    print ("*******************************************************")
    
    # Step 2:  convert pngs into jpegs.
    out_colmap = os.path.join(args.aux_colmap_dir,args.seq_name)
    out_jpg_image_dir = os.path.join(out_colmap, 'dense', "images")
    os.makedirs(f'"{out_jpg_image_dir}"', exist_ok=True)
    run_str = f'./convert_imglist_to_jpg.sh "{args.in_png_image_dir}" img_list_temp_png.txt "{out_jpg_image_dir}" {args.out_imglist_txt}'
    print ("Step 2: We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 2 done")
    print ()
    print ("*******************************************************")
    
    # Step 3
    out_kapt = os.path.join(args.aux_kapture_dir,args.seq_name)
    run_str = f'kapture_import_bundler.py --input "{args.bundler_file}" --image-path "{out_jpg_image_dir}" --add-reconstruction --output "{out_kapt}"  --image-list "{args.out_imglist_txt}"'
    print ("Step 3: Conversion from bundler to kapture. We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 3 done")
    print ()
    print ("*******************************************************")


    # Step 4
    run_str = f'kapture_export_colmap.py -i "{out_kapt}" -txt "{out_colmap}/reconstruction" -db "{out_colmap}/{args.seq_name}.db"'
    print ("Step 4: Conversion from kapture to colmap. We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 4 done")
    print ()
    print ("*******************************************************")

    
    # Step 5
    poolsize = max(1, os.cpu_count() - 1)
    run_str = f'python -utt process.py --root "{out_colmap}" --depth_dir "{args.depth_dir}" --seq "{args.seq_name}" --th {args.depth_th_3dpt} --n {args.depth_resize_to} --poolsize {poolsize}'
    print ("Step 5: Cleaning-up depth maps: keep only those, which are supported by sparse 3d point cloud. We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 5 done")
    print ()
    print ("*******************************************************")

    
    # Step 6
    run_str = f'python -utt pairs.py --root "{args.aux_colmap_dir}"  --seq "{args.seq_name}" --dilation {args.dilation} --th {args.inl_ratio_th}'
    print ("Step 6: Creating image pairs. We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 6 done")
    print ()
    print ("*******************************************************")

    
        
    # Step 7
    run_str = f'python -utt genDataFromCOLMAP.py --root "{args.aux_colmap_dir}"  --seq "{args.seq_name}" --dilation {args.dilation} --th {args.inl_ratio_th} --n {args.depth_resize_to}'
    print ("Step 7: Creating image pairs. We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 7 done")
    print ()
    print ("*******************************************************")

    
    # Step 8
    #run_str = f'python -utt genTripletsFromData.py --root "{args.aux_colmap_dir}"  --seq "{args.seq_name}"  --seed {args.seed}'
    run_str = f'python -utt genTripletsFromSequentialData.py --root "{args.aux_colmap_dir}"  --seq "{args.seq_name}"  --seed {args.seed}'
    print ("Step 8: Creating image triplets. We are going the run the following command:")
    print (run_str)
    #out = subprocess.run(run_str, shell=True)
    print ("Step 8 done")
    print ()
    print ("*******************************************************")
    
        
    # Step 8
    run_str = f'python -utt rotate_image_seq.py --root "{args.aux_colmap_dir}"  --seq "{args.seq_name}"'
    print ("Step 9: Rotating images to have (mostly and if possible) upright orientation. Only 90/180/270 rotatations are considered. We are going the run the following command:")
    print (run_str)
    out = subprocess.run(run_str, shell=True)
    print ("Step 9 done")
    print ()
    print ("*******************************************************")

        
    print ("DONE!!!!")
    print (f"Now you can use {out_colmap}/set_100 directory")

        
        
        