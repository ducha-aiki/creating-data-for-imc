
import sys
import os
sys.path.append('thirdparty/')
import numpy as np
from colmap.scripts.python.read_write_model import (read_model,
                                                    qvec2rotmat,
                                                    read_cameras_text,
                                                    read_images_text,
                                                    read_points3D_text)

import numpy as np
import h5py
import os
import deepdish as dd
# Colmap
from colmap.scripts.python.read_dense import read_array


# # Required files in a directory
# - images/
# - calibration/
# - visibility/
# - new-vis-pairs/
# - depth-
# - images.txt
# - calibration.txt


import shutil

def read_colmap_txt(dir_name):
    cameras = read_cameras_text(os.path.join(dir_name, 'cameras.txt'))
    images = read_images_text(os.path.join(dir_name, 'images.txt'))
    points = read_points3D_text(os.path.join(dir_name, 'points3D.txt'))
    return  cameras, images, points


def genDataFromColmap(data_loc_master, data_list):
    for d in data_list:
        print('Working on {}'.format(d))
        data_loc = data_loc_master + d + '/'
        
        if not os.path.exists(data_loc+'all/'):
            os.makedirs(data_loc+'all/')

        # ----Tast 1: Setup images.txt----
        print("  setting up images txt ...")
        # First read images.bin
        cameras, images, points = read_colmap_txt(os.path.join(data_loc,'reconstruction'))
    
        # images has keys and each key has image name `name`
        # First get keys in a sorted order to a list and write `name` 
        # along with path to a `images.txt` file.
        image_keys = list(images.keys())
        image_keys.sort()

        images_txt = data_loc + 'all/images.txt'
        with open(images_txt, 'w') as f:
            for key in image_keys:
                f.write('images/'+images[key].name+'\n')

        # ----Task 2: Setup images directory----
        print("  setting up images directory ...")
        src = data_loc + 'dense/images/'
        dest = data_loc + 'all/images/'
        try:
            shutil.copytree(src, dest)
        except:
            pass
        # ----Task 3: Setup depth_maps.txt----
        print("  setting up depth_maps txt ...")
        # We have image_keys from above use them to write depth_maps.txt
        calibration_txt = data_loc + 'all/depth_maps.txt'
        with open(calibration_txt, 'w') as f:
            for key in image_keys:
                f.write('calibration/'+images[key].name[:-4]+'.h5\n')

        # ----Task 4: Setup depth map directory----
        print("  setting up depth_maps directory ...")
        src = data_loc + 'dense/stereo/depth_maps_clean_200_th_0.20/'
        dest = data_loc + 'all/depth_maps/'
        shutil.copytree(src, dest)

        # ----Task 5: Setup calibration.txt----
        print("  setting up calibration txt ...")
        # We have image_keys from above use them to write calibration.txt
        calibration_txt = data_loc + 'all/calibration.txt'
        with open(calibration_txt, 'w') as f:
            for key in image_keys:
                f.write('calibration/calibration_'+images[key].name[:-4]+'.h5\n')

        # ----Task 6: Setup calibration directory----
        print("  setting up calibration directory ...")

        # From calibration, I will only require:
        # Camera Intrinsics, K: for normalizing Keypoints
        # Rotation matrix, R: for computing geodesic distance
        # translation vector, t: for computing geodesic distance

        # Use cameras.bin to get K
        #cameras = read_cameras_binary(data_loc + 'dense/sparse/cameras.bin')

        # Generate calibration directory
        cal_dir = data_loc + 'all/calibration'
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)

        for key in image_keys:
            pars = cameras[key].params
            #K = np.array([[pars[0], 0, pars[2]], [0, pars[1], pars[3]], [0, 0, 1]])
            K = np.array([[pars[0], 0, pars[1]], [0, pars[0], pars[2]], [0, 0, 1]])
            q = images[key].qvec
            R = qvec2rotmat(q)
            T = images[key].tvec
            # Save K, q, R, T to a file
            # First get the file name
            file_name = data_loc + 'all/calibration/calibration_' + images[key].name[:-4] + '.h5'
            with h5py.File(file_name, 'w') as f:
                f.create_dataset('K', data=K)
                f.create_dataset('q', data=q)
                f.create_dataset('R', data=R)
                f.create_dataset('T', data=T)


        # --------Task 5: Setup covisibility directory--------        
        print("  setting up covisibility directory ...")
        # Read pairs file
        # each pair contains [bbox1, bbox2, visibility1, visibility2, Number of shared matches]
        pairs = dd.io.load(data_loc + 'dense/stereo/pairs-dilation-0.00.h5')
        
        # Create a visibility directory if it doesn't exist
        vis_dir = data_loc + 'all/new-vis-pairs'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        #cameras, images, points = read_model(path=data_loc + '/dense/sparse', ext='.bin')
        # loop through different covisibility threshold
        for th in np.arange(0, 1, 0.1):
            keys = []
            for k in pairs:
                if all([v >= th for v in pairs[k][2:4]]):
                    a, b = k
                    im1 = images[a].name[:-4]#[0]
                    im2 = images[b].name[:-4]#[0]
                    keys.append('-'.join(sorted([im1, im2],reverse=True)))
            print (keys)
            dd.io.save('{}/keys-th-{:0.1f}.dd'.format(vis_dir, th), keys)

        # --------Task 6: Setup visibility.txt--------        
        print("  setting up visibility txt")

        # We have image_keys from above use them to write visibility.txt
        visibility_txt = data_loc + 'all/visibility.txt'
        with open(visibility_txt, 'w') as f:
            for key in image_keys:
                f.write('visibility/vis_'+images[key].name[:-4]+'.txt\n')


        # --------Task 7: Setup visibility threshold directory--------        
        print("  setting up visibility threshold directory ...")
        # Read pairs file
        # each pair contains [bbox1, bbox2, visibility1, visibility2, Number of shared matches]
        pairs = dd.io.load(data_loc + 'dense/stereo/pairs-dilation-0.00.h5')
        
        # Create a visibility directory if it doesn't exist
        vis_dir = data_loc + 'all/visibility'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        # For each key in `image_keys`, find the pairs
        for key in image_keys:
            pairs_key = []
            for p in pairs:
                if p[0] == key:
                    pairs_key.append(p)
            # For each image, generate a visibility file
            # Store num matches from the information in pairs
            # If same image, keep -1
            # If no mathes, keep 0
            
            # First get the file name
            file_name = data_loc + 'all/visibility/vis_' + images[key].name[:-4] + '.txt'
            
            # Open file and fill contents
            with open(file_name, 'w') as f:
                for key2 in image_keys:
                    if key2 == key:
                        f.write('-1\n')
                        continue
                    pair_exist = False
                    for p in pairs_key:
                        if p[1] == key2:
                            pair_exist = True
                            break
                    if pair_exist:
                        # Get the matches
                        f.write(str(pairs[(key,key2)][4])+'\n')
                    else:
                        f.write('0\n')

if __name__ == '__main__':
    root = '/home/old-ufo/datasets/tree/'
    data_loc_master = root#'/home/yuhe/workspace/sfm_benchmark/'
    seqs = ['tree_in_colmap']
    data_list = seqs#['united_states_capitol_tmp']
    genDataFromColmap(data_loc_master, data_list)
