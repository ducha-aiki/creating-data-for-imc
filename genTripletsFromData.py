
import sys
sys.path.append('thirdparty/')
import os
import numpy as np
import h5py
import itertools
from six.moves import xrange
import random
from shutil import copyfile
import deepdish as dd
from tqdm import tqdm
from path_helper import get_fullpath_list
from load_helper import load_vis
from argparse import ArgumentParser
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))


def get_image_pairs(vis_list, num_images, vis_th):
    vis = load_vis(vis_list)
    image_pairs = []
    for ii, jj in itertools.product(xrange(num_images),
                                    xrange(num_images)):
        if ii != jj:
            if vis[ii][jj] > vis_th:
                image_pairs.append((ii, jj))
    return image_pairs

def write_images_txt(set_loc, set_xx_key, data_loc):
    images_txt = os.path.join(set_loc ,'images.txt')
    with open(images_txt, 'w') as f:
        for key in set_xx_key:
            image_path = "images/{}.jpg".format(key)
            f.write(image_path + '\n')

def write_visibility_txt(set_loc, set_xx_key, data_loc):
    images_txt = os.path.join(set_loc ,'visibility.txt')
    with open(images_txt, 'w') as f:
        for key in set_xx_key:
            visibility_path = "visibility/vis_{}.txt".format(key)
            f.write(visibility_path + '\n')

def write_subset_images_txt(set_loc, set_xx_key, data_loc, idx, set_name):
    dir1 = os.path.join(set_loc, 'sub_set')
    if not os.path.exists(dir1):
        os.makedirs(dir1)    
    images_txt = os.path.join(dir1,   set_name + '_{:03.0f}.txt'.format(idx))
    with open(images_txt, 'w') as f:
        for key in set_xx_key:
            image_path = "images/{}.jpg".format(key)
            f.write(image_path + '\n')    

def write_depth_maps_txt(set_loc, set_xx_key, data_loc):
    images_txt = os.path.join(set_loc,'depth_maps.txt')
    with open(images_txt, 'w') as f:
        for key in set_xx_key:
            image_path = "depth_maps/{}.h5".format(key)
            f.write(image_path + '\n')

def write_calibration_txt(set_loc, set_xx_key, data_loc):
    calibration_txt = os.path.join(set_loc,'calibration.txt')
    with open(calibration_txt, 'w') as f:
        for key in set_xx_key:
            f.write('calibration/calibration_'+key+'.h5\n')           

def copy_calibration_files(set_loc, set_xx_key, data_loc):
    src_cal_dir = os.path.join(data_loc , 'calibration')
    dst_cal_dir = os.path.join(set_loc,'calibration')
    if not os.path.exists(dst_cal_dir):
        os.makedirs(dst_cal_dir)
    for key in set_xx_key:
        copyfile(os.path.join(src_cal_dir, 'calibration_' + key + '.h5'),
                 os.path.join(dst_cal_dir, 'calibration_' + key + '.h5'))

def copy_images(set_loc, set_xx_key, data_loc):
    src_images_dir = os.path.join(data_loc,'images')
    dst_images_dir = os.path.join(set_loc, 'images')
    if not os.path.exists(dst_images_dir):
        os.makedirs(dst_images_dir)
    for key in set_xx_key:
        copyfile(os.path.join(src_images_dir, key + '.jpg'),
                 os.path.join(dst_images_dir, key + '.jpg'))

def copy_depth_maps(set_loc, set_xx_key, data_loc,):
    src_depth_dir = os.path.join(data_loc,'depth_maps')
    dst_depth_dir =  os.path.join(set_loc, 'depth_maps')
    if not os.path.exists(dst_depth_dir):
        os.makedirs(dst_depth_dir)
    for key in set_xx_key:
        in_img = os.path.join(src_depth_dir, key + '.h5')
        if not os.path.isfile(in_img):
            print (f"Depth map {in_img} is missing, skipping")
            continue
        # copyfile(src_depth_dir + key + '.jpg', dst_images_dir + key + '.jpg')
        copyfile(in_img, os.path.join(dst_depth_dir, key + '.h5'))

def write_new_vis_pairs(set_loc, set_xx_key, data_loc):
    src_vis_dir = os.path.join(data_loc, 'new-vis-pairs')
    dst_vis_dir =os.path.join(set_loc, 'new-vis-pairs')
    if not os.path.exists(dst_vis_dir):
        os.makedirs(dst_vis_dir)
    for th in np.arange(0, 1, 0.1):
        valid_pairs = []
        pairs= dd.io.load(os.path.join(src_vis_dir,'keys-th-{:0.1f}.dd'.format(th)))
        for pair in pairs:
            if pair.split('-')[0] in set_xx_key and pair.split('-')[1] in set_xx_key:
                valid_pairs.append(pair)
        print (f"Valid pairs for th {th} is {len(valid_pairs)}")
        np.save('{}keys-th-{:0.1f}.npy'.format(dst_vis_dir, th), valid_pairs)

def write_visibility_files(set_loc, set_xx_key, data_loc, set_xx_idx ):
    vis_list = get_fullpath_list(data_loc, "visibility")
    vis_dir = os.path.join(set_loc,  'visibility')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # For each key in `set_xx`, load its visibility file and 
    # feed the current visibility file with the order
    vis = load_vis(vis_list)
    for idx_key, name_key in zip(set_xx_idx, set_xx_key):
        vis_key = vis[idx_key]

        
        vis_file_name = 'vis_' + name_key + '.txt'

        # For each image, generate a visibility file
        # Store num matches from the information in pairs
        # If same image, keep -1
        # If no mathes, keep 0
        # First get the file name
        file_name = os.path.join(vis_dir,vis_file_name)

        # Open file and fill contents
        with open(file_name, 'w') as f:
            for key2 in set_xx_idx:
                f.write(str(vis_key[key2])+'\n')

def gen_pair_dict(image_pairs):
    pairs_dict = {}
    for pair in image_pairs:
        if pair[0] in pairs_dict.keys():
            pairs_dict[pair[0]].append(pair[1])
        else:
            pairs_dict[pair[0]] = [pair[1]]
    keys_list = list(pairs_dict.keys())
    keys_list.sort()
    keys_list = keys_list[::-1]
    return keys_list , pairs_dict   

def gen_triplets(keys_list,pairs_dict):
    triplets = []
    for key in keys_list:
        new_keys_list = pairs_dict[key]
        for key2 in new_keys_list:
            if key2 in pairs_dict:
                list2 = pairs_dict[key2]
                # Get intersection of list, list2
                list3 = intersection(new_keys_list, list2)
                list3.sort()
                for i in list3:
                    triplets.append([key, key2, i])
    return triplets   

def gen_set(triplets, set_size):
    set_100 = []
    temp_triplet = random.choice(triplets)
    while len(set_100) < set_size:
        temp_triplet = random.choice(triplets)
        if len(set_100) < 98:
            # Do whatever
            for i in temp_triplet:
                if i not in set_100:
                    set_100.append(i)
        elif len(set_100) == 98:
            # Find a triplet which has 2 unique values
            counter = 0
            values_to_append = []
            for i in temp_triplet:
                if i not in set_100:
                    values_to_append.append(i)
                    counter = counter + 1
            if counter < 3:
                set_100 = set_100 + values_to_append
        elif len(set_100) == 99:
            # Find a triplet which has 1 unique values
            counter = 0
            values_to_append = []
            for i in temp_triplet:
                if i not in set_100:
                    values_to_append.append(i)
                    counter = counter + 1
            if counter < 2:
                set_100 = set_100 + values_to_append
    return set_100

def get_10bag( _triplets_list):
    ret_list = []
    while len(ret_list) < 10:
        _sample_1 = random.choice(_triplets_list)
        
        # Check for values not in ret_list
        counter = 0
        values_to_append = []
        for i in _sample_1:
            if i not in ret_list:
                values_to_append.append(i)
                counter = counter + 1
        if len(ret_list) < 8:
            ret_list = ret_list + values_to_append
        elif len(ret_list) == 8:
            if len(values_to_append) < 3:
                ret_list = ret_list + values_to_append
        elif len(ret_list) == 9:
            if len(values_to_append) == 1:
                ret_list = ret_list + values_to_append
    ret_list.sort()
    ret_list = ret_list[::-1]
    return ret_list
def get_25bag( _triplets_list):
    ret_list = []
    while len(ret_list) < 25:
        _sample_1 = random.choice(_triplets_list)
        
        # Check for values not in ret_list
        counter = 0
        values_to_append = []
        for i in _sample_1:
            if i not in ret_list:
                values_to_append.append(i)
                counter = counter + 1
        if len(ret_list) < 23:
            ret_list = ret_list + values_to_append
        elif len(ret_list) == 23:
            if len(values_to_append) < 3:
                ret_list = ret_list + values_to_append
        elif len(ret_list) == 24:
            if len(values_to_append) == 1:
                ret_list = ret_list + values_to_append
    ret_list.sort()
    ret_list = ret_list[::-1]
    return ret_list
# data_loc_master = '../../../scratch/sfm/data/'
# data_list = ['reichstag', 'milan_cathedral', 'london_bridge', 'sagrada_familia', 
#              'florence_cathedral_side', 'mount_rushmore', 'st_pauls_cathedral', 
#              'lincoln_memorial_statue', 'piazza_san_marco', 'united_states_capitol']


def compute_image_pairs(vis_list, num_images, vis_th):
    vis = load_vis(vis_list)
    image_pairs = []
    for ii, jj in itertools.product(xrange(num_images),
                                    xrange(num_images)):
        if ii != jj:
            if vis[ii][jj] > vis_th:
                image_pairs.append((ii, jj))
    return image_pairs

def images_idx2images_key(set_xx_idx, images_list):
    out = []
    for idx in set_xx_idx:
        fname = os.path.basename(images_list[idx])
        out.append(fname[:-4])
    return out

def gen_subset(data_loc_master, data_list, max_num_pairs):
    for d in data_list:
        print('Working on {}'.format(d))
        print('Create set_100')
        data_loc = os.path.join(os.path.join(data_loc_master, d), 'all')
        set_loc = os.path.join(os.path.join(data_loc_master, d), 'set_100')
        if not os.path.exists(set_loc):
            os.makedirs(set_loc)

        # load image pairs
        vis_th = 100
        vis_list = get_fullpath_list(data_loc, "visibility")
        images_list = get_fullpath_list(data_loc, "images")
        image_pairs = compute_image_pairs(vis_list, len(images_list), vis_th)

        # get number of images
        num_images = len(images_list)

        # get a subset of image pairs to prevent 
        image_pairs = random.sample(image_pairs, min(max_num_pairs, len(image_pairs)))
        
        # Generate pairs dict
        keys_list, pairs_dict = gen_pair_dict(image_pairs)

        # Generate triplets
        triplets = gen_triplets(keys_list, pairs_dict)
        print('Triplets (Before) =', len(triplets))
        # Run a while loop on triplets to sample 100 images
        set_size = min(100,num_images)
        set_100_idx = gen_set(triplets, set_size)
        set_100_idx = sorted(set_100_idx)
        set_100_key = images_idx2images_key(set_100_idx, images_list)
        write_images_txt(set_loc, set_100_key, data_loc)
        write_depth_maps_txt(set_loc, set_100_key, data_loc)
        write_calibration_txt(set_loc, set_100_key, data_loc)
        write_visibility_txt(set_loc, set_100_key, data_loc)

        copy_images(set_loc, set_100_key, data_loc)
        copy_depth_maps(set_loc, set_100_key, data_loc)
        copy_calibration_files(set_loc,  set_100_key, data_loc)
        write_visibility_files(set_loc,  set_100_key, data_loc, set_100_idx)
        write_new_vis_pairs(set_loc, set_100_key, data_loc)


        # Now compute new set of triplet from set_100 - Reduces search space
        print('Gen new triplets')
        vis_list = get_fullpath_list(set_loc, "visibility")
        images_list = get_fullpath_list(set_loc, "images")
        image_pairs = compute_image_pairs(vis_list, len(images_list), vis_th)

        # Generate pairs dict
        keys_list, pairs_dict = gen_pair_dict(image_pairs)

        # Generate triplets
        triplets = gen_triplets(keys_list, pairs_dict)
        print('Triplets (After) =', len(triplets))


        # 3bag gen
        selected_triplets = []
        for idx in tqdm(range(100)):
            if not os.path.exists(set_loc):
                os.makedirs(set_loc)
            while True:
                current_sample = random.choice(triplets)
                # Memoization
                if current_sample not in selected_triplets:
                    selected_triplets.append(current_sample)
                    current_sample = sorted(current_sample)
                    write_subset_images_txt(set_loc, images_idx2images_key(current_sample, images_list), data_loc, idx, '3bag')
                    break

        selected_5bags = []
        for idx in tqdm(range(100)):
            if not os.path.exists(set_loc):
                os.makedirs(set_loc)
            while True:
                # First get a triplet
                current_sample = random.choice(triplets)
                # Check if it's a valid triplet
                temp_sample = random.choice(triplets)
                counter = 0
                values_to_append = []
                for i in temp_sample:
                    if i not in current_sample:
                        values_to_append.append(i)
                        counter = counter + 1
                if counter==2:
                    current_sample = current_sample + values_to_append
                    current_sample.sort()
                    current_sample = current_sample[::-1]
                    if current_sample not in selected_5bags:
                        selected_5bags.append(current_sample)
                        current_sample = sorted(current_sample)
                        write_subset_images_txt(set_loc, images_idx2images_key(current_sample, images_list), data_loc, idx, '5bag')
                        break

        selected_10bags = []
        for idx in tqdm(range(100)):
            while True:
                # First get a triplet
                current_sample = get_10bag(triplets)
                
                if current_sample not in selected_10bags:
                    selected_10bags.append(current_sample)
                    current_sample = sorted(current_sample)
                    write_subset_images_txt(set_loc, images_idx2images_key(current_sample, images_list), data_loc, idx, '10bag')
                    break

        selected_25bags = []
        for idx in tqdm(range(100)):
            if not os.path.exists(set_loc):
                os.makedirs(set_loc)
            while True:
                # First get a triplet
                current_sample = get_25bag( triplets)
                
                if current_sample not in selected_25bags:
                    selected_25bags.append(current_sample)
                    current_sample = sorted(current_sample)
                    write_subset_images_txt(set_loc, images_idx2images_key(current_sample, images_list), data_loc, idx, '25bag')
                    break

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    params = parser.parse_args()
    random.seed(params.seed)
    max_num_pairs = 100000
    gen_subset(params.root, [params.seq], max_num_pairs)
