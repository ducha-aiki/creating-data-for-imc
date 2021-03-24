import sys
sys.path.append('thirdparty/')
import numpy as np
from colmap.scripts.python.read_write_model import (read_model,
                                                    qvec2rotmat,
                                                    read_cameras_text,
                                                    read_images_text,
                                                    read_points3D_text)
from imageio import imread, imsave
import os
from os import makedirs
from os.path import isdir
from tqdm import tqdm
from skimage.transform import resize as imresize
import h5py
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser
from IPython import embed
from time import time, sleep
import deepdish as dd
def read_colmap_txt(dir_name):
    cameras = read_cameras_text(os.path.join(dir_name, 'cameras.txt'))
    images = read_images_text(os.path.join(dir_name, 'images.txt'))
    points = read_points3D_text(os.path.join(dir_name, 'points3D.txt'))
    return  cameras, images, points

def find_bb(xy, imsize, dilation_factor):
    height, width = imsize[0], imsize[1]
    z = np.zeros(imsize, dtype=np.uint16)
    #print ("xy before uint", xyxy)
    
    xy = np.round(xy).astype(np.uint16)
    valid = np.array([False] * xy.shape[0])
    for i, p in enumerate(xy):
        if p[0] >= 0 and p[0] < width and p[1] >= 0 and p[1] < height:
            valid[i] = True
    #print ("xy before", xy)
    xy = xy[valid, :]
    #print ("xy after", xy)
    z[xy[:, 1], xy[:, 0]] = 1
    ix = np.where(z.sum(axis=0) > 0)[0]
    iy = np.where(z.sum(axis=1) > 0)[0]

    if ix.size < 3 or iy.size < 3:
        print ("Fail", ix.size, iy.size)
        return None, 0

    bb = [ix.min(), ix.max(), iy.min(), iy.max()]

    # Compute ratio between bb and original image size (in terms of area)
    ratio = ((bb[1] - bb[0]) * (bb[3] - bb[2])) / (height * width)

    # dilate it a little bit
    # large values will preserve more scale changes overall
    dx = bb[1] - bb[0]
    dy = bb[3] - bb[2]
    bb = [max(np.round(bb[0] - dx / 2 * (dilation_factor)), 0),
          min(np.round(bb[1] + dx / 2 * (dilation_factor)), width - 1),
          max(np.round(bb[2] - dy / 2 * (dilation_factor)), 0),
          min(np.round(bb[3] + dy / 2 * (dilation_factor)), height - 1)]

    # recompute and force 1:1 aspect ratio
    dx = bb[1] - bb[0]
    dy = bb[3] - bb[2]

    # embed()
    # let's not overthink this
    # 1. try to dilate into a square
    if dx == dy:
        return bb, ratio
    if dy > dx:
        diff = (np.floor((dy - dx) / 2), np.ceil((dy - dx) / 2))
        bb_dilated = np.round([bb[0] - diff[0], bb[1] + diff[1], bb[2], bb[3]])
    else:
        diff = (np.floor((dx - dy) / 2), np.ceil((dx - dy) / 2))
        bb_dilated = np.round([bb[0], bb[1], bb[2] - diff[0], bb[3] + diff[1]])

    if bb_dilated[0] >= 0 and bb_dilated[1] < width and bb_dilated[2] >= 0 and bb_dilated[3] < height:
        return bb_dilated, ratio

    # 2. otherwise, shrink into a square
    if dy > dx:
        diff = (np.floor((dy - dx) / 2), np.ceil((dy - dx) / 2))
        bb_shrunk = np.round([bb[0], bb[1], bb[2] + diff[0], bb[3] - diff[1]])
    else:
        diff = (np.floor((dx - dy) / 2), np.ceil((dx - dy) / 2))
        bb_shrunk = np.round([bb[0] + diff[0], bb[1] - diff[1], bb[2], bb[3]])

    return bb_shrunk, ratio


def check_pair(args):
    t_start = time()

    cam1, cam2, (h1, w1), (h2, w2), im1, im2, points, src, idx1, idx2, dilation = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]
    #verbose = True
    verbose = False

    # Do not use these values! They are before rescaling/rectification
    # w1, h1 = cam1.width, cam1.height
    # w2, h2 = cam2.width, cam2.height
    #raise ValueError("rRRR")
    xy1 = im1.xys
    xy2 = im2.xys
    #print ("xy1", xy1)
    #print ("xy2", xy2)
    
    
    p1 = im1.point3D_ids
    p2 = im2.point3D_ids
    v1 = p1 > 0
    v2 = p2 > 0

    common = np.intersect1d(p1[v1], p2[v2])
    #if verbose:
    #    print (p1[v1], p2[v2])

    # Filter out pairs with very few matches
    if common.size < 5:
        #if verbose:
        #    print(f'Processing pair ({idx1}, {idx2}): ignoring (num points < 5) [{time() - t_start:.2f} s.]')
        #    sys.stdout.flush()
        return idx1, idx2, None, None, 0, 0, common.size
    #else:
    #    print("Enough!")
    #    print (common)
    # Some images have many points outside boundaries?
    # Probably a problem with rescaling small images?
    # Ignore them if this happens
    # oob = ((xy1[:, 0] < 0) | (xy1[:, 1] < 0) | (xy1[:, 0] > w1) | (xy1[:, 1] > h1)).sum() + \
    #       ((xy2[:, 0] < 0) | (xy2[:, 1] < 0) | (xy2[:, 0] > w2) | (xy2[:, 1] > h2)).sum()
    # if oob.sum() > 200:
    #     if verbose:
    #         print(f'Processing pair ({idx1}, {idx2}): ignoring (oob) [{time() - t_start:.2f} s.]')
    #         sys.stdout.flush()
    #     return idx1, idx2, None, None, 0, 0, 0

    # print(f'Processing pair ({idx1}, {idx2}): not yet')
    # Find valid indices for each image
    common1, common2 = [], []
    for c in common:
        common1.append(np.where(p1 == c)[0][0])
        common2.append(np.where(p2 == c)[0][0])

    # if idx1 == 1364 and idx2 == 1361:
    #     embed()
    #print ("common1", common1)
    
    bb1, ratio1 = find_bb(xy1[common1, :], (h1, w1), dilation)
    #print (bb1, ratio1)
    #print ("common2", common1)
    
    bb2, ratio2 = find_bb(xy2[common2, :], (h2, w2), dilation)
    #print (bb2, ratio2)
    # embed()

    # Dunno why
    if bb1 is None or bb2 is None:
        if verbose:
            print(f'Could not find matching bounding boxes [{time() - t_start:.2f} s.]')
            sys.stdout.flush()
        return idx1, idx2, None, None, ratio1, ratio2, 0

    # Sanity check
    if bb1[0] < 0 or bb1[1] >= w1 or bb1[2] < 0 or bb1[3] >= h1:
        raise RuntimeError(f'Bounding box outside image boundaries (image 1: ({idx1, idx2}))')
    if bb2[0] < 0 or bb2[1] >= w2 or bb2[2] < 0 or bb2[3] >= h2:
        raise RuntimeError(f'Bounding box outside image boundaries (image 2: ({idx1, idx2}))')

    # print(f'Box 1: [{bb1[0]}, {bb1[1]}, {bb1[2]}, {bb1[3]}] ratio y/x: {(bb1[3] - bb1[1]) / (bb1[2] - bb1[0])}')
    # print(f'Box 2: [{bb2[0]}, {bb2[1]}, {bb2[2]}, {bb2[3]}] ratio y/x: {(bb2[3] - bb2[1]) / (bb2[2] - bb2[0])}')

    # Find final set of points in common
    xy1s = xy1[common1, :]
    xy2s = xy2[common2, :]
    in1 = np.where((xy1s[:, 0] >= bb1[0]) &
                   (xy1s[:, 0] < bb1[1]) &
                   (xy1s[:, 1] >= bb1[2]) &
                   (xy1s[:, 1] < bb1[3]))[0]
    in2 = np.where((xy2s[:, 0] >= bb2[0]) &
                   (xy2s[:, 0] < bb2[1]) &
                   (xy2s[:, 1] >= bb2[2]) &
                   (xy2s[:, 1] < bb2[3]))[0]

    final = np.intersect1d(in1, in2)

    if verbose:
        print(f'Processing pair ({idx1}, {idx2}): {final.size} matches [{time() - t_start:.2f} s.]')
        sys.stdout.flush()
    return idx1, idx2, bb1, bb2, ratio1, ratio2, final.size


def parse_seq(root, seq, dilation):
    t = time()
    src =  os.path.join(root, seq)
    # Parse reconstruction
    print(f'Processing: "{seq}"')
    cameras, images, points = read_colmap_txt(os.path.join(src,'reconstruction'))
    print(f'Cameras: {len(cameras)}')
    print(f'Images: {len(images)}')
    print(f'3D points: {len(points)}')
    indices = [i for i in cameras]

    # Get 3d points
    xyz, rgb = [], []
    for i in points:
        xyz.append(points[i].xyz)
        rgb.append(points[i].rgb)
    xyz = np.array(xyz)
    rgb = np.array(rgb)
    
    # Lazy count
    num = 0
    for i in indices:
        for j in indices:
            if i > j:
                num += 1
    print(f'Number of pairs: {num}')

    # Must retrieve size of rectified images
    t = time()
    print('Collecting rectified image sizes...')
    imsize = {}
    for i in indices:
        imsize[i] = imread(f'{src}/dense/images/{images[i].name}').shape[:2]
    print(f'Done [{time() - t:0.2f} s.]')

    count = 0
    args = []
    for i in indices:
        for j in indices:
            if i > j:
                # args.append((cameras[i], cameras[j], images[i], images[j], xyz, src, i, j, dilation))
                args.append((cameras[i], cameras[j], imsize[i], imsize[j], images[i], images[j], xyz, src, i, j, dilation))
                count += 1

    # i, j = 1181, 1180
    # i, j = 108, 13
    # r = check_pair((cameras[i], cameras[j], imsize[i], imsize[j], images[i], images[j], xyz, src, i, j, dilation))
    # raise RuntimeError("stop")

    num_proc = int(.7 * cpu_count())
    pool = Pool(processes=num_proc)

    # embed()
    if True:
        pool_res = pool.map(check_pair, args)
    else:
        pool_res = []
        for _arg in args:
            pool_res.append(check_pair(_arg))
    print(f'Done! {(time() - t)/60:.1f} min.')

    # returns: idx1, idx2, bb1, bb2, ratio1, ratio2, final.size
    res_dict = {}
    for r in pool_res:
        res_dict[r[0], r[1]] = [r[2], r[3], r[4], r[5], r[6]]

    return res_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--dilation", type=float, default=0)
    parser.add_argument("--th", type=float, default=0.1)
    params = parser.parse_args()
    th = params.th
    r = parse_seq(params.root, params.seq, params.dilation)
    n, n_th = 0, 0
    for k in r:
        if r[k][0] is not None and r[k][1] is not None:
            n += 1
        if r[k][0] is not None and r[k][1] is not None and r[k][2] >= th and r[k][3] >= th:
            n_th += 1
    print(f'Valid pairs: {n}/{len(r)} ({n/len(r)*100:.1f}%)')
    print(f'Valid pairs at ratio threshold {th:.2f}: {n_th}/{len(r)} ({n_th/len(r)*100:.1f}%)')
    t = time()
    out_dir = os.path.join(params.root, params.seq)
    dd.io.save(os.path.join(out_dir, f'dense/stereo/pairs-dilation-{params.dilation:.2f}.h5'), r)
    with open(os.path.join(out_dir, f'dense/stereo/pairs-dilation-{params.dilation:.2f}.txt'), 'w') as f:
        f.write(f'{0} {n} {len(r)} {n/len(r)*100:.1f}\n')
        f.write(f'{th} {n_th} {len(r)} {n_th/len(r)*100:.1f}\n')
    print(f'Saved! {(time() - t)/60:.1f} min.')
