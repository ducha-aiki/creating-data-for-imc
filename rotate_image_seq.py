import sys
sys.path.append('../sfm_benchmark')

import os
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from path_helper import get_fullpath_list
from load_helper import load_calib
from io_helper import save_h5, load_h5
from argparse import ArgumentParser

def roatat_image(r, t, k, image, depth):
    # init axis in cam cooridnate
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    # rotate to world coordinate
    x_rot_axis = np.matmul(r.T, x_axis)
    y_rot_axis = np.matmul(r.T, y_axis)
    # get normal vector for plane formed by y axis and rotated x axis
    z_rot_axis = np.cross(x_rot_axis, y_axis)
    z_rot_axis = z_rot_axis / np.linalg.norm(z_rot_axis)
    # project rotated y axis to y_axis-x_rotated_axis plane
    y_proj_axis = (y_rot_axis - np.dot(y_rot_axis, z_axis) * z_axis)
    y_proj_axis = y_proj_axis / np.linalg.norm(y_proj_axis)
    # calculate angle between projected rotated y axis and x_axis
    angle = math.acos(np.dot(y_axis, y_proj_axis))

    # continue if image is upright
    #if angle < math.pi / 4:
    #    return r, t, k, image, depth, False
    # check if image is roateted 90 degree
    print("found image not upright")
    if True:#abs(angle) < math.pi / 4 * 3:
        # calculate new extrinsic and intrinsic
        direct = -1#y_rot_axis[0] / abs(y_rot_axis[0])
        r_90 = R.from_euler('Z', [-direct * 90], degrees=True).as_dcm()
        r = np.matmul(r_90, r)
        t = np.matmul(r_90, t)
        f_x = k[0, 0]
        f_y = k[1, 1]
        c_x = k[0, 2]
        c_y = k[1, 2]
        k[0, 0] = f_y
        k[1, 1] = f_x
        k[0, 2] = c_y
        k[1, 2] = c_x
        # rotate image
        trans_img = cv2.transpose(image)
        if direct == -1:
            image = cv2.flip(trans_img, 1)
            depth = np.rot90(depth, 3)
        else:
            image = cv2.flip(trans_img, 0)
            depth = np.rot90(depth)
    # check if image is upside down
    else:
        # calculate new extrinsic and intrinsic
        direct = -y_rot_axis[0] / abs(y_rot_axis[0])
        r_180 = R.from_euler('Z', [-direct * 180], degrees=True).as_dcm()
        r = np.matmul(r_180, r)
        t = np.matmul(r_180, t)
        # rotate image
        image = cv2.flip(image, -1)
        depth = np.rot90(depth, 2)

    return np.squeeze(r), np.squeeze(t), k, image, depth, True


def main(data_base_dir,data_list):

    for data in data_list:
        print("processing {}".format(data))
        data_dir = os.path.join(data_base_dir, data, "set_100_new")
        image_dir = os.path.join(data_dir, "images")
        depth_dir = os.path.join(data_dir, "depth_maps")
        calib_dir = os.path.join(data_dir, "calibration")
        calib_list = get_fullpath_list(data_dir, "calibration")
        calib_dict = load_calib(calib_list)
        for key, calib in calib_dict.items():
            r = np.asarray(calib["R"])
            t = np.asarray(calib["T"])
            k = np.asarray(calib["K"])

            # get calibration file path
            calib_file_name = "calibration_" + key + ".h5"
            calib_path = os.path.join(calib_dir, calib_file_name)

            # get image file path
            image_file_name = key + ".jpg"
            image_path = os.path.join(image_dir, image_file_name)

            # read depth_map
            depth_file_name = key + ".h5"
            depth_path = os.path.join(depth_dir, depth_file_name)

            # load depth map and image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            try:
                depth_dict = load_h5(depth_path)
                #print (image.shape)
                #print (depth_dict.keys(), depth_dict["min_distance"].shape)
                #print (depth_dict["depth"])
                #print (depth_dict["depth"].shape)
            except:
                depth_dict = {"depth": np.zeros((image.shape[0], image.shape[1])),
                              "min_distance": np.zeros((image.shape[0], image.shape[1]))}
            depth_map = depth_dict["depth"]

            r, t, k, image, depth_map, changed_flag = roatat_image(r, t, k, image, depth_map)

            calib["T"] = t
            calib["R"] = r
            calib["K"] = k
            if changed_flag:
                # svae calibration file
                save_h5(calib, calib_path )

                # save rotated image
                result = cv2.imwrite(
                        os.path.join(image_dir, key + ".jpg"),
                        image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                # save depth map
                depth_dict["depth"] = depth_map
                save_h5(depth_dict, depth_path )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    params = parser.parse_args()
    main(params.root, [params.seq])
