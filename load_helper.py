import os
import cv2
import numpy as np
import h5py

def save_h5(dict_to_save, filename):
    """Saves dictionary to hdf5 file"""

    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


def load_h5(filename):
    """Loads dictionary from hdf5 file"""

    dict_to_load = {}
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]#.value
    return dict_to_load

def load_image(image_path,
               use_color_image=False,
               input_width=512,
               crop_center=True):
    """
    Loads image and do preprocessing.

    :param image_path: Fullpath to the image.
    :param use_color_image: Flag to read color/gray image
    :param input_width: Width of the image for scaling
    :param crop_center: Flag to crop while scaling

    :return: Tuple of (Color/Gray image, scale_factor)
    """
    # Assuming all images in the directory are color images
    image = cv2.imread(image_path)
    if not use_color_image:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop center and resize image into something reasonable
    scale_factor = 1.0
    if crop_center:
        rows, cols = image.shape[:2]
        if rows > cols:
            cut = (rows - cols) // 2
            img_cropped = image[cut:cut + cols, :]
        else:
            cut = (cols - rows) // 2
            img_cropped = image[:, cut:cut + rows]
        scale_factor = float(input_width) / float(img_cropped.shape[0])
        image = cv2.resize(img_cropped, (input_width, input_width))

    return (image, scale_factor)


def load_depth(depth_path):
    return load_h5(depth_path)["depth"]


def load_vis(vis_fullpath_list, subset_index=None):
    """
    Given fullpath_list load all visibility ranges

    :param vis_fullpath_list: Full path list of visibility

    :return: List of visibility
    """
    vis = []
    if subset_index is None:
        for vis_file in vis_fullpath_list:
            # Load visibility
            vis.append(np.loadtxt(vis_file).flatten().astype("float32"))
    else:
        for idx in subset_index:
            tmp_vis = np.loadtxt(
                vis_fullpath_list[idx]).flatten().astype("float32")
            tmp_vis = tmp_vis[subset_index]
            vis.append(tmp_vis)
    return vis


def load_calib(calib_fullpath_list, subset_index=None):
    """Load all calibration files and create a dictionary."""

    calib = {}
    if subset_index is None:
        for _calib_file in calib_fullpath_list:
            img_name = os.path.splitext(
                os.path.basename(_calib_file))[0].replace("calibration_", "")
            # _calib_file.split(
            #     "/")[-1].replace("calibration_", "")[:-3]
            # # Don't know why, but rstrip .h5 also strips
            # # more then necssary sometimes!
            # #
            # # img_name = _calib_file.split(
            # #     "/")[-1].replace("calibration_", "").rstrip(".h5")
            calib[img_name] = load_h5(_calib_file)
    else:
        for idx in subset_index:
            _calib_file = calib_fullpath_list[idx]
            img_name = os.path.splitext(
                os.path.basename(_calib_file))[0].replace("calibration_", "")
            calib[img_name] = load_h5(_calib_file)
    return calib
