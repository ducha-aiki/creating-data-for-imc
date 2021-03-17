import json
from jsmin import jsmin

import cv2
import h5py
import numpy as np


def load_image(image_path,
               use_color_image=False,
               input_width=512,
               crop_center=True,
               force_rgb=False):
    """
    Loads image and do preprocessing.

    :param image_path: Fullpath to the image.
    :param use_color_image: Flag to read color/gray image
    :param input_width: Width of the image for scaling
    :param crop_center: Flag to crop while scaling
    :param force_rgb: Flag to convert color image from BGR to RGB

    :return: Tuple of (Color/Gray image, scale_factor)
    """
    # Assuming all images in the directory are color images
    image = cv2.imread(image_path)
    if not use_color_image:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif force_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


def build_composite_image(image_path1,
                          image_path2,
                          axis=1,
                          margin=0,
                          background=1):
    """
    Load two images and returns a composite image.

    :param image_path1: Fullpath to image 1.
    :param image_path2: Fullpath to image 2.
    :param margin: Space between images
    :param background: 0 (black) or 1 (white)

    :return: (Composite image,
                (vertical_offset1, vertical_offset2),
                (horizontal_offset1, horizontal_offset2))
    """
    if background != 0 and background != 1:
        background = 1
    if axis != 0 and axis != 1:
        raise RuntimeError('Axis must be 0 (vertical) or 1 (horizontal')

    im1 = cv2.imread(image_path1)
    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    elif im1.ndim == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError('invalid image format')

    im2 = cv2.imread(image_path2)
    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    elif im2.ndim == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError('invalid image format')

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3),
                             dtype=np.uint8) + 255 * background
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3),
                             dtype=np.uint8) + 255 * background
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin
    composite[voff1:voff1 + h1, hoff1:hoff1 + w1, :] = im1
    composite[voff2:voff2 + h2, hoff2:hoff2 + w2, :] = im2

    return (composite, (voff1, voff2), (hoff1, hoff2))


def load_json(json_path):
    """Loads json file."""
    with open(json_path) as js_file:
        out = parse_json(js_file.read())
    return out


def parse_json(str1):
    # print ('got', str1)
    minified = jsmin(str1).replace('\n', ' ')
    minified = minified.replace(',]', ']')
    minified = minified.replace(',}', '}')
    if minified.startswith('"') and minified.endswith('"'):
        minified = minified[1:-1]
    # print ('parsing', minified)

    json_load = json.loads(minified)
    return json_load


def add_method_json_to_cfg(cfg, json_method):
    ''' Parsing new format of method '''
    cfg_js = json_method['config']
    cfg.label = json_method['label']
    cfg.method_kp = cfg_js['keypoint']
    cfg.num_kp = int(cfg_js['num_keypoints'])
    cfg.method_desc = cfg_js['descriptor']
    cfg.method_match = cfg_js['matcher']
    cfg.method_geom = cfg_js['method_geom']

    cfg.refine_inliers = cfg_js['refine_inliers']
    assert 'distance' in cfg.method_match
    assert 'filtering' in cfg.method_match
    assert 'num_nn' in cfg.method_match
    assert 'method' in cfg.method_match
    assert 'symmetric' in cfg.method_match
    cfg.num_nn = cfg.method_match['num_nn']

    # copy the entire dict here to save it on pack_res.py
    cfg.full_method_dict = json_method

    return cfg


def add_method_str_to_cfg(cfg, str_method):
    ''' Parsing old format of method '''

    raise RuntimeError('Deprecated: please use the new config format')

    tmp = str_method.split("_")
    if len(tmp) != 6:
        raise RuntimeError("Bad number of arguments: {}".format(str_method))
    cfg.method_kp = tmp[0]
    cfg.num_kp = tmp[1]
    cfg.method_desc = tmp[2]
    cfg.method_match = {}
    cfg.method_match['method'] = tmp[3]
    cfg.method_match['num_nn'] = int(tmp[4])
    cfg.num_nn = int(tmp[4])
    cfg.ratio_th = int(tmp[5]) / 100.0
    cfg.method_match['filtering'] = {}
    cfg.method_match['filtering']['type'] = 'snn_ratio'
    cfg.method_match['filtering']['threshold'] = cfg.ratio_th
    cfg.method_match['symmetric'] = {}
    cfg.method_match['symmetric']['enabled'] = False

    return cfg


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
            dict_to_load[key] = f[key].value
    return dict_to_load
