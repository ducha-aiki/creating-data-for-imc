import os
import numpy as np


def get_eval_path(mode, cfg):
    if mode == "feature":
        return get_feature_path(cfg)
    elif mode == "match":
        return get_match_path(cfg)
    elif mode == "filter":
        return get_filter_path(cfg)
    elif mode == "model":
        return get_geom_path(cfg)
    elif mode == "stereo":
        return get_stereo_path(cfg)
    else:
        raise ValueError("Unknown job type")


def get_eval_file(mode, cfg, job_id=None):
    if job_id:
        return os.path.join(get_eval_path(mode, cfg),
                            "{}.{}".format(job_id, mode))
    else:
        try:
            file_list = os.listdir(get_eval_path(mode, cfg))
            valid_file = [
                file for file in file_list if file.split('.')[-1] == mode
            ]
            if len(valid_file) == 0:
                return None
            elif len(valid_file) == 1:
                return os.path.join(get_eval_path(mode, cfg), valid_file[0])
            else:
                print("Should never be here")
                import IPython
                IPython.embed()
                return None
        except FileNotFoundError:
            os.makedirs(get_eval_path(mode, cfg))
            return None


def get_data_path(cfg):
    """Returns where the per-dataset results folder is stored.

    TODO: This probably should be done in a neater way.
    """

    # Get data directory for "set_100"
    return os.path.join(cfg.path_data, cfg.dataset_name,
                        "set_{}".format(cfg.num_max_set))


def get_base_path(cfg):
    """Returns where the per-dataset results folder is stored."""

    return os.path.join(cfg.path_results, cfg.dataset_name)


def get_feature_path(cfg):
    """Returns where the keypoints and descriptor results folder is stored.

    Method names converted to lower-case."""

    return os.path.join(
        get_base_path(cfg), "{}_{}_{}".format(cfg.method_kp.lower(),
                                              cfg.num_kp,
                                              cfg.method_desc.lower()))


def get_kp_file(cfg):
    """Returns the path to the keypoint file."""

    return os.path.join(get_feature_path(cfg), "keypoints.h5")


def get_scale_file(cfg):
    """Returns the path to the scale file."""

    return os.path.join(get_feature_path(cfg), "scales.h5")


def get_score_file(cfg):
    """Returns the path to the score file."""

    return os.path.join(get_feature_path(cfg), "scores.h5")


def get_angle_file(cfg):
    """Returns the path to the angle file."""

    return os.path.join(get_feature_path(cfg), "angles.h5")


def get_affine_file(cfg):
    """Returns the path to the angle file."""

    return os.path.join(get_feature_path(cfg), "affine.h5")


def get_desc_file(cfg):
    """Returns the path to the descriptor file."""

    return os.path.join(get_feature_path(cfg), "descriptors.h5")


def get_match_name(cfg):
    """Return folder name for the matching model.

    Converted to lower-case to avoid conflicts."""

    if isinstance(cfg.method_match, str):
        match_method = cfg.method_match
    elif isinstance(cfg.method_match, dict):
        # Make a custom string for the matching folder
        match_method = [cfg.method_match['method']]

        # flann/bf
        if cfg.method_match['flann']:
            match_method += ['flann']
        else:
            match_method += ['bf']

        # number of neighbours
        match_method += ['numnn-{}'.format(cfg.method_match['num_nn'])]

        # distance
        match_method += ['dist-{}'.format(cfg.method_match['distance'])]

        # 2-way matching
        if not cfg.method_match['symmetric']['enabled']:
            match_method += ['nosym']
        else:
            match_method += [
                'sym-{}'.format(cfg.method_match['symmetric']['reduce'])
            ]

        # filtering
        if cfg.method_match['filtering']['type'] == 'none':
            match_method += ['nofilter']
        elif cfg.method_match['filtering']['type'].lower() in [
                'snn_ratio_pairwise', 'snn_ratio_vs_last'
        ]:
            match_method += [
                'filter-snn-{}'.format(
                    cfg.method_match['filtering']['threshold'])
            ]
        elif cfg.method_match['filtering']['type'].lower(
        ) == 'fginn_ratio_pairwise':
            match_method += [
                'filter-fginn-pairwise-{}-{}'.format(
                    cfg.method_match['filtering']['threshold'],
                    cfg.method_match['filtering']['fginn_radius'])
            ]
        else:
            raise ValueError("Unknown filtering type")

        # distance filtering
        if 'descriptor_distance_filter' in cfg.method_match:
            if 'threshold' in cfg.method_match['descriptor_distance_filter']:
                max_dist = cfg.method_match['descriptor_distance_filter'][
                    'threshold']
                match_method += ['maxdist-{:.03f}'.format(max_dist)]

        # Refine with CNe
        # if cfg.refine_inliers:
        #     match_method += ["CNe"]

        match_method = '_'.join(match_method)
    else:
        raise ValueError('Unknown method_match {}'.format(str(
            cfg.method_match)))

    return match_method.lower()

def get_filter_path(cfg):
    # TODO After adding more filters, change refine flag to some string options
    if cfg.refine_inliers:
        return os.path.join(get_match_path(cfg),"cne")
    else:
        return os.path.join(get_match_path(cfg),"no_filter")

def get_match_path(cfg):
    """Returns where the match results folder is stored."""
    return os.path.join(get_feature_path(cfg), get_match_name(cfg))


def get_match_file(cfg):
    """Returns the path to the match file."""

    return os.path.join(get_match_path(cfg), "matches.h5")


def get_match_cost_file(cfg):
    """Returns the path to the match file."""

    return os.path.join(get_match_path(cfg), "matching_cost.h5")


def get_geom_name(cfg):
    """Return folder name for the geometry model.

    Converted to lower-case to avoid conflicts."""

    method = cfg.method_geom['method'].lower()

    # Temporary fix
    if method == 'cv2-patched-ransac-f':
        label = "_".join([
            method, 'th', str(cfg.method_geom['threshold']),
            'conf', str(cfg.method_geom['confidence'])
        ])
    elif method in ['cv2-ransac-e', 'cv2-ransac-f']:
        label = "_".join([
            method, 'th', str(cfg.method_geom['threshold']),
            'conf', str(cfg.method_geom['confidence']),
        ])
    elif method in ['cmp-degensac-f', 'cmp-degensac-f-laf', 'cmp-gc-ransac-e']:
        label = "_".join([
            method,
            'th', str(cfg.method_geom['threshold']),
            'conf', str(cfg.method_geom['confidence']),
            'max_iter', str(cfg.method_geom['max_iter']),
            'error', str(cfg.method_geom['error_type']),
            'degencheck', str(cfg.method_geom['degeneracy_check'])
        ])
    elif method in ['cmp-gc-ransac-f', 'skimage-ransac-f', 'cmp-magsac-f']:
        label = "_".join([
            method,
            'th', str(cfg.method_geom['threshold']),
            'conf', str(cfg.method_geom['confidence']),
            'max_iter', str(cfg.method_geom['max_iter'])
        ])
    elif method in ['cv2-lmeds-e', 'cv2-lmeds-f']:
        label = "_".join([method, 'conf', str(cfg.method_geom['confidence'])])
    elif method in ['intel-dfe-f']:
        label = "_".join([method, 'th', str(cfg.method_geom['threshold']), 'postprocess', str(cfg.method_geom['postprocess'])])
    elif method in ['cv2-7pt', 'cv2-8pt']:
        label = method
    else:
        raise ValueError('Unknown method for E/F estimation')

    return label.lower()


def get_geom_path(cfg):
    """Returns where the match results folder is stored."""

    geom_name = get_geom_name(cfg)
    return os.path.join(get_filter_path(cfg), geom_name)


def get_geom_file(cfg):
    """Returns the path to the match file."""

    return os.path.join(get_geom_path(cfg), "essential.h5")


def get_geom_inl_file(cfg):
    """Returns the path to the match file."""
    return os.path.join(get_geom_path(cfg), "essential_inliers.h5")


def get_geom_cost_file(cfg):
    """Returns the path to the geom cost file."""
    return os.path.join(get_geom_path(cfg), "geom_cost.h5")


def get_cne_temp_path(cfg):
    return os.path.join(get_filter_path(cfg), "temp_cne")


def get_filter_match_file(cfg):
    return os.path.join(get_filter_path(cfg), "matches_inlier.h5")


def get_filter_cost_file(cfg):
    return os.path.join(get_filter_path(cfg), "matches_inlier_cost.h5")


def get_cne_data_dump_path(cfg):
    return os.path.join(get_cne_temp_path(cfg), "data_dump")


def get_stereo_path(cfg):
    """Returns the path to where the stereo results are stored."""

    return os.path.join(get_geom_path(cfg), "set_{}".format(cfg.num_max_set))


def get_stereo_pose_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (pose errors)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg),
                        "stereo_pose_errors{}.h5".format(label))


def get_repeatability_score_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (pose errors)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg),
                        "repeatability_score_file{}.h5".format(label))


def get_stereo_epipolar_pre_match_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (epipolar distances)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(get_stereo_path(cfg),
                        "stereo_epipolar_pre_match_errors{}.h5".format(label))


def get_stereo_epipolar_refined_match_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (epipolar distances)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(
        get_stereo_path(cfg),
        "stereo_epipolar_refined_match_errors{}.h5".format(label))


def get_stereo_epipolar_final_match_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (epipolar distances)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(
        get_stereo_path(cfg),
        "stereo_epipolar_final_match_errors{}.h5".format(label))


def get_stereo_depth_projection_pre_match_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (epipolar distances)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(
        get_stereo_path(cfg),
        "stereo_projection_errors_pre_match{}.h5".format(label))


def get_stereo_depth_projection_refined_match_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (epipolar distances)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(
        get_stereo_path(cfg),
        "stereo_projection_errors_refined_match{}.h5".format(label))


def get_stereo_depth_projection_final_match_file(cfg, th=None):
    """Returns the path to where the stereo results are.  (epipolar distances)

    """

    label = "" if th is None else "-th-{:s}".format(th)
    return os.path.join(
        get_stereo_path(cfg),
        "stereo_projection_errors_final_match{}.h5".format(label))


def get_colmap_path(cfg):
    """Returns the path to where the colmap results are stored."""

    return os.path.join(get_filter_path(cfg),"multiview",
                        "bag_size_{}".format(cfg.bag_size),
                        "bag_id_{:05d}".format(cfg.bag_id))


def get_colmap_mark_file(cfg):
    """Returns the path to where the colmap results are.  (pose errors)

    """

    return os.path.join(get_colmap_path(cfg), "colmap_has_run")


def get_colmap_pose_file(cfg):
    """Returns the path to where the colmap results are.  (pose errors)

    """

    return os.path.join(get_colmap_path(cfg), "colmap_pose_errors.h5")


def get_colmap_output_path(cfg):
    """Returns the path to where the colmap outputs are.

    """

    return os.path.join(get_colmap_path(cfg), "colmap")


def get_colmap_temp_path(cfg):
    """Returns the path to where the colmap working path should be.

    TODO: Do we want to use slurm temp directory?

    """

    return os.path.join(get_colmap_path(cfg), "temp_colmap")


def parse_file_to_list(file_name, data_dir):
    """
    Parses filenames from the given text file using the `data_dir`

    :param file_name: File with list of file names
    :param data_dir: Full path location appended to the filename

    :return: List of full paths to the file names
    """

    fullpath_list = []
    with open(file_name, "r") as f:
        while True:
            # Read a single line
            line = f.readline()
            # Check the `line` type
            if not isinstance(line, str):
                line = line.decode("utf-8")
            if not line:
                break
            # Strip `\n` at the end and append to the `fullpath_list`
            fullpath_list.append(os.path.join(data_dir, line.rstrip("\n")))
    return fullpath_list


def get_fullpath_list(data_dir, key):
    """
    Returns the full-path lists to image info in `data_dir`

    :param data_dir: Path to the location of dataset
    :param key: Which item to retrieve from

    :return: Tuple containing fullpath lists for the key item
    """
    # Read the list of images, homography and geometry
    list_file = os.path.join(data_dir, "{}.txt".format(key))

    # Parse files to fullpath lists
    fullpath_list = parse_file_to_list(list_file, data_dir)

    return fullpath_list


def get_item_name_list(fullpath_list):
    """Returns each item name in the full path list, excluding the extension"""

    return [os.path.splitext(os.path.basename(_s))[0] for _s in fullpath_list]


def get_stereo_viz_folder(cfg):
    """Returns the path to the matching results visualization folder.

    """

    # Ideally we would keep the PDFs but support is broken, see viz_colmap.py
    match_method = get_match_name(cfg)

    base = os.path.join(
        cfg.path_visualization, "{}-{}-{}-{}".format(cfg.method_kp.upper(),
                                                     cfg.method_desc.upper(),
                                                     cfg.num_kp,
                                                     match_method.upper()),
        cfg.dataset_name)
    return os.path.join(base, 'stereo-png'), os.path.join(base, 'stereo-jpg')


def get_colmap_viz_folder(cfg):
    """Returns the path to the MVS results visualization folder.

    Same convention as for the json file
    """
    # Ideally we would keep the PDFs but support is broken, see viz_colmap.py
    match_method = get_match_name(cfg)

    base = os.path.join(
        cfg.path_visualization, "{}-{}-{}-{}".format(cfg.method_kp.upper(),
                                                     cfg.method_desc.upper(),
                                                     cfg.num_kp,
                                                     match_method.upper()),
        cfg.dataset_name)
    return os.path.join(base, 'mvs-png'), os.path.join(base, 'mvs-jpg')


def get_pairs_per_threshold(data_dir):
    pairs = {}
    for th in np.arange(0, 1, 0.1):
        pairs['{:0.1f}'.format(th)] = np.load(
            '{}/new-vis-pairs/keys-th-{:0.1f}.npy'.format(data_dir, th))
    return pairs
