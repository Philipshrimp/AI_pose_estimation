import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src import matcher, optimizer


def get_pose(cfg_data):
    root_path = cfg_data['root_path']
    src_index = cfg_data['src_index']
    dst_index = cfg_data['dst_index']

    intrinsics = np.array(cfg_data['intrinsics'])
    dist_coeffs = np.array(cfg_data['dist_coeffs'])
    img_width = cfg_data['img_width']
    img_height = cfg_data['img_height']

    src_pts, dst_pts, total_time = matcher.compute_correspondence_matching(root_path, src_index, dst_index)
    src_pts, dst_pts = matcher.reject_outliers(src_pts, dst_pts)

    K = optimizer.get_optimal_intrinsics(intrinsics, dist_coeffs, img_width, img_height)

    R, t, min_avg_error, total_time = optimizer.get_optimal_pose(src_pts, dst_pts, K, total_time)

    print("---Final result---")
    print("Average error: ", min_avg_error)
    print("Total computation time: ", total_time)

    return R, t
