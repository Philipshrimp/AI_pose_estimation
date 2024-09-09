import os
import time
import cv2
import numpy as np
import yaml
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image,load_and_mask_image, rbd
from lightglue import viz2d


def compute_correspondence_matching(root_path, src_index, dst_index, max_keypoints=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # This will be removed
    print("Initializing...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    _ = extractor.extract(dummy_input)
    _ = matcher({'image0': _, 'image1': _})
    print("Initialization done")
    # This will be removed

    image0 = load_and_mask_image(root_path + f'/{src_index}.bmp').to(device)
    image1 = load_and_mask_image(root_path + f'/{dst_index}.bmp').to(device)

    start_time = time.time()
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({'image0': feats0, 'image1': feats1})

    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]

    end_time = time.time()

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    src_pts = m_kpts0.cpu().numpy()
    dst_pts = m_kpts1.cpu().numpy()
    total_time = end_time - start_time

    return src_pts, dst_pts, total_time

def reject_outliers(src_pts, dst_pts):
    mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 2.0)[1]
    mask = mask.ravel().astype(bool)

    new_src = src_pts[mask]
    new_dst = dst_pts[mask]

    return new_src, new_dst

def get_optimal_intrinsics(intrinsic, dist_coeffs, img_width, img_height):
    K, _ = cv2.getOptimalNewCameraMatrix(
        intrinsic, 
        dist_coeffs, 
        (img_width, img_height), 
        alpha=0, 
        centerPrincipalPoint=True)

    return K

def get_optimal_pose(src_pts, dst_pts, K, total_time):
    min_avg_error = float('inf')
    best_R = None
    best_t = None

    while (min_avg_error > 2.0):
        pose_start = time.time()
        R, t = compute_pose(src_pts, dst_pts, K)

        P1 = np.hstack((K, np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))

        pts3D = triangulate_points(P1, P2, src_pts, dst_pts)

        error1 = calculate_reprojection_error(pts3D, src_pts, K, np.eye(3), np.zeros((3, 1)))
        error2 = calculate_reprojection_error(pts3D, dst_pts, K, R, t)
        avg_error = (error1 + error2) / 2

        total_time = total_time + (time.time() - pose_start)

        if (avg_error < min_avg_error):
            min_avg_error = avg_error
            best_R = R
            best_t = t

        if (total_time >= 1.6):
            break

    return best_R, best_t, min_avg_error, total_time

def compute_fundamental_matrix_m_estimator(src_pts, dst_pts, max_iterations=5000):
    best_F = None
    best_error = float('inf') 

    for _ in range(max_iterations):
        indices = np.random.choice(len(src_pts), 8, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        F = cv2.findFundamentalMat(src_sample, dst_sample, cv2.FM_8POINT)[0]
        
        if F is None:
            continue

        src_pts_h = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
        dst_pts_h = np.hstack((dst_pts, np.ones((dst_pts.shape[0], 1))))
        errors = np.abs(np.einsum('ij,jk->i', src_pts_h, F @ dst_pts_h.T))

        threshold_index = int(len(errors) * 0.1)
        top_errors = np.partition(errors, threshold_index)[:threshold_index]

        top_mean_error = np.mean(top_errors)
        mean_error = np.mean(errors)

        if mean_error < best_error:
            best_error = mean_error
            best_top_error = top_mean_error
            best_F = F

        if (best_error < 5.0) and (best_top_error < 1.0):
            return best_F
    
    return best_F

def compute_pose(src_pts, dst_pts, K):
    F = compute_fundamental_matrix_m_estimator(src_pts, dst_pts)

    E = K.T @ F @ K

    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)
    
    return R, t

def triangulate_points(P1, P2, pts1, pts2):
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    pts3D = pts4D[:3, :] / pts4D[3, :]
    return pts3D.T

def calculate_reprojection_error(pts3D, pts2D, K, R, t):
    if pts3D.shape[0] == 3:
        pts3D = pts3D.T

    pts3D_cam = R @ pts3D.T + t

    pts2D_proj, _ = cv2.projectPoints(pts3D_cam.T, np.zeros(3), np.zeros(3), K, None)
    pts2D_proj = pts2D_proj.reshape(-1, 2)

    errors = np.sqrt(np.sum((pts2D - pts2D_proj)**2, axis=1))

    threshold_index = int(len(errors) * 0.1)
    top_errors = np.partition(errors, threshold_index)[:threshold_index]

    mean_error = np.mean(top_errors)

    return mean_error

def visualize(root_path, src_index, dst_index, m_kpts0, m_kpts1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_img0 = load_image(root_path + f'/{src_index}.bmp').to(device)
    original_img1 = load_image(root_path + f'/{dst_index}.bmp').to(device)

    axes = viz2d.plot_images([original_img0, original_img1])

    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.save_plot(f'results/matches_{dst_index}.png')

def get_pose(cfg_data):
    root_path = cfg_data['root_path']
    src_index = cfg_data['src_index']
    dst_index = cfg_data['dst_index']

    intrinsics = np.array(cfg_data['intrinsics'])
    dist_coeffs = np.array(cfg_data['dist_coeffs'])
    img_width = cfg_data['img_width']
    img_height = cfg_data['img_height']

    src_pts, dst_pts, total_time = compute_correspondence_matching(root_path, src_index, dst_index)
    src_pts, dst_pts = reject_outliers(src_pts, dst_pts)

    K = get_optimal_intrinsics(intrinsics, dist_coeffs, img_width, img_height)

    R, t, min_avg_error, total_time = get_optimal_pose(src_pts, dst_pts, K, total_time)

    print("---Final result---")
    print("Average error: ", min_avg_error)
    print("Total computation time: ", total_time)

    return R, t


with open('config.yaml', 'r') as file:
    cfg_data = yaml.load(file, Loader=yaml.FullLoader)

R, t = get_pose(cfg_data)

print(R)
print(t)
