import os
import time
import cv2
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image,load_and_mask_image, rbd
from lightglue import viz2d


def compute_fundamental_matrix_m_estimator(src_pts, dst_pts, max_iterations=10000):
    """Compute fundamental matrix using M-estimator with Huber loss."""
    best_F = None
    best_error = float('inf') 

    for i in range(max_iterations):
        # 랜덤하게 8개의 점 선택
        indices = np.random.choice(len(src_pts), 8, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        # 기본적인 fundamental matrix 계산
        F = cv2.findFundamentalMat(src_sample, dst_sample, cv2.FM_8POINT)[0]

        # 모든 점에 대해 재투영 오차 계산
        src_pts_h = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
        dst_pts_h = np.hstack((dst_pts, np.ones((dst_pts.shape[0], 1))))
        errors = np.abs(np.einsum('ij,jk->i', src_pts_h, F @ dst_pts_h.T))

        # 상위 10%의 오차 선택
        threshold_index = int(len(errors) * 0.1)
        top_errors = np.partition(errors, threshold_index)[:threshold_index]

        # 평균 계산
        top_mean_error = np.mean(top_errors)
        mean_error = np.mean(errors)

        if mean_error < best_error:
            best_error = mean_error
            best_top_error = top_mean_error
            best_F = F

        if (best_error < 5.0) and (best_top_error < 1.0):
            print(i)
            break

    print("Top errors: ", best_top_error)
    print("Best errors: ", best_error)
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

    # 3D 점을 카메라 좌표계로 변환
    pts3D_cam = R @ pts3D.T + t

    # 3D 점을 2D로 투영
    pts2D_proj, _ = cv2.projectPoints(pts3D_cam.T, np.zeros(3), np.zeros(3), K, None)
    pts2D_proj = pts2D_proj.reshape(-1, 2)

    # 재투영 오차 계산
    errors = np.sqrt(np.sum((pts2D - pts2D_proj)**2, axis=1))

    # 상위 10%의 오차 선택
    threshold_index = int(len(errors) * 0.1)
    top_errors = np.partition(errors, threshold_index)[:threshold_index]

    # 평균 계산
    mean_error = np.mean(top_errors)

    return mean_error

# SuperPoint+LightGlue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

# This will be removed
print("Initializing...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)
_ = extractor.extract(dummy_input)
_ = matcher({'image0': _, 'image1': _})
print("Initialization done")
# This will be removed

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
src_index = 0
dst_index = 1
root_path = '../data/raw'

original_img0 = load_image(root_path + f'/{src_index}.bmp')
original_img1 = load_image(root_path + f'/{dst_index}.bmp')

image0 = load_and_mask_image(root_path + f'/{src_index}.bmp').to(device)
image1 = load_and_mask_image(root_path + f'/{dst_index}.bmp').to(device)

start_time = time.time()
print("Starting feature extraction...")

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
time_interval1 = time.time()
feats1 = extractor.extract(image1)
time_interval2 = time.time()

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})

end_time = time.time()

feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([original_img0, original_img1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)

viz2d.save_plot(f'results/matches_{dst_index}.png')
print(f"Time taken: {time_interval1 - start_time} seconds")
print(f"Time taken: {time_interval2 - time_interval1} seconds")
print(f"Time taken: {end_time - time_interval2} seconds")
print(f"Time taken: {end_time - start_time} seconds")

# NumPy 배열로 변환
src_pts = m_kpts0.cpu().numpy()
dst_pts = m_kpts1.cpu().numpy()

original_K = np.array([[1.71326283e+03, 0.00000000e+00, 1.26716548e+03],
              [0.00000000e+00, 1.71201684e+03, 1.06652107e+03],
              [0, 0, 1]])
dist_coeffs = np.array([1.70168395e+01, -4.91002680e+00, -1.75329962e-03, -4.80750653e-05,  -1.21918723e+00, 1.76062725e+01,  2.02562080e+00, -4.35890100e+00])

K, _ = cv2.getOptimalNewCameraMatrix(original_K, dist_coeffs, (image0.shape[1], image0.shape[0]), alpha=0, centerPrincipalPoint=True)

pose_time = time.time()

R, t = compute_pose(src_pts, dst_pts, K)

P1 = np.hstack((K, np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))

pts3D = triangulate_points(P1, P2, src_pts, dst_pts)

error1 = calculate_reprojection_error(pts3D, src_pts, K, np.eye(3), np.zeros((3, 1)))
error2 = calculate_reprojection_error(pts3D, dst_pts, K, R, t)

# 평균 재투영 오차
avg_error = (error1 + error2) / 2

print(f"Time taken: {time.time() - pose_time} seconds")
print(f"첫 번째 이미지 재투영 오차: {error1:.4f} 픽셀")
print(f"두 번째 이미지 재투영 오차: {error2:.4f} 픽셀")
print(f"평균 재투영 오차: {avg_error:.4f} 픽셀")
print("======================================")
