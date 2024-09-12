import cv2
import numpy as np
import time

from src import pose_estimator


def triangulate_points(P1, P2, pts1, pts2):
    pts4D = np.zeros((4, pts1.shape[0]))

    for i in range(pts1.shape[0]):
        A = np.array([
            pts1[i, 0] * P1[2, :] - P1[0, :],
            pts1[i, 1] * P1[2, :] - P1[1, :],
            pts2[i, 0] * P2[2, :] - P2[0, :],
            pts2[i, 1] * P2[2, :] - P2[1, :]
        ])
        
        _, _, Vt = np.linalg.svd(A)
        pts4D[:, i] = Vt[-1]

    pts3D = pts4D[:3, :] / pts4D[3, :]
    
    return pts3D.T

def calculate_reprojection_error(pts3D, pts2D, K, R, t):
    if pts3D.shape[0] == 3:
        pts3D = pts3D.T

    pts3D_cam = R @ pts3D.T + t
    pts2D_proj_hom = K @ pts3D_cam
    pts2D_proj = pts2D_proj_hom[:2, :] / pts2D_proj_hom[2, :]
    pts2D_proj = pts2D_proj.T

    errors = np.sqrt(np.sum((pts2D - pts2D_proj)**2, axis=1))

    threshold_index = int(len(errors) * 0.1)
    top_errors = np.partition(errors, threshold_index)[:threshold_index]

    mean_error = np.mean(top_errors)

    return mean_error

def get_avg_error(src_pts, dst_pts, K, R, t):
    P1 = np.hstack((K, np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    pts3D = triangulate_points(P1, P2, src_pts, dst_pts)

    error1 = calculate_reprojection_error(pts3D, src_pts, K, np.eye(3), np.zeros((3, 1)))
    error2 = calculate_reprojection_error(pts3D, dst_pts, K, R, t)

    avg_error = (error1 + error2) / 2

    return avg_error


def get_optimal_pose(src_pts, dst_pts, K, total_time):
    min_avg_error = float('inf')
    best_R = None
    best_t = None

    while (min_avg_error > 2.0):
        pose_start = time.time()
        R1, R2, t = pose_estimator.compute_pose(src_pts, dst_pts, K)

        avg_first = get_avg_error(src_pts, dst_pts, K, R1, t)
        avg_second = get_avg_error(src_pts, dst_pts, K, R2, t)

        if (avg_first < avg_second):
            avg_error = avg_first
            R = R1
        else:
            avg_error = avg_second
            R = R2

        total_time = total_time + (time.time() - pose_start)

        if (avg_error < min_avg_error):
            min_avg_error = avg_error
            best_R = R
            best_t = t

        if (total_time >= 1.6):
            break

    return best_R, best_t, min_avg_error, total_time

def get_optimal_intrinsics(intrinsic, dist_coeffs, img_width, img_height):
    # [TODO] REPLACE OpenCV!!
    K, _ = cv2.getOptimalNewCameraMatrix(
        intrinsic, 
        dist_coeffs, 
        (img_width, img_height), 
        alpha=0, 
        centerPrincipalPoint=True)

    return K
