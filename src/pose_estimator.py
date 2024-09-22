import numpy as np


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    
    T = np.array([[1/std, 0, -mean[0]/std],
                  [0, 1/std, -mean[1]/std],
                  [0, 0, 1]])
    
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_h.T).T
    
    return pts_norm, T

def construct_A(pts1, pts2):
    A = []
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i, 0], pts1[i, 1]
        x2, y2 = pts2[i, 0], pts2[i, 1]
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    return np.array(A)

def compute_fundamental_matrix(src_pts, dst_pts):
    src_norm, T1 = normalize_points(src_pts)
    dst_norm, T2 = normalize_points(dst_pts)

    A = construct_A(src_norm[:, :2], dst_norm[:, :2])

    U, S, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_norm)
    S[-1] = 0
    F_norm = U @ np.diag(S) @ Vt

    F = T2.T @ F_norm @ T1

    return F

def compute_m_estimator(src_pts, dst_pts, max_iterations=5000):
    best_F = None
    best_error = float('inf') 

    for _ in range(max_iterations):
        indices = np.random.choice(len(src_pts), 8, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        F = compute_fundamental_matrix(src_sample, dst_sample)

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
    F = compute_m_estimator(src_pts, dst_pts)

    E = K.T @ F @ K

    U, _, Vt = np.linalg.svd(E)

    W = np.array([[0, -1, 0], 
                  [1,  0, 0], 
                  [0,  0, 1]])
    
    R1 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0:
        R1 = -R1

    R2 = U @ W @ Vt
    if np.linalg.det(R2) < 0:
        R2 = -R2

    t_x = -U[:, 2]
    t = (t_x / np.linalg.norm(t_x)).reshape(-1, 1)
    
    return R1, R2, t
