import numpy as np


def compute_fundamental_matrix(src_pts, dst_pts):
    src_pts_h = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
    dst_pts_h = np.hstack((dst_pts, np.ones((dst_pts.shape[0], 1))))

    A = np.array([[src_pts_h[i][0] * dst_pts_h[i][0], 
                    src_pts_h[i][0] * dst_pts_h[i][1], 
                    src_pts_h[i][0], 
                    src_pts_h[i][1] * dst_pts_h[i][0], 
                    src_pts_h[i][1] * dst_pts_h[i][1], 
                    src_pts_h[i][1], 
                    dst_pts_h[i][0], 
                    dst_pts_h[i][1], 
                    1] for i in range(len(src_pts_h))])

    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

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
