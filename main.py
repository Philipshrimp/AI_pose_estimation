import numpy as np

def match_features(desc1, desc2, threshold=0.75):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist < threshold:
            matches.append((i, min_dist_idx, min_dist))
    
    matches = sorted(matches, key=lambda x: x[2])
    return matches

def prosac(matches, keypoints1, keypoints2, max_trials=1000, threshold=1.0):
    best_inliers = []
    best_model = None

    for trial in range(max_trials):
        # 1. Progressively select matches based on their sorted order
        subset = matches[:min(trial+1, len(matches))]
        
        # 2. Randomly sample a subset of the matches
        sampled_match = subset[np.random.randint(0, len(subset))]

        # 3. Compute a transformation model using the sampled matches (e.g., homography)
        idx1, idx2, _ = sampled_match
        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]

        # Here, we're simulating a simple model. Replace this with an actual model estimation.
        model = kp2 - kp1  # Dummy model: simple translation (replace with actual transformation)

        # 4. Evaluate the model by counting the number of inliers
        inliers = []
        for idx1, idx2, _ in matches:
            kp1 = keypoints1[idx1]
            kp2 = keypoints2[idx2]
            projected_kp1 = kp1 + model  # Apply the transformation (dummy model)

            if np.linalg.norm(projected_kp1 - kp2) < threshold:
                inliers.append((idx1, idx2))

        # 5. Update the best model if the current one is better
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model

    return best_inliers, best_model

# 예시 사용법
desc1 = np.random.rand(500, 256)  # 이미지 1의 디스크립터 (예시)
desc2 = np.random.rand(500, 256)  # 이미지 2의 디스크립터 (예시)

keypoints1 = np.random.rand(500, 2) * 100  # 이미지 1의 키포인트 (예시)
keypoints2 = keypoints1 + np.random.rand(500, 2) * 5  # 이미지 2의 키포인트 (약간의 변형 추가)

matches = match_features(desc1, desc2)
inliers, model = prosac(matches, keypoints1, keypoints2)

print(f"Found {len(inliers)} inliers using PROSAC")
