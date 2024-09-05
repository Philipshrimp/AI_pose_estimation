import os
import time
import cv2
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image,load_and_mask_image, rbd
from lightglue import viz2d


# SuperPoint+LightGlue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
# device = torch.device("cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

ttt = time.time()
dummy_input = torch.randn(1, 3, 224, 224).to(device)
_ = extractor.extract(dummy_input)
_ = matcher({'image0': _, 'image1': _})
print(f"Time taken: {time.time() - ttt} seconds")

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
dst_index = 4
root_path = '../data/raw'

original_img0 = load_image(root_path + '/1.bmp')
original_img1 = load_image(root_path + f'/{dst_index}.bmp')

image0 = load_and_mask_image(root_path + '/1.bmp').to(device)
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


# [TODO] 
# Implement the function to calculate the transformation matrix

# NumPy 배열로 변환
src_pts = m_kpts0.cpu().numpy()
dst_pts = m_kpts1.cpu().numpy()

# # 기본 카메라 매트릭스 (실제 카메라 매개변수로 대체해야 함)
K = np.array([[1.71326283e+03, 0.00000000e+00, 1.26716548e+03],
              [0.00000000e+00, 1.71201684e+03, 1.06652107e+03],
              [0, 0, 1]])

d = [1.70168395e+01, -4.91002680e+00, -1.75329962e-03, -4.80750653e-05,  -1.21918723e+00, 1.76062725e+01,  2.02562080e+00, -4.35890100e+00]

# # Essential 매트릭스 계산
E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# # Essential 매트릭스에서 R과 t 복구
_, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

print("회전 행렬:")
print(R)
print("\n평행이동 벡터:")
print(t)

# # 변환 행렬 계산
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t.reshape(3)

print("\n변환 행렬:")
print(T)