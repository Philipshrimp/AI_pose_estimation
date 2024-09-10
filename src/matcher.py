import cv2
import numpy as np
import time
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_and_mask_image, rbd


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
    # [TODO] REPLACE OpenCV!!
    mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 2.0)[1]
    mask = mask.ravel().astype(bool)

    new_src = src_pts[mask]
    new_dst = dst_pts[mask]

    return new_src, new_dst
