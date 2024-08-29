import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

from lightglue import LightGlue, SuperPoint
from lightglue import match_pair
from lightglue.utils import load_and_mask_image, rbd
from lightglue import viz2d


# SuperPoint+LightGlue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_and_mask_image('../data/elp_avg/average0.bmp').to(device)
image1 = load_and_mask_image('../data/elp_avg/average1.bmp').to(device)

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})

feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

viz2d.save_plot('results/matches.png')
