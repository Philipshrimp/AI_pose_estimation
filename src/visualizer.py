import torch

from lightglue import viz2d
from lightglue.utils import load_image


def visualize(root_path, src_index, dst_index, m_kpts0, m_kpts1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_img0 = load_image(root_path + f'/{src_index}.bmp').to(device)
    original_img1 = load_image(root_path + f'/{dst_index}.bmp').to(device)

    axes = viz2d.plot_images([original_img0, original_img1])

    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.save_plot(f'results/matches_{dst_index}.png')
