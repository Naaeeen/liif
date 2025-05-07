# liif/datasets/single_npy.py
import numpy as np
import torch
from torch.utils.data import Dataset
from .datasets import register          # <- 必须从当前包引入注册器

@register('single-npy')                 # 名称必须与 YAML 完全一致
class SingleNpyDataset(Dataset):
    """
    Dataset that loads a single image (img.npy) and its coordinate grid (coord.npy)
    and returns flattened (coord, cell, gt) so that LIIF wrappers can sample from it.
    """

    def __init__(self, img_path, coord_path, cache=None):
        # 1. load image and coord
        img   = np.load(img_path)   # H×W×3, float32 in [0,1]
        coord = np.load(coord_path) # H×W×2, float32 in [0,1]

        h, w = img.shape[:2]
        cell = np.stack([
            np.full_like(coord[..., 0], 1./w),
            np.full_like(coord[..., 1], 1./h)
        ], axis=-1)

        # flatten so that wrapper can index randomly
        self.data = dict(
            inp  = img.transpose(2,0,1).astype(np.float32),    # 3×H×W
            coord= coord.reshape(-1, 2).astype(np.float32),    # (N,2)
            cell = cell.reshape(-1, 2).astype(np.float32),     # (N,2)
            gt   = img.reshape(-1, 3).astype(np.float32)       # (N,3)
        )

    def __len__(self):
        return 1                    # one sample = the whole image

    def __getitem__(self, idx):
        return {k: torch.from_numpy(v) for k, v in self.data.items()}
