import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class PatchDataset(Dataset):
    """
    簡易データセット: 画像をランダム生成またはフォルダから読み込み、
    grid_h×grid_w のパッチインデックスで dataset_id/image_id を生成。
    本番では実画像読み込みに差し替え。
    """

    def __init__(self, length: int, in_ch: int, H: int, W: int, grid_h: int, grid_w: int, num_datasets: int = 4, image_id_start: int = 0):
        self.length = length
        self.in_ch = in_ch
        self.H = H
        self.W = W
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_datasets = num_datasets
        self.image_id_start = image_id_start

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        img = torch.rand(self.in_ch, self.H, self.W)
        L = self.grid_h * self.grid_w
        dataset_id = torch.randint(0, max(1, self.num_datasets), (L,))
        image_id = torch.full(
            (L,), self.image_id_start + idx, dtype=torch.long)
        # 目標（ダミー）
        mask = (torch.rand(1, self.H, self.W) > 0.5).float()
        cls = torch.randint(0, 6, (L,))
        return img, dataset_id, image_id, mask, cls


def build_loader(batch_size: int, in_ch: int, H: int, W: int, grid_h: int, grid_w: int, num_workers: int = 2, length: int = 16):
    ds = PatchDataset(length=length, in_ch=in_ch, H=H,
                      W=W, grid_h=grid_h, grid_w=grid_w)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
