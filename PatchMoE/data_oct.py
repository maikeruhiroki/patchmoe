import os
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class OCTFolderDataset(Dataset):
    """
    画像フォルダからOCT画像を読み込み、gridに合わせて dataset_id / image_id を生成。
    ディレクトリ構成例:
      root/
        datasetA/*.png
        datasetB/*.png
    データセット名で dataset_id を割当。
    """

    def __init__(self, root: str, grid_h: int, grid_w: int, image_size: int = 256):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.image_size = image_size
        self.items = []
        subdirs = sorted([d for d in glob.glob(
            os.path.join(root, '*')) if os.path.isdir(d)])
        self.name_to_id = {}
        self.id_to_name = {}
        for didx, d in enumerate(subdirs):
            name = os.path.basename(d.rstrip('/'))
            self.name_to_id[name] = didx
            self.id_to_name[didx] = name
            for p in sorted(glob.glob(os.path.join(d, '*.png')) + glob.glob(os.path.join(d, '*.jpg'))):
                self.items.append((p, didx))
        self.tx = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, dataset_id_val = self.items[idx]
        img = Image.open(path).convert('RGB')
        img = self.tx(img)
        L = self.grid_h * self.grid_w
        dataset_id = torch.full((L,), dataset_id_val, dtype=torch.long)
        image_id = torch.full((L,), idx, dtype=torch.long)
        mask = (torch.rand(1, self.image_size, self.image_size) > 0.5).float()
        cls = torch.randint(0, 6, (L,))
        return img, dataset_id, image_id, mask, cls


def build_oct_loader(root: str, batch_size: int, grid_h: int, grid_w: int, num_workers: int = 2):
    ds = OCTFolderDataset(root=root, grid_h=grid_h, grid_w=grid_w)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
