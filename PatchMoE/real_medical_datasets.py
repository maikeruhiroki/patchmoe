#!/usr/bin/env python3
"""
実際の医用画像データセット用の実装
論文で使用される4つのデータセットに対応:
- DRIVE (Retinal Vessel)
- HV_NIR (NIR Vessel) 
- Kvasir-SEG (Polyp Segmentation)
- Synapse (Multi-organ)
"""
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple, Optional
import glob


class MedicalDatasetBase(Dataset):
    """医用画像データセットの基底クラス"""

    def __init__(self,
                 root_dir: str,
                 image_size: int = 512,
                 grid_h: int = 16,
                 grid_w: int = 16,
                 num_classes: int = 6,
                 dataset_id: int = 0):
        self.root_dir = root_dir
        self.image_size = image_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_classes = num_classes
        self.dataset_id = dataset_id

        # 画像とマスクのパスを取得
        self.image_paths, self.mask_paths = self._load_paths()

        # 前処理
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """サブクラスで実装"""
        raise NotImplementedError

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 画像とマスクを読み込み
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 前処理
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # パッチ情報を生成
        L = self.grid_h * self.grid_w
        dataset_ids = torch.full((L,), self.dataset_id, dtype=torch.long)
        image_ids = torch.full((L,), idx, dtype=torch.long)

        # クラスラベルを生成（マスクから）
        patch_classes = self._extract_patch_classes(mask)

        return image, dataset_ids, image_ids, mask, patch_classes

    def _extract_patch_classes(self, mask: torch.Tensor) -> torch.Tensor:
        """マスクから各パッチのクラスラベルを抽出"""
        L = self.grid_h * self.grid_w

        # マスクをグリッドサイズにリサイズ
        mask_resized = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(self.grid_h, self.grid_w),
            mode='nearest'
        ).squeeze(0).squeeze(0)

        patch_classes = torch.zeros(L, dtype=torch.long)
        for i in range(L):
            h_idx = i // self.grid_w
            w_idx = i % self.grid_w
            patch_value = mask_resized[h_idx, w_idx]
            # グレー値をクラスIDに変換
            patch_classes[i] = min(
                int(patch_value.item() / 50), self.num_classes - 1)

        return patch_classes


class DRIVEDataset(MedicalDatasetBase):
    """DRIVE (Retinal Vessel) データセット"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, dataset_id=0, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """DRIVEデータセットのパスを読み込み"""
        image_dir = os.path.join(self.root_dir, 'images')
        mask_dir = os.path.join(self.root_dir, 'masks')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            # データセットが存在しない場合はダミーデータを生成
            return self._generate_dummy_paths()

        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

        # 分割
        if self.split == 'train':
            image_paths = image_paths[:int(0.8 * len(image_paths))]
            mask_paths = mask_paths[:int(0.8 * len(mask_paths))]
        else:
            image_paths = image_paths[int(0.8 * len(image_paths)):]
            mask_paths = mask_paths[int(0.8 * len(mask_paths)):]

        return image_paths, mask_paths

    def _generate_dummy_paths(self) -> Tuple[List[str], List[str]]:
        """ダミーデータを生成"""
        print(
            f"DRIVE dataset not found at {self.root_dir}, generating dummy data...")
        return [], []


class HVNIRDataset(MedicalDatasetBase):
    """HV_NIR (NIR Vessel) データセット"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, dataset_id=1, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """HV_NIRデータセットのパスを読み込み"""
        image_dir = os.path.join(self.root_dir, 'images')
        mask_dir = os.path.join(self.root_dir, 'masks')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            return self._generate_dummy_paths()

        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

        # 分割
        if self.split == 'train':
            image_paths = image_paths[:int(0.8 * len(image_paths))]
            mask_paths = mask_paths[:int(0.8 * len(mask_paths))]
        else:
            image_paths = image_paths[int(0.8 * len(image_paths)):]
            mask_paths = mask_paths[int(0.8 * len(mask_paths)):]

        return image_paths, mask_paths

    def _generate_dummy_paths(self) -> Tuple[List[str], List[str]]:
        """ダミーデータを生成"""
        print(
            f"HV_NIR dataset not found at {self.root_dir}, generating dummy data...")
        return [], []


class KvasirSEGDataset(MedicalDatasetBase):
    """Kvasir-SEG (Polyp Segmentation) データセット"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, dataset_id=2, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """Kvasir-SEGデータセットのパスを読み込み"""
        image_dir = os.path.join(self.root_dir, 'images')
        mask_dir = os.path.join(self.root_dir, 'masks')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            return self._generate_dummy_paths()

        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

        # 分割
        if self.split == 'train':
            image_paths = image_paths[:int(0.8 * len(image_paths))]
            mask_paths = mask_paths[:int(0.8 * len(mask_paths))]
        else:
            image_paths = image_paths[int(0.8 * len(image_paths)):]
            mask_paths = mask_paths[int(0.8 * len(mask_paths)):]

        return image_paths, mask_paths

    def _generate_dummy_paths(self) -> Tuple[List[str], List[str]]:
        """ダミーデータを生成"""
        print(
            f"Kvasir-SEG dataset not found at {self.root_dir}, generating dummy data...")
        return [], []


class SynapseDataset(MedicalDatasetBase):
    """Synapse (Multi-organ) データセット"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, dataset_id=3, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """Synapseデータセットのパスを読み込み"""
        image_dir = os.path.join(self.root_dir, 'images')
        mask_dir = os.path.join(self.root_dir, 'masks')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            return self._generate_dummy_paths()

        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

        # 分割
        if self.split == 'train':
            image_paths = image_paths[:int(0.8 * len(image_paths))]
            mask_paths = mask_paths[:int(0.8 * len(mask_paths))]
        else:
            image_paths = image_paths[int(0.8 * len(image_paths)):]
            mask_paths = mask_paths[int(0.8 * len(mask_paths)):]

        return image_paths, mask_paths

    def _generate_dummy_paths(self) -> Tuple[List[str], List[str]]:
        """ダミーデータを生成"""
        print(
            f"Synapse dataset not found at {self.root_dir}, generating dummy data...")
        return [], []


class UnifiedMedicalDataset(Dataset):
    """複数の医用画像データセットを統合"""

    def __init__(self,
                 dataset_configs: List[Dict],
                 image_size: int = 512,
                 grid_h: int = 16,
                 grid_w: int = 16,
                 num_classes: int = 6):
        self.image_size = image_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_classes = num_classes

        # 各データセットを読み込み
        self.datasets = []
        self.dataset_offsets = [0]

        for config in dataset_configs:
            dataset_type = config['type']
            root_dir = config['root_dir']
            split = config.get('split', 'train')

            if dataset_type == 'drive':
                dataset = DRIVEDataset(root_dir, split,
                                       image_size=image_size, grid_h=grid_h,
                                       grid_w=grid_w, num_classes=num_classes)
            elif dataset_type == 'hv_nir':
                dataset = HVNIRDataset(root_dir, split,
                                       image_size=image_size, grid_h=grid_h,
                                       grid_w=grid_w, num_classes=num_classes)
            elif dataset_type == 'kvasir_seg':
                dataset = KvasirSEGDataset(root_dir, split,
                                           image_size=image_size, grid_h=grid_h,
                                           grid_w=grid_w, num_classes=num_classes)
            elif dataset_type == 'synapse':
                dataset = SynapseDataset(root_dir, split,
                                         image_size=image_size, grid_h=grid_h,
                                         grid_w=grid_w, num_classes=num_classes)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            self.datasets.append(dataset)
            self.dataset_offsets.append(
                self.dataset_offsets[-1] + len(dataset))

        # データセットマッピングを保存
        self.dataset_mapping = {
            'drive': 0,
            'hv_nir': 1,
            'kvasir_seg': 2,
            'synapse': 3
        }

    def __len__(self):
        return self.dataset_offsets[-1]

    def __getitem__(self, idx):
        # どのデータセットに属するかを判定
        dataset_idx = 0
        for i, offset in enumerate(self.dataset_offsets[1:], 1):
            if idx < offset:
                dataset_idx = i - 1
                break

        # データセット内のインデックスに変換
        local_idx = idx - self.dataset_offsets[dataset_idx]

        return self.datasets[dataset_idx][local_idx]


def build_unified_medical_loader(dataset_configs: List[Dict],
                                 batch_size: int = 4,
                                 num_workers: int = 2,
                                 shuffle: bool = True,
                                 **kwargs) -> DataLoader:
    """統合医用画像データセットのDataLoaderを構築"""

    dataset = UnifiedMedicalDataset(dataset_configs, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def create_dummy_medical_datasets(output_dir: str = "/workspace/medical_datasets"):
    """ダミーの医用画像データセットを生成（テスト用）"""
    import numpy as np
    from PIL import Image, ImageDraw

    datasets = [
        {'name': 'drive', 'description': 'Retinal Vessel'},
        {'name': 'hv_nir', 'description': 'NIR Vessel'},
        {'name': 'kvasir_seg', 'description': 'Polyp Segmentation'},
        {'name': 'synapse', 'description': 'Multi-organ'}
    ]

    for dataset in datasets:
        dataset_dir = os.path.join(output_dir, dataset['name'])
        images_dir = os.path.join(dataset_dir, 'images')
        masks_dir = os.path.join(dataset_dir, 'masks')

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # 各データセット用のダミー画像を生成
        for i in range(20):  # 20枚の画像
            # 画像生成
            img = Image.new('RGB', (512, 512), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)

            # データセット固有の特徴を追加
            if dataset['name'] == 'drive':
                # 網膜血管のような構造
                for _ in range(10):
                    x1, y1 = np.random.randint(0, 512, 2)
                    x2, y2 = x1 + \
                        np.random.randint(10, 50), y1 + \
                        np.random.randint(10, 50)
                    draw.line([x1, y1, x2, y2], fill=(255, 255, 255), width=2)
            elif dataset['name'] == 'hv_nir':
                # NIR血管のような構造
                for _ in range(8):
                    x1, y1 = np.random.randint(0, 512, 2)
                    x2, y2 = x1 + \
                        np.random.randint(15, 60), y1 + \
                        np.random.randint(15, 60)
                    draw.line([x1, y1, x2, y2], fill=(200, 200, 200), width=3)
            elif dataset['name'] == 'kvasir_seg':
                # ポリープのような構造
                for _ in range(5):
                    x, y = np.random.randint(100, 400, 2)
                    r = np.random.randint(20, 50)
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255))
            elif dataset['name'] == 'synapse':
                # 多臓器のような構造
                for _ in range(6):
                    x, y = np.random.randint(50, 450, 2)
                    w, h = np.random.randint(30, 80, 2)
                    draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 255))

            # マスク生成
            mask = Image.new('L', (512, 512), color=0)
            mask_draw = ImageDraw.Draw(mask)

            # 対応するマスクを生成
            if dataset['name'] in ['drive', 'hv_nir']:
                # 血管マスク
                for _ in range(5):
                    x1, y1 = np.random.randint(0, 512, 2)
                    x2, y2 = x1 + \
                        np.random.randint(10, 50), y1 + \
                        np.random.randint(10, 50)
                    mask_draw.line([x1, y1, x2, y2], fill=255, width=2)
            else:
                # 臓器/ポリープマスク
                for _ in range(3):
                    x, y = np.random.randint(50, 450, 2)
                    r = np.random.randint(20, 50)
                    mask_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

            # 保存
            img.save(os.path.join(images_dir, f'{i:03d}.png'))
            mask.save(os.path.join(masks_dir, f'{i:03d}.png'))

        print(
            f"Generated dummy dataset: {dataset['name']} ({dataset['description']})")

    print(f"All dummy datasets generated in: {output_dir}")


if __name__ == "__main__":
    # ダミーデータセットを生成
    create_dummy_medical_datasets()

    # 統合データセットのテスト
    dataset_configs = [
        {'type': 'drive', 'root_dir': '/workspace/medical_datasets/drive', 'split': 'train'},
        {'type': 'hv_nir', 'root_dir': '/workspace/medical_datasets/hv_nir',
            'split': 'train'},
        {'type': 'kvasir_seg',
            'root_dir': '/workspace/medical_datasets/kvasir_seg', 'split': 'train'},
        {'type': 'synapse', 'root_dir': '/workspace/medical_datasets/synapse',
            'split': 'train'},
    ]

    loader = build_unified_medical_loader(dataset_configs, batch_size=2)

    print(f"Unified dataset created with {len(loader.dataset)} samples")
    print(f"Dataset mapping: {loader.dataset.dataset_mapping}")

    # テストバッチ
    for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(loader):
        print(f"Batch {batch_idx}: images={images.shape}, dataset_ids={dataset_ids.shape}, "
              f"image_ids={image_ids.shape}, masks={masks.shape}, classes={classes.shape}")
        if batch_idx >= 2:
            break
