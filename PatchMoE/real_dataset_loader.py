#!/usr/bin/env python3
"""
実際の医用画像データセット用の改良されたデータローダー
Kaggle等から取得した実際のデータセットに対応
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path
import cv2


class RealMedicalDataset(Dataset):
    """実際の医用画像データセット用の改良されたデータセットクラス"""

    def __init__(self,
                 root_dir: str,
                 dataset_name: str,
                 image_size: int = 512,
                 grid_h: int = 16,
                 grid_w: int = 16,
                 num_classes: int = 6,
                 dataset_id: int = 0,
                 split: str = 'train',
                 split_ratio: float = 0.8,
                 augmentation: bool = True):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_classes = num_classes
        self.dataset_id = dataset_id
        self.split = split
        self.split_ratio = split_ratio
        self.augmentation = augmentation

        # データセット情報を読み込み
        self.dataset_info = self._load_dataset_info()

        # 画像とマスクのパスを取得
        self.image_paths, self.mask_paths = self._load_paths()

        # 前処理パイプライン
        self.transform = self._get_image_transform()
        self.mask_transform = self._get_mask_transform()

        print(
            f"Loaded {dataset_name} dataset: {len(self.image_paths)} samples")

    def _load_dataset_info(self) -> Dict:
        """データセット情報を読み込み"""
        info_file = self.root_dir / self.dataset_name / "dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                return json.load(f)
        else:
            return {"name": self.dataset_name, "status": "unknown"}

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """画像とマスクのパスを読み込み"""
        dataset_dir = self.root_dir / self.dataset_name
        images_dir = dataset_dir / "images"
        masks_dir = dataset_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"Dataset directories not found: {images_dir}, {masks_dir}")

        # 画像ファイルを取得
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(images_dir / ext)))

        # マスクファイルを取得
        mask_paths = []
        for ext in image_extensions:
            mask_paths.extend(glob.glob(str(masks_dir / ext)))

        # パスをソート
        image_paths = sorted(image_paths)
        mask_paths = sorted(mask_paths)

        # 画像とマスクの対応を確認
        if len(image_paths) != len(mask_paths):
            print(
                f"Warning: Mismatch in image/mask count: {len(image_paths)} images, {len(mask_paths)} masks")
            # 最小数に合わせる
            min_count = min(len(image_paths), len(mask_paths))
            image_paths = image_paths[:min_count]
            mask_paths = mask_paths[:min_count]

        # 分割
        total_count = len(image_paths)
        train_count = int(total_count * self.split_ratio)

        if self.split == 'train':
            image_paths = image_paths[:train_count]
            mask_paths = mask_paths[:train_count]
        else:
            image_paths = image_paths[train_count:]
            mask_paths = mask_paths[train_count:]

        return image_paths, mask_paths

    def _get_image_transform(self):
        """画像用の前処理パイプライン"""
        transforms = [
            T.Resize((self.image_size, self.image_size)),
        ]

        if self.augmentation and self.split == 'train':
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1),
            ])

        transforms.append(T.ToTensor())

        return T.Compose(transforms)

    def _get_mask_transform(self):
        """マスク用の前処理パイプライン"""
        return T.Compose([
            T.Resize((self.image_size, self.image_size),
                     interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """マスクの前処理"""
        # グレースケールに変換
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # 二値化（必要に応じて）
        if mask.max() > 1:
            mask = (mask > 128).astype(np.uint8) * 255

        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 画像とマスクを読み込み
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 画像読み込み
        image = Image.open(image_path).convert('RGB')

        # マスク読み込みと前処理
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        mask_array = self._preprocess_mask(mask_array)
        mask = Image.fromarray(mask_array)

        # 前処理適用
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

            # グレー値をクラスIDに変換（より細かい分類）
            if patch_value < 50:
                patch_classes[i] = 0  # 背景
            elif patch_value < 100:
                patch_classes[i] = 1  # 低密度血管
            elif patch_value < 150:
                patch_classes[i] = 2  # 中密度血管
            elif patch_value < 200:
                patch_classes[i] = 3  # 高密度血管
            else:
                patch_classes[i] = 4  # 血管中心

        return patch_classes


class UnifiedRealMedicalDataset(Dataset):
    """複数の実際の医用画像データセットを統合"""

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
            dataset_name = config['name']
            root_dir = config['root_dir']
            split = config.get('split', 'train')
            dataset_id = config.get('dataset_id', 0)
            augmentation = config.get('augmentation', True)

            dataset = RealMedicalDataset(
                root_dir=root_dir,
                dataset_name=dataset_name,
                image_size=image_size,
                grid_h=grid_h,
                grid_w=grid_w,
                num_classes=num_classes,
                dataset_id=dataset_id,
                split=split,
                augmentation=augmentation
            )

            self.datasets.append(dataset)
            self.dataset_offsets.append(
                self.dataset_offsets[-1] + len(dataset))

        # データセットマッピングを保存
        self.dataset_mapping = {config['name']: config.get('dataset_id', i)
                                for i, config in enumerate(dataset_configs)}

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


def build_real_medical_loader(dataset_configs: List[Dict],
                              batch_size: int = 4,
                              num_workers: int = 2,
                              shuffle: bool = True,
                              **kwargs) -> DataLoader:
    """実際の医用画像データセットのDataLoaderを構築"""

    dataset = UnifiedRealMedicalDataset(dataset_configs, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def create_sample_configs():
    """サンプル設定を作成"""
    return [
        {
            'name': 'drive',
            'root_dir': '/workspace/real_medical_datasets',
            'split': 'train',
            'dataset_id': 0,
            'augmentation': True
        },
        {
            'name': 'stare',
            'root_dir': '/workspace/real_medical_datasets',
            'split': 'train',
            'dataset_id': 1,
            'augmentation': True
        },
        {
            'name': 'chase',
            'root_dir': '/workspace/real_medical_datasets',
            'split': 'train',
            'dataset_id': 2,
            'augmentation': True
        }
    ]


if __name__ == "__main__":
    # サンプル設定でテスト
    configs = create_sample_configs()

    try:
        loader = build_real_medical_loader(configs, batch_size=2)
        print(f"Real dataset created with {len(loader.dataset)} samples")
        print(f"Dataset mapping: {loader.dataset.dataset_mapping}")

        # テストバッチ
        for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(loader):
            print(f"Batch {batch_idx}: images={images.shape}, dataset_ids={dataset_ids.shape}, "
                  f"image_ids={image_ids.shape}, masks={masks.shape}, classes={classes.shape}")
            if batch_idx >= 2:
                break

    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please run download_real_datasets.py first to download the datasets.")
