import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
from typing import Dict, List, Tuple
import glob
import cv2


class KaggleMedicalDatasetBase(Dataset):
    """Kaggle医用画像データセットの基底クラス"""

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
            # グレー値をクラスIDに変換（血管/ポリープ/臓器 = 1, 背景 = 0）
            if patch_value.item() > 128:  # 閾値で二値化
                patch_classes[i] = 1
            else:
                patch_classes[i] = 0

        return patch_classes


class DRIVEKaggleDataset(KaggleMedicalDatasetBase):
    """DRIVE Kaggleデータセット"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """DRIVEデータセットのパスを読み込み"""
        if self.split == 'train':
            images_dir = os.path.join(
                self.root_dir, 'DRIVE', 'training', 'images')
            masks_dir = os.path.join(
                self.root_dir, 'DRIVE', 'training', '1st_manual')
        else:
            images_dir = os.path.join(self.root_dir, 'DRIVE', 'test', 'images')
            masks_dir = os.path.join(self.root_dir, 'DRIVE', 'test', 'mask')

        image_paths = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.gif')))

        if not image_paths or not mask_paths:
            print(f"Warning: No images or masks found for DRIVE {self.split}")
            return [], []

        # 画像とマスクのファイル名が一致するようにフィルタリング
        filtered_image_paths = []
        filtered_mask_paths = []
        for img_path in image_paths:
            img_name_base = os.path.basename(img_path).split('_')[0]  # '21'
            for mask_path in mask_paths:
                mask_name_base = os.path.basename(
                    mask_path).split('_')[0]  # '21'
                if img_name_base == mask_name_base:
                    filtered_image_paths.append(img_path)
                    filtered_mask_paths.append(mask_path)
                    break

        return filtered_image_paths, filtered_mask_paths


class KvasirSEGKaggleDataset(KaggleMedicalDatasetBase):
    """Kvasir-SEG Kaggleデータセット"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """Kvasir-SEGデータセットのパスを読み込み"""
        images_dir = os.path.join(
            self.root_dir, 'Kvasir-SEG', 'Kvasir-SEG', 'images')
        masks_dir = os.path.join(
            self.root_dir, 'Kvasir-SEG', 'Kvasir-SEG', 'masks')

        image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.jpg')))

        if not image_paths or not mask_paths:
            print(f"Warning: No images or masks found for Kvasir-SEG")
            return [], []

        # 画像とマスクのファイル名が一致するようにフィルタリング
        filtered_image_paths = []
        filtered_mask_paths = []
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(masks_dir, img_name)
            if os.path.exists(mask_path):
                filtered_image_paths.append(img_path)
                filtered_mask_paths.append(mask_path)

        # 訓練/検証分割
        total_samples = len(filtered_image_paths)
        train_size = int(0.8 * total_samples)

        if self.split == 'train':
            return filtered_image_paths[:train_size], filtered_mask_paths[:train_size]
        else:
            return filtered_image_paths[train_size:], filtered_mask_paths[train_size:]


class SynapseKaggleDataset(KaggleMedicalDatasetBase):
    """Synapse Kaggleデータセット"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """Synapseデータセットのパスを読み込み"""
        train_dir = os.path.join(self.root_dir, 'Synapse', 'train_npz')

        npz_files = sorted(glob.glob(os.path.join(train_dir, '*.npz')))

        if not npz_files:
            print(f"Warning: No npz files found for Synapse")
            return [], []

        # 訓練/検証分割
        total_samples = len(npz_files)
        train_size = int(0.8 * total_samples)

        if self.split == 'train':
            selected_files = npz_files[:train_size]
        else:
            selected_files = npz_files[train_size:]

        return selected_files, selected_files  # 同じファイルから画像とマスクを読み込む

    def __getitem__(self, idx):
        # npzファイルから画像とマスクを読み込み
        npz_path = self.image_paths[idx]
        data = np.load(npz_path)

        # 画像とマスクを取得
        image = data['image']  # [H, W, C]
        mask = data['label']   # [H, W]

        # 画像のチャンネル数を確認・調整
        if len(image.shape) == 2:  # [H, W] -> [H, W, 1]
            image = np.expand_dims(image, axis=2)
        if image.shape[2] == 1:  # [H, W, 1] -> [H, W, 3] (グレースケールをRGBに変換)
            image = np.repeat(image, 3, axis=2)

        # PIL Imageに変換
        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

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


class RetinaBloodVesselKaggleDataset(KaggleMedicalDatasetBase):
    """Retina Blood Vessel Kaggleデータセット（HV_NIRの代替）"""

    def __init__(self, root_dir: str, split: str = 'train', **kwargs):
        self.split = split
        super().__init__(root_dir, **kwargs)

    def _load_paths(self) -> Tuple[List[str], List[str]]:
        """Retina Blood Vesselデータセットのパスを読み込み"""
        if self.split == 'train':
            images_dir = os.path.join(self.root_dir, 'Data', 'train', 'image')
            masks_dir = os.path.join(self.root_dir, 'Data', 'train', 'mask')
        else:
            images_dir = os.path.join(self.root_dir, 'Data', 'test', 'image')
            masks_dir = os.path.join(self.root_dir, 'Data', 'test', 'mask')

        image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.png')))

        if not image_paths or not mask_paths:
            print(
                f"Warning: No images or masks found for Retina Blood Vessel {self.split}")
            return [], []

        # 画像とマスクのファイル名が一致するようにフィルタリング
        filtered_image_paths = []
        filtered_mask_paths = []
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(masks_dir, img_name)
            if os.path.exists(mask_path):
                filtered_image_paths.append(img_path)
                filtered_mask_paths.append(mask_path)

        return filtered_image_paths, filtered_mask_paths


class UnifiedKaggleMedicalDataset(Dataset):
    """複数のKaggle医用画像データセットを統合"""

    def __init__(self,
                 dataset_configs: List[Dict],
                 image_size: int = 512,
                 grid_h: int = 16,
                 grid_w: int = 16,
                 num_classes: int = 2):  # 二値分類（血管/ポリープ/臓器 vs 背景）
        self.image_size = image_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_classes = num_classes

        # 各データセットを読み込み
        self.datasets = []
        self.dataset_offsets = [0]
        self.dataset_mapping = {}

        for i, config in enumerate(dataset_configs):
            dataset_type = config['type']
            root_dir = config['root_dir']
            split = config.get('split', 'train')

            if dataset_type == 'drive':
                dataset = DRIVEKaggleDataset(root_dir, split,
                                             image_size=image_size, grid_h=grid_h,
                                             grid_w=grid_w, num_classes=num_classes,
                                             dataset_id=i)
            elif dataset_type == 'kvasir_seg':
                dataset = KvasirSEGKaggleDataset(root_dir, split,
                                                 image_size=image_size, grid_h=grid_h,
                                                 grid_w=grid_w, num_classes=num_classes,
                                                 dataset_id=i)
            elif dataset_type == 'synapse':
                dataset = SynapseKaggleDataset(root_dir, split,
                                               image_size=image_size, grid_h=grid_h,
                                               grid_w=grid_w, num_classes=num_classes,
                                               dataset_id=i)
            elif dataset_type == 'retina_blood_vessel':
                dataset = RetinaBloodVesselKaggleDataset(root_dir, split,
                                                         image_size=image_size, grid_h=grid_h,
                                                         grid_w=grid_w, num_classes=num_classes,
                                                         dataset_id=i)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            self.datasets.append(dataset)
            self.dataset_offsets.append(
                self.dataset_offsets[-1] + len(dataset))
            self.dataset_mapping[dataset_type] = i

        print(f"Unified Kaggle dataset created with {len(self)} samples.")
        print(f"Dataset mapping: {self.dataset_mapping}")

    def __len__(self):
        return self.dataset_offsets[-1]

    def __getitem__(self, idx):
        # どのデータセットに属するかを判定
        dataset_idx = 0
        for i in range(1, len(self.dataset_offsets)):
            if idx < self.dataset_offsets[i]:
                dataset_idx = i - 1
                break

        # データセット内のインデックスに変換
        local_idx = idx - self.dataset_offsets[dataset_idx]

        return self.datasets[dataset_idx][local_idx]


def build_kaggle_medical_loader(dataset_configs: List[Dict],
                                batch_size: int = 4,
                                num_workers: int = 2,
                                shuffle: bool = True,
                                **kwargs) -> DataLoader:
    """統合Kaggle医用画像データセットのDataLoaderを構築"""

    dataset = UnifiedKaggleMedicalDataset(dataset_configs, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == '__main__':
    # 統合データセットのテスト
    dataset_configs = [
        {'type': 'drive', 'root_dir': '/workspace/real_medical_datasets_kaggle',
            'split': 'train'},
        {'type': 'kvasir_seg',
            'root_dir': '/workspace/real_medical_datasets_kaggle', 'split': 'train'},
        {'type': 'synapse', 'root_dir': '/workspace/real_medical_datasets_kaggle',
            'split': 'train'},
        {'type': 'retina_blood_vessel',
            'root_dir': '/workspace/real_medical_datasets_kaggle', 'split': 'train'},
    ]

    loader = build_kaggle_medical_loader(
        dataset_configs, batch_size=2, num_classes=2, num_workers=0)

    print(f"Unified Kaggle dataset created with {len(loader.dataset)} samples")
    print(f"Dataset mapping: {loader.dataset.dataset_mapping}")

    # テストバッチ
    for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Dataset IDs shape: {dataset_ids.shape}")
        print(f"  Image IDs shape: {image_ids.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Classes shape: {classes.shape}")
        print(f"  Dataset IDs unique: {torch.unique(dataset_ids)}")
        if batch_idx >= 2:
            break
