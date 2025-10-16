import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import random


class MedicalSegmentationDataset(Dataset):
    """
    医学画像セグメンテーション用のより現実的なデータセット
    論文で使用されるような複雑な構造を持つ画像を生成
    """

    def __init__(self, length: int, image_size: int = 512, num_classes: int = 6,
                 num_datasets: int = 4, grid_h: int = 16, grid_w: int = 16):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_datasets = num_datasets
        self.grid_h = grid_h
        self.grid_w = grid_w

    def __len__(self):
        return self.length

    def generate_medical_image(self, idx: int) -> Image.Image:
        """医学画像のような複雑な構造を持つ画像を生成"""
        img = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 背景ノイズ（医学画像の特徴的なノイズ）
        noise = np.random.normal(0, 15, (self.image_size, self.image_size, 3))
        img_array = np.array(img).astype(np.float32) + noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        # 複数の解剖学的構造を描画
        structures = self._generate_anatomical_structures(idx)

        for structure in structures:
            if structure['type'] == 'organ':
                self._draw_organ(draw, structure)
            elif structure['type'] == 'vessel':
                self._draw_vessel(draw, structure)
            elif structure['type'] == 'lesion':
                self._draw_lesion(draw, structure)
            elif structure['type'] == 'tissue':
                self._draw_tissue(draw, structure)

        # 医学画像らしいフィルタリング
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        return img

    def _generate_anatomical_structures(self, idx: int) -> list:
        """解剖学的構造を生成"""
        random.seed(idx)
        structures = []

        # 臓器（大きな楕円形）
        for _ in range(random.randint(1, 3)):
            structures.append({
                'type': 'organ',
                'center': (random.randint(100, 400), random.randint(100, 400)),
                'size': (random.randint(80, 150), random.randint(60, 120)),
                'color': (random.randint(80, 150), random.randint(80, 150), random.randint(80, 150)),
                'class': 1
            })

        # 血管（細長い線）
        for _ in range(random.randint(2, 5)):
            structures.append({
                'type': 'vessel',
                'start': (random.randint(50, 450), random.randint(50, 450)),
                'end': (random.randint(50, 450), random.randint(50, 450)),
                'width': random.randint(3, 8),
                'color': (random.randint(120, 200), random.randint(100, 180), random.randint(100, 180)),
                'class': 2
            })

        # 病変（不規則な形状）
        for _ in range(random.randint(1, 4)):
            structures.append({
                'type': 'lesion',
                'center': (random.randint(100, 400), random.randint(100, 400)),
                'radius': random.randint(20, 60),
                'irregularity': random.uniform(0.3, 0.8),
                'color': (random.randint(150, 255), random.randint(100, 200), random.randint(100, 200)),
                'class': 3
            })

        # 組織（複雑なパターン）
        for _ in range(random.randint(2, 4)):
            structures.append({
                'type': 'tissue',
                'center': (random.randint(100, 400), random.randint(100, 400)),
                'size': (random.randint(60, 120), random.randint(60, 120)),
                'pattern': random.choice(['striped', 'spotted', 'gradient']),
                'color': (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180)),
                'class': 4
            })

        return structures

    def _draw_organ(self, draw: ImageDraw.Draw, structure: dict):
        """臓器を描画"""
        center = structure['center']
        size = structure['size']
        color = structure['color']

        # 楕円形の臓器
        bbox = [center[0] - size[0]//2, center[1] - size[1]//2,
                center[0] + size[0]//2, center[1] + size[1]//2]
        draw.ellipse(bbox, fill=color, outline=None)

    def _draw_vessel(self, draw: ImageDraw.Draw, structure: dict):
        """血管を描画"""
        start = structure['start']
        end = structure['end']
        width = structure['width']
        color = structure['color']

        # 曲線的な血管
        mid_x = (start[0] + end[0]) // 2 + random.randint(-50, 50)
        mid_y = (start[1] + end[1]) // 2 + random.randint(-50, 50)

        points = [start, (mid_x, mid_y), end]
        draw.line(points, fill=color, width=width)

    def _draw_lesion(self, draw: ImageDraw.Draw, structure: dict):
        """病変を描画"""
        center = structure['center']
        radius = structure['radius']
        irregularity = structure['irregularity']
        color = structure['color']

        # 不規則な形状の病変
        points = []
        num_points = 8
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            r = radius * (1 + random.uniform(-irregularity, irregularity))
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            points.append((x, y))

        draw.polygon(points, fill=color, outline=None)

    def _draw_tissue(self, draw: ImageDraw.Draw, structure: dict):
        """組織を描画"""
        center = structure['center']
        size = structure['size']
        pattern = structure['pattern']
        color = structure['color']

        bbox = [center[0] - size[0]//2, center[1] - size[1]//2,
                center[0] + size[0]//2, center[1] + size[1]//2]

        if pattern == 'striped':
            # 縞模様
            for i in range(0, size[0], 8):
                if i % 16 < 8:
                    draw.rectangle([bbox[0] + i, bbox[1], bbox[0] + i + 8, bbox[3]],
                                   fill=color, outline=None)
        elif pattern == 'spotted':
            # 斑点模様
            for _ in range(10):
                x = random.randint(bbox[0], bbox[2])
                y = random.randint(bbox[1], bbox[3])
                draw.ellipse([x-3, y-3, x+3, y+3], fill=color, outline=None)
        else:  # gradient
            # グラデーション
            draw.rectangle(bbox, fill=color, outline=None)

    def generate_segmentation_mask(self, structures: list) -> Image.Image:
        """セグメンテーションマスクを生成"""
        mask = Image.new('L', (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(mask)

        for structure in structures:
            class_id = structure['class']
            color = class_id * 50  # クラスIDに応じたグレー値

            if structure['type'] == 'organ':
                center = structure['center']
                size = structure['size']
                bbox = [center[0] - size[0]//2, center[1] - size[1]//2,
                        center[0] + size[0]//2, center[1] + size[1]//2]
                draw.ellipse(bbox, fill=color, outline=None)

            elif structure['type'] == 'vessel':
                start = structure['start']
                end = structure['end']
                width = structure['width']
                mid_x = (start[0] + end[0]) // 2 + random.randint(-50, 50)
                mid_y = (start[1] + end[1]) // 2 + random.randint(-50, 50)
                points = [start, (mid_x, mid_y), end]
                draw.line(points, fill=color, width=width)

            elif structure['type'] == 'lesion':
                center = structure['center']
                radius = structure['radius']
                irregularity = structure['irregularity']
                points = []
                num_points = 8
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    r = radius * \
                        (1 + random.uniform(-irregularity, irregularity))
                    x = center[0] + r * np.cos(angle)
                    y = center[1] + r * np.sin(angle)
                    points.append((x, y))
                draw.polygon(points, fill=color, outline=None)

            elif structure['type'] == 'tissue':
                center = structure['center']
                size = structure['size']
                bbox = [center[0] - size[0]//2, center[1] - size[1]//2,
                        center[0] + size[0]//2, center[1] + size[1]//2]
                draw.rectangle(bbox, fill=color, outline=None)

        return mask

    def __getitem__(self, idx):
        # 医学画像を生成
        img = self.generate_medical_image(idx)

        # 対応する構造を再生成（同じseedで）
        random.seed(idx)
        structures = self._generate_anatomical_structures(idx)

        # セグメンテーションマスクを生成
        mask = self.generate_segmentation_mask(structures)

        # テンソルに変換
        img_tensor = T.ToTensor()(img)
        mask_tensor = T.ToTensor()(mask)

        # パッチIDを生成
        L = self.grid_h * self.grid_w
        dataset_id = torch.randint(0, self.num_datasets, (L,))
        image_id = torch.full((L,), idx, dtype=torch.long)

        # クラスラベル（各パッチの主要クラス）- マスクから抽出
        patch_classes = torch.zeros(L, dtype=torch.long)
        # マスクから各パッチの主要クラスを抽出（簡略化）
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),
            size=(self.grid_h, self.grid_w),
            mode='nearest'
        ).squeeze(0).squeeze(0)

        for i in range(L):
            h_idx = i // self.grid_w
            w_idx = i % self.grid_w
            patch_value = mask_resized[h_idx, w_idx]
            # グレー値をクラスIDに変換（0からnum_classes-1の範囲に正規化）
            if patch_value.item() == 0:
                patch_classes[i] = 0  # 背景
            else:
                # 1-5のクラスにマッピング（元の構造クラス+1）
                patch_classes[i] = min(int(patch_value.item() / 50), self.num_classes - 1)

        return img_tensor, dataset_id, image_id, mask_tensor, patch_classes


def build_medical_loader(batch_size: int, length: int = 100, image_size: int = 512,
                         num_classes: int = 6, num_datasets: int = 4,
                         grid_h: int = 16, grid_w: int = 16, num_workers: int = 2):
    """医学画像データローダーを構築"""
    dataset = MedicalSegmentationDataset(
        length=length, image_size=image_size, num_classes=num_classes,
        num_datasets=num_datasets, grid_h=grid_h, grid_w=grid_w
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
