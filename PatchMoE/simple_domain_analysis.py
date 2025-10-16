#!/usr/bin/env python3
"""
シンプルなドメイン適応分析
実際のKaggleデータセットを使用した分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob


class SimpleMedicalDataset(Dataset):
    """シンプルな医用画像データセット"""

    def __init__(self, data_dir, dataset_id, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        # 画像とマスクのパスを取得
        self.image_paths = sorted(
            glob.glob(os.path.join(data_dir, 'train', 'image', '*.png')))
        self.mask_paths = sorted(
            glob.glob(os.path.join(data_dir, 'train', 'mask', '*.png')))

        print(
            f"Dataset {dataset_id}: Found {len(self.image_paths)} images, {len(self.mask_paths)} masks")

    def __len__(self):
        return min(len(self.image_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        # 画像とマスクを読み込み
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 前処理
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # データセットIDと画像ID
        dataset_id = torch.tensor(self.dataset_id, dtype=torch.long)
        image_id = torch.tensor(idx, dtype=torch.long)

        # クラス（バイナリセグメンテーション）
        classes = torch.tensor(1, dtype=torch.long)  # 背景と前景の2クラス

        return image, dataset_id, image_id, mask, classes


class SimpleDomainAnalyzer:
    """シンプルなドメイン分析器"""

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_names = ['DRIVE', 'Kvasir-SEG',
                              'Synapse', 'Retina Blood Vessel']

    def create_dataloaders(self):
        """データローダーを作成"""
        dataloaders = []

        # 各データセットのデータローダーを作成
        for i, dataset_name in enumerate(self.dataset_names):
            data_dir = f'/workspace/real_medical_datasets_kaggle/Data'
            dataset = SimpleMedicalDataset(data_dir, i)

            if len(dataset) > 0:
                dataloader = DataLoader(
                    dataset, batch_size=4, shuffle=True, num_workers=0)
                dataloaders.append(dataloader)
                print(
                    f"Created dataloader for {dataset_name} with {len(dataset)} samples")
            else:
                print(f"No data found for {dataset_name}")
                dataloaders.append(None)

        return dataloaders

    def analyze_dataset_distribution(self, dataloaders):
        """データセット分布の分析"""
        print("データセット分布を分析中...")

        distribution = {}
        for i, (dataloader, name) in enumerate(zip(dataloaders, self.dataset_names)):
            if dataloader is not None:
                distribution[name] = len(dataloader.dataset)
            else:
                distribution[name] = 0

        # 可視化
        plt.figure(figsize=(10, 6))
        names = list(distribution.keys())
        counts = list(distribution.values())

        plt.bar(names, counts, color=['red', 'blue', 'green', 'orange'])
        plt.title('データセット分布')
        plt.xlabel('データセット')
        plt.ylabel('サンプル数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/workspace/outputs/patchmoe_kaggle/dataset_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return distribution

    def analyze_image_statistics(self, dataloaders):
        """画像統計の分析"""
        print("画像統計を分析中...")

        stats = {}

        for i, (dataloader, name) in enumerate(zip(dataloaders, self.dataset_names)):
            if dataloader is None:
                continue

            print(f"Analyzing {name}...")

            # 統計情報を収集
            pixel_values = []
            mask_values = []

            for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(tqdm(dataloader, desc=f"Processing {name}")):
                if batch_idx >= 10:  # 最初の10バッチのみ
                    break

                pixel_values.append(images.numpy().flatten())
                mask_values.append(masks.numpy().flatten())

            if pixel_values:
                all_pixels = np.concatenate(pixel_values)
                all_masks = np.concatenate(mask_values)

                stats[name] = {
                    'pixel_mean': float(np.mean(all_pixels)),
                    'pixel_std': float(np.std(all_pixels)),
                    'pixel_min': float(np.min(all_pixels)),
                    'pixel_max': float(np.max(all_pixels)),
                    'mask_mean': float(np.mean(all_masks)),
                    'mask_std': float(np.std(all_masks)),
                    'samples_analyzed': len(all_pixels)
                }

        # 統計の可視化
        if stats:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            metrics = ['pixel_mean', 'pixel_std', 'mask_mean', 'mask_std']
            metric_names = ['Pixel Mean', 'Pixel Std', 'Mask Mean', 'Mask Std']

            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                names = list(stats.keys())
                values = [stats[name][metric] for name in names]

                axes[i].bar(names, values, color=[
                            'red', 'blue', 'green', 'orange'])
                axes[i].set_title(f'{metric_name} by Dataset')
                axes[i].set_ylabel(metric_name)
                axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('/workspace/outputs/patchmoe_kaggle/image_statistics.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        return stats

    def generate_analysis_report(self, distribution, stats):
        """分析レポートの生成"""
        report = {
            'dataset_distribution': distribution,
            'image_statistics': stats,
            'analysis_summary': {
                'total_datasets': len(self.dataset_names),
                'datasets_with_data': sum(1 for count in distribution.values() if count > 0),
                'total_samples': sum(distribution.values())
            }
        }

        # レポート保存
        with open('/workspace/outputs/patchmoe_kaggle/simple_domain_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("分析レポートを生成しました: /workspace/outputs/patchmoe_kaggle/simple_domain_analysis_report.json")

        return report


def main():
    """メイン実行関数"""
    print("シンプルなドメイン適応分析を開始します...")

    # 出力ディレクトリの作成
    os.makedirs('/workspace/outputs/patchmoe_kaggle', exist_ok=True)

    # 分析器の初期化
    analyzer = SimpleDomainAnalyzer()

    # データローダーの作成
    print("データローダーを作成中...")
    dataloaders = analyzer.create_dataloaders()

    # データセット分布の分析
    distribution = analyzer.analyze_dataset_distribution(dataloaders)

    # 画像統計の分析
    stats = analyzer.analyze_image_statistics(dataloaders)

    # 分析レポートの生成
    report = analyzer.generate_analysis_report(distribution, stats)

    print("シンプルなドメイン適応分析が完了しました！")
    print(f"分析されたデータセット数: {report['analysis_summary']['datasets_with_data']}")
    print(f"総サンプル数: {report['analysis_summary']['total_samples']}")


if __name__ == "__main__":
    main()

