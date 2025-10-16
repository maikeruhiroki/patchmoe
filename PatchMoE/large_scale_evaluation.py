#!/usr/bin/env python3
"""
大規模データセットでの検証
より多くの医用画像データでの検証
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
import time
from collections import defaultdict


class LargeScaleMedicalDataset(Dataset):
    """大規模医用画像データセット"""

    def __init__(self, data_dir, dataset_id, transform=None, mask_transform=None, max_samples=None):
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

        # 最大サンプル数の制限
        if max_samples and len(self.image_paths) > max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]

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


class LargeScaleEvaluator:
    """大規模データセット評価器"""

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_names = ['DRIVE', 'Kvasir-SEG',
                              'Synapse', 'Retina Blood Vessel']

    def create_scaled_dataloaders(self, scale_factors=[1.0, 2.0, 4.0, 8.0]):
        """スケールファクターに基づいてデータローダーを作成"""
        dataloaders_by_scale = {}

        for scale in scale_factors:
            print(f"\nCreating dataloaders for scale factor: {scale}x")
            dataloaders = []

            for i, dataset_name in enumerate(self.dataset_names):
                data_dir = f'/workspace/real_medical_datasets_kaggle/Data'
                max_samples = int(80 * scale)  # 基本80サンプルからスケール

                dataset = LargeScaleMedicalDataset(
                    data_dir, i, max_samples=max_samples)

                if len(dataset) > 0:
                    dataloader = DataLoader(
                        dataset, batch_size=4, shuffle=True, num_workers=0)
                    dataloaders.append(dataloader)
                    print(f"  {dataset_name}: {len(dataset)} samples")
                else:
                    print(f"  {dataset_name}: No data found")
                    dataloaders.append(None)

            dataloaders_by_scale[scale] = dataloaders

        return dataloaders_by_scale

    def evaluate_performance_scaling(self, dataloaders_by_scale):
        """性能スケーリングの評価"""
        print("性能スケーリングを評価中...")

        results = {}

        for scale, dataloaders in dataloaders_by_scale.items():
            print(f"\nEvaluating scale {scale}x...")

            scale_results = {
                'total_samples': 0,
                'dataset_counts': {},
                'processing_times': {},
                'memory_usage': {}
            }

            for i, (dataloader, name) in enumerate(zip(dataloaders, self.dataset_names)):
                if dataloader is None:
                    continue

                print(f"  Processing {name}...")
                start_time = time.time()

                # データローダーの処理時間を測定
                batch_count = 0
                for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(tqdm(dataloader, desc=f"Processing {name}")):
                    batch_count += 1
                    if batch_idx >= 10:  # 最初の10バッチのみ
                        break

                end_time = time.time()
                processing_time = end_time - start_time

                scale_results['dataset_counts'][name] = len(dataloader.dataset)
                scale_results['processing_times'][name] = processing_time
                scale_results['total_samples'] += len(dataloader.dataset)

                print(
                    f"    Samples: {len(dataloader.dataset)}, Time: {processing_time:.2f}s")

            results[scale] = scale_results

        return results

    def analyze_scalability(self, results):
        """スケーラビリティの分析"""
        print("スケーラビリティを分析中...")

        scales = sorted(results.keys())
        total_samples = [results[scale]['total_samples'] for scale in scales]
        avg_processing_times = []

        for scale in scales:
            times = list(results[scale]['processing_times'].values())
            if times:
                avg_processing_times.append(np.mean(times))
            else:
                avg_processing_times.append(0)

        # スケーラビリティの可視化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # サンプル数 vs スケール
        axes[0, 0].plot(scales, total_samples, 'bo-',
                        linewidth=2, markersize=8)
        axes[0, 0].set_title('Total Samples vs Scale Factor')
        axes[0, 0].set_xlabel('Scale Factor')
        axes[0, 0].set_ylabel('Total Samples')
        axes[0, 0].grid(True, alpha=0.3)

        # 処理時間 vs スケール
        axes[0, 1].plot(scales, avg_processing_times,
                        'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Average Processing Time vs Scale Factor')
        axes[0, 1].set_xlabel('Scale Factor')
        axes[0, 1].set_ylabel('Processing Time (s)')
        axes[0, 1].grid(True, alpha=0.3)

        # スループット（サンプル/秒）
        throughput = [samples / time if time > 0 else 0 for samples,
                      time in zip(total_samples, avg_processing_times)]
        axes[1, 0].plot(scales, throughput, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Throughput vs Scale Factor')
        axes[1, 0].set_xlabel('Scale Factor')
        axes[1, 0].set_ylabel('Samples/Second')
        axes[1, 0].grid(True, alpha=0.3)

        # 効率性（スループット/スケール）
        efficiency = [t / scale if scale > 0 else 0 for t,
                      scale in zip(throughput, scales)]
        axes[1, 1].plot(scales, efficiency, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Efficiency vs Scale Factor')
        axes[1, 1].set_xlabel('Scale Factor')
        axes[1, 1].set_ylabel('Efficiency (Samples/s per Scale)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/workspace/outputs/patchmoe_kaggle/scalability_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'scales': scales,
            'total_samples': total_samples,
            'avg_processing_times': avg_processing_times,
            'throughput': throughput,
            'efficiency': efficiency
        }

    def evaluate_memory_usage(self, dataloaders_by_scale):
        """メモリ使用量の評価"""
        print("メモリ使用量を評価中...")

        memory_results = {}

        for scale, dataloaders in dataloaders_by_scale.items():
            print(f"Evaluating memory usage for scale {scale}x...")

            scale_memory = {}
            total_samples = 0

            for i, (dataloader, name) in enumerate(zip(dataloaders, self.dataset_names)):
                if dataloader is None:
                    continue

                # データセットサイズの推定
                dataset_size = len(dataloader.dataset)
                total_samples += dataset_size

                # メモリ使用量の推定（画像サイズ512x512x3、マスク512x512）
                image_memory = dataset_size * 512 * 512 * 3 * 4  # float32
                mask_memory = dataset_size * 512 * 512 * 4  # float32
                total_memory = (image_memory + mask_memory) / (1024**3)  # GB

                scale_memory[name] = {
                    'samples': dataset_size,
                    'estimated_memory_gb': total_memory
                }

            memory_results[scale] = {
                'total_samples': total_samples,
                'datasets': scale_memory,
                'total_estimated_memory_gb': sum(d['estimated_memory_gb'] for d in scale_memory.values())
            }

        # メモリ使用量の可視化
        scales = sorted(memory_results.keys())
        total_memory = [memory_results[scale]
                        ['total_estimated_memory_gb'] for scale in scales]

        plt.figure(figsize=(10, 6))
        plt.plot(scales, total_memory, 'bo-', linewidth=2, markersize=8)
        plt.title('Estimated Memory Usage vs Scale Factor')
        plt.xlabel('Scale Factor')
        plt.ylabel('Estimated Memory Usage (GB)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/workspace/outputs/patchmoe_kaggle/memory_usage_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return memory_results

    def generate_large_scale_report(self, results, scalability, memory_results):
        """大規模評価レポートの生成"""
        report = {
            'evaluation_summary': {
                'scales_evaluated': sorted(results.keys()),
                'max_scale': max(results.keys()),
                'total_samples_at_max_scale': results[max(results.keys())]['total_samples']
            },
            'scalability_analysis': scalability,
            'memory_analysis': memory_results,
            'performance_metrics': {
                'scales': sorted(results.keys()),
                'total_samples': [results[scale]['total_samples'] for scale in sorted(results.keys())],
                'avg_processing_times': [np.mean(list(results[scale]['processing_times'].values())) for scale in sorted(results.keys())]
            }
        }

        # レポート保存
        with open('/workspace/outputs/patchmoe_kaggle/large_scale_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("大規模評価レポートを生成しました: /workspace/outputs/patchmoe_kaggle/large_scale_evaluation_report.json")

        return report


def main():
    """メイン実行関数"""
    print("大規模データセットでの検証を開始します...")

    # 出力ディレクトリの作成
    os.makedirs('/workspace/outputs/patchmoe_kaggle', exist_ok=True)

    # 評価器の初期化
    evaluator = LargeScaleEvaluator()

    # スケールファクターに基づくデータローダーの作成
    scale_factors = [1.0, 2.0, 4.0, 8.0]
    dataloaders_by_scale = evaluator.create_scaled_dataloaders(scale_factors)

    # 性能スケーリングの評価
    results = evaluator.evaluate_performance_scaling(dataloaders_by_scale)

    # スケーラビリティの分析
    scalability = evaluator.analyze_scalability(results)

    # メモリ使用量の評価
    memory_results = evaluator.evaluate_memory_usage(dataloaders_by_scale)

    # 大規模評価レポートの生成
    report = evaluator.generate_large_scale_report(
        results, scalability, memory_results)

    print("大規模データセットでの検証が完了しました！")
    print(f"評価されたスケール: {report['evaluation_summary']['scales_evaluated']}")
    print(
        f"最大スケールでの総サンプル数: {report['evaluation_summary']['total_samples_at_max_scale']}")


if __name__ == "__main__":
    main()

