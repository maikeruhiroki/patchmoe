#!/usr/bin/env python3
"""
ドメイン適応性能の詳細分析
異なる医用画像ドメイン間の適応性能を評価
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from tqdm import tqdm
import os
import json
from collections import defaultdict

from PatchMoE.model import PatchMoEModel
from PatchMoE.config import PatchMoEConfig
from PatchMoE.kaggle_dataset_loader import build_kaggle_medical_loader
from PatchMoE.eval import dice_score, miou


class DomainAdaptationAnalyzer:
    """ドメイン適応性能分析クラス"""

    def __init__(self, model_path, config_path=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        # モデル設定
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = PatchMoEConfig(**config_dict)
        else:
            self.config = PatchMoEConfig()

        # モデル読み込み（保存されたモデルの設定に合わせる）
        self.model = PatchMoEModel(
            in_ch=3,
            feat_dim=128,
            num_datasets=4,  # 保存されたモデルは4データセット用
            num_images=10000,  # 保存されたモデルの設定
            grid_h=16,
            grid_w=16,
            num_patches=None,
            num_classes=2,  # 保存されたモデルは2クラス用
            num_layers=8,
            num_heads=8,
            num_queries=256,  # 保存されたモデルの設定
            gate_top_k=2,
            gate_capacity=1.0,
            gate_noise=1.0,
            backbone='resnet50',  # 保存されたモデルはResNet50使用
            pretrained_backbone=True,
            experts_per_device=4,
            use_multiscale=True
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # データセット名
        self.dataset_names = ['DRIVE', 'Kvasir-SEG',
                              'Synapse', 'Retina Blood Vessel']

    def extract_features(self, data_loader):
        """特徴量抽出"""
        features = []
        dataset_ids = []
        image_ids = []
        labels = []

        with torch.no_grad():
            for batch_idx, (images, d_ids, img_ids, masks, classes) in enumerate(tqdm(data_loader, desc="特徴量抽出")):
                images = images.to(self.device)
                d_ids = d_ids.to(self.device)
                img_ids = img_ids.to(self.device)

                # モデルから特徴量を抽出
                outputs = self.model(images, d_ids, img_ids)

                if len(outputs) >= 3:
                    cls_logits, pred_masks, patch_features = outputs[:3]

                    if patch_features is not None:
                        # パッチ特徴量を平均化
                        patch_features_avg = patch_features.mean(
                            dim=1)  # [batch_size, feature_dim]
                        features.append(patch_features_avg.cpu().numpy())
                        dataset_ids.extend(d_ids.cpu().numpy())
                        image_ids.extend(img_ids.cpu().numpy())
                        labels.extend(classes.cpu().numpy())

        if features:
            features = np.vstack(features)
            return features, np.array(dataset_ids), np.array(image_ids), np.array(labels)
        else:
            return None, None, None, None

    def analyze_domain_separation(self, features, dataset_ids):
        """ドメイン分離の分析"""
        print("ドメイン分離分析を実行中...")

        # t-SNE可視化
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # ドメイン分離の可視化
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue', 'green', 'orange']

        for i, dataset_name in enumerate(self.dataset_names):
            mask = dataset_ids == i
            if np.any(mask):
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c=colors[i], label=dataset_name, alpha=0.6, s=20)

        plt.title('ドメイン分離の可視化 (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/workspace/outputs/patchmoe_kaggle/domain_separation_tsne.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # シルエット分析
        try:
            silhouette_avg = silhouette_score(features, dataset_ids)
            print(f"シルエット係数: {silhouette_avg:.4f}")
        except:
            print("シルエット係数の計算に失敗しました")
            silhouette_avg = 0.0

        return features_2d, silhouette_avg

    def analyze_expert_usage(self, data_loader):
        """エキスパート使用状況の分析"""
        print("エキスパート使用状況を分析中...")

        expert_usage = defaultdict(list)
        dataset_expert_usage = defaultdict(lambda: defaultdict(int))

        with torch.no_grad():
            for batch_idx, (images, d_ids, img_ids, masks, classes) in enumerate(tqdm(data_loader, desc="エキスパート分析")):
                images = images.to(self.device)
                d_ids = d_ids.to(self.device)
                img_ids = img_ids.to(self.device)

                # モデルから出力を取得
                outputs = self.model(images, d_ids, img_ids)

                if len(outputs) >= 4:
                    cls_logits, pred_masks, patch_features, moe_aux_losses = outputs[:4]

                    # MoEの補助損失からエキスパート使用状況を推定
                    if moe_aux_losses:
                        for aux_loss in moe_aux_losses:
                            if hasattr(aux_loss, 'expert_usage'):
                                expert_usage['total'].append(
                                    aux_loss.expert_usage.cpu().numpy())

                                for i, d_id in enumerate(d_ids):
                                    dataset_name = self.dataset_names[d_id.item(
                                    )]
                                    expert_usage[dataset_name].append(
                                        aux_loss.expert_usage[i].cpu().numpy())

        # エキスパート使用状況の可視化
        if expert_usage:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            for i, (dataset_name, usage_data) in enumerate(expert_usage.items()):
                if i < 4 and usage_data:
                    usage_array = np.array(usage_data)
                    if len(usage_array.shape) > 1:
                        usage_array = usage_array.mean(axis=0)

                    axes[i].bar(range(len(usage_array)), usage_array)
                    axes[i].set_title(f'{dataset_name} - エキスパート使用状況')
                    axes[i].set_xlabel('エキスパートID')
                    axes[i].set_ylabel('使用頻度')
                    axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('/workspace/outputs/patchmoe_kaggle/expert_usage_analysis.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        return expert_usage

    def cross_domain_evaluation(self, data_loader):
        """クロスドメイン評価"""
        print("クロスドメイン評価を実行中...")

        results = {}

        with torch.no_grad():
            for batch_idx, (images, d_ids, img_ids, masks, classes) in enumerate(tqdm(data_loader, desc="クロスドメイン評価")):
                images = images.to(self.device)
                d_ids = d_ids.to(self.device)
                img_ids = img_ids.to(self.device)
                masks = masks.to(self.device)
                classes = classes.to(self.device)

                # 各ドメインでテスト
                for target_domain in range(len(self.dataset_names)):
                    target_d_ids = torch.full_like(d_ids, target_domain)

                    outputs = self.model(images, target_d_ids, img_ids)

                    if len(outputs) >= 2:
                        cls_logits, pred_masks = outputs[:2]

                        # 予測マスクの処理
                        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                        if pred_binary.shape != masks.shape:
                            pred_binary = torch.nn.functional.interpolate(
                                pred_binary, size=masks.shape[-2:], mode='bilinear', align_corners=False
                            )

                        # メトリクス計算
                        dice = dice_score(pred_binary, masks)
                        miou_val = miou(pred_binary, masks)

                        if target_domain not in results:
                            results[target_domain] = {'dice': [], 'miou': []}

                        results[target_domain]['dice'].append(
                            dice.item() if hasattr(dice, 'item') else dice)
                        results[target_domain]['miou'].append(
                            miou_val.item() if hasattr(miou_val, 'item') else miou_val)

        # 結果の可視化
        self.visualize_cross_domain_results(results)

        return results

    def visualize_cross_domain_results(self, results):
        """クロスドメイン結果の可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Dice係数の可視化
        dice_data = []
        miou_data = []
        domain_labels = []

        for domain_id, metrics in results.items():
            if metrics['dice']:
                dice_data.append(np.mean(metrics['dice']))
                miou_data.append(np.mean(metrics['miou']))
                domain_labels.append(self.dataset_names[domain_id])

        if dice_data:
            axes[0].bar(domain_labels, dice_data, color='skyblue', alpha=0.7)
            axes[0].set_title('クロスドメイン Dice係数')
            axes[0].set_ylabel('Dice係数')
            axes[0].set_xticklabels(domain_labels, rotation=45)
            axes[0].grid(True, alpha=0.3)

            axes[1].bar(domain_labels, miou_data,
                        color='lightcoral', alpha=0.7)
            axes[1].set_title('クロスドメイン mIoU')
            axes[1].set_ylabel('mIoU')
            axes[1].set_xticklabels(domain_labels, rotation=45)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/workspace/outputs/patchmoe_kaggle/cross_domain_evaluation.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_analysis_report(self, features, dataset_ids, expert_usage, cross_domain_results):
        """分析レポートの生成"""
        report = {
            'analysis_summary': {
                'total_samples': len(features) if features is not None else 0,
                'num_domains': len(self.dataset_names),
                'feature_dimension': features.shape[1] if features is not None else 0
            },
            'domain_distribution': {},
            'expert_usage_summary': {},
            'cross_domain_performance': {}
        }

        # ドメイン分布
        if dataset_ids is not None:
            unique, counts = np.unique(dataset_ids, return_counts=True)
            for domain_id, count in zip(unique, counts):
                report['domain_distribution'][self.dataset_names[domain_id]] = int(
                    count)

        # エキスパート使用状況
        for dataset_name, usage_data in expert_usage.items():
            if usage_data:
                usage_array = np.array(usage_data)
                if len(usage_array.shape) > 1:
                    usage_array = usage_array.mean(axis=0)
                report['expert_usage_summary'][dataset_name] = {
                    'mean_usage': float(np.mean(usage_array)),
                    'std_usage': float(np.std(usage_array)),
                    'max_usage': float(np.max(usage_array))
                }

        # クロスドメイン性能
        for domain_id, metrics in cross_domain_results.items():
            if metrics['dice']:
                report['cross_domain_performance'][self.dataset_names[domain_id]] = {
                    'mean_dice': float(np.mean(metrics['dice'])),
                    'mean_miou': float(np.mean(metrics['miou'])),
                    'std_dice': float(np.std(metrics['dice'])),
                    'std_miou': float(np.std(metrics['miou']))
                }

        # レポート保存
        with open('/workspace/outputs/patchmoe_kaggle/domain_adaptation_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("分析レポートを生成しました: /workspace/outputs/patchmoe_kaggle/domain_adaptation_report.json")

        return report


def main():
    """メイン実行関数"""
    print("ドメイン適応性能の詳細分析を開始します...")

    # 出力ディレクトリの作成
    os.makedirs('/workspace/outputs/patchmoe_kaggle', exist_ok=True)

    # 分析器の初期化
    model_path = '/workspace/outputs/patchmoe_kaggle/best_model.pt'
    analyzer = DomainAdaptationAnalyzer(model_path)

    # データセット設定
    dataset_configs = [
        {
            'name': 'DRIVE',
            'type': 'drive',
            'root_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'data_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'dataset_id': 0
        },
        {
            'name': 'Kvasir-SEG',
            'type': 'kvasir_seg',
            'root_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'data_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'dataset_id': 1
        },
        {
            'name': 'Synapse',
            'type': 'synapse',
            'root_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'data_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'dataset_id': 2
        },
        {
            'name': 'Retina Blood Vessel',
            'type': 'retina_blood_vessel',
            'root_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'data_dir': '/workspace/real_medical_datasets_kaggle/Data',
            'dataset_id': 3
        }
    ]

    # データローダーの準備
    train_loader = build_kaggle_medical_loader(
        dataset_configs=dataset_configs,
        batch_size=4,
        num_workers=0
    )
    val_loader = build_kaggle_medical_loader(
        dataset_configs=dataset_configs,
        batch_size=4,
        num_workers=0
    )

    # 特徴量抽出
    print("特徴量を抽出中...")
    features, dataset_ids, image_ids, labels = analyzer.extract_features(
        train_loader)

    if features is not None:
        # ドメイン分離分析
        features_2d, silhouette_score = analyzer.analyze_domain_separation(
            features, dataset_ids)

        # エキスパート使用状況分析
        expert_usage = analyzer.analyze_expert_usage(train_loader)

        # クロスドメイン評価
        cross_domain_results = analyzer.cross_domain_evaluation(train_loader)

        # 分析レポート生成
        report = analyzer.generate_analysis_report(
            features, dataset_ids, expert_usage, cross_domain_results
        )

        print("ドメイン適応分析が完了しました！")
        print(f"シルエット係数: {silhouette_score:.4f}")
        print(f"分析されたサンプル数: {len(features)}")
        print(f"ドメイン数: {len(analyzer.dataset_names)}")

    else:
        print("特徴量の抽出に失敗しました。")


if __name__ == "__main__":
    main()
