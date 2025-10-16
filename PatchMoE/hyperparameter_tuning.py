#!/usr/bin/env python3
"""
PatchMoEのハイパーパラメータ最適化スクリプト
MoE安定化パラメータの詳細チューニング
"""
import os
import json
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple

from PatchMoE.model import PatchMoEModel
from PatchMoE.medical_dataset import build_medical_loader
from PatchMoE.losses import dice_loss, FocalLoss
from PatchMoE.contrastive import PatchContrastiveLoss


class HyperparameterTuner:
    """ハイパーパラメータ最適化クラス"""

    def __init__(self, base_config: Dict, device: str = 'cuda:0'):
        self.base_config = base_config
        self.device = device
        self.results = []

    def get_hyperparameter_grid(self) -> List[Dict]:
        """最適化するハイパーパラメータのグリッドを定義"""
        grid = {
            'top_k': [1, 2, 4],
            'capacity_factor': [0.8, 1.0, 1.2],
            'gate_noise': [0.0, 0.5, 1.0],
            'experts_per_device': [2, 4],
            'num_layers': [6, 8],
            'num_heads': [4, 8],  # feat_dim=128と互いに素になるように調整
            'lr': [1e-4, 5e-4],
            'weight_decay': [1e-2]
        }

        # 重要なパラメータの組み合わせを生成（次元制約を考慮）
        important_params = ['top_k', 'capacity_factor', 'gate_noise', 'experts_per_device']
        combinations = list(itertools.product(*[grid[k] for k in important_params]))

        configs = []
        for combo in combinations[:8]:  # 上位8組み合わせをテスト
            config = self.base_config.copy()
            # 次元制約を満たすように調整
            config['feat_dim'] = 128  # 固定
            config['num_heads'] = 8   # 128の約数
            for i, param in enumerate(important_params):
                config[param] = combo[i]

            # 追加のパラメータをランダムに選択
            config['num_layers'] = np.random.choice(grid['num_layers'])
            config['num_heads'] = np.random.choice(grid['num_heads'])
            config['lr'] = np.random.choice(grid['lr'])
            config['weight_decay'] = np.random.choice(grid['weight_decay'])

            configs.append(config)

        return configs

    def train_and_evaluate(self, config: Dict, epochs: int = 2) -> Dict:
        """指定された設定で学習・評価を実行"""
        print(f"Testing config: {config}")

        # データローダー作成
        train_loader = build_medical_loader(
            batch_size=config.get('batch_size', 2),
            length=100,
            image_size=512,
            num_classes=config.get('num_classes', 6),
            num_datasets=config.get('num_datasets', 4),
            grid_h=config.get('grid_h', 16),
            grid_w=config.get('grid_w', 16),
            num_workers=0
        )

        # モデル作成
        model = PatchMoEModel(
            feat_dim=config.get('feat_dim', 128),
            grid_h=config.get('grid_h', 16),
            grid_w=config.get('grid_w', 16),
            num_classes=config.get('num_classes', 6),
            num_layers=config.get('num_layers', 8),
            num_heads=config.get('num_heads', 8),
            num_queries=config.get('grid_h', 16) * config.get('grid_w', 16),
            num_datasets=config.get('num_datasets', 4),
            num_images=config.get('num_images_cap', 100000),
            gate_top_k=config.get('top_k', 2),
            gate_capacity=config.get('capacity_factor', 1.0),
            gate_noise=config.get('gate_noise', 1.0),
            experts_per_device=config.get('experts_per_device', 4),
            use_multiscale=True,
            backbone='resnet50',
            pretrained_backbone=True
        ).to(self.device)

        # オプティマイザー
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-2)
        )

        # 損失関数
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        contrastive_loss = PatchContrastiveLoss(temperature=0.1)

        # 学習ループ
        model.train()
        total_losses = []
        dice_scores = []

        for epoch in range(epochs):
            epoch_losses = []
            epoch_dices = []

            for batch_idx, (img, dataset_id, image_id, mask, cls) in enumerate(train_loader):
                if batch_idx >= 10:  # 短時間で評価
                    break

                img = img.to(self.device)
                dataset_id = dataset_id.to(self.device)
                image_id = image_id.to(self.device)
                mask = mask.to(self.device)
                cls = cls.to(self.device)

                optimizer.zero_grad()

                # フォワードパス
                logits, pred_mask = model(img, dataset_id, image_id)

                # 損失計算
                cls_loss = focal_loss(
                    logits.view(-1, logits.size(-1)), cls.view(-1))
                dice_loss_val = dice_loss(pred_mask, mask)

                # コントラスト損失（簡略化）
                contrast_loss = contrastive_loss(
                    logits.view(-1, logits.size(-1)),
                    torch.zeros(logits.size(0) * logits.size(1),
                                device=self.device)
                )

                # MoEロードバランシング損失
                lb_loss = 0.0
                for layer in model.decoder.layers:
                    if hasattr(layer, 'last_moe_aux') and layer.last_moe_aux is not None:
                        lb_loss += layer.last_moe_aux

                total_loss = cls_loss + dice_loss_val + 0.1 * contrast_loss + 0.01 * lb_loss

                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())

                # Dice係数計算
                with torch.no_grad():
                    pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
                    dice = 2 * (pred_binary * mask).sum() / \
                        (pred_binary.sum() + mask.sum() + 1e-6)
                    epoch_dices.append(dice.item())

            total_losses.extend(epoch_losses)
            dice_scores.extend(epoch_dices)

        # 最終評価
        avg_loss = np.mean(total_losses[-10:])  # 最後の10バッチの平均
        avg_dice = np.mean(dice_scores[-10:])

        result = {
            'config': config,
            'avg_loss': avg_loss,
            'avg_dice': avg_dice,
            'final_loss': total_losses[-1] if total_losses else float('inf'),
            'final_dice': dice_scores[-1] if dice_scores else 0.0
        }

        print(f"Result: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")
        return result

    def optimize(self, output_dir: str, max_configs: int = 10):
        """ハイパーパラメータ最適化を実行"""
        os.makedirs(output_dir, exist_ok=True)

        configs = self.get_hyperparameter_grid()
        configs = configs[:max_configs]  # 制限

        print(f"Testing {len(configs)} configurations...")

        for i, config in enumerate(configs):
            print(f"\n--- Configuration {i+1}/{len(configs)} ---")
            try:
                result = self.train_and_evaluate(config)
                self.results.append(result)

                # 中間結果を保存
                with open(os.path.join(output_dir, 'tuning_results.json'), 'w') as f:
                    json.dump(self.results, f, indent=2)

            except Exception as e:
                print(f"Error in config {i+1}: {e}")
                continue

        # 最良の設定を選択
        if self.results:
            best_result = max(self.results, key=lambda x: x['avg_dice'])
            print(f"\nBest configuration found:")
            print(f"Dice Score: {best_result['avg_dice']:.4f}")
            print(f"Loss: {best_result['avg_loss']:.4f}")
            print(f"Config: {best_result['config']}")

            # 最良の設定を保存
            with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
                json.dump(best_result, f, indent=2)

            return best_result

        return None


def main():
    """メイン実行関数"""
    base_config = {
        'feat_dim': 128,
        'grid_h': 16,
        'grid_w': 16,
        'num_classes': 6,
        'num_datasets': 4,
        'num_images_cap': 100000,
        'batch_size': 2
    }

    tuner = HyperparameterTuner(base_config)
    best_config = tuner.optimize(
        '/workspace/outputs/hyperparameter_tuning', max_configs=8)

    if best_config:
        print(f"\n🎉 Best hyperparameters found!")
        print(f"Best Dice Score: {best_config['avg_dice']:.4f}")
        return best_config
    else:
        print("❌ No valid configurations found")
        return None


if __name__ == '__main__':
    main()
