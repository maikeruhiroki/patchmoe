#!/usr/bin/env python3
"""
実際の医用画像データセットを使用したPatchMoEの訓練スクリプト
Kaggle等から取得した実際のデータセットに対応
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from PatchMoE.model import PatchMoEModel
from PatchMoE.real_dataset_loader import build_real_medical_loader
from PatchMoE.losses import PatchMoECombinedLoss
from PatchMoE.contrastive import AdvancedPatchContrastiveLoss, DomainAdaptiveContrastiveLoss
from PatchMoE.eval import dice_score as dice_coefficient, miou


class RealDatasetTrainer:
    """実際の医用画像データセット用の訓練クラス"""

    def __init__(self, config: dict, device: str = 'cuda:0'):
        self.config = config
        self.device = device

        # 出力ディレクトリ
        self.output_dir = config.get(
            'output_dir', '/workspace/outputs/patchmoe_real')
        os.makedirs(self.output_dir, exist_ok=True)

        # モデル初期化
        self.model = self._build_model()
        self.model.to(device)

        # 損失関数
        self.criterion = PatchMoECombinedLoss(
            num_classes=config['num_classes'],
            dice_weight=config.get('dice_weight', 1.0),
            focal_weight=config.get('focal_weight', 1.0),
            contrastive_weight=config.get('contrastive_weight', 0.1),
            moe_weight=config.get('moe_weight', 0.01)
        )

        # 対照学習損失
        self.contrastive_criterion = AdvancedPatchContrastiveLoss(
            temperature=config.get('contrastive_temperature', 0.07),
            hard_negative_weight=config.get('hard_negative_weight', 2.0),
            domain_separation_weight=config.get(
                'domain_separation_weight', 1.5)
        )

        # オプティマイザー
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-2)
        )

        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('lr', 1e-4) * 0.01
        )

        # 訓練履歴
        self.train_history = {
            'loss': [],
            'dice': [],
            'miou': [],
            'contrastive_loss': [],
            'moe_loss': []
        }

        self.val_history = {
            'loss': [],
            'dice': [],
            'miou': []
        }

        # 最良の性能
        self.best_dice = 0.0
        self.best_miou = 0.0

    def _build_model(self):
        """モデルを構築"""
        return PatchMoEModel(
            in_ch=self.config.get('in_channels', 3),
            feat_dim=self.config.get('feat_dim', 128),
            grid_h=self.config.get('grid_h', 16),
            grid_w=self.config.get('grid_w', 16),
            num_classes=self.config.get('num_classes', 6),
            num_layers=self.config.get('num_layers', 8),
            num_heads=self.config.get('num_heads', 8),
            num_queries=self.config.get('num_queries', 256),
            num_datasets=self.config.get('num_datasets', 3),
            num_images=self.config.get('num_images_cap', 100000),
            gate_top_k=self.config.get('top_k', 2),
            gate_capacity=self.config.get('capacity_factor', 1.0),
            gate_noise=self.config.get('gate_noise', 1.0),
            backbone=self.config.get('backbone', 'resnet50'),
            pretrained_backbone=self.config.get('pretrained_backbone', True),
            experts_per_device=self.config.get('experts_per_device', 4),
            use_multiscale=self.config.get('use_multiscale', True)
        )

    def _build_data_loaders(self):
        """データローダーを構築"""
        # データセット設定
        dataset_configs = [
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

        val_dataset_configs = [
            {
                'name': 'drive',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'val',
                'dataset_id': 0,
                'augmentation': False
            },
            {
                'name': 'stare',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'val',
                'dataset_id': 1,
                'augmentation': False
            },
            {
                'name': 'chase',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'val',
                'dataset_id': 2,
                'augmentation': False
            }
        ]

        # 訓練データローダー
        train_loader = build_real_medical_loader(
            dataset_configs,
            batch_size=self.config.get('batch_size', 4),
            num_workers=self.config.get('num_workers', 2),
            shuffle=True,
            image_size=self.config.get('image_size', 512),
            grid_h=self.config.get('grid_h', 16),
            grid_w=self.config.get('grid_w', 16),
            num_classes=self.config.get('num_classes', 6)
        )

        # 検証データローダー
        val_loader = build_real_medical_loader(
            val_dataset_configs,
            batch_size=self.config.get('batch_size', 4),
            num_workers=self.config.get('num_workers', 2),
            shuffle=False,
            image_size=self.config.get('image_size', 512),
            grid_h=self.config.get('grid_h', 16),
            grid_w=self.config.get('grid_w', 16),
            num_classes=self.config.get('num_classes', 6)
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader):
        """1エポックの訓練"""
        self.model.train()
        epoch_losses = []
        epoch_dices = []
        epoch_mious = []
        epoch_contrastive_losses = []
        epoch_moe_losses = []

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(pbar):
            images = images.to(self.device)
            dataset_ids = dataset_ids.to(self.device)
            image_ids = image_ids.to(self.device)
            masks = masks.to(self.device)
            classes = classes.to(self.device)

            self.optimizer.zero_grad()

            # フォワードパス
            logits, pred_masks = self.model(images, dataset_ids, image_ids)

            # パッチ特徴量を取得（対照学習用）
            patch_features = logits  # [B, L, D]

            # 対照学習損失
            contrastive_loss = self.contrastive_criterion(
                patch_features, dataset_ids, image_ids
            )

            # MoE補助損失
            moe_aux_losses = []
            for layer in self.model.decoder.layers:
                if hasattr(layer, 'last_moe_aux') and layer.last_moe_aux is not None:
                    moe_aux_losses.append(layer.last_moe_aux)

            moe_aux_loss = sum(moe_aux_losses) if moe_aux_losses else torch.tensor(
                0.0, device=self.device)

            # 統合損失
            loss_dict = self.criterion(
                logits, pred_masks, classes, masks,
                contrastive_loss=contrastive_loss,
                moe_aux_loss=moe_aux_loss
            )

            total_loss = loss_dict['total_loss']
            total_loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # メトリクス計算
            with torch.no_grad():
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                # サイズを合わせる
                if pred_binary.shape != masks.shape:
                    pred_binary = torch.nn.functional.interpolate(
                        pred_binary, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                dice = dice_coefficient(pred_binary, masks)
                miou_val = miou(pred_binary, masks)

                epoch_losses.append(total_loss.item())
                epoch_dices.append(dice if isinstance(
                    dice, float) else dice.item())
                epoch_mious.append(miou_val if isinstance(
                    miou_val, float) else miou_val.item())
                epoch_contrastive_losses.append(contrastive_loss.item())
                epoch_moe_losses.append(moe_aux_loss.item())

                # プログレスバー更新
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{dice if isinstance(dice, float) else dice.item():.4f}',
                    'mIoU': f'{miou_val if isinstance(miou_val, float) else miou_val.item():.4f}'
                })

        return {
            'loss': np.mean(epoch_losses),
            'dice': np.mean(epoch_dices),
            'miou': np.mean(epoch_mious),
            'contrastive_loss': np.mean(epoch_contrastive_losses),
            'moe_loss': np.mean(epoch_moe_losses)
        }

    def validate_epoch(self, val_loader):
        """1エポックの検証"""
        self.model.eval()
        epoch_losses = []
        epoch_dices = []
        epoch_mious = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(pbar):
                images = images.to(self.device)
                dataset_ids = dataset_ids.to(self.device)
                image_ids = image_ids.to(self.device)
                masks = masks.to(self.device)
                classes = classes.to(self.device)

                # フォワードパス
                logits, pred_masks = self.model(images, dataset_ids, image_ids)

                # 損失計算
                loss_dict = self.criterion(logits, pred_masks, classes, masks)
                total_loss = loss_dict['total_loss']

                # メトリクス計算
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                # サイズを合わせる
                if pred_binary.shape != masks.shape:
                    pred_binary = torch.nn.functional.interpolate(
                        pred_binary, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                dice = dice_coefficient(pred_binary, masks)
                miou_val = miou(pred_binary, masks)

                epoch_losses.append(total_loss.item())
                epoch_dices.append(dice if isinstance(
                    dice, float) else dice.item())
                epoch_mious.append(miou_val if isinstance(
                    miou_val, float) else miou_val.item())

                # プログレスバー更新
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{dice if isinstance(dice, float) else dice.item():.4f}',
                    'mIoU': f'{miou_val if isinstance(miou_val, float) else miou_val.item():.4f}'
                })

        return {
            'loss': np.mean(epoch_losses),
            'dice': np.mean(epoch_dices),
            'miou': np.mean(epoch_mious)
        }

    def train(self):
        """訓練の実行"""
        print("Building data loaders...")
        try:
            train_loader, val_loader = self._build_data_loaders()
        except FileNotFoundError as e:
            print(f"Dataset not found: {e}")
            print("Please run download_real_datasets.py first to download the datasets.")
            return

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        epochs = self.config.get('epochs', 100)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # 訓練
            train_metrics = self.train_epoch(train_loader)

            # 検証
            val_metrics = self.validate_epoch(val_loader)

            # スケジューラー更新
            self.scheduler.step()

            # 履歴更新
            for key in self.train_history:
                self.train_history[key].append(train_metrics[key])
            for key in self.val_history:
                self.val_history[key].append(val_metrics[key])

            # 最良モデル保存
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_miou = val_metrics['miou']

                # モデル保存
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'config': self.config,
                    'best_dice': self.best_dice,
                    'best_miou': self.best_miou,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }, os.path.join(self.output_dir, 'best_model.pt'))

                # 最良メトリクス保存
                with open(os.path.join(self.output_dir, 'best_metrics.json'), 'w') as f:
                    json.dump({
                        'dice': self.best_dice,
                        'miou': self.best_miou,
                        'epoch': epoch
                    }, f, indent=2)

            # ログ出力
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, mIoU: {train_metrics['miou']:.4f}")
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, mIoU: {val_metrics['miou']:.4f}")
            print(
                f"Best  - Dice: {self.best_dice:.4f}, mIoU: {self.best_miou:.4f}")

        # 最終結果の可視化
        self.plot_training_history()

        print(f"\nTraining completed!")
        print(f"Best Dice: {self.best_dice:.4f}")
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"Results saved to: {self.output_dir}")

    def plot_training_history(self):
        """訓練履歴の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 損失
        axes[0, 0].plot(self.train_history['loss'], label='Train')
        axes[0, 0].plot(self.val_history['loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Dice係数
        axes[0, 1].plot(self.train_history['dice'], label='Train')
        axes[0, 1].plot(self.val_history['dice'], label='Validation')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # mIoU
        axes[1, 0].plot(self.train_history['miou'], label='Train')
        axes[1, 0].plot(self.val_history['miou'], label='Validation')
        axes[1, 0].set_title('mIoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mIoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 対照学習損失
        axes[1, 1].plot(self.train_history['contrastive_loss'],
                        label='Contrastive Loss')
        axes[1, 1].plot(self.train_history['moe_loss'], label='MoE Loss')
        axes[1, 1].set_title('Auxiliary Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                    'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train PatchMoE on Real Medical Datasets')
    parser.add_argument('--output_dir', type=str, default='/workspace/outputs/patchmoe_real',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    args = parser.parse_args()

    # 設定
    config = {
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-2,
        'num_workers': 2,

        # モデル設定
        'in_channels': 3,
        'feat_dim': 128,
        'grid_h': 16,
        'grid_w': 16,
        'num_classes': 6,
        'num_layers': 8,
        'num_heads': 8,
        'num_queries': 256,
        'num_datasets': 3,
        'num_images_cap': 100000,
        'experts_per_device': 4,

        # MoE設定
        'top_k': 2,
        'capacity_factor': 1.0,
        'gate_noise': 1.0,

        # バックボーン設定
        'backbone': 'resnet50',
        'pretrained_backbone': True,
        'use_multiscale': True,

        # 損失重み
        'dice_weight': 1.0,
        'focal_weight': 1.0,
        'contrastive_weight': 0.1,
        'moe_weight': 0.01,

        # 対照学習設定
        'contrastive_temperature': 0.07,
        'hard_negative_weight': 2.0,
        'domain_separation_weight': 1.5,

        # 画像設定
        'image_size': 512,
    }

    # 訓練実行
    trainer = RealDatasetTrainer(config, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
