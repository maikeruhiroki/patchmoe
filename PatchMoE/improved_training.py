#!/usr/bin/env python3
"""
改善されたPatchMoE訓練スクリプト
低いDice/mIoUスコアの問題を解決
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import glob

from PatchMoE.model import PatchMoEModel
from PatchMoE.config import PatchMoEConfig
from PatchMoE.eval import dice_score, miou


class ImprovedMedicalDataset:
    """改善された医用画像データセット"""

    def __init__(self, data_dir, transform=None, mask_transform=None, max_samples=None):
        self.data_dir = data_dir
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

        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]

        print(
            f"Found {len(self.image_paths)} images, {len(self.mask_paths)} masks")

    def __len__(self):
        return min(len(self.image_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        # 画像とマスクを読み込み
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # 前処理
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # データセットIDと画像ID（バッチ次元を追加）
        dataset_id = torch.tensor([0], dtype=torch.long)  # 単一データセット
        image_id = torch.tensor([idx], dtype=torch.long)

        # クラス（バイナリセグメンテーション）
        classes = torch.tensor([1], dtype=torch.long)

        return image, dataset_id, image_id, mask, classes


class ImprovedTrainer:
    """改善された訓練器"""

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # モデルの初期化
        self.model = PatchMoEModel(
            in_ch=3,
            feat_dim=128,
            num_datasets=1,  # 単一データセット
            num_images=1000,
            grid_h=16,
            grid_w=16,
            num_patches=256,  # パッチ数を明示的に指定
            num_classes=2,
            num_layers=6,  # レイヤー数を削減
            num_heads=8,
            num_queries=100,  # クエリ数を削減
            gate_top_k=2,
            gate_capacity=1.0,
            gate_noise=0.1,  # ノイズを削減
            backbone='simple',  # シンプルなバックボーン
            pretrained_backbone=False,
            experts_per_device=2,  # エキスパート数を削減
            use_multiscale=False
        ).to(device)

        # オプティマイザー
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        # 損失関数
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader, epoch):
        """訓練エポック"""
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_miou = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(pbar):
            images = images.to(self.device)
            dataset_ids = dataset_ids.to(self.device)
            image_ids = image_ids.to(self.device)
            masks = masks.to(self.device)
            classes = classes.to(self.device)

            # 前向き伝播
            self.optimizer.zero_grad()

            # モデル出力
            outputs = self.model(images, dataset_ids, image_ids)

            if len(outputs) >= 2:
                cls_logits, pred_masks = outputs[:2]

                # 予測マスクのサイズ調整
                if pred_masks.shape != masks.shape:
                    pred_masks = torch.nn.functional.interpolate(
                        pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )

                # 損失計算
                loss = self.criterion(pred_masks, masks)

                # 後向き伝播
                loss.backward()
                self.optimizer.step()

                # メトリクス計算
                with torch.no_grad():
                    pred_binary = torch.sigmoid(pred_masks)
                    dice = dice_score(pred_binary, masks)
                    miou_val = miou(pred_binary, masks)

                    epoch_loss += loss.item()
                    epoch_dice += dice.item() if hasattr(dice, 'item') else dice
                    epoch_miou += miou_val.item() if hasattr(miou_val, 'item') else miou_val

                # プログレスバー更新
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item() if hasattr(dice, "item") else dice:.4f}',
                    'mIoU': f'{miou_val.item() if hasattr(miou_val, "item") else miou_val:.4f}'
                })

        return {
            'loss': epoch_loss / len(dataloader),
            'dice': epoch_dice / len(dataloader),
            'miou': epoch_miou / len(dataloader)
        }

    def validate_epoch(self, dataloader, epoch):
        """検証エポック"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_miou = 0.0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validation {epoch+1}")
            for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(pbar):
                images = images.to(self.device)
                dataset_ids = dataset_ids.to(self.device)
                image_ids = image_ids.to(self.device)
                masks = masks.to(self.device)
                classes = classes.to(self.device)

                # モデル出力
                outputs = self.model(images, dataset_ids, image_ids)

                if len(outputs) >= 2:
                    cls_logits, pred_masks = outputs[:2]

                    # 予測マスクのサイズ調整
                    if pred_masks.shape != masks.shape:
                        pred_masks = torch.nn.functional.interpolate(
                            pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False
                        )

                    # 損失計算
                    loss = self.criterion(pred_masks, masks)

                    # メトリクス計算
                    pred_binary = torch.sigmoid(pred_masks)
                    dice = dice_score(pred_binary, masks)
                    miou_val = miou(pred_binary, masks)

                    epoch_loss += loss.item()
                    epoch_dice += dice.item() if hasattr(dice, 'item') else dice
                    epoch_miou += miou_val.item() if hasattr(miou_val, 'item') else miou_val

                    # プログレスバー更新
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Dice': f'{dice.item() if hasattr(dice, "item") else dice:.4f}',
                        'mIoU': f'{miou_val.item() if hasattr(miou_val, "item") else miou_val:.4f}'
                    })

        return {
            'loss': epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
            'dice': epoch_dice / len(dataloader) if len(dataloader) > 0 else 0.0,
            'miou': epoch_miou / len(dataloader) if len(dataloader) > 0 else 0.0
        }

    def train(self, train_loader, val_loader, epochs):
        """訓練実行"""
        print("改善されたPatchMoE訓練を開始します...")

        # 訓練履歴
        history = {
            'train_loss': [], 'train_dice': [], 'train_miou': [],
            'val_loss': [], 'val_dice': [], 'val_miou': []
        }

        best_dice = 0.0
        best_miou = 0.0

        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")

            # 訓練
            train_metrics = self.train_epoch(train_loader, epoch)

            # 検証
            val_metrics = self.validate_epoch(val_loader, epoch)

            # スケジューラー更新
            self.scheduler.step()

            # 履歴更新
            history['train_loss'].append(train_metrics['loss'])
            history['train_dice'].append(train_metrics['dice'])
            history['train_miou'].append(train_metrics['miou'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_dice'].append(val_metrics['dice'])
            history['val_miou'].append(val_metrics['miou'])

            # ベストモデル保存
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                best_miou = val_metrics['miou']

                # モデル保存
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'dice': best_dice,
                    'miou': best_miou
                }, '/workspace/outputs/patchmoe_kaggle/improved_best_model.pt')

                # メトリクス保存
                with open('/workspace/outputs/patchmoe_kaggle/improved_best_metrics.json', 'w') as f:
                    json.dump({
                        'dice': best_dice,
                        'miou': best_miou,
                        'epoch': epoch + 1
                    }, f)

            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, mIoU: {train_metrics['miou']:.4f}")
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, mIoU: {val_metrics['miou']:.4f}")
            print(f"Best  - Dice: {best_dice:.4f}, mIoU: {best_miou:.4f}")

        # 訓練履歴の可視化
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """訓練履歴の可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Dice
        axes[1].plot(history['train_dice'], label='Train Dice', color='blue')
        axes[1].plot(history['val_dice'], label='Val Dice', color='red')
        axes[1].set_title('Training and Validation Dice Score')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # mIoU
        axes[2].plot(history['train_miou'], label='Train mIoU', color='blue')
        axes[2].plot(history['val_miou'], label='Val mIoU', color='red')
        axes[2].set_title('Training and Validation mIoU')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mIoU')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/workspace/outputs/patchmoe_kaggle/improved_training_history.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """メイン実行関数"""
    print("改善されたPatchMoE訓練を開始します...")

    # 出力ディレクトリの作成
    os.makedirs('/workspace/outputs/patchmoe_kaggle', exist_ok=True)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 設定
    config = PatchMoEConfig()
    config.epochs = 10
    config.batch_size = 4
    config.lr = 1e-4

    # データセット
    data_dir = '/workspace/real_medical_datasets_kaggle/Data'
    dataset = ImprovedMedicalDataset(data_dir, max_samples=80)

    # データローダー
    train_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # 訓練器
    trainer = ImprovedTrainer(config, device)

    # 訓練実行
    history = trainer.train(train_loader, val_loader, config.epochs)

    print("改善されたPatchMoE訓練が完了しました！")
    print(f"最終的なベスト Dice: {max(history['val_dice']):.4f}")
    print(f"最終的なベスト mIoU: {max(history['val_miou']):.4f}")


if __name__ == "__main__":
    main()
