#!/usr/bin/env python3
"""
シンプルなセグメンテーションモデル
PatchMoEの複雑さを避けて、基本的なセグメンテーション性能を確認
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSegmentationModel(nn.Module):
    """シンプルなセグメンテーションモデル"""

    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # エンコーダー
        self.encoder = nn.Sequential(
            # ブロック1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 512 -> 256

            # ブロック2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256 -> 128

            # ブロック3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            # ブロック4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
        )

        # デコーダー
        self.decoder = nn.Sequential(
            # アップサンプリング1
            nn.ConvTranspose2d(512, 256, 2, stride=2),  # 32 -> 64
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # アップサンプリング2
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 64 -> 128
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # アップサンプリング3
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 128 -> 256
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # アップサンプリング4
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 256 -> 512
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 最終出力
            nn.Conv2d(32, 1, 1),  # バイナリセグメンテーション用に1チャンネル
        )

    def forward(self, x):
        # エンコーダー
        encoded = self.encoder(x)

        # デコーダー
        decoded = self.decoder(encoded)

        return decoded


class SimpleTrainer:
    """シンプルな訓練器"""

    def __init__(self, device):
        self.device = device
        self.model = SimpleSegmentationModel().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
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
            masks = masks.to(self.device)

            # 前向き伝播
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 損失計算
            loss = self.criterion(outputs, masks)

            # 後向き伝播
            loss.backward()
            self.optimizer.step()

            # メトリクス計算
            with torch.no_grad():
                pred_binary = torch.sigmoid(outputs)
                dice = self.calculate_dice(pred_binary, masks)
                miou_val = self.calculate_miou(pred_binary, masks)

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
                masks = masks.to(self.device)

                # 前向き伝播
                outputs = self.model(images)

                # 損失計算
                loss = self.criterion(outputs, masks)

                # メトリクス計算
                pred_binary = torch.sigmoid(outputs)
                dice = self.calculate_dice(pred_binary, masks)
                miou_val = self.calculate_miou(pred_binary, masks)

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

    def calculate_dice(self, pred, target, eps=1e-6):
        """Dice係数の計算"""
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + eps) / (union + eps)
        return dice

    def calculate_miou(self, pred, target, eps=1e-6):
        """mIoUの計算"""
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        miou = (intersection + eps) / (union + eps)
        return miou

    def train(self, train_loader, val_loader, epochs):
        """訓練実行"""
        print("シンプルなセグメンテーションモデルの訓練を開始します...")

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
                }, '/workspace/outputs/patchmoe_kaggle/simple_best_model.pt')

                # メトリクス保存
                import json
                with open('/workspace/outputs/patchmoe_kaggle/simple_best_metrics.json', 'w') as f:
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

        return history


def main():
    """メイン実行関数"""
    print("シンプルなセグメンテーションモデルの訓練を開始します...")

    # 出力ディレクトリの作成
    import os
    os.makedirs('/workspace/outputs/patchmoe_kaggle', exist_ok=True)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # データセット（既存のImprovedMedicalDatasetを使用）
    from PatchMoE.improved_training import ImprovedMedicalDataset
    from torch.utils.data import DataLoader

    data_dir = '/workspace/real_medical_datasets_kaggle/Data'
    dataset = ImprovedMedicalDataset(data_dir, max_samples=80)

    # データローダー
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0
    )

    # 訓練器
    trainer = SimpleTrainer(device)

    # 訓練実行
    history = trainer.train(train_loader, val_loader, epochs=10)

    print("シンプルなセグメンテーションモデルの訓練が完了しました！")
    print(f"最終的なベスト Dice: {max(history['val_dice']):.4f}")
    print(f"最終的なベスト mIoU: {max(history['val_miou']):.4f}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
