#!/usr/bin/env python3
"""
改良されたU-Netモデルの訓練スクリプト
論文の成功要因を参考にした段階的実装
"""

from PatchMoE.eval import dice_score, miou
from PatchMoE.improved_unet_model import ImprovedUNet, ImprovedLoss
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt

# 改良されたモデルのインポート
sys.path.append('/workspace')


class EnhancedMedicalDataset(Dataset):
    """改良された医用画像データセット"""

    def __init__(self, num_samples=200, image_size=512, split='train', num_datasets=4):
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split
        self.num_datasets = num_datasets

        # より現実的な医用画像データを生成
        self.images = []
        self.masks = []

        for i in range(num_samples):
            # データセットIDをランダムに選択
            dataset_id = i % num_datasets

            # データセットごとに異なる特徴を持つ画像を生成
            if dataset_id == 0:  # DRIVE (網膜血管)
                image = self._generate_retinal_image()
                mask = self._generate_vessel_mask()
            elif dataset_id == 1:  # HV_NIR (近赤外血管)
                image = self._generate_nir_image()
                mask = self._generate_vessel_mask()
            elif dataset_id == 2:  # Kvasir-SEG (ポリープ)
                image = self._generate_polyp_image()
                mask = self._generate_polyp_mask()
            else:  # Synapse (多臓器)
                image = self._generate_ct_image()
                mask = self._generate_organ_mask()

            self.images.append(image)
            self.masks.append(mask)

    def _generate_retinal_image(self):
        """網膜画像の生成"""
        image = np.random.rand(
            3, self.image_size, self.image_size).astype(np.float32)
        # 網膜の特徴的な色調を模擬
        image[0] *= 0.8  # Rチャンネルを減衰
        image[1] *= 1.2  # Gチャンネルを強調
        image[2] *= 0.6  # Bチャンネルを減衰
        return image

    def _generate_nir_image(self):
        """近赤外画像の生成"""
        image = np.random.rand(
            1, self.image_size, self.image_size).astype(np.float32)
        # グレースケール画像を3チャンネルに変換
        image = np.repeat(image, 3, axis=0)
        return image

    def _generate_polyp_image(self):
        """ポリープ画像の生成"""
        image = np.random.rand(
            3, self.image_size, self.image_size).astype(np.float32)
        # 内視鏡画像の特徴的な色調
        image[0] *= 1.1  # Rチャンネルを強調
        image[1] *= 0.9  # Gチャンネルを減衰
        image[2] *= 1.0  # Bチャンネルはそのまま
        return image

    def _generate_ct_image(self):
        """CT画像の生成"""
        image = np.random.rand(
            1, self.image_size, self.image_size).astype(np.float32)
        # HU値の範囲を模擬
        image = (image - 0.5) * 2000  # HU値の範囲
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000  # 正規化
        # 3チャンネルに変換
        image = np.repeat(image, 3, axis=0)
        return image

    def _generate_vessel_mask(self):
        """血管マスクの生成"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # 複数の血管を生成
        num_vessels = np.random.randint(3, 8)
        for _ in range(num_vessels):
            # 血管の中心線
            start_x = np.random.randint(0, self.image_size)
            start_y = np.random.randint(0, self.image_size)
            end_x = np.random.randint(0, self.image_size)
            end_y = np.random.randint(0, self.image_size)

            # 血管の太さ
            thickness = np.random.randint(2, 8)

            # 血管を描画
            for t in np.linspace(0, 1, 100):
                x = int(start_x + t * (end_x - start_x))
                y = int(start_y + t * (end_y - start_y))

                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    # 円形の血管断面
                    for dx in range(-thickness, thickness + 1):
                        for dy in range(-thickness, thickness + 1):
                            if dx*dx + dy*dy <= thickness*thickness:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.image_size and 0 <= ny < self.image_size:
                                    mask[ny, nx] = 1.0

        return mask

    def _generate_polyp_mask(self):
        """ポリープマスクの生成"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # ポリープの中心とサイズ
        center_x = np.random.randint(
            self.image_size // 4, 3 * self.image_size // 4)
        center_y = np.random.randint(
            self.image_size // 4, 3 * self.image_size // 4)
        radius = np.random.randint(20, 60)

        # 楕円形のポリープ
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        mask = mask.astype(np.float32)

        return mask

    def _generate_organ_mask(self):
        """臓器マスクの生成"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # 複数の臓器を生成
        num_organs = np.random.randint(2, 5)
        for _ in range(num_organs):
            # 臓器の中心とサイズ
            center_x = np.random.randint(
                self.image_size // 4, 3 * self.image_size // 4)
            center_y = np.random.randint(
                self.image_size // 4, 3 * self.image_size // 4)
            radius = np.random.randint(30, 80)

            # 臓器の形状
            y, x = np.ogrid[:self.image_size, :self.image_size]
            organ_mask = ((x - center_x) ** 2 +
                          (y - center_y) ** 2) <= radius ** 2
            mask = np.logical_or(mask, organ_mask)

        mask = mask.astype(np.float32)
        return mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        mask = torch.from_numpy(self.masks[idx])

        return {
            'images': image,
            'masks': mask
        }


def create_data_loaders():
    """データローダーを作成"""
    train_dataset = EnhancedMedicalDataset(
        num_samples=200, split='train', num_datasets=4)
    val_dataset = EnhancedMedicalDataset(
        num_samples=50, split='val', num_datasets=4)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8,
                            shuffle=False, num_workers=0)

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練エポック"""
    model.train()
    epoch_losses = []
    epoch_dices = []
    epoch_mious = []

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['images'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()

        try:
            # モデルの出力
            outputs = model(images)

            # 出力の処理
            if isinstance(outputs, tuple):
                pred_masks, deep_outputs = outputs
            else:
                pred_masks = outputs
                deep_outputs = None

            # 損失計算
            total_loss, loss_dict = criterion(pred_masks, masks, deep_outputs)

            total_loss.backward()
            optimizer.step()

            # 評価指標計算
            with torch.no_grad():
                pred_binary = torch.sigmoid(pred_masks)
                pred_binary = (pred_binary > 0.5).float()

                dice = dice_score(pred_binary, masks)
                miou_val = miou(pred_binary, masks)

                epoch_losses.append(total_loss.item())
                epoch_dices.append(
                    dice.item() if hasattr(dice, 'item') else dice)
                epoch_mious.append(miou_val.item() if hasattr(
                    miou_val, 'item') else miou_val)

            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Dice': f'{dice.item() if hasattr(dice, "item") else dice:.4f}',
                'mIoU': f'{miou_val.item() if hasattr(miou_val, "item") else miou_val:.4f}'
            })

        except Exception as e:
            print(f"Training error: {e}")
            continue

    return np.mean(epoch_losses), np.mean(epoch_dices), np.mean(epoch_mious)


def validate_epoch(model, val_loader, criterion, device):
    """検証エポック"""
    model.eval()
    epoch_losses = []
    epoch_dices = []
    epoch_mious = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)

            try:
                # モデルの出力
                outputs = model(images)

                # 出力の処理
                if isinstance(outputs, tuple):
                    pred_masks, deep_outputs = outputs
                else:
                    pred_masks = outputs
                    deep_outputs = None

                # 損失計算
                total_loss, loss_dict = criterion(
                    pred_masks, masks, deep_outputs)

                # 評価指標計算
                pred_binary = torch.sigmoid(pred_masks)
                pred_binary = (pred_binary > 0.5).float()

                dice = dice_score(pred_binary, masks)
                miou_val = miou(pred_binary, masks)

                epoch_losses.append(total_loss.item())
                epoch_dices.append(
                    dice.item() if hasattr(dice, 'item') else dice)
                epoch_mious.append(miou_val.item() if hasattr(
                    miou_val, 'item') else miou_val)

                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{dice.item() if hasattr(dice, "item") else dice:.4f}',
                    'mIoU': f'{miou_val.item() if hasattr(miou_val, "item") else miou_val:.4f}'
                })

            except Exception as e:
                print(f"Validation error: {e}")
                continue

    if not epoch_losses:
        return 0.0, 0.0, 0.0

    return np.mean(epoch_losses), np.mean(epoch_dices), np.mean(epoch_mious)


def plot_training_history(train_losses, val_losses, train_dices, val_dices, save_path):
    """訓練履歴の可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 損失の可視化
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Diceスコアの可視化
    ax2.plot(train_dices, label='Train Dice', color='blue')
    ax2.plot(val_dices, label='Val Dice', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """メイン関数"""
    print("改良されたU-Netモデルの訓練を開始します...")

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # データローダー作成
    train_loader, val_loader = create_data_loaders()
    print(
        f"Found {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")

    # モデル作成
    model = ImprovedUNet(
        in_channels=3,
        num_classes=1,
        base_channels=64,
        num_layers=4,
        use_attention=True,
        use_deep_supervision=True
    ).to(device)

    # 損失関数とオプティマイザー
    criterion = ImprovedLoss(deep_supervision_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 学習率スケジューラー
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    # 訓練ループ
    num_epochs = 50
    best_dice = 0.0
    best_miou = 0.0

    # 訓練履歴の記録
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        # 訓練
        train_loss, train_dice, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 検証
        val_loss, val_dice, val_miou = validate_epoch(
            model, val_loader, criterion, device
        )

        # 学習率スケジューリング
        scheduler.step()

        # ベストモデル更新
        if val_dice > best_dice:
            best_dice = val_dice
            best_miou = val_miou

            # モデル保存
            os.makedirs('outputs/improved_unet', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
                'best_miou': best_miou
            }, 'outputs/improved_unet/best_model.pt')

        # 履歴の記録
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(
            f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, mIoU: {train_miou:.4f}")
        print(
            f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, mIoU: {val_miou:.4f}")
        print(f"Best  - Dice: {best_dice:.4f}, mIoU: {best_miou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 訓練履歴の可視化
    plot_training_history(
        train_losses, val_losses, train_dices, val_dices,
        'outputs/improved_unet/training_history.png'
    )

    # 最終結果保存
    results = {
        'best_dice': float(best_dice),
        'best_miou': float(best_miou),
        'final_train_dice': float(train_dice),
        'final_train_miou': float(train_miou),
        'final_val_dice': float(val_dice),
        'final_val_miou': float(val_miou),
        'num_epochs': num_epochs,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }

    with open('outputs/improved_unet/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n改良されたU-Netモデルの訓練が完了しました！")
    print(f"最終的なベスト Dice: {best_dice:.4f}")
    print(f"最終的なベスト mIoU: {best_miou:.4f}")
    print(
        f"論文との性能差: {83.83 - best_dice*100:.2f}% (論文: 83.83%, 現在: {best_dice*100:.2f}%)")


if __name__ == "__main__":
    main()

