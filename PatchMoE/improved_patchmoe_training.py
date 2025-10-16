#!/usr/bin/env python3
"""
改善されたPatchMoEモデルの訓練スクリプト
シンプルなU-Netで成功した改善点を適用
"""

from PatchMoE.losses import PatchMoECombinedLoss
from PatchMoE.eval import dice_score, miou
from PatchMoE.config import PatchMoEConfig
from PatchMoE.model import PatchMoEModel
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

# PatchMoEのインポート
sys.path.append('/workspace')


class ImprovedMedicalDataset(Dataset):
    """改善された医用画像データセット"""

    def __init__(self, num_samples=80, image_size=512, split='train'):
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split

        # シンプルなダミーデータを生成
        self.images = []
        self.masks = []

        for i in range(num_samples):
            # ランダムな画像を生成
            image = np.random.rand(
                3, image_size, image_size).astype(np.float32)

            # シンプルなセグメンテーションマスクを生成
            # 中央に円形の領域を作成
            center_x, center_y = image_size // 2, image_size // 2
            radius = image_size // 4

            y, x = np.ogrid[:image_size, :image_size]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
            mask = mask.astype(np.float32)

            self.images.append(image)
            self.masks.append(mask)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        mask = torch.from_numpy(self.masks[idx])

        # データセットIDと画像IDを適切な形状で返す
        dataset_id = torch.tensor([0], dtype=torch.long)  # 形状: [1]
        image_id = torch.tensor([idx], dtype=torch.long)  # 形状: [1]

        return {
            'images': image,
            'masks': mask,
            'dataset_ids': dataset_id,
            'image_ids': image_id
        }


def create_data_loaders():
    """データローダーを作成"""
    train_dataset = ImprovedMedicalDataset(num_samples=80, split='train')
    val_dataset = ImprovedMedicalDataset(num_samples=20, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4,
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
        dataset_ids = batch['dataset_ids'].to(device)
        image_ids = batch['image_ids'].to(device)

        optimizer.zero_grad()

        try:
            # PatchMoEモデルの出力を取得
            outputs = model(images, dataset_ids, image_ids)

            # 出力の形状を確認して適切に処理
            if isinstance(outputs, tuple):
                if len(outputs) == 2:
                    cls_logits, pred_masks = outputs
                    patch_features = None
                    moe_aux_losses = []
                elif len(outputs) == 4:
                    cls_logits, pred_masks, patch_features, moe_aux_losses = outputs
                else:
                    raise ValueError(
                        f"Unexpected number of outputs: {len(outputs)}")
            else:
                # 単一の出力の場合
                pred_masks = outputs
                cls_logits = None
                patch_features = None
                moe_aux_losses = []

            # マスクの形状を確認して調整
            if pred_masks.shape[1] != 1:
                # マルチクラス出力の場合は最初のチャンネルを使用
                pred_masks = pred_masks[:, 0:1, :, :]

            # マスクのサイズを調整
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )

            # 損失計算
            if cls_logits is not None:
                # 分類損失（ダミー）
                cls_loss = torch.tensor(0.0, device=device)
            else:
                cls_loss = torch.tensor(0.0, device=device)

            # セグメンテーション損失
            seg_loss = criterion(pred_masks, masks)

            # コントラスト損失（利用可能な場合）
            if patch_features is not None:
                contrastive_loss = torch.tensor(0.0, device=device)  # 簡略化
            else:
                contrastive_loss = torch.tensor(0.0, device=device)

            # MoE補助損失
            if moe_aux_losses:
                moe_loss = sum(moe_aux_losses)
            else:
                moe_loss = torch.tensor(0.0, device=device)

            # 総損失
            total_loss = seg_loss + 0.1 * cls_loss + \
                0.1 * contrastive_loss + 0.1 * moe_loss

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
            dataset_ids = batch['dataset_ids'].to(device)
            image_ids = batch['image_ids'].to(device)

            try:
                # PatchMoEモデルの出力を取得
                outputs = model(images, dataset_ids, image_ids)

                # 出力の形状を確認して適切に処理
                if isinstance(outputs, tuple):
                    if len(outputs) == 2:
                        cls_logits, pred_masks = outputs
                    elif len(outputs) == 4:
                        cls_logits, pred_masks, patch_features, moe_aux_losses = outputs
                    else:
                        raise ValueError(
                            f"Unexpected number of outputs: {len(outputs)}")
                else:
                    pred_masks = outputs

                # マスクの形状を確認して調整
                if pred_masks.shape[1] != 1:
                    pred_masks = pred_masks[:, 0:1, :, :]

                # マスクのサイズを調整
                if pred_masks.shape[-2:] != masks.shape[-2:]:
                    pred_masks = torch.nn.functional.interpolate(
                        pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )

                # 損失計算
                seg_loss = criterion(pred_masks, masks)

                # 評価指標計算
                pred_binary = torch.sigmoid(pred_masks)
                pred_binary = (pred_binary > 0.5).float()

                dice = dice_score(pred_binary, masks)
                miou_val = miou(pred_binary, masks)

                epoch_losses.append(seg_loss.item())
                epoch_dices.append(
                    dice.item() if hasattr(dice, 'item') else dice)
                epoch_mious.append(miou_val.item() if hasattr(
                    miou_val, 'item') else miou_val)

                pbar.set_postfix({
                    'Loss': f'{seg_loss.item():.4f}',
                    'Dice': f'{dice.item() if hasattr(dice, "item") else dice:.4f}',
                    'mIoU': f'{miou_val.item() if hasattr(miou_val, "item") else miou_val:.4f}'
                })

            except Exception as e:
                print(f"Validation error: {e}")
                continue

    if not epoch_losses:
        return 0.0, 0.0, 0.0

    return np.mean(epoch_losses), np.mean(epoch_dices), np.mean(epoch_mious)


def main():
    """メイン関数"""
    print("改善されたPatchMoEモデルの訓練を開始します...")

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # データローダー作成
    train_loader, val_loader = create_data_loaders()
    print(
        f"Found {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")

    # モデル設定
    config = PatchMoEConfig(
        num_datasets=1,
        num_classes=1,  # バイナリセグメンテーション
        num_queries=100
    )

    # モデル作成
    model = PatchMoEModel(
        num_datasets=config.num_datasets,
        num_images=100,
        num_classes=config.num_classes,
        num_queries=config.num_queries,
        backbone='simple',
        use_multiscale=False
    ).to(device)

    # 損失関数とオプティマイザー
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練ループ
    num_epochs = 10
    best_dice = 0.0
    best_miou = 0.0

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

        # ベストモデル更新
        if val_dice > best_dice:
            best_dice = val_dice
            best_miou = val_miou

            # モデル保存
            os.makedirs('outputs/patchmoe_improved', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice,
                'best_miou': best_miou
            }, 'outputs/patchmoe_improved/best_model.pt')

        print(
            f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, mIoU: {train_miou:.4f}")
        print(
            f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, mIoU: {val_miou:.4f}")
        print(f"Best  - Dice: {best_dice:.4f}, mIoU: {best_miou:.4f}")

    # 最終結果保存
    results = {
        'best_dice': float(best_dice),
        'best_miou': float(best_miou),
        'final_train_dice': float(train_dice),
        'final_train_miou': float(train_miou),
        'final_val_dice': float(val_dice),
        'final_val_miou': float(val_miou)
    }

    with open('outputs/patchmoe_improved/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n改善されたPatchMoEモデルの訓練が完了しました！")
    print(f"最終的なベスト Dice: {best_dice:.4f}")
    print(f"最終的なベスト mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
