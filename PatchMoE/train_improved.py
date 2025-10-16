#!/usr/bin/env python3
"""
改良されたPatchMoE学習スクリプト
- 改良された損失関数
- 対照学習の統合
- より安定した学習
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Any

from PatchMoE.model import PatchMoEModel
from PatchMoE.medical_dataset import build_medical_loader
from PatchMoE.losses import PatchMoECombinedLoss
from PatchMoE.contrastive import AdvancedPatchContrastiveLoss, DomainAdaptiveContrastiveLoss


class ImprovedPatchMoETrainer:
    """改良されたPatchMoE学習クラス"""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda:0'):
        self.config = config
        self.device = device
        
        # モデル初期化
        self.model = PatchMoEModel(
            in_ch=3,
            feat_dim=config['feat_dim'],
            grid_h=config['grid_h'],
            grid_w=config['grid_w'],
            num_datasets=config['num_datasets'],
            num_images=config['num_images_cap'],
            num_classes=config['num_classes'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            num_queries=config['num_queries'],
            gate_top_k=config['top_k'],
            gate_capacity=config['capacity_factor'],
            gate_noise=config['gate_noise'],
            experts_per_device=config['experts_per_device'],
            backbone='resnet50',
            pretrained_backbone=True,
            use_multiscale=True
        ).to(device)
        
        # 損失関数
        self.criterion = PatchMoECombinedLoss(
            num_classes=config['num_classes'],
            dice_weight=1.0,
            focal_weight=1.0,
            contrastive_weight=0.1,
            moe_weight=0.01
        )
        
        # 対照学習損失
        self.contrastive_criterion = AdvancedPatchContrastiveLoss(
            temperature=0.07,
            hard_negative_weight=2.0,
            domain_separation_weight=1.5
        )
        
        # オプティマイザー
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'], eta_min=1e-6
        )
        
        # データローダー
        self.train_loader = build_medical_loader(
            batch_size=config['batch_size'],
            length=1000,
            image_size=512,
            num_classes=config['num_classes'],
            num_datasets=config['num_datasets'],
            grid_h=config['grid_h'],
            grid_w=config['grid_w'],
            num_workers=0
        )
        
        # ログ設定
        os.makedirs(config['out_dir'], exist_ok=True)
        self.writer = SummaryWriter(config['out_dir'])
        
        # メトリクス追跡
        self.best_dice = 0.0
        self.best_miou = 0.0
        
    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor, 
                       mask_logits: torch.Tensor, mask_targets: torch.Tensor) -> Dict[str, float]:
        """メトリクス計算"""
        # Dice係数
        probs = torch.softmax(mask_logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        
        # マルチクラスDice
        dice_scores = []
        for c in range(mask_logits.size(1)):
            pred_c = (pred == c).float()
            target_c = (mask_targets == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice.item())
            
        dice = np.mean(dice_scores)
        
        # mIoU
        iou_scores = []
        for c in range(mask_logits.size(1)):
            pred_c = (pred == c).float()
            target_c = (mask_targets == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            iou = intersection / (union + 1e-6)
            iou_scores.append(iou.item())
            
        miou = np.mean(iou_scores)
        
        return {'dice': dice, 'miou': miou}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        total_metrics = {'dice': 0.0, 'miou': 0.0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, dataset_ids, image_ids, masks, patch_classes) in enumerate(pbar):
            images = images.to(self.device)
            dataset_ids = dataset_ids.to(self.device)
            image_ids = image_ids.to(self.device)
            masks = masks.to(self.device)
            patch_classes = patch_classes.to(self.device)
            
            self.optimizer.zero_grad()
            
            # フォワードパス
            logits, mask_logits = self.model(images, dataset_ids, image_ids)
            
            # 対照学習のための特徴抽出（PPEの出力を使用）
            with torch.no_grad():
                # PPE特徴を取得（簡略化）
                B, L, C = logits.shape
                patch_ids = torch.arange(L, device=self.device).unsqueeze(0).repeat(B, 1)
                
                # データセットIDと画像IDをパッチ数に合わせて拡張
                dataset_ids_expanded = dataset_ids[:, 0:1].repeat(1, L)
                image_ids_expanded = image_ids[:, 0:1].repeat(1, L)
                
                coords = torch.stack([dataset_ids_expanded, image_ids_expanded, patch_ids], dim=-1)
                ppe_features = self.model.ppe(coords)
            
            # 対照学習損失
            contrastive_loss = self.contrastive_criterion(
                ppe_features, dataset_ids[:, 0:1].repeat(1, L), 
                image_ids[:, 0:1].repeat(1, L)
            )
            
            # MoE負荷分散損失
            moe_aux_loss = torch.tensor(0.0, device=self.device)
            
            # マスクを適切なサイズにリサイズ
            target_masks = F.interpolate(masks, size=mask_logits.shape[2:], mode='nearest').squeeze(1)
            
            # 統合損失計算
            loss_dict = self.criterion(
                logits, mask_logits, patch_classes, target_masks,
                contrastive_loss, moe_aux_loss
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            
            # グラデーションクリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # メトリクス計算
            with torch.no_grad():
                metrics = self.compute_metrics(logits, patch_classes, mask_logits, target_masks)
                total_metrics['dice'] += metrics['dice']
                total_metrics['miou'] += metrics['miou']
            
            total_loss += loss.item()
            num_batches += 1
            
            # プログレスバー更新
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.3f}',
                'mIoU': f'{metrics["miou"]:.3f}',
                'Contrastive': f'{contrastive_loss.item():.4f}'
            })
            
            # ログ記録
            if batch_idx % self.config['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Total', loss.item(), global_step)
                self.writer.add_scalar('Loss/Classification', loss_dict['cls_loss'].item(), global_step)
                self.writer.add_scalar('Loss/Segmentation', loss_dict['seg_loss'].item(), global_step)
                self.writer.add_scalar('Loss/Contrastive', contrastive_loss.item(), global_step)
                self.writer.add_scalar('Metrics/Dice', metrics['dice'], global_step)
                self.writer.add_scalar('Metrics/mIoU', metrics['miou'], global_step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # 平均メトリクス
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss, **avg_metrics}
    
    def train(self):
        """学習実行"""
        print(f"🚀 改良されたPatchMoE学習を開始")
        print(f"📊 設定: {json.dumps(self.config, indent=2)}")
        
        for epoch in range(self.config['epochs']):
            # 学習
            train_metrics = self.train_epoch(epoch)
            
            # スケジューラー更新
            self.scheduler.step()
            
            # エポック結果表示
            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Dice: {train_metrics['dice']:.3f}")
            print(f"  mIoU: {train_metrics['miou']:.3f}")
            
            # ベストモデル保存
            if train_metrics['dice'] > self.best_dice:
                self.best_dice = train_metrics['dice']
                self.best_miou = train_metrics['miou']
                
                # チェックポイント保存
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_dice': self.best_dice,
                    'best_miou': self.best_miou,
                    'config': self.config
                }
                
                torch.save(checkpoint, os.path.join(self.config['out_dir'], 'best_model.pt'))
                
                # メトリクス保存
                with open(os.path.join(self.config['out_dir'], 'best_metrics.json'), 'w') as f:
                    json.dump({'dice': self.best_dice, 'miou': self.best_miou}, f, indent=2)
                
                print(f"  ✅ 新しいベストモデル保存 (Dice: {self.best_dice:.3f})")
            
            print("-" * 50)
        
        print(f"🎯 学習完了! ベスト結果:")
        print(f"  Dice: {self.best_dice:.3f}")
        print(f"  mIoU: {self.best_miou:.3f}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='改良されたPatchMoE学習')
    parser.add_argument('--config', type=str, default=None, help='設定ファイルパス')
    parser.add_argument('--output_dir', type=str, default='/workspace/outputs/patchmoe_improved', help='出力ディレクトリ')
    parser.add_argument('--epochs', type=int, default=10, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    args = parser.parse_args()
    
    # 設定
    config = {
        'feat_dim': 128,
        'grid_h': 16,
        'grid_w': 16,
        'num_classes': 6,
        'num_datasets': 4,
        'num_images_cap': 100000,
        'num_queries': 25,
        'num_layers': 8,
        'num_heads': 8,
        'top_k': 2,
        'capacity_factor': 1.0,
        'gate_noise': 1.0,
        'experts_per_device': 4,
        'lr': args.lr,
        'weight_decay': 1e-2,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'out_dir': args.output_dir,
        'log_interval': 10
    }
    
    # 学習実行
    trainer = ImprovedPatchMoETrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
