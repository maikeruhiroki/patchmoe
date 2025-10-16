#!/usr/bin/env python3
"""
高度技術を統合したPatchMoE学習スクリプト
- Pareto最適化
- ドメイン適応
- 転移学習
- マルチタスク学習
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

from PatchMoE.advanced_techniques import (
    AdvancedPatchMoE, ParetoOptimizer, DomainAdaptationModule,
    create_advanced_patchmoe
)
from PatchMoE.medical_dataset import build_medical_loader
from PatchMoE.losses import PatchMoECombinedLoss
from PatchMoE.contrastive import AdvancedPatchContrastiveLoss


class AdvancedPatchMoETrainer:
    """高度技術を統合したPatchMoE学習クラス"""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda:0'):
        self.config = config
        self.device = device
        
        # 高度技術統合モデルを作成
        self.model = create_advanced_patchmoe(
            base_config=config,
            advanced_config=config.get('advanced', {})
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
        
        # Pareto最適化器
        self.pareto_optimizer = ParetoOptimizer(num_objectives=4)
        
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
        
        # Pareto最適化のための損失追跡
        pareto_losses = []
        
        pbar = tqdm(self.train_loader, desc=f'Advanced Epoch {epoch}')
        for batch_idx, (images, dataset_ids, image_ids, masks, patch_classes) in enumerate(pbar):
            images = images.to(self.device)
            dataset_ids = dataset_ids.to(self.device)
            image_ids = image_ids.to(self.device)
            masks = masks.to(self.device)
            patch_classes = patch_classes.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 高度技術統合モデルのフォワードパス
            logits, mask_logits, additional_losses = self.model(
                images, dataset_ids, image_ids, dataset_ids[:, 0:1]
            )
            
            # マスクを適切なサイズにリサイズ
            target_masks = F.interpolate(masks, size=mask_logits.shape[2:], mode='nearest').squeeze(1)
            
            # 対照学習のための特徴抽出
            with torch.no_grad():
                B, L, C = logits.shape
                patch_ids = torch.arange(L, device=self.device).unsqueeze(0).repeat(B, 1)
                dataset_ids_expanded = dataset_ids[:, 0:1].repeat(1, L)
                image_ids_expanded = image_ids[:, 0:1].repeat(1, L)
                coords = torch.stack([dataset_ids_expanded, image_ids_expanded, patch_ids], dim=-1)
                ppe_features = self.model.base_model.ppe(coords)
            
            # 対照学習損失
            contrastive_loss = self.contrastive_criterion(
                ppe_features, dataset_ids_expanded, image_ids_expanded
            )
            
            # 統合損失計算
            loss_dict = self.criterion(
                logits, mask_logits, patch_classes, target_masks,
                contrastive_loss, torch.tensor(0.0, device=self.device)
            )
            
            # 高度技術の損失を追加
            total_loss_tensor = loss_dict['total_loss']
            
            if 'domain_loss' in additional_losses:
                domain_loss = additional_losses['domain_loss']
                total_loss_tensor += 0.1 * domain_loss
                loss_dict['domain_loss'] = domain_loss
                
            if 'kd_loss' in additional_losses:
                kd_loss = additional_losses['kd_loss']
                total_loss_tensor += 0.1 * kd_loss
                loss_dict['kd_loss'] = kd_loss
                
            # Pareto最適化のための損失収集
            pareto_losses.append(torch.stack([
                loss_dict['cls_loss'],
                loss_dict['seg_loss'],
                contrastive_loss,
                additional_losses.get('domain_loss', torch.tensor(0.0, device=self.device))
            ]))
            
            # バックプロパゲーション
            total_loss_tensor.backward()
            
            # グラデーションクリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # メトリクス計算
            with torch.no_grad():
                metrics = self.compute_metrics(logits, patch_classes, mask_logits, target_masks)
                total_metrics['dice'] += metrics['dice']
                total_metrics['miou'] += metrics['miou']
            
            total_loss += total_loss_tensor.item()
            num_batches += 1
            
            # プログレスバー更新
            pbar.set_postfix({
                'Loss': f'{total_loss_tensor.item():.4f}',
                'Dice': f'{metrics["dice"]:.3f}',
                'mIoU': f'{metrics["miou"]:.3f}',
                'Contrastive': f'{contrastive_loss.item():.4f}',
                'Domain': f'{additional_losses.get("domain_loss", torch.tensor(0.0)).item():.4f}'
            })
            
            # ログ記録
            if batch_idx % self.config['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Total', total_loss_tensor.item(), global_step)
                self.writer.add_scalar('Loss/Classification', loss_dict['cls_loss'].item(), global_step)
                self.writer.add_scalar('Loss/Segmentation', loss_dict['seg_loss'].item(), global_step)
                self.writer.add_scalar('Loss/Contrastive', contrastive_loss.item(), global_step)
                
                if 'domain_loss' in additional_losses:
                    self.writer.add_scalar('Loss/Domain', additional_losses['domain_loss'].item(), global_step)
                if 'kd_loss' in additional_losses:
                    self.writer.add_scalar('Loss/KnowledgeDistillation', additional_losses['kd_loss'].item(), global_step)
                    
                self.writer.add_scalar('Metrics/Dice', metrics['dice'], global_step)
                self.writer.add_scalar('Metrics/mIoU', metrics['miou'], global_step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Pareto最適化による重み調整
        if len(pareto_losses) > 0:
            avg_pareto_losses = torch.stack(pareto_losses).mean(dim=0)
            optimized_weights = self.pareto_optimizer.update_weights(avg_pareto_losses)
            print(f"Pareto最適化重み: {optimized_weights.cpu().numpy()}")
        
        # 平均メトリクス
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss, **avg_metrics}
    
    def train(self):
        """学習実行"""
        print(f"🚀 高度技術統合PatchMoE学習を開始")
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
                
                torch.save(checkpoint, os.path.join(self.config['out_dir'], 'best_advanced_model.pt'))
                
                # メトリクス保存
                with open(os.path.join(self.config['out_dir'], 'best_advanced_metrics.json'), 'w') as f:
                    json.dump({'dice': self.best_dice, 'miou': self.best_miou}, f, indent=2)
                
                print(f"  ✅ 新しいベストモデル保存 (Dice: {self.best_dice:.3f})")
            
            print("-" * 50)
        
        print(f"🎯 高度技術統合学習完了! ベスト結果:")
        print(f"  Dice: {self.best_dice:.3f}")
        print(f"  mIoU: {self.best_miou:.3f}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='高度技術統合PatchMoE学習')
    parser.add_argument('--config', type=str, default=None, help='設定ファイルパス')
    parser.add_argument('--output_dir', type=str, default='/workspace/outputs/patchmoe_advanced', help='出力ディレクトリ')
    parser.add_argument('--epochs', type=int, default=5, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=2, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    parser.add_argument('--use_domain_adaptation', action='store_true', help='ドメイン適応を使用')
    parser.add_argument('--use_knowledge_distillation', action='store_true', help='知識蒸留を使用')
    args = parser.parse_args()
    
    # 設定
    config = {
        'in_ch': 3,
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
        'log_interval': 10,
        'advanced': {
            'num_domains': 4,
            'use_domain_adaptation': args.use_domain_adaptation,
            'use_knowledge_distillation': args.use_knowledge_distillation
        }
    }
    
    # 学習実行
    trainer = AdvancedPatchMoETrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
