#!/usr/bin/env python3
"""
PatchMoE高度技術実装
- Pareto最適化
- 転移学習
- ドメイン適応
- 知識蒸留
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools


class ParetoOptimizer:
    """
    Pareto最適化による複数目的最適化
    複数の損失（分類、セグメンテーション、対照学習）のバランスを自動調整
    """
    
    def __init__(self, num_objectives: int = 3, alpha: float = 0.1, 
                 beta: float = 0.1, gamma: float = 0.1):
        self.num_objectives = num_objectives
        self.alpha = alpha  # 学習率
        self.beta = beta    # 重み更新率
        self.gamma = gamma  # 正則化係数
        
        # 重みを動的に調整（デバイスは後で設定）
        self.weights = torch.ones(num_objectives) / num_objectives
        self.gradients = torch.zeros(num_objectives)
        
    def update_weights(self, losses: torch.Tensor) -> torch.Tensor:
        """損失に基づいて重みを動的に更新"""
        # 損失の勾配を計算
        current_grads = torch.autograd.grad(
            losses.sum(), losses, retain_graph=True, allow_unused=True
        )[0]
        
        if current_grads is not None:
            # デバイスを統一
            if self.gradients.device != current_grads.device:
                self.gradients = self.gradients.to(current_grads.device)
            
            self.gradients = self.beta * self.gradients + (1 - self.beta) * current_grads
            
            # 重みを更新（勾配の大きさに反比例）
            grad_norms = torch.norm(self.gradients, dim=0)
            grad_norms = torch.clamp(grad_norms, min=1e-8)
            
            new_weights = 1.0 / grad_norms
            new_weights = new_weights / new_weights.sum()
            
            # デバイスを統一
            if self.weights.device != new_weights.device:
                self.weights = self.weights.to(new_weights.device)
            
            # スムーズな更新
            self.weights = self.alpha * new_weights + (1 - self.alpha) * self.weights
            
        return self.weights


class DomainAdaptationModule(nn.Module):
    """
    ドメイン適応モジュール
    異なるデータセット間の特徴を統一
    """
    
    def __init__(self, feat_dim: int = 128, num_domains: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_domains = num_domains
        
        # ドメイン識別器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, num_domains)
        )
        
        # ドメイン不変特徴抽出器
        self.feature_extractor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # グラデーション反転層
        self.grl = GradientReversalLayer(alpha=1.0)
        
    def forward(self, features: torch.Tensor, domain_labels: torch.Tensor = None):
        """
        Args:
            features: [B, L, D] 入力特徴
            domain_labels: [B, L] ドメインラベル
        """
        # ドメイン不変特徴を抽出
        domain_invariant = self.feature_extractor(features)
        
        # ドメイン分類（グラデーション反転付き）
        domain_logits = self.domain_classifier(self.grl(domain_invariant))
        
        return domain_invariant, domain_logits


class GradientReversalFunction(torch.autograd.Function):
    """グラデーション反転関数"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """グラデーション反転層"""
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class KnowledgeDistillationModule(nn.Module):
    """
    知識蒸留モジュール
    教師モデルから学生モデルへ知識を転移
    """
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # 教師モデルを凍結
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        # 教師モデルの予測
        with torch.no_grad():
            teacher_logits, teacher_masks = self.teacher_model(*inputs)
            
        # 学生モデルの予測
        student_logits, student_masks = self.student_model(*inputs)
        
        # 知識蒸留損失
        kd_loss = self.knowledge_distillation_loss(
            student_logits, teacher_logits, self.temperature
        )
        
        # マスク蒸留損失
        mask_kd_loss = F.mse_loss(student_masks, teacher_masks)
        
        return student_logits, student_masks, kd_loss, mask_kd_loss
    
    def knowledge_distillation_loss(self, student_logits: torch.Tensor, 
                                   teacher_logits: torch.Tensor, 
                                   temperature: float) -> torch.Tensor:
        """知識蒸留損失"""
        # ソフトマックスで温度を適用
        student_soft = F.softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KLダイバージェンス損失
        kd_loss = F.kl_div(
            student_soft.log(), teacher_soft, reduction='batchmean'
        ) * (temperature ** 2)
        
        return kd_loss


class TransferLearningModule(nn.Module):
    """
    転移学習モジュール
    事前学習済みモデルから特定タスクへ転移
    """
    
    def __init__(self, pretrained_model: nn.Module, num_classes: int = 6,
                 freeze_backbone: bool = True):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        
        if freeze_backbone:
            # バックボーンを凍結
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
                
        # タスク固有のヘッド
        feat_dim = getattr(pretrained_model, 'feat_dim', 128)
        self.task_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, num_classes)
        )
        
        # 適応層
        self.adaptation_layer = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, dataset_ids: torch.Tensor, 
                image_ids: torch.Tensor):
        # 事前学習済み特徴を抽出
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.pretrained_model(x, dataset_ids, image_ids)
            
        # タスク適応
        adapted_features = self.adaptation_layer(features)
        
        # タスク固有の予測
        task_logits = self.task_head(adapted_features)
        
        return task_logits


class AdvancedPatchMoE(nn.Module):
    """
    高度技術を統合したPatchMoE
    """
    
    def __init__(self, base_model: nn.Module, num_domains: int = 4,
                 use_domain_adaptation: bool = True,
                 use_knowledge_distillation: bool = False,
                 teacher_model: nn.Module = None):
        super().__init__()
        self.base_model = base_model
        self.num_domains = num_domains
        self.use_domain_adaptation = use_domain_adaptation
        self.use_knowledge_distillation = use_knowledge_distillation
        
        # ドメイン適応モジュール
        if use_domain_adaptation:
            self.domain_adaptation = DomainAdaptationModule(
                feat_dim=128, num_domains=num_domains
            )
            
        # 知識蒸留モジュール
        if use_knowledge_distillation and teacher_model is not None:
            self.knowledge_distillation = KnowledgeDistillationModule(
                teacher_model, base_model
            )
            
        # Pareto最適化器
        self.pareto_optimizer = ParetoOptimizer(num_objectives=3)
        
    def forward(self, images: torch.Tensor, dataset_ids: torch.Tensor, 
                image_ids: torch.Tensor, domain_labels: torch.Tensor = None):
        
        # ベースモデルの予測
        logits, masks = self.base_model(images, dataset_ids, image_ids)
        
        losses = {}
        
        # ドメイン適応
        if self.use_domain_adaptation:
            # PPE特徴を取得（簡略化）
            B, L, C = logits.shape
            patch_ids = torch.arange(L, device=images.device).unsqueeze(0).repeat(B, 1)
            dataset_ids_expanded = dataset_ids[:, 0:1].repeat(1, L)
            image_ids_expanded = image_ids[:, 0:1].repeat(1, L)
            coords = torch.stack([dataset_ids_expanded, image_ids_expanded, patch_ids], dim=-1)
            ppe_features = self.base_model.ppe(coords)
            
            # ドメイン適応
            domain_invariant, domain_logits = self.domain_adaptation(
                ppe_features, domain_labels
            )
            
            # ドメイン分類損失
            if domain_labels is not None:
                # ドメインラベルを適切なサイズに調整
                domain_labels_expanded = domain_labels[:, 0:1].repeat(1, domain_logits.size(1))
                domain_loss = F.cross_entropy(
                    domain_logits.view(-1, self.num_domains),
                    domain_labels_expanded.view(-1)
                )
                losses['domain_loss'] = domain_loss
                
        # 知識蒸留
        if self.use_knowledge_distillation:
            kd_loss, mask_kd_loss = self.knowledge_distillation(
                (images, dataset_ids, image_ids)
            )
            losses['kd_loss'] = kd_loss
            losses['mask_kd_loss'] = mask_kd_loss
            
        return logits, masks, losses


class MultiTaskLearningModule(nn.Module):
    """
    マルチタスク学習モジュール
    複数の関連タスクを同時に学習
    """
    
    def __init__(self, shared_backbone: nn.Module, tasks: List[str],
                 task_heads: Dict[str, nn.Module]):
        super().__init__()
        self.shared_backbone = shared_backbone
        self.tasks = tasks
        self.task_heads = nn.ModuleDict(task_heads)
        
        # タスク重み（動的に調整）
        self.task_weights = nn.Parameter(torch.ones(len(tasks)))
        
    def forward(self, x: torch.Tensor, dataset_ids: torch.Tensor, 
                image_ids: torch.Tensor, task_targets: Dict[str, torch.Tensor]):
        
        # 共有特徴を抽出
        shared_features = self.shared_backbone(x, dataset_ids, image_ids)
        
        # 各タスクの予測
        task_outputs = {}
        task_losses = {}
        
        for task in self.tasks:
            if task in self.task_heads:
                task_output = self.task_heads[task](shared_features)
                task_outputs[task] = task_output
                
                # タスク固有の損失計算
                if task in task_targets:
                    if task == 'segmentation':
                        task_losses[task] = F.cross_entropy(
                            task_output, task_targets[task]
                        )
                    elif task == 'classification':
                        task_losses[task] = F.binary_cross_entropy_with_logits(
                            task_output, task_targets[task]
                        )
                        
        return task_outputs, task_losses


def create_advanced_patchmoe(base_config: Dict, advanced_config: Dict) -> AdvancedPatchMoE:
    """高度技術を統合したPatchMoEを作成"""
    from PatchMoE.model import PatchMoEModel
    
    # ベースモデルを作成
    base_model = PatchMoEModel(
        in_ch=base_config['in_ch'],
        feat_dim=base_config['feat_dim'],
        grid_h=base_config['grid_h'],
        grid_w=base_config['grid_w'],
        num_datasets=base_config['num_datasets'],
        num_images=base_config['num_images_cap'],
        num_classes=base_config['num_classes'],
        num_layers=base_config['num_layers'],
        num_heads=base_config['num_heads'],
        num_queries=base_config['num_queries'],
        gate_top_k=base_config['top_k'],
        gate_capacity=base_config['capacity_factor'],
        gate_noise=base_config['gate_noise'],
        experts_per_device=base_config['experts_per_device'],
        backbone='resnet50',
        pretrained_backbone=True,
        use_multiscale=True
    )
    
    # 高度技術を統合
    advanced_model = AdvancedPatchMoE(
        base_model=base_model,
        num_domains=advanced_config.get('num_domains', 4),
        use_domain_adaptation=advanced_config.get('use_domain_adaptation', True),
        use_knowledge_distillation=advanced_config.get('use_knowledge_distillation', False)
    )
    
    return advanced_model
