#!/usr/bin/env python3
"""
PatchMoEの成功要因を参考にした高精度セグメンテーションモデル
論文の設計思想を段階的に実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class PatchPositionEmbedding(nn.Module):
    """パッチ位置埋め込み（論文の3D座標埋め込み）"""

    def __init__(self, embed_dim: int = 256, max_datasets: int = 8, max_images: int = 1000, max_patches: int = 100):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_datasets = max_datasets
        self.max_images = max_images
        self.max_patches = max_patches

        # 3D座標埋め込み（dataset_id, image_id, patch_id）
        self.dataset_embed = nn.Embedding(max_datasets, embed_dim)
        self.image_embed = nn.Embedding(max_images, embed_dim)
        self.patch_embed = nn.Embedding(max_patches, embed_dim)

        # 位置情報の統合
        self.position_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, dataset_ids: torch.Tensor, image_ids: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dataset_ids: [B, L] - バッチ内の各パッチのデータセットID
            image_ids: [B, L] - バッチ内の各パッチの画像ID  
            patch_ids: [B, L] - バッチ内の各パッチのパッチID
        Returns:
            position_embeddings: [B, L, embed_dim]
        """
        B, L = dataset_ids.shape

        # 各座標の埋め込み
        dataset_emb = self.dataset_embed(dataset_ids)  # [B, L, embed_dim]
        image_emb = self.image_embed(image_ids)        # [B, L, embed_dim]
        patch_emb = self.patch_embed(patch_ids)        # [B, L, embed_dim]

        # 3D座標の統合
        # [B, L, embed_dim*3]
        combined = torch.cat([dataset_emb, image_emb, patch_emb], dim=-1)
        position_emb = self.position_mlp(combined)  # [B, L, embed_dim]

        return position_emb


class ExpertNetwork(nn.Module):
    """専門家ネットワーク（MoEのExpert）"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GateNetwork(nn.Module):
    """ゲートネットワーク（Expert選択）"""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            expert_weights: [B, L, num_experts] - 各Expertの重み
            expert_indices: [B, L, top_k] - 選択されたExpertのインデックス
        """
        logits = self.gate(x)  # [B, L, num_experts]

        # Top-K選択
        top_k_weights, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        return top_k_weights, top_k_indices


class MoELayer(nn.Module):
    """Mixture of Experts層"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 複数のExpert
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

        # ゲートネットワーク
        self.gate = GateNetwork(input_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, input_dim] - 入力特徴
        Returns:
            output: [B, L, output_dim] - 出力特徴
            load_balancing_loss: ロードバランシング損失
        """
        B, L, _ = x.shape

        # ゲートによるExpert選択
        expert_weights, expert_indices = self.gate(x)  # [B, L, top_k]

        # 各Expertの出力を計算
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # [B, L, output_dim]
        # [B, L, num_experts, output_dim]
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # 選択されたExpertの出力を重み付きで統合
        selected_outputs = torch.gather(
            expert_outputs,
            dim=2,
            index=expert_indices.unsqueeze(-1).expand(-1, -
                                                      1, -1, expert_outputs.size(-1))
        )  # [B, L, top_k, output_dim]

        # 重み付き平均
        weighted_output = torch.sum(
            selected_outputs * expert_weights.unsqueeze(-1),
            dim=2
        )  # [B, L, output_dim]

        # ロードバランシング損失の計算
        expert_usage = expert_weights.mean(
            dim=1)  # [B, top_k] - バッチ内でのExpert使用率
        load_balancing_loss = self._compute_load_balancing_loss(expert_usage)

        return weighted_output, load_balancing_loss

    def _compute_load_balancing_loss(self, expert_usage: torch.Tensor) -> torch.Tensor:
        """ロードバランシング損失の計算"""
        # 各Expertの使用率の分散を最小化
        expert_usage_mean = expert_usage.mean()
        load_balancing_loss = torch.var(expert_usage, dim=-1).mean()
        return load_balancing_loss


class ContrastiveLoss(nn.Module):
    """コントラスト損失（異ドメイン間の特徴分離）"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, dataset_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, L, D] - パッチ特徴
            dataset_ids: [B, L] - データセットID
        """
        B, L, D = features.shape

        # 特徴を正規化
        features_norm = F.normalize(features, dim=-1)

        # 同じデータセット内のパッチを正例、異なるデータセットのパッチを負例とする
        contrastive_loss = 0.0
        num_pairs = 0

        for i in range(B):
            for j in range(L):
                anchor_feature = features_norm[i, j]  # [D]
                anchor_dataset = dataset_ids[i, j]

                # 正例と負例の計算
                positive_mask = (dataset_ids[i] == anchor_dataset)
                negative_mask = (dataset_ids[i] != anchor_dataset)

                if positive_mask.sum() > 1 and negative_mask.sum() > 0:
                    # 正例との類似度
                    positive_sim = torch.mm(
                        anchor_feature.unsqueeze(0),
                        features_norm[i, positive_mask].T
                    ) / self.temperature

                    # 負例との類似度
                    negative_sim = torch.mm(
                        anchor_feature.unsqueeze(0),
                        features_norm[i, negative_mask].T
                    ) / self.temperature

                    # InfoNCE損失
                    all_sim = torch.cat([positive_sim, negative_sim], dim=1)
                    labels = torch.zeros(
                        1, dtype=torch.long, device=features.device)

                    contrastive_loss += F.cross_entropy(all_sim, labels)
                    num_pairs += 1

        return contrastive_loss / max(num_pairs, 1)


class EnhancedPatchMoE(nn.Module):
    """改良されたPatchMoEモデル"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        patch_size: int = 64,
        embed_dim: int = 256,
        num_experts: int = 8,
        num_layers: int = 6,
        num_heads: int = 8,
        max_datasets: int = 8,
        max_images: int = 1000
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # パッチ化と埋め込み
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置埋め込み
        self.position_embed = PatchPositionEmbedding(
            embed_dim, max_datasets, max_images, 100)

        # Transformerエンコーダー
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        # MoE層
        self.moe_layer = MoELayer(
            embed_dim, embed_dim * 2, embed_dim, num_experts)

        # セグメンテーションヘッド
        self.segmentation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 損失関数
        self.contrastive_loss = ContrastiveLoss()

    def forward(
        self,
        images: torch.Tensor,
        dataset_ids: torch.Tensor,
        image_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W] - 入力画像
            dataset_ids: [B, L] - データセットID
            image_ids: [B, L] - 画像ID
        Returns:
            segmentation_masks: [B, num_classes, H, W] - セグメンテーション結果
            contrastive_loss: コントラスト損失
            load_balancing_loss: ロードバランシング損失
        """
        B, C, H, W = images.shape

        # パッチ化
        # [B, embed_dim, H//patch_size, W//patch_size]
        patches = self.patch_embed(images)
        patches = patches.flatten(2).transpose(1, 2)  # [B, L, embed_dim]

        # パッチIDの生成
        L = patches.size(1)
        patch_ids = torch.arange(
            L, device=images.device).unsqueeze(0).expand(B, -1)

        # 位置埋め込み
        position_emb = self.position_embed(dataset_ids, image_ids, patch_ids)
        patches = patches + position_emb

        # Transformerエンコーダー
        encoded_patches = self.transformer_encoder(patches)

        # MoE層
        moe_output, load_balancing_loss = self.moe_layer(encoded_patches)

        # セグメンテーションヘッド
        segmentation_logits = self.segmentation_head(
            moe_output)  # [B, L, num_classes]

        # 元の画像サイズに復元
        segmentation_logits = segmentation_logits.transpose(
            1, 2)  # [B, num_classes, L]
        segmentation_logits = segmentation_logits.view(
            B, self.num_classes, H//self.patch_size, W//self.patch_size)
        segmentation_logits = F.interpolate(segmentation_logits, size=(
            H, W), mode='bilinear', align_corners=False)

        # コントラスト損失の計算
        contrastive_loss = self.contrastive_loss(encoded_patches, dataset_ids)

        return segmentation_logits, contrastive_loss, load_balancing_loss


class EnhancedLoss(nn.Module):
    """改良された損失関数"""

    def __init__(self, contrastive_weight: float = 0.1, load_balancing_weight: float = 0.01):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.load_balancing_weight = load_balancing_weight

        # セグメンテーション損失
        self.segmentation_loss = nn.BCEWithLogitsLoss()

        # Dice損失
        self.dice_loss = self._dice_loss

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice損失の計算"""
        smooth = 1e-5
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        contrastive_loss: torch.Tensor,
        load_balancing_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: [B, C, H, W] - 予測結果
            target: [B, C, H, W] - 正解マスク
            contrastive_loss: コントラスト損失
            load_balancing_loss: ロードバランシング損失
        """
        # セグメンテーション損失
        bce_loss = self.segmentation_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        segmentation_loss = bce_loss + dice_loss

        # 総損失
        total_loss = (
            segmentation_loss +
            self.contrastive_weight * contrastive_loss +
            self.load_balancing_weight * load_balancing_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'segmentation_loss': segmentation_loss.item(),
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'load_balancing_loss': load_balancing_loss.item()
        }

        return total_loss, loss_dict

