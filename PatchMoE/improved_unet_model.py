#!/usr/bin/env python3

"""
論文の成功要因を参考にした改良U-Netモデル
PatchMoEの設計思想を段階的に実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """位置エンコーディング（論文の3D座標埋め込みの簡略版）"""

    def __init__(self, embed_dim: int, max_len: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim

        # 学習可能な位置埋め込み
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] - 入力特徴
        Returns:
            x + pos_embed: [B, L, D] - 位置埋め込みを加算した特徴
        """
        B, L, D = x.shape
        pos_embed = self.pos_embed[:, :L, :].expand(B, -1, -1)
        return x + pos_embed


class AttentionBlock(nn.Module):
    """改良されたアテンションブロック"""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # クエリ、キー、バリューの計算
        q = self.q_conv(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.k_conv(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.v_conv(x).view(B, self.num_heads, self.head_dim, H * W)

        # アテンション計算
        q = q.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        k = k.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]
        v = v.transpose(-2, -1)  # [B, num_heads, H*W, head_dim]

        # スケールドドットプロダクトアテンション
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # アテンション適用
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)

        return self.out_conv(out)


class ResidualBlock(nn.Module):
    """改良された残差ブロック"""

    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差接続のための1x1畳み込み
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # アテンション
        if use_attention:
            self.attention = AttentionBlock(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = self.attention(out)

        out += residual
        out = self.relu(out)

        return out


class ImprovedUNet(nn.Module):
    """改良されたU-Netモデル"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        num_layers: int = 4,
        use_attention: bool = True,
        use_deep_supervision: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_deep_supervision = use_deep_supervision

        # エンコーダー
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # 最初の層
        self.encoder.append(ResidualBlock(
            in_channels, base_channels, use_attention))

        # エンコーダー層
        for i in range(num_layers):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.encoder.append(ResidualBlock(in_ch, out_ch, use_attention))

        # ボトルネック
        self.bottleneck = ResidualBlock(
            base_channels * (2 ** num_layers),
            base_channels * (2 ** num_layers),
            use_attention
        )

        # デコーダー
        self.decoder = nn.ModuleList()
        for i in range(num_layers, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            self.decoder.append(ResidualBlock(in_ch, out_ch, use_attention))

        # 最終層
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)

        # 深層監視用のヘッド
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(num_layers):
                self.deep_supervision_heads.append(
                    nn.Conv2d(base_channels *
                              (2 ** (num_layers - i - 1)), num_classes, 1)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # エンコーダー
        encoder_outputs = []
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            encoder_outputs.append(x)
            if i < len(self.encoder) - 1:
                x = self.pool(x)

        # ボトルネック
        x = self.bottleneck(x)

        # デコーダー
        for i, decoder_layer in enumerate(self.decoder):
            # スキップ接続
            skip_connection = encoder_outputs[-(i + 1)]
            x = F.interpolate(
                x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_connection], dim=1)
            x = decoder_layer(x)

        # 最終出力
        output = self.final_conv(x)

        if self.use_deep_supervision and self.training:
            # 深層監視の出力も返す
            deep_outputs = []
            for i, head in enumerate(self.deep_supervision_heads):
                deep_output = head(encoder_outputs[-(i + 1)])
                deep_output = F.interpolate(
                    deep_output, size=output.shape[2:], mode='bilinear', align_corners=False)
                deep_outputs.append(deep_output)
            return output, deep_outputs

        return output


class ImprovedLoss(nn.Module):
    """改良された損失関数"""

    def __init__(self, deep_supervision_weight: float = 0.4):
        super().__init__()
        self.deep_supervision_weight = deep_supervision_weight

        # セグメンテーション損失
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = self._dice_loss
        self.focal_loss = self._focal_loss

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice損失の計算"""
        smooth = 1e-5
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal損失の計算"""
        pred = torch.sigmoid(pred)

        # バイナリクロスエントロピー
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # Focal重み
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = alpha * (1 - pt) ** gamma

        return (focal_weight * bce).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, deep_outputs: Optional[list] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: [B, C, H, W] - 予測結果
            target: [B, C, H, W] - 正解マスク
            deep_outputs: 深層監視の出力（オプション）
        """
        # メイン損失
        bce_loss = self.bce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)

        main_loss = bce_loss + dice_loss + focal_loss

        # 深層監視損失
        deep_loss = 0.0
        if deep_outputs is not None:
            for deep_output in deep_outputs:
                deep_bce = self.bce_loss(deep_output, target)
                deep_dice = self.dice_loss(deep_output, target)
                deep_focal = self.focal_loss(deep_output, target)
                deep_loss += deep_bce + deep_dice + deep_focal

        # 総損失
        total_loss = main_loss + self.deep_supervision_weight * deep_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item(),
            'deep_loss': deep_loss.item() if deep_outputs is not None else 0.0
        }

        return total_loss, loss_dict
