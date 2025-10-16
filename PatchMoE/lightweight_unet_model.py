#!/usr/bin/env python3
"""
軽量で実用的なU-Netモデル
論文の成功要因を参考にした効率的な実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class LightweightUNet(nn.Module):
    """軽量U-Netモデル"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,  # より小さなベースチャンネル
        num_layers: int = 3,       # より少ない層数
        use_attention: bool = False,  # アテンションを無効化
        use_deep_supervision: bool = False  # 深層監視を無効化
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        # エンコーダー
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # 最初の層
        self.encoder.append(self._make_layer(in_channels, base_channels))

        # エンコーダー層
        for i in range(num_layers):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.encoder.append(self._make_layer(in_ch, out_ch))

        # ボトルネック
        self.bottleneck = self._make_layer(
            base_channels * (2 ** num_layers),
            base_channels * (2 ** num_layers)
        )

        # デコーダー
        self.decoder = nn.ModuleList()
        for i in range(num_layers, 0, -1):
            # スキップ接続とアップサンプリング後の特徴を結合するため、チャンネル数を調整
            in_ch = base_channels * \
                (2 ** i) + base_channels * (2 ** (i - 1))  # アップサンプル後 + スキップ接続
            out_ch = base_channels * (2 ** (i - 1))
            self.decoder.append(self._make_layer(in_ch, out_ch))

        # 最終層
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """シンプルな畳み込み層を作成"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

        return output


class LightweightLoss(nn.Module):
    """軽量損失関数"""

    def __init__(self):
        super().__init__()

        # セグメンテーション損失
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = self._dice_loss

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice損失の計算"""
        smooth = 1e-5
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: [B, C, H, W] - 予測結果
            target: [B, C, H, W] - 正解マスク
        """
        # メイン損失
        bce_loss = self.bce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)

        total_loss = bce_loss + dice_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item()
        }

        return total_loss, loss_dict
