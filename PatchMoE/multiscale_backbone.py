import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from typing import Dict, List, Tuple


class MultiScaleResNetBackbone(nn.Module):
    """
    マルチスケール特徴抽出を行うResNet50バックボーン
    FPNスタイルで複数解像度の特徴マップを出力
    """

    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()

        # ResNet50の事前学習済みモデル
        resnet = models.resnet50(pretrained=pretrained)

        # 各層の出力を取得
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels, 1/4 scale
        self.layer2 = resnet.layer2  # 512 channels, 1/8 scale
        self.layer3 = resnet.layer3  # 1024 channels, 1/16 scale
        self.layer4 = resnet.layer4  # 2048 channels, 1/32 scale

        # 特徴マップを統一次元に投影
        self.lateral_conv1 = nn.Conv2d(256, out_dim, 1)
        self.lateral_conv2 = nn.Conv2d(512, out_dim, 1)
        self.lateral_conv3 = nn.Conv2d(1024, out_dim, 1)
        self.lateral_conv4 = nn.Conv2d(2048, out_dim, 1)

        # FPNのトップダウン接続
        self.fpn_conv1 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.fpn_conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.fpn_conv3 = nn.Conv2d(out_dim, out_dim, 3, padding=1)
        self.fpn_conv4 = nn.Conv2d(out_dim, out_dim, 3, padding=1)

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        入力: [B, 3, H, W]
        出力: {
            'c2': [B, out_dim, H/4, W/4],   # 1/4 scale
            'c3': [B, out_dim, H/8, W/8],   # 1/8 scale
            'c4': [B, out_dim, H/16, W/16], # 1/16 scale
            'c5': [B, out_dim, H/32, W/32]  # 1/32 scale
        }
        """
        # ResNetの基本ブロック
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)  # 1/4 scale
        c3 = self.layer2(c2)  # 1/8 scale
        c4 = self.layer3(c3)  # 1/16 scale
        c5 = self.layer4(c4)  # 1/32 scale

        # 側方接続で次元統一
        p5 = self.lateral_conv4(c5)
        p4 = self.lateral_conv3(c4)
        p3 = self.lateral_conv2(c3)
        p2 = self.lateral_conv1(c2)

        # トップダウン接続（FPN）
        p4 = p4 + \
            nn.functional.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p3 = p3 + \
            nn.functional.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p2 = p2 + \
            nn.functional.interpolate(p3, size=p2.shape[-2:], mode='nearest')

        # 最終的な特徴マップ
        p5 = self.fpn_conv4(p5)
        p4 = self.fpn_conv3(p4)
        p3 = self.fpn_conv2(p3)
        p2 = self.fpn_conv1(p2)

        return {
            'c2': p2,  # 1/4 scale - 高解像度、低レベル特徴
            'c3': p3,  # 1/8 scale
            'c4': p4,  # 1/16 scale
            'c5': p5   # 1/32 scale - 低解像度、高レベル特徴
        }

    @staticmethod
    def get_preprocess(pretrained: bool = True):
        """ResNet50用の前処理"""
        if pretrained:
            return torch.nn.Sequential(
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        else:
            return torch.nn.Sequential(
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            )


class MultiScaleFeatureFusion(nn.Module):
    """
    マルチスケール特徴を統合するモジュール
    """

    def __init__(self, feat_dim: int, target_scale: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.target_scale = target_scale

        # 各スケールの特徴を統合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_dim * 4, feat_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim * 2, feat_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        マルチスケール特徴を統合して目標解像度に出力
        """
        # すべての特徴を同じ解像度にリサイズ
        target_size = features['c2'].shape[-2:]  # c2の解像度を基準

        c2 = features['c2']
        c3 = nn.functional.interpolate(
            features['c3'], size=target_size, mode='bilinear', align_corners=False)
        c4 = nn.functional.interpolate(
            features['c4'], size=target_size, mode='bilinear', align_corners=False)
        c5 = nn.functional.interpolate(
            features['c5'], size=target_size, mode='bilinear', align_corners=False)

        # 特徴を連結
        fused = torch.cat([c2, c3, c4, c5], dim=1)

        # 統合
        output = self.fusion_conv(fused)

        return output
