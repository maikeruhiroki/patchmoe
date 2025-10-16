import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    """ResNet-50 のconv4出力を特徴として使用し、1x1でdim整合。"""

    def __init__(self, out_dim: int = 128, pretrained: bool = False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=weights)
        # stem + layer1..4 を利用。conv5前の出力を使用
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.proj = nn.Conv2d(2048, out_dim, 1)

    @staticmethod
    def get_preprocess(pretrained: bool):
        if pretrained:
            return ResNet50_Weights.IMAGENET1K_V2.transforms()
        # デフォルト: [0,1] を想定
        return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        return x
