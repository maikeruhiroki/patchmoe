import torch
from torch import nn


class PatchMoEHead(nn.Module):
    """
    シンプルな出力ヘッド:
      - 入力: [B, L, D]
      - 出力: クラス logits [B, L, C] と マスク [B, 1, H, W]（任意のリシェイプ）
    学術検証用の簡易版として、パッチ列を (H, W) に並べ替えてアップサンプル。
    """

    def __init__(self, model_dim: int, num_classes: int, grid_h: int, grid_w: int, upsample: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.classifier = nn.Linear(model_dim, num_classes)
        self.mask_proj = nn.Linear(model_dim, 1)
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear',
                                    align_corners=False) if upsample > 1 else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        assert L == self.grid_h * self.grid_w, "L must equal grid_h*grid_w for mask reshape"
        logits = self.classifier(x)  # [B, L, C]

        mask = self.mask_proj(x)  # [B, L, 1]
        mask = mask.transpose(1, 2).contiguous().view(
            B, 1, self.grid_h, self.grid_w)
        mask = self.upsample(mask)
        return logits, mask
