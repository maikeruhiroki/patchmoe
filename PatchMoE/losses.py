import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + eps
    loss = 1 - (num / den)
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        w = self.alpha * (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        loss = w * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def load_balancing_aux_loss(moe_output) -> torch.Tensor:
    # Tutelの y.l_aux を利用。None の場合は 0。
    if hasattr(moe_output, 'l_aux') and moe_output.l_aux is not None:
        return moe_output.l_aux.mean() if hasattr(moe_output.l_aux, 'mean') else moe_output.l_aux
    return torch.tensor(0.0, device=moe_output.device if isinstance(moe_output, torch.Tensor) else 'cpu')


class ImprovedDiceLoss(nn.Module):
    """改良されたDice損失：クラス不均衡に対応"""

    def __init__(self, smooth: float = 1e-6, class_weights: torch.Tensor = None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # マルチクラス対応
        # [B, C, H, W] and [B, H, W]
        if logits.dim() == 4 and targets.dim() == 3:
            probs = torch.softmax(logits, dim=1)
            targets_one_hot = F.one_hot(
                targets.long(), num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
        else:
            probs = torch.sigmoid(logits)
            targets_one_hot = targets

        # 各クラスのDice損失を計算
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # クラス重みを適用
        if self.class_weights is not None:
            dice = dice * self.class_weights.to(dice.device)

        return 1.0 - dice.mean()


class AdaptiveFocalLoss(nn.Module):
    """適応的Focal損失：困難なサンプルにより重点"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # パッチレベルの分類損失（logits: [B, L, C], targets: [B, L]）
        if logits.dim() == 3 and targets.dim() == 2:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            focal_loss = focal_loss.view(logits.size(0), logits.size(1))
        elif logits.dim() == 4 and targets.dim() == 3:  # マルチクラス画像レベル
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:  # バイナリ
            probs = torch.sigmoid(logits)
            pt = torch.where(targets == 1, probs, 1 - probs)
            focal_loss = self.alpha * \
                (1 - pt) ** self.gamma * \
                F.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none')

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PatchMoECombinedLoss(nn.Module):
    """PatchMoE用の統合損失関数"""

    def __init__(self, num_classes: int = 6, dice_weight: float = 1.0,
                 focal_weight: float = 1.0, contrastive_weight: float = 0.1,
                 moe_weight: float = 0.01):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.contrastive_weight = contrastive_weight
        self.moe_weight = moe_weight

        # クラス不均衡を考慮した重み（背景クラスを軽く）
        class_weights = torch.ones(num_classes)
        class_weights[0] = 0.1  # 背景クラスの重みを下げる
        class_weights = class_weights / class_weights.sum() * num_classes

        self.dice_loss = ImprovedDiceLoss(class_weights=class_weights)
        self.focal_loss = AdaptiveFocalLoss()

    def forward(self, logits: torch.Tensor, mask_logits: torch.Tensor,
                targets: torch.Tensor, mask_targets: torch.Tensor,
                contrastive_loss: torch.Tensor = None,
                moe_aux_loss: torch.Tensor = None) -> dict:

        # 分類損失（パッチレベルの分類）
        cls_loss = self.focal_loss(logits, targets)

        # セグメンテーション損失（画像レベルのセグメンテーション）
        # サイズを合わせる
        if mask_logits.shape != mask_targets.shape:
            mask_logits = torch.nn.functional.interpolate(
                mask_logits, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False
            )
        seg_loss = self.dice_loss(mask_logits, mask_targets)

        # 総損失
        total_loss = (self.focal_weight * cls_loss +
                      self.dice_weight * seg_loss)

        # 対照学習損失
        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss

        # MoE負荷分散損失
        if moe_aux_loss is not None:
            total_loss += self.moe_weight * moe_aux_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'seg_loss': seg_loss,
            'contrastive_loss': contrastive_loss if contrastive_loss is not None else torch.tensor(0.0),
            'moe_loss': moe_aux_loss if moe_aux_loss is not None else torch.tensor(0.0)
        }
