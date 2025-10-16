import os
import torch
from torch import nn
from torch.optim import AdamW

from PatchMoE.model import PatchMoEModel
from PatchMoE.contrastive import PatchContrastiveLoss
from PatchMoE.losses import dice_loss, FocalLoss


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ハイパラ（軽量）
    B, C, H, W = 2, 3, 64, 64
    grid_h, grid_w = 8, 8
    num_classes = 3

    model = PatchMoEModel(in_ch=C, feat_dim=64, grid_h=grid_h, grid_w=grid_w, num_classes=num_classes,
                          num_layers=3, num_heads=4, num_queries=grid_h * grid_w).to(device)

    # ダミーデータローダ（擬似）
    images = torch.randn(B, C, H, W, device=device)
    dataset_ids = torch.randint(0, 4, (B, grid_h * grid_w), device=device)
    image_ids = torch.randint(0, 128, (B, grid_h * grid_w), device=device)

    # ターゲット（クラスとマスク）
    # クラス: 最初の L 個に対し one-hot
    L = grid_h * grid_w
    cls_targets = torch.randint(0, num_classes, (B, L), device=device)
    cls_targets_onehot = nn.functional.one_hot(
        cls_targets, num_classes=num_classes).float()
    # マスク: バイナリ（簡易）
    mask_targets = (torch.rand(B, 1, H, W, device=device) > 0.5).float()

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    focal = FocalLoss()
    contrast = PatchContrastiveLoss(temperature=0.2)

    model.train()
    optimizer.zero_grad()
    logits, mask = model(images, dataset_ids, image_ids)

    # 損失計算
    # クラス（最小例: L トークン分に対して平均）
    cls_logits = logits
    cls_loss = nn.functional.cross_entropy(
        cls_logits.view(-1, num_classes), cls_targets.view(-1))

    # マスク: リサイズ合わせ
    mask = nn.functional.interpolate(
        mask, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)
    d_loss = dice_loss(mask, mask_targets)
    f_loss = focal(mask, mask_targets)

    # Contrastive: デコーダ出力トークンを使用（モデル内部の queries を直接は返さないため、logits生成直前の特徴は簡易に logits から近似せず、省略可）
    # ここでは簡便に kv 側 id で PPE 近傍を使うため、疑似として cls_logits を特徴とみなす（実運用ではデコーダ中間特徴を返す設計に変更推奨）
    with torch.no_grad():
        feats_for_contrast = cls_logits.detach()
    con_loss = contrast(feats_for_contrast, dataset_ids, image_ids)

    total = cls_loss + 0.8 * d_loss + 1.0 * f_loss + 0.2 * con_loss
    total.backward()
    optimizer.step()

    print({
        'cls': float(cls_loss.item()),
        'dice': float(d_loss.item()),
        'focal': float(f_loss.item()),
        'contrast': float(con_loss.item()),
        'total': float(total.item()),
    })


if __name__ == '__main__':
    main()
