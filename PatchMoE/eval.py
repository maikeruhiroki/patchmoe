import os
import argparse
import json
import torch
from PIL import Image
import numpy as np


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    num = 2 * (pred * target).sum().item() + eps
    den = (pred.pow(2) + target.pow(2)).sum().item() + eps
    return num / den


def miou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    inter = (pred * target).sum().item()
    union = (pred + target - pred * target).sum().item() + eps
    return inter / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--out', type=str,
                        default='/workspace/outputs/patchmoe')
    args = parser.parse_args()

    def load_mask(path):
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = Image.open(path).convert('L')
            arr = np.array(img).astype('float32') / 255.0
            return torch.from_numpy(arr)
        t = torch.load(path)
        if isinstance(t, dict) and 'mask' in t:
            return torch.sigmoid(t['mask'])[0, 0]
        return t

    mask = load_mask(args.pred)
    gt_mask = load_mask(args.gt)

    # サイズが異なる場合はリサイズ
    if mask.shape != gt_mask.shape:
        print(f"Resizing prediction from {mask.shape} to {gt_mask.shape}")
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=gt_mask.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

    d = dice_score(mask, gt_mask)
    m = miou(mask, gt_mask)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump({'dice': d, 'miou': m}, f)
    print({'dice': d, 'miou': m})


if __name__ == '__main__':
    main()
