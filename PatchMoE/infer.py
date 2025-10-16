import os
import argparse
import csv
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from PatchMoE.model import PatchMoEModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str,
                        default='/workspace/outputs/patchmoe')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--experts_per_device', type=int, default=None)
    parser.add_argument('--feat_dim', type=int, default=None)
    parser.add_argument('--grid_h', type=int, default=None)
    parser.add_argument('--grid_w', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--num_queries', type=int, default=None)
    parser.add_argument('--num_datasets', type=int, default=None)
    parser.add_argument('--num_images', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt

    # チェックポイント作成時と同じグローバル専門家数に合わせるため、単GPU推論では
    # training の (experts_per_device * WORLD_SIZE) と等しくなるよう指定してください。
    # ckpt から cfg を復元
    cfg = ckpt.get('cfg', {})

    def pick(k, default=None):
        v = getattr(args, k)
        return v if v is not None else cfg.get(k, default)

    grid_h = pick('grid_h', 8)
    grid_w = pick('grid_w', 8)
    nqs = pick('num_queries', grid_h * grid_w)
    model = PatchMoEModel(
        feat_dim=pick('feat_dim', 128),
        grid_h=grid_h,
        grid_w=grid_w,
        num_layers=pick('num_layers', 3),
        num_heads=pick('num_heads', 4),
        num_queries=nqs,
        num_datasets=pick('num_datasets', 8),
        num_images=pick('num_images', 100000),
        experts_per_device=pick('experts_per_device', 2),
    )
    # DDP学習時のexperts_per_deviceと同じ値を使用（グローバル専門家数は自動調整）
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    with torch.no_grad():
        if args.image and os.path.exists(args.image):
            img = Image.open(args.image).convert('RGB')
            img = img.resize((256, 256))
            x = TF.to_tensor(img).unsqueeze(0).to(device)
        else:
            x = torch.randn(1, 3, 256, 256, device=device)
        L = model.grid_h * model.grid_w
        dataset_ids = torch.zeros(1, L, dtype=torch.long, device=device)
        image_ids = torch.zeros(1, L, dtype=torch.long, device=device)
        logits, mask = model(x, dataset_ids, image_ids)
    os.makedirs(args.out, exist_ok=True)
    # PNG保存（ファイル名は入力名があればそれに紐づけ）
    out_mask = os.path.join(
        args.out, 'mask.png' if not args.image else f"mask_{os.path.basename(args.image)}.png")
    mask_img = (torch.sigmoid(mask)[0, 0].cpu().clamp(
        0, 1) * 255).byte().numpy()
    Image.fromarray(mask_img).save(out_mask)
    # CSV保存（クラス確率）
    probs = torch.softmax(logits[0], dim=-1).mean(dim=0).cpu().tolist()
    out_csv = os.path.join(args.out, 'classes.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'prob'])
        for i, p in enumerate(probs):
            w.writerow([i, p])
    print('saved:', out_mask, out_csv)


if __name__ == '__main__':
    main()
