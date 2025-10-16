import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler

from PatchMoE.config import PatchMoEConfig
from PatchMoE.backbone_resnet import ResNet50Backbone
from PatchMoE.model import PatchMoEModel
from PatchMoE.data import PatchDataset
from PatchMoE.contrastive import PatchContrastiveLoss
from PatchMoE.losses import dice_loss, FocalLoss
from PatchMoE.medical_dataset import build_medical_loader


def setup():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    num_cuda = torch.cuda.device_count()
    use_nccl = torch.cuda.is_available() and (num_cuda >= world_size)
    dist.init_process_group(backend='nccl' if use_nccl else 'gloo')


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--out', type=str,
                        default='/workspace/outputs/patchmoe')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--backbone', type=str,
                        default='simple', choices=['simple', 'resnet50'])
    parser.add_argument('--pretrained_backbone', action='store_true')
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--capacity', type=float, default=None)
    parser.add_argument('--gate_noise', type=float, default=None)
    parser.add_argument('--experts_per_device', type=int, default=None)
    parser.add_argument('--use_multiscale', action='store_true',
                        help='Use multi-scale feature extraction')
    parser.add_argument('--use_medical_dataset', action='store_true',
                        help='Use medical dataset instead of dummy data')
    args = parser.parse_args()

    setup()
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    num_cuda = torch.cuda.device_count()
    requested = int(os.environ.get('LOCAL_WORLD_SIZE',
                    os.environ.get('WORLD_SIZE', '1')))
    use_cuda_all = torch.cuda.is_available() and (num_cuda >= requested)
    use_cuda = use_cuda_all and (local_rank < num_cuda)
    device = torch.device(f'cuda:{local_rank}' if use_cuda else 'cpu')

    cfg = PatchMoEConfig()
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.capacity is not None:
        cfg.capacity_factor = args.capacity
    if args.gate_noise is not None:
        cfg.gate_noise = args.gate_noise
    if args.experts_per_device is not None:
        cfg.experts_per_device = args.experts_per_device
    # 実際に使用するクエリ数はグリッドと一致
    cfg.num_queries = cfg.grid_h * cfg.grid_w

    # Data
    if args.use_medical_dataset:
        # 医学画像データセットを使用
        ds = build_medical_loader(
            batch_size=cfg.batch_size,
            length=200,  # より多くのサンプル
            image_size=512,
            num_classes=cfg.num_classes,
            num_datasets=cfg.num_datasets,
            grid_h=cfg.grid_h,
            grid_w=cfg.grid_w,
            num_workers=cfg.num_workers
        ).dataset
    elif args.data_root:
        from PatchMoE.data_oct import OCTFolderDataset
        ds = OCTFolderDataset(
            root=args.data_root, grid_h=cfg.grid_h, grid_w=cfg.grid_w, image_size=256)
        if rank == 0 and hasattr(ds, 'name_to_id'):
            import json
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, 'dataset_mapping.json'), 'w') as f:
                json.dump(ds.name_to_id, f, indent=2)
    else:
        ds = PatchDataset(length=64, in_ch=cfg.in_channels, H=256, W=256,
                          grid_h=cfg.grid_h, grid_w=cfg.grid_w, num_datasets=cfg.num_datasets)
    sampler = DistributedSampler(ds) if dist.is_initialized() else None
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=(
        sampler is None), num_workers=cfg.num_workers)

    # Model (use ResNet50 backbone via PatchMoEModel integration by swapping backbone if needed)
    # クエリ数はグリッドに一致させる
    model = PatchMoEModel(in_ch=cfg.in_channels, feat_dim=cfg.feat_dim, grid_h=cfg.grid_h, grid_w=cfg.grid_w,
                          num_classes=cfg.num_classes, num_layers=cfg.num_layers, num_heads=cfg.num_heads, num_queries=cfg.num_queries,
                          num_datasets=cfg.num_datasets, num_images=cfg.num_images_cap,
                          gate_top_k=cfg.top_k, gate_capacity=cfg.capacity_factor, gate_noise=cfg.gate_noise,
                          backbone=args.backbone, pretrained_backbone=args.pretrained_backbone,
                          experts_per_device=cfg.experts_per_device, use_multiscale=args.use_multiscale).to(device)
    # 既存モデルは簡易バックボーン。同APIで差し替えたい場合はPatchMoEModelに引数で選択実装可。

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    optimizer = AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay)
    focal = FocalLoss()
    contrast = PatchContrastiveLoss(temperature=0.2)

    os.makedirs(args.out, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=args.out) if (rank == 0) else None

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for it, (img, dataset_id, image_id, mask_t, cls_t) in enumerate(dl):
            img = img.to(device)
            dataset_id = dataset_id.to(device)
            image_id = image_id.to(device)
            mask_t = mask_t.to(device)
            cls_t = cls_t.to(device)

            optimizer.zero_grad()
            logits, mask = model(img, dataset_id, image_id)
            L = logits.size(1)
            cls_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, cfg.num_classes), cls_t.view(-1))
            mask = torch.nn.functional.interpolate(
                mask, size=mask_t.shape[-2:], mode='bilinear', align_corners=False)
            d_loss = dice_loss(mask, mask_t)
            f_loss = focal(mask, mask_t)
            with torch.no_grad():
                feats_for_contrast = logits.detach()
            con_loss = contrast(feats_for_contrast, dataset_id, image_id)
            # lb_loss（ロードバランシング補助）
            lb_loss = 0.0
            dec = model.module.decoder if isinstance(
                model, DDP) else model.decoder
            for lyr in dec.layers:
                if getattr(lyr, 'last_moe_aux', None) is not None:
                    lb = lyr.last_moe_aux
                    try:
                        lb = lb.mean()
                    except Exception:
                        pass
                    lb_loss = lb_loss + lb

            total = cls_loss + 0.8 * d_loss + 1.0 * f_loss + 0.2 * con_loss + 0.01 * \
                (lb_loss if not isinstance(lb_loss, float)
                 else torch.tensor(lb_loss, device=device))
            total.backward()
            optimizer.step()

            if it % cfg.log_interval == 0 and rank == 0:
                scalars = {'loss': float(total.item()), 'cls': float(cls_loss.item()), 'dice': float(
                    d_loss.item()), 'focal': float(f_loss.item()), 'contrast': float(con_loss.item())}
                if not isinstance(lb_loss, float):
                    scalars['lb'] = float(lb_loss.item())
                print({'e': epoch, 'it': it, **scalars})
                if writer:
                    for k, v in scalars.items():
                        writer.add_scalar(
                            k, v, global_step=epoch * len(dl) + it)

        if rank == 0:
            # cfg と dataset mapping を一緒に保存
            state = model.state_dict() if not isinstance(
                model, DDP) else model.module.state_dict()
            extras = {}
            map_path = os.path.join(args.out, 'dataset_mapping.json')
            if os.path.exists(map_path):
                import json
                with open(map_path, 'r') as f:
                    extras['dataset_mapping'] = json.load(f)
            extras['cfg'] = cfg.__dict__
            ckpt = {'model': state, 'epoch': epoch, **extras}
            torch.save(ckpt, os.path.join(args.out, f'ckpt_{epoch}.pt'))

    if writer:
        writer.close()
    cleanup()


if __name__ == '__main__':
    main()
