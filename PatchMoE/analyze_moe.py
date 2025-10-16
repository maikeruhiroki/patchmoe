#!/usr/bin/env python3
"""
MoE専門家使用状況の分析スクリプト
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PatchMoE.model import PatchMoEModel
from PatchMoE.config import PatchMoEConfig


def analyze_moe_experts(ckpt_path: str, output_dir: str):
    """MoE専門家の使用状況を分析"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # チェックポイント読み込み
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('cfg', {})

    # モデル構築
    model = PatchMoEModel(
        feat_dim=cfg.get('feat_dim', 128),
        grid_h=cfg.get('grid_h', 16),
        grid_w=cfg.get('grid_w', 16),
        num_layers=cfg.get('num_layers', 8),
        num_heads=cfg.get('num_heads', 8),
        num_queries=cfg.get('num_queries', 256),
        num_datasets=cfg.get('num_datasets', 8),
        num_images=cfg.get('num_images_cap', 100000),
        experts_per_device=cfg.get('experts_per_device', 4),
        gate_top_k=cfg.get('top_k', 2),
        gate_capacity=cfg.get('capacity_factor', 1.0),
        gate_noise=cfg.get('gate_noise', 1.0),
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().to(device)

    # テストデータ生成
    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512, device=device)
        L = model.grid_h * model.grid_w
        dataset_ids = torch.zeros(1, L, dtype=torch.long, device=device)
        image_ids = torch.zeros(1, L, dtype=torch.long, device=device)

        # フォワードパス実行
        logits, mask = model(x, dataset_ids, image_ids)

        # MoE専門家使用状況を収集
        expert_usage = []
        for layer_idx, layer in enumerate(model.decoder.layers):
            if hasattr(layer.moe_ffn, 'last_moe_aux'):
                aux_info = layer.moe_ffn.last_moe_aux
                if aux_info is not None:
                    expert_usage.append({
                        'layer': layer_idx,
                        'aux_loss': aux_info.get('l_aux', 0),
                        'gate_scores': aux_info.get('gate_scores', None),
                        'expert_counts': aux_info.get('expert_counts', None)
                    })

    # 結果可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. レイヤー別aux loss
    layers = [info['layer'] for info in expert_usage]
    aux_losses = [info['aux_loss'] for info in expert_usage]
    axes[0, 0].bar(layers, aux_losses)
    axes[0, 0].set_title('Load Balancing Loss by Layer')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Aux Loss')

    # 2. 専門家使用分布（最初のレイヤー）
    if expert_usage and expert_usage[0]['expert_counts'] is not None:
        expert_counts = expert_usage[0]['expert_counts'].cpu().numpy()
        axes[0, 1].bar(range(len(expert_counts)), expert_counts)
        axes[0, 1].set_title('Expert Usage Count (Layer 0)')
        axes[0, 1].set_xlabel('Expert ID')
        axes[0, 1].set_ylabel('Usage Count')

    # 3. ゲートスコア分布
    if expert_usage and expert_usage[0]['gate_scores'] is not None:
        gate_scores = expert_usage[0]['gate_scores'].cpu().numpy()
        axes[1, 0].hist(gate_scores.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('Gate Scores Distribution (Layer 0)')
        axes[1, 0].set_xlabel('Gate Score')
        axes[1, 0].set_ylabel('Frequency')

    # 4. セグメンテーションマスク
    mask_np = torch.sigmoid(mask)[0, 0].cpu().numpy()
    im = axes[1, 1].imshow(mask_np, cmap='viridis')
    axes[1, 1].set_title('Segmentation Mask')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/moe_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 統計情報出力
    print(f"MoE Analysis Results:")
    print(f"Number of layers: {len(expert_usage)}")
    print(f"Average aux loss: {np.mean(aux_losses):.4f}")
    print(f"Total aux loss: {np.sum(aux_losses):.4f}")

    if expert_usage and expert_usage[0]['expert_counts'] is not None:
        expert_counts = expert_usage[0]['expert_counts'].cpu().numpy()
        print(f"Expert usage variance: {np.var(expert_counts):.4f}")
        print(
            f"Most used expert: {np.argmax(expert_counts)} (count: {np.max(expert_counts)})")
        print(
            f"Least used expert: {np.argmin(expert_counts)} (count: {np.min(expert_counts)})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str,
                        default='/workspace/outputs/patchmoe_analysis')
    args = parser.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    analyze_moe_experts(args.ckpt, args.out)
