import os
import torch

from PatchMoE import PatchPositionEmbedding, TutelMoEFFN, PatchMoEDecoderLayer


def test_forward_smoke():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    B, L, D = 2, 8, 64
    num_datasets, num_images, num_patches = 3, 100, 64

    ppe = PatchPositionEmbedding(
        num_datasets=num_datasets,
        num_images=num_images,
        num_patches=num_patches,
        model_dim=D,
    ).to(device)

    moe_ffn = TutelMoEFFN(
        model_dim=D,
        hidden_size_per_expert=128,
        num_experts_per_device=2,
        top_k=2,
    ).to(device)

    layer = PatchMoEDecoderLayer(model_dim=D, num_heads=4, moe_ffn=moe_ffn).to(device)

    tokens = torch.randn(B, L, D, device=device)
    coords = torch.stack([
        torch.zeros(B, L, dtype=torch.long, device=device),  # dataset_id=0
        torch.arange(L, device=device).unsqueeze(0).repeat(B, 1),  # image_id (dummy)
        torch.arange(L, device=device).unsqueeze(0).repeat(B, 1),  # patch_id (dummy)
    ], dim=-1)

    pos = ppe(coords)
    out = layer(tokens, pos)
    assert out.shape == (B, L, D)


if __name__ == "__main__":
    test_forward_smoke()
    print("smoke ok")


