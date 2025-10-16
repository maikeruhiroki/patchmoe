import torch

from PatchMoE import PatchPositionEmbedding, TutelMoEFFN, PatchMoEDecoderLayer
from PatchMoE.head import PatchMoEHead
from PatchMoE.contrastive import PatchContrastiveLoss


def test_e2e_forward_and_loss():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    B, H, W, D = 2, 4, 4, 64
    L = H * W
    num_datasets, num_images, num_patches = 3, 128, L
    num_classes = 5

    # Modules
    ppe = PatchPositionEmbedding(
        num_datasets, num_images, num_patches, model_dim=D).to(device)
    moe_ffn = TutelMoEFFN(
        model_dim=D, hidden_size_per_expert=128, num_experts_per_device=2).to(device)
    decoder = PatchMoEDecoderLayer(
        model_dim=D, num_heads=4, moe_ffn=moe_ffn).to(device)
    head = PatchMoEHead(model_dim=D, num_classes=num_classes,
                        grid_h=H, grid_w=W, upsample=2).to(device)
    criterion = PatchContrastiveLoss(temperature=0.2)

    # Fake data
    tokens = torch.randn(B, L, D, device=device)
    dataset_ids = torch.randint(0, num_datasets, (B, L), device=device)
    # 同一画像IDを各バッチに複数回含めてポジティブを生成
    image_ids = torch.randint(0, 8, (B, L), device=device)
    patch_ids = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)
    coords = torch.stack([dataset_ids, image_ids, patch_ids], dim=-1)

    # Forward
    pos = ppe(coords)
    x = decoder(tokens, pos)
    logits, mask = head(x)

    # Loss
    loss = criterion(x, dataset_ids, image_ids)

    assert logits.shape == (B, L, num_classes)
    assert mask.dim() == 4 and mask.size(0) == B
    assert loss.item() == loss.item()  # not NaN


if __name__ == "__main__":
    test_e2e_forward_and_loss()
    print("e2e ok")
