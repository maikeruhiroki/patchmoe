import torch

from PatchMoE import PatchPositionEmbedding, TutelMoEFFN
from PatchMoE.decoder_xattn import PatchMoEXAttnDecoderLayer
from PatchMoE.head import PatchMoEHead


def test_e2e_xattn():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    B, H, W, D = 2, 4, 4, 64
    L = H * W
    Nq = 25
    num_datasets, num_images, num_patches = 3, 256, L
    num_classes = 6

    # Key/Value 側
    kv_tokens = torch.randn(B, L, D, device=device)
    dataset_ids = torch.randint(0, num_datasets, (B, L), device=device)
    image_ids = torch.randint(0, 16, (B, L), device=device)
    patch_ids = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)
    ppe = PatchPositionEmbedding(
        num_datasets, num_images, num_patches, model_dim=D).to(device)
    ppe_kv = ppe(torch.stack([dataset_ids, image_ids, patch_ids], dim=-1))

    # Query 側（学習可能クエリ想定のダミー）
    queries = torch.randn(B, Nq, D, device=device)

    # デコーダ
    moe_ffn = TutelMoEFFN(
        model_dim=D, hidden_size_per_expert=128, num_experts_per_device=2).to(device)
    layer = PatchMoEXAttnDecoderLayer(
        model_dim=D, num_heads=4, moe_ffn=moe_ffn).to(device)

    x = layer(kv_tokens, ppe_kv, queries)

    # 出力ヘッド（ここでは Nq を  H*W にマッピングするために最初の L 個のみ使用）
    x_hw = x[:, :L, :]
    head = PatchMoEHead(model_dim=D, num_classes=num_classes,
                        grid_h=H, grid_w=W, upsample=2).to(device)
    logits, mask = head(x_hw)

    assert logits.shape == (B, L, num_classes)
    assert mask.shape[0] == B


if __name__ == "__main__":
    test_e2e_xattn()
    print("e2e_xattn ok")
