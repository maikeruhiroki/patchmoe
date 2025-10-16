import torch
from torch import nn


class PatchPositionEmbedding(nn.Module):
    """
    PatchMoE用のパッチ位置埋め込み。

    入力は (dataset_id, image_id, patch_id) の3要素。以下いずれかの形式を受け付ける。
      - forward(dataset_ids, image_ids, patch_ids): いずれも [B, L] のLongTensor
      - forward(coords): coords は [B, L, 3] のLongTensor

    出力は [B, L, model_dim] の埋め込み表現。
    """

    def __init__(
        self,
        num_datasets: int,
        num_images: int,
        num_patches: int,
        model_dim: int,
        embedding_dim: int = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        embed_dim = embedding_dim or model_dim

        self.dataset_embedding = nn.Embedding(num_datasets, embed_dim)
        self.image_embedding = nn.Embedding(num_images, embed_dim)
        self.patch_embedding = nn.Embedding(num_patches, embed_dim)

        self.proj = nn.Linear(embed_dim * 3, model_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.dataset_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.image_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.patch_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, *args):
        if len(args) == 1:
            coords = args[0]
            assert coords.dim() == 3 and coords.size(-1) == 3, "coords must be [B, L, 3]"
            dataset_ids = coords[..., 0]
            image_ids = coords[..., 1]
            patch_ids = coords[..., 2]
        elif len(args) == 3:
            dataset_ids, image_ids, patch_ids = args
            assert dataset_ids.shape == image_ids.shape == patch_ids.shape, "IDs must share shape [B, L]"
        else:
            raise ValueError("forward expects (coords) or (dataset_ids, image_ids, patch_ids)")

        d_emb = self.dataset_embedding(dataset_ids)
        i_emb = self.image_embedding(image_ids)
        p_emb = self.patch_embedding(patch_ids)

        concat = torch.cat([d_emb, i_emb, p_emb], dim=-1)
        out = self.proj(concat)
        return self.dropout(out)


