import torch
from torch import nn
from .decoder_xattn import PatchMoEXAttnDecoderLayer


class PatchMoEDecoderStack(nn.Module):
    """
    学習可能クエリを持つ Cross-Attn → MoE FFN の L層スタック。
    - learnable_queries: [Nq, D] を学習。
    - 前向きでは B バッチに複製して使用。
    """

    def __init__(self, model_dim: int, num_heads: int, moe_ffn_factory, num_layers: int = 3, num_queries: int = 25, dropout: float = 0.1) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.randn(num_queries, model_dim) * 0.02)

        layers = []
        for _ in range(num_layers):
            layers.append(PatchMoEXAttnDecoderLayer(
                model_dim=model_dim, num_heads=num_heads, moe_ffn=moe_ffn_factory(), dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, kv_tokens: torch.Tensor, ppe_kv: torch.Tensor) -> torch.Tensor:
        B = kv_tokens.size(0)
        queries = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)
        x = queries
        for layer in self.layers:
            x = layer(kv_tokens, ppe_kv, x)
        return x
