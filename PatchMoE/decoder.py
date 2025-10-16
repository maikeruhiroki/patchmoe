import torch
from torch import nn


class PatchMoEDecoderLayer(nn.Module):
    """
    簡易Transformerデコーダ層:
      - 入力: token表現 [B, L, D] と PPE [B, L, D]（同次元）
      - Self-Attn → 残差・LN → MoE-FFN → 残差・LN
    実運用ではPETRv2のQuery・Cross-Attnへ置換/拡張する想定。
    """

    def __init__(self, model_dim: int, num_heads: int, moe_ffn: nn.Module, dropout: float = 0.1) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(model_dim)

        self.moe_ffn = moe_ffn
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)

    def forward(self, tokens: torch.Tensor, ppe: torch.Tensor) -> torch.Tensor:
        assert tokens.shape == ppe.shape, "tokens と ppe は同じ形状 [B, L, D] である必要があります"
        x = tokens + ppe

        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + self.attn_dropout(attn_out))

        # MoE-FFN
        ffn_out = self.moe_ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_out))
        return x


