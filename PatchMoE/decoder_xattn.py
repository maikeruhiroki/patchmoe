import torch
from torch import nn


class PatchMoEXAttnDecoderLayer(nn.Module):
    """
    図5.2に対応: Query × (Key, Value) の Cross-Attention → MoE FFN。

    入力:
      - kv_tokens: [B, L_kv, D]   (バックボーン＋PPEを加えたKey/Valueの元特徴)
      - ppe_kv:    [B, L_kv, D]   (PPE。kv_tokensに加算して使用)
      - queries:   [B, Nq,  D]    (タスク埋め込みや学習可能クエリ)

    出力:
      - out:       [B, Nq,  D]
    """

    def __init__(self, model_dim: int, num_heads: int, moe_ffn: nn.Module, dropout: float = 0.1) -> None:
        super().__init__()
        self.model_dim = model_dim

        self.pre_xattn_norm_q = nn.LayerNorm(model_dim)
        self.pre_xattn_norm_kv = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True)
        self.xattn_dropout = nn.Dropout(dropout)
        self.post_xattn_norm = nn.LayerNorm(model_dim)

        self.moe_ffn = moe_ffn
        self.ffn_dropout = nn.Dropout(dropout)
        self.post_ffn_norm = nn.LayerNorm(model_dim)
        self.last_moe_aux = None

    def forward(self, kv_tokens: torch.Tensor, ppe_kv: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        assert kv_tokens.shape == ppe_kv.shape, "kv_tokens と ppe_kv は同形状 [B, L, D]"
        # Key/Value は PPE を加えた特徴
        kv = self.pre_xattn_norm_kv(kv_tokens + ppe_kv)
        q = self.pre_xattn_norm_q(queries)

        # Cross-Attention: Q × (K,V)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = self.post_xattn_norm(q + self.xattn_dropout(attn_out))

        # MoE FFN + 残差
        ffn_out = self.moe_ffn(x)
        # Tutelのl_auxを保持
        self.last_moe_aux = getattr(ffn_out, 'l_aux', None)
        x = self.post_ffn_norm(x + self.ffn_dropout(ffn_out))
        return x
