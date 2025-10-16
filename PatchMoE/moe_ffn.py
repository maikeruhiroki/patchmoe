import os
import sys
import torch
from torch import nn

# DDPサブプロセスでも tutel 名前空間を解決できるようパスを追加
workspace_path = '/workspace'
tutel_build = '/workspace/tutel/build/lib.linux-x86_64-cpython-38'
for p in (workspace_path, tutel_build):
    if p not in sys.path and os.path.isdir(p):
        sys.path.append(p)

try:
    # standard install layout (pip package)
    from tutel import moe as tutel_moe  # type: ignore
except Exception:
    try:
        # editable repo layout under namespace 'tutel.tutel'
        from tutel.tutel import moe as tutel_moe  # type: ignore
    except Exception:
        try:
            # submodule layout: 'tutel.tutel.moe'
            from tutel.tutel.moe import moe_layer as _moe_layer  # type: ignore

            class _Shim:
                moe_layer = _moe_layer
            tutel_moe = _Shim()
        except Exception:
            tutel_moe = None


class TutelMoEFFN(nn.Module):
    """
    TutelのMoE層でTransformerのFFNを置き換えるための薄いラッパー。
    入力/出力は [..., model_dim] を想定。
    """

    def __init__(
        self,
        model_dim: int,
        hidden_size_per_expert: int,
        num_experts_per_device: int = 2,
        top_k: int = 2,
        capacity_factor: float = 1.0,
        gate_noise: float = 1.0,
        activation: str = "relu",
        fp32_gate: bool = True,
    ) -> None:
        super().__init__()
        assert tutel_moe is not None, "tutel.moe をインポートできません。'pip install -e /workspace/tutel' または PYTHONPATH の確認をしてください。"

        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()

        # Tutel expects experts['count_per_node'] instead of num_experts_per_device
        self.moe = tutel_moe.moe_layer(
            gate_type={
                'type': 'top',
                'k': top_k,
                'capacity_factor': capacity_factor,
                'fp32_gate': fp32_gate,
                'gate_noise': gate_noise,
            },
            model_dim=model_dim,
            experts={
                'count_per_node': num_experts_per_device,
                'type': 'ffn',
                'hidden_size_per_expert': hidden_size_per_expert,
                'activation_fn': lambda x: act_fn(x),
            },
            scan_expert_func=lambda name, p: setattr(
                p, 'skip_allreduce', True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.view(-1, orig_shape[-1])
        y2d = self.moe(x2d)
        y = y2d.view(*orig_shape)

        # l_auxを属性として追加
        if hasattr(self.moe, 'l_aux'):
            y.l_aux = self.moe.l_aux

        return y
