# PatchMoE (minimal skeleton)

This is a minimal, framework-agnostic skeleton to reproduce key building blocks of PatchMoE:

- Patch Position Embedding (PPE): maps (dataset_id, image_id, patch_id) -> model_dim
- Tutel-based MoE FFN: replaces Transformer FFN with Microsoft Tutel MoE
- A simple decoder layer: Self-Attn + MoE-FFN with residual + LayerNorm

## Install

- Ensure Tutel is available in the Python env. In this workspace Tutel sources exist under `tutel/`.
  If not installed, install it in editable mode:

```bash
pip install -e /workspace/tutel
```

## Quick smoke test

```bash
python -m PatchMoE.tests.smoke_test
```

## Integration notes

- PETRv2-style position encoder: replace 3D coordinates with (dataset_id, image_id, patch_id) and feed as PPE.
- Replace Transformer FFN with `TutelMoEFFN`. For DDP, mark local expert params with `skip_allreduce` (already set in this wrapper).
- mmseg/mmcv/mmdet integration should follow the pinned versions recommended by the author.

## Next steps

- Add contrastive loss around PPE outputs (patch similarity) to match the paper.
- Wire to PETR head (`petr_head_seg.py`) cross-attention and final heads (linear + upsample).
