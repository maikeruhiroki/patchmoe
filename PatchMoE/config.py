from dataclasses import dataclass


@dataclass
class PatchMoEConfig:
    # Model
    in_channels: int = 3
    feat_dim: int = 128
    grid_h: int = 16
    grid_w: int = 16
    num_classes: int = 6
    num_layers: int = 8
    num_heads: int = 8
    num_queries: int = 25
    num_datasets: int = 8
    num_images_cap: int = 100000
    experts_per_device: int = 4

    # MoE gate tuning
    top_k: int = 2
    capacity_factor: float = 1.0
    gate_noise: float = 1.0

    # Train
    lr: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 1
    batch_size: int = 4
    num_workers: int = 0

    # Logging / ckpt
    out_dir: str = "/workspace/outputs/patchmoe"
    log_interval: int = 10
