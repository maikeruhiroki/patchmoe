import torch
from torch import nn
import torch.nn.functional as F
from .ppe import PatchPositionEmbedding
from .moe_ffn import TutelMoEFFN
from .decoder_stack import PatchMoEDecoderStack
from .head import PatchMoEHead
from .backbone_resnet import ResNet50Backbone
from .multiscale_backbone import MultiScaleResNetBackbone, MultiScaleFeatureFusion


class SimpleBackbone(nn.Module):
    """簡易CNNバックボーン（スモーク用）。本番はResNet等に差し替え。"""

    def __init__(self, in_ch: int = 3, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, feat_dim, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchMoEModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        feat_dim: int = 64,
        grid_h: int = 8,
        grid_w: int = 8,
        num_datasets: int = 4,
        num_images: int = 1024,
        num_patches: int = None,
        num_classes: int = 6,
        num_layers: int = 3,
        num_heads: int = 4,
        num_queries: int = 25,
        gate_top_k: int = 2,
        gate_capacity: float = 1.0,
        gate_noise: float = 1.0,
        backbone: str = 'simple',
        pretrained_backbone: bool = False,
        experts_per_device: int = 2,
        use_multiscale: bool = False,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        L = grid_h * grid_w
        num_patches = num_patches or L

        if backbone == 'resnet50':
            if use_multiscale:
                self.backbone = MultiScaleResNetBackbone(
                    out_dim=feat_dim, pretrained=pretrained_backbone)
                self.feature_fusion = MultiScaleFeatureFusion(feat_dim)
                self.preprocess = MultiScaleResNetBackbone.get_preprocess(
                    pretrained_backbone)
            else:
                self.backbone = ResNet50Backbone(
                    out_dim=feat_dim, pretrained=pretrained_backbone)
                self.feature_fusion = None
                self.preprocess = ResNet50Backbone.get_preprocess(
                    pretrained_backbone)
        else:
            self.backbone = SimpleBackbone(in_ch=in_ch, feat_dim=feat_dim)
            self.feature_fusion = None
            self.preprocess = nn.Identity()
        self.proj = nn.Conv2d(feat_dim, feat_dim, 1)
        self.ppe = PatchPositionEmbedding(
            num_datasets, num_images, num_patches, model_dim=feat_dim)

        def moe_factory():
            # ゲート設定を渡す
            m = TutelMoEFFN(model_dim=feat_dim, hidden_size_per_expert=feat_dim * 2, num_experts_per_device=experts_per_device,
                            top_k=gate_top_k, capacity_factor=gate_capacity, gate_noise=gate_noise)
            return m

        self.decoder = PatchMoEDecoderStack(model_dim=feat_dim, num_heads=num_heads,
                                            moe_ffn_factory=moe_factory, num_layers=num_layers, num_queries=num_queries)
        self.head = PatchMoEHead(
            model_dim=feat_dim, num_classes=num_classes, grid_h=grid_h, grid_w=grid_w, upsample=4)

    def forward(self, images: torch.Tensor, dataset_ids: torch.Tensor, image_ids: torch.Tensor):
        B, _, H, W = images.shape
        images = self.preprocess(images)

        if self.feature_fusion is not None:
            # マルチスケール特徴抽出
            multiscale_feats = self.backbone(images)
            feats = self.feature_fusion(multiscale_feats)  # [B, C, H, W]
        else:
            feats = self.backbone(images)  # [B, C, H, W]
        feats = self.proj(feats)
        # グリッド解像度に整合
        feats = F.adaptive_avg_pool2d(feats, (self.grid_h, self.grid_w))
        # パッチ展開
        kv_tokens = feats.flatten(2).transpose(1, 2)   # [B, L, C]
        L = kv_tokens.size(1)
        patch_ids = torch.arange(
            L, device=images.device).unsqueeze(0).repeat(B, 1)
        coords = torch.stack([dataset_ids, image_ids, patch_ids], dim=-1)
        ppe_kv = self.ppe(coords)                 # [B, L, C]

        queries_out = self.decoder(kv_tokens, ppe_kv)  # [B, Nq, C]
        # ヘッドに渡すトークン数をグリッドに合わせる
        # クエリ数がグリッドサイズより小さい場合は拡張
        if queries_out.size(1) < self.grid_h * self.grid_w:
            # クエリをグリッドサイズまで拡張（最後のクエリを繰り返し）
            pad_size = self.grid_h * self.grid_w - queries_out.size(1)
            last_query = queries_out[:, -1:, :].repeat(1, pad_size, 1)
            queries_out = torch.cat([queries_out, last_query], dim=1)
        
        x_hw = queries_out[:, : (self.grid_h * self.grid_w), :]
        logits, mask = self.head(x_hw)
        return logits, mask
