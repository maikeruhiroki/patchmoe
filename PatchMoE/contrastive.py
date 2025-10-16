import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class PatchContrastiveLoss(nn.Module):
    """
    簡易 NT-Xent 風コントラスト損失。

    入力:
      - feats: [B, L, D] 特徴
      - dataset_ids: [B, L] LongTensor
      - image_ids: [B, L] LongTensor
    ルール:
      - 同一 (dataset_id, image_id) 内の異なるパッチを positives とし、それ以外を negatives とする。
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8, pos_weight: float = 1.0, neg_weight: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, feats: torch.Tensor, dataset_ids: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
        B, L, D = feats.shape
        x = F.normalize(feats.reshape(B * L, D), dim=-1, eps=self.eps)

        # 類似度行列 [N, N]
        sim = torch.matmul(x, x.t()) / self.temperature

        # マスク作成
        did = dataset_ids.reshape(-1)
        iid = image_ids.reshape(-1)
        same_img = (did.unsqueeze(1) == did.unsqueeze(0)) & (
            iid.unsqueeze(1) == iid.unsqueeze(0))
        eye = torch.eye(B * L, dtype=torch.bool, device=feats.device)
        pos_mask = same_img & (~eye)
        neg_mask = ~same_img

        # それぞれの i について、複数 positive がある場合は平均で扱う
        # log-softmax 計算
        sim_exp = torch.exp(sim)
        weights = pos_mask.float() * self.pos_weight + neg_mask.float() * self.neg_weight
        denom = (sim_exp * weights).sum(dim=1) + self.eps
        # positive の重み平均
        pos_sum = (sim_exp * (pos_mask.float() * self.pos_weight)).sum(dim=1)
        # 損失: -log( pos / (pos+neg) ) を平均
        loss = -torch.log((pos_sum + self.eps) / denom)
        # positive が一つも無い要素は無視（例えば L=1）
        valid = pos_mask.any(dim=1)
        if valid.any():
            return loss[valid].mean()
        return loss.mean()


class AdvancedPatchContrastiveLoss(nn.Module):
    """
    改良された対照学習損失：
    - ドメイン間の分離を強化
    - ハードネガティブマイニング
    - 温度パラメータの適応的調整
    """
    
    def __init__(self, temperature: float = 0.07, eps: float = 1e-8, 
                 hard_negative_weight: float = 2.0, domain_separation_weight: float = 1.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.hard_negative_weight = hard_negative_weight
        self.domain_separation_weight = domain_separation_weight
        
    def forward(self, feats: torch.Tensor, dataset_ids: torch.Tensor, 
                image_ids: torch.Tensor, patch_ids: torch.Tensor = None) -> torch.Tensor:
        B, L, D = feats.shape
        x = F.normalize(feats.reshape(B * L, D), dim=-1, eps=self.eps)
        
        # 類似度行列
        sim = torch.matmul(x, x.t()) / self.temperature
        
        # マスク作成
        did = dataset_ids.reshape(-1)
        iid = image_ids.reshape(-1)
        
        # 同一画像内の異なるパッチをpositive
        same_img = (did.unsqueeze(1) == did.unsqueeze(0)) & (iid.unsqueeze(1) == iid.unsqueeze(0))
        eye = torch.eye(B * L, dtype=torch.bool, device=feats.device)
        pos_mask = same_img & (~eye)
        
        # 異なるデータセット間をnegative
        diff_dataset = (did.unsqueeze(1) != did.unsqueeze(0))
        
        # 同一データセット内の異なる画像をnegative
        same_dataset_diff_img = (did.unsqueeze(1) == did.unsqueeze(0)) & (iid.unsqueeze(1) != iid.unsqueeze(0))
        
        # ハードネガティブマイニング：類似度が高いnegativeを重く扱う
        sim_exp = torch.exp(sim)
        neg_weights = torch.ones_like(sim)
        neg_weights[diff_dataset] *= self.domain_separation_weight
        neg_weights[same_dataset_diff_img] *= 1.0
        
        # ハードネガティブの重み調整
        hard_neg_threshold = torch.quantile(sim[diff_dataset], 0.8) if diff_dataset.any() else 0
        hard_neg_mask = diff_dataset & (sim > hard_neg_threshold)
        neg_weights[hard_neg_mask] *= self.hard_negative_weight
        
        # 損失計算
        pos_sum = (sim_exp * pos_mask.float()).sum(dim=1)
        neg_sum = (sim_exp * neg_weights * (~pos_mask).float()).sum(dim=1)
        
        loss = -torch.log((pos_sum + self.eps) / (pos_sum + neg_sum + self.eps))
        
        # 有効なpositiveがある要素のみを考慮
        valid = pos_mask.any(dim=1)
        if valid.any():
            return loss[valid].mean()
        return loss.mean()


class DomainAdaptiveContrastiveLoss(nn.Module):
    """
    ドメイン適応型対照学習損失：
    - データセット間の特徴分離を強化
    - ドメイン不変特徴の学習を促進
    """
    
    def __init__(self, temperature: float = 0.1, domain_weight: float = 2.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.domain_weight = domain_weight
        
    def forward(self, feats: torch.Tensor, dataset_ids: torch.Tensor, 
                image_ids: torch.Tensor) -> torch.Tensor:
        B, L, D = feats.shape
        x = F.normalize(feats.reshape(B * L, D), dim=-1)
        
        sim = torch.matmul(x, x.t()) / self.temperature
        sim_exp = torch.exp(sim)
        
        did = dataset_ids.reshape(-1)
        iid = image_ids.reshape(-1)
        
        # 同一画像内の異なるパッチをpositive
        same_img = (did.unsqueeze(1) == did.unsqueeze(0)) & (iid.unsqueeze(1) == iid.unsqueeze(0))
        eye = torch.eye(B * L, dtype=torch.bool, device=feats.device)
        pos_mask = same_img & (~eye)
        
        # 異なるデータセット間を強くnegative
        diff_dataset = (did.unsqueeze(1) != did.unsqueeze(0))
        
        # 損失計算（ドメイン間分離を強化）
        pos_sum = (sim_exp * pos_mask.float()).sum(dim=1)
        neg_sum = (sim_exp * diff_dataset.float() * self.domain_weight + 
                  sim_exp * (~pos_mask & ~diff_dataset).float()).sum(dim=1)
        
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        
        valid = pos_mask.any(dim=1)
        if valid.any():
            return loss[valid].mean()
        return loss.mean()
