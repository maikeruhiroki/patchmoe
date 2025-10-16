#!/usr/bin/env python3
"""
PatchMoEのハイパーパラメータ最適化スクリプト
Optunaを使用して最適なパラメータを探索
"""
import os
import torch
import optuna
import json
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime

from PatchMoE.model import PatchMoEModel
from PatchMoE.real_dataset_loader import build_real_medical_loader
from PatchMoE.losses import PatchMoECombinedLoss
from PatchMoE.contrastive import AdvancedPatchContrastiveLoss
from PatchMoE.eval import dice_score as dice_coefficient, miou


class HyperparameterOptimizer:
    """ハイパーパラメータ最適化クラス"""

    def __init__(self, device: str = 'cuda:0', n_trials: int = 50):
        self.device = device
        self.n_trials = n_trials
        self.best_score = 0.0
        self.best_params = None

        # 出力ディレクトリ
        self.output_dir = '/workspace/outputs/hyperparameter_optimization'
        os.makedirs(self.output_dir, exist_ok=True)

        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def objective(self, trial: optuna.Trial) -> float:
        """Optunaの目的関数"""
        try:
            # ハイパーパラメータの提案
            params = {
                # 学習率
                'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),

                # 重み減衰
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),

                # バッチサイズ
                'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8]),

                # 特徴次元
                'feat_dim': trial.suggest_categorical('feat_dim', [64, 128, 256]),

                # レイヤー数
                'num_layers': trial.suggest_int('num_layers', 4, 12),

                # ヘッド数
                'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),

                # MoE設定
                'top_k': trial.suggest_int('top_k', 1, 4),
                'capacity_factor': trial.suggest_float('capacity_factor', 0.5, 2.0),
                'gate_noise': trial.suggest_float('gate_noise', 0.1, 2.0),
                'experts_per_device': trial.suggest_categorical('experts_per_device', [2, 4, 8]),

                # 損失重み
                'dice_weight': trial.suggest_float('dice_weight', 0.5, 2.0),
                'focal_weight': trial.suggest_float('focal_weight', 0.5, 2.0),
                'contrastive_weight': trial.suggest_float('contrastive_weight', 0.01, 0.5),
                'moe_weight': trial.suggest_float('moe_weight', 0.001, 0.1),

                # 対照学習設定
                'contrastive_temperature': trial.suggest_float('contrastive_temperature', 0.01, 0.2),
                'hard_negative_weight': trial.suggest_float('hard_negative_weight', 1.0, 3.0),
                'domain_separation_weight': trial.suggest_float('domain_separation_weight', 0.5, 2.0),
            }

            # モデル訓練と評価
            score = self._train_and_evaluate(params, trial.number)

            # 最良スコアの更新
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.logger.info(
                    f"New best score: {score:.4f} with params: {params}")

            return score

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0

    def _train_and_evaluate(self, params: Dict[str, Any], trial_number: int) -> float:
        """モデルの訓練と評価"""
        # データローダー構築
        dataset_configs = [
            {
                'name': 'drive',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'train',
                'dataset_id': 0,
                'augmentation': True
            },
            {
                'name': 'stare',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'train',
                'dataset_id': 1,
                'augmentation': True
            },
            {
                'name': 'chase',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'train',
                'dataset_id': 2,
                'augmentation': True
            }
        ]

        val_dataset_configs = [
            {
                'name': 'drive',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'val',
                'dataset_id': 0,
                'augmentation': False
            },
            {
                'name': 'stare',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'val',
                'dataset_id': 1,
                'augmentation': False
            },
            {
                'name': 'chase',
                'root_dir': '/workspace/real_medical_datasets',
                'split': 'val',
                'dataset_id': 2,
                'augmentation': False
            }
        ]

        try:
            train_loader = build_real_medical_loader(
                dataset_configs,
                batch_size=params['batch_size'],
                num_workers=2,
                shuffle=True,
                image_size=512,
                grid_h=16,
                grid_w=16,
                num_classes=6
            )

            val_loader = build_real_medical_loader(
                val_dataset_configs,
                batch_size=params['batch_size'],
                num_workers=2,
                shuffle=False,
                image_size=512,
                grid_h=16,
                grid_w=16,
                num_classes=6
            )
        except Exception as e:
            self.logger.error(f"Failed to build data loaders: {e}")
            return 0.0

        # モデル構築
        model = PatchMoEModel(
            in_ch=3,
            feat_dim=params['feat_dim'],
            grid_h=16,
            grid_w=16,
            num_classes=6,
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            num_queries=256,
            num_datasets=3,
            num_images=100000,
            gate_top_k=params['top_k'],
            gate_capacity=params['capacity_factor'],
            gate_noise=params['gate_noise'],
            backbone='resnet50',
            pretrained_backbone=True,
            experts_per_device=params['experts_per_device'],
            use_multiscale=True
        ).to(self.device)

        # 損失関数
        criterion = PatchMoECombinedLoss(
            num_classes=6,
            dice_weight=params['dice_weight'],
            focal_weight=params['focal_weight'],
            contrastive_weight=params['contrastive_weight'],
            moe_weight=params['moe_weight']
        )

        contrastive_criterion = AdvancedPatchContrastiveLoss(
            temperature=params['contrastive_temperature'],
            hard_negative_weight=params['hard_negative_weight'],
            domain_separation_weight=params['domain_separation_weight']
        )

        # オプティマイザー
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        # 短時間訓練（3エポック）
        best_val_dice = 0.0

        for epoch in range(3):
            # 訓練
            model.train()
            for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(train_loader):
                if batch_idx >= 10:  # 短縮訓練
                    break

                images = images.to(self.device)
                dataset_ids = dataset_ids.to(self.device)
                image_ids = image_ids.to(self.device)
                masks = masks.to(self.device)
                classes = classes.to(self.device)

                optimizer.zero_grad()

                # フォワードパス
                logits, pred_masks = model(images, dataset_ids, image_ids)

                # 対照学習損失
                contrastive_loss = contrastive_criterion(
                    logits, dataset_ids, image_ids)

                # MoE補助損失
                moe_aux_losses = []
                for layer in model.decoder.layers:
                    if hasattr(layer, 'last_moe_aux') and layer.last_moe_aux is not None:
                        moe_aux_losses.append(layer.last_moe_aux)

                moe_aux_loss = sum(moe_aux_losses) if moe_aux_losses else torch.tensor(
                    0.0, device=self.device)

                # 統合損失
                loss_dict = criterion(
                    logits, pred_masks, classes, masks,
                    contrastive_loss=contrastive_loss,
                    moe_aux_loss=moe_aux_loss
                )

                total_loss = loss_dict['total_loss']
                total_loss.backward()

                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)

                optimizer.step()

            # 検証
            model.eval()
            val_dices = []

            with torch.no_grad():
                for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(val_loader):
                    if batch_idx >= 5:  # 短縮評価
                        break

                    images = images.to(self.device)
                    dataset_ids = dataset_ids.to(self.device)
                    image_ids = image_ids.to(self.device)
                    masks = masks.to(self.device)
                    classes = classes.to(self.device)

                    logits, pred_masks = model(images, dataset_ids, image_ids)

                    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                    if pred_binary.shape != masks.shape:
                        pred_binary = torch.nn.functional.interpolate(
                            pred_binary, size=masks.shape[-2:], mode='bilinear', align_corners=False
                        )

                    dice = dice_coefficient(pred_binary, masks)
                    val_dices.append(dice if isinstance(
                        dice, float) else dice.item())

            avg_val_dice = np.mean(val_dices) if val_dices else 0.0
            best_val_dice = max(best_val_dice, avg_val_dice)

        return best_val_dice

    def optimize(self):
        """最適化の実行"""
        self.logger.info(
            f"Starting hyperparameter optimization with {self.n_trials} trials")

        # Optunaスタディの作成
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )

        # 最適化実行
        study.optimize(self.objective, n_trials=self.n_trials)

        # 結果の保存
        self._save_results(study)

        return study.best_params, study.best_value

    def _save_results(self, study: optuna.Study):
        """結果の保存"""
        # 最良パラメータ
        best_params = study.best_params
        best_value = study.best_value

        results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }

        # JSONファイルに保存
        with open(os.path.join(self.output_dir, 'optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # 最良パラメータを個別ファイルに保存
        with open(os.path.join(self.output_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)

        # 可視化
        try:
            import matplotlib.pyplot as plt

            # 最適化履歴
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # スコアの履歴
            trial_numbers = [t.number for t in study.trials]
            trial_values = [
                t.value for t in study.trials if t.value is not None]

            ax1.plot(trial_numbers[:len(trial_values)],
                     trial_values, 'b-', alpha=0.6)
            ax1.axhline(y=best_value, color='r', linestyle='--',
                        label=f'Best: {best_value:.4f}')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Validation Dice Score')
            ax1.set_title('Optimization History')
            ax1.legend()
            ax1.grid(True)

            # パラメータ重要度
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            values = list(importance.values())

            ax2.barh(params, values)
            ax2.set_xlabel('Importance')
            ax2.set_title('Parameter Importance')
            ax2.grid(True, axis='x')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,
                        'optimization_plots.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except ImportError:
            self.logger.warning(
                "Matplotlib not available, skipping visualization")

        self.logger.info(f"Optimization completed!")
        self.logger.info(f"Best score: {best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Results saved to: {self.output_dir}")


def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization for PatchMoE')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of optimization trials')
    parser.add_argument('--device', type=str,
                        default='cuda:0', help='Device to use')

    args = parser.parse_args()

    # 最適化実行
    optimizer = HyperparameterOptimizer(
        device=args.device, n_trials=args.n_trials)
    best_params, best_score = optimizer.optimize()

    print(f"\nOptimization Results:")
    print(f"Best Score: {best_score:.4f}")
    print(f"Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
