import os
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging
import datetime
import matplotlib.pyplot as plt

from PatchMoE.config import PatchMoEConfig
from PatchMoE.model import PatchMoEModel
from PatchMoE.kaggle_dataset_loader import build_kaggle_medical_loader, UnifiedKaggleMedicalDataset
from PatchMoE.losses import PatchMoECombinedLoss
from PatchMoE.contrastive import PatchContrastiveLoss
from PatchMoE.eval import dice_score as dice_coefficient, miou


class KaggleMedicalDatasetTrainer:
    """Kaggle医用画像データセット用の訓練クラス"""

    def __init__(self, cfg: PatchMoEConfig, device: torch.device, output_dir: str):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_logging()
        self.logger.info(f"Configuration: {cfg}")

        self.train_loader, self.val_loader = self._build_loaders()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()
        self.scheduler = self._build_scheduler()

        self.best_val_dice = -1.0
        self.best_val_miou = -1.0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_miou': [], 'val_miou': []
        }

    def _setup_logging(self):
        log_file = os.path.join(
            self.output_dir, f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler()
                            ])
        self.logger = logging.getLogger(__name__)

    def _build_loaders(self):
        self.logger.info("Building data loaders for Kaggle datasets...")
        # Kaggleデータセット設定
        dataset_configs = [
            {'type': 'drive', 'root_dir': '/workspace/real_medical_datasets_kaggle',
                'split': 'train'},
            {'type': 'kvasir_seg',
                'root_dir': '/workspace/real_medical_datasets_kaggle', 'split': 'train'},
            {'type': 'synapse', 'root_dir': '/workspace/real_medical_datasets_kaggle',
                'split': 'train'},
            {'type': 'retina_blood_vessel',
                'root_dir': '/workspace/real_medical_datasets_kaggle', 'split': 'train'},
        ]

        train_loader = build_kaggle_medical_loader(
            dataset_configs=[
                cfg for cfg in dataset_configs if cfg['split'] == 'train'],
            batch_size=self.cfg.batch_size,
            image_size=512,
            num_classes=self.cfg.num_classes,
            grid_h=self.cfg.grid_h,
            grid_w=self.cfg.grid_w,
            num_workers=self.cfg.num_workers,
            shuffle=True
        )

        val_loader = build_kaggle_medical_loader(
            dataset_configs=[
                cfg for cfg in dataset_configs if cfg['split'] == 'val'],
            batch_size=self.cfg.batch_size,
            image_size=512,
            num_classes=self.cfg.num_classes,
            grid_h=self.cfg.grid_h,
            grid_w=self.cfg.grid_w,
            num_workers=self.cfg.num_workers,
            shuffle=False
        )

        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")

        return train_loader, val_loader

    def _build_model(self) -> PatchMoEModel:
        self.logger.info("Building model...")
        # num_datasetsはUnifiedKaggleMedicalDatasetで実際にロードされたデータセットの数に合わせる
        num_datasets_in_loader = len(self.train_loader.dataset.datasets)
        model = PatchMoEModel(
            in_ch=self.cfg.in_channels,
            feat_dim=self.cfg.feat_dim,
            grid_h=self.cfg.grid_h,
            grid_w=self.cfg.grid_w,
            num_classes=self.cfg.num_classes,
            num_layers=self.cfg.num_layers,
            num_heads=self.cfg.num_heads,
            num_queries=self.cfg.num_queries,
            num_datasets=num_datasets_in_loader,
            num_images=self.cfg.num_images_cap,
            gate_top_k=self.cfg.top_k,
            gate_capacity=self.cfg.capacity_factor,
            gate_noise=self.cfg.gate_noise,
            backbone='resnet50',
            pretrained_backbone=True,
            experts_per_device=self.cfg.experts_per_device,
            use_multiscale=True
        ).to(self.device)
        self.logger.info(
            "Model built successfully with ResNet50 backbone and multiscale features.")
        return model

    def _build_optimizer(self) -> optim.Optimizer:
        self.logger.info("Building optimizer...")
        return optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def _build_criterion(self) -> PatchMoECombinedLoss:
        self.logger.info("Building loss criterion...")
        return PatchMoECombinedLoss(
            num_classes=self.cfg.num_classes,
            dice_weight=1.0,
            focal_weight=1.0,
            contrastive_weight=0.1,
            moe_weight=0.01
        )

    def _build_scheduler(self):
        self.logger.info("Building learning rate scheduler...")
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        epoch_losses = []
        epoch_dices = []
        epoch_mious = []
        epoch_contrastive_losses = []
        epoch_moe_losses = []

        pbar = tqdm(loader, desc=f"Training", leave=False)
        for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(pbar):
            images = images.to(self.device)
            dataset_ids = dataset_ids.to(self.device)
            image_ids = image_ids.to(self.device)
            masks = masks.to(self.device)
            classes = classes.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images, dataset_ids, image_ids)
            if len(outputs) == 2:
                cls_logits, pred_masks = outputs
                patch_features = None
                moe_aux_losses = []
            else:
                cls_logits, pred_masks, patch_features, moe_aux_losses = outputs

            # Contrastive lossの計算
            contrastive_loss = torch.tensor(0.0, device=self.device)
            if patch_features is not None:
                contrastive_loss = PatchContrastiveLoss()(
                    patch_features, dataset_ids, image_ids)

            loss_dict = self.criterion(
                cls_logits, pred_masks, classes, masks,
                contrastive_loss=contrastive_loss,
                moe_aux_loss=sum(moe_aux_losses) if moe_aux_losses else torch.tensor(
                    0.0, device=self.device)
            )
            total_loss = loss_dict['total_loss']

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                if pred_binary.shape != masks.shape:
                    pred_binary = torch.nn.functional.interpolate(
                        pred_binary, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                dice = dice_coefficient(pred_binary, masks)
                miou_val = miou(pred_binary, masks)

                epoch_losses.append(total_loss.item())
                epoch_dices.append(dice if isinstance(
                    dice, float) else dice.item())
                epoch_mious.append(miou_val if isinstance(
                    miou_val, float) else miou_val.item())
                epoch_contrastive_losses.append(
                    loss_dict['contrastive_loss'].item())
                epoch_moe_losses.append(loss_dict['moe_loss'].item())

                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{dice if isinstance(dice, float) else dice.item():.4f}',
                    'mIoU': f'{miou_val if isinstance(miou_val, float) else miou_val.item():.4f}'
                })

        self.scheduler.step()

        return {
            'loss': sum(epoch_losses) / len(epoch_losses),
            'dice': sum(epoch_dices) / len(epoch_dices),
            'miou': sum(epoch_mious) / len(epoch_mious),
            'contrastive_loss': sum(epoch_contrastive_losses) / len(epoch_contrastive_losses),
            'moe_loss': sum(epoch_moe_losses) / len(epoch_moe_losses),
        }

    def validate_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        epoch_losses = []
        epoch_dices = []
        epoch_mious = []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Validation", leave=False)
            for batch_idx, (images, dataset_ids, image_ids, masks, classes) in enumerate(pbar):
                images = images.to(self.device)
                dataset_ids = dataset_ids.to(self.device)
                image_ids = image_ids.to(self.device)
                masks = masks.to(self.device)
                classes = classes.to(self.device)

                outputs = self.model(images, dataset_ids, image_ids)

                if len(outputs) == 2:
                    cls_logits, pred_masks = outputs
                    patch_features = None
                    moe_aux_losses = []
                else:
                    cls_logits, pred_masks, patch_features, moe_aux_losses = outputs

                # Contrastive lossの計算
                contrastive_loss = torch.tensor(0.0, device=self.device)
                if patch_features is not None:
                    contrastive_loss = PatchContrastiveLoss()(
                        patch_features, dataset_ids, image_ids)

                loss_dict = self.criterion(
                    cls_logits, pred_masks, classes, masks,
                    contrastive_loss=contrastive_loss,
                    moe_aux_loss=sum(moe_aux_losses) if moe_aux_losses else torch.tensor(
                        0.0, device=self.device)
                )
                total_loss = loss_dict['total_loss']

                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                if pred_binary.shape != masks.shape:
                    pred_binary = torch.nn.functional.interpolate(
                        pred_binary, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                dice = dice_coefficient(pred_binary, masks)
                miou_val = miou(pred_binary, masks)

                epoch_losses.append(total_loss.item())
                epoch_dices.append(dice if isinstance(
                    dice, float) else dice.item())
                epoch_mious.append(miou_val if isinstance(
                    miou_val, float) else miou_val.item())

                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Dice': f'{dice if isinstance(dice, float) else dice.item():.4f}',
                    'mIoU': f'{miou_val if isinstance(miou_val, float) else miou_val.item():.4f}'
                })

        # 検証データが空の場合の処理
        if len(epoch_losses) == 0:
            return {
                'loss': 0.0,
                'dice': 0.0,
                'miou': 0.0,
            }

        return {
            'loss': sum(epoch_losses) / len(epoch_losses),
            'dice': sum(epoch_dices) / len(epoch_dices),
            'miou': sum(epoch_mious) / len(epoch_mious),
        }

    def plot_training_history(self):
        """訓練履歴をプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.training_history['train_loss'],
                        label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Dice Score
        axes[0, 1].plot(self.training_history['train_dice'],
                        label='Train Dice')
        axes[0, 1].plot(self.training_history['val_dice'], label='Val Dice')
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # mIoU
        axes[1, 0].plot(self.training_history['train_miou'],
                        label='Train mIoU')
        axes[1, 0].plot(self.training_history['val_miou'], label='Val mIoU')
        axes[1, 0].set_title('Training and Validation mIoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mIoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Combined metrics
        axes[1, 1].plot(self.training_history['val_dice'],
                        label='Val Dice', marker='o')
        axes[1, 1].plot(self.training_history['val_miou'],
                        label='Val mIoU', marker='s')
        axes[1, 1].set_title('Validation Metrics Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                    'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.cfg.epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")
            train_metrics = self.train_epoch(self.train_loader)
            val_metrics = self.validate_epoch(self.val_loader)

            # 履歴を更新
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_dice'].append(train_metrics['dice'])
            self.training_history['val_dice'].append(val_metrics['dice'])
            self.training_history['train_miou'].append(train_metrics['miou'])
            self.training_history['val_miou'].append(val_metrics['miou'])

            self.logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, mIoU: {train_metrics['miou']:.4f}")
            self.logger.info(
                f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, mIoU: {val_metrics['miou']:.4f}")

            # 最良モデルの保存
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.best_val_miou = val_metrics['miou']

                model_path = os.path.join(self.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(
                    f"Saved best model to {model_path} with Dice: {self.best_val_dice:.4f}")

                metrics_path = os.path.join(
                    self.output_dir, "best_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'dice': self.best_val_dice,
                        'miou': self.best_val_miou,
                        'epoch': epoch + 1
                    }, f, indent=2)

        # 訓練履歴をプロット
        self.plot_training_history()

        self.logger.info("Training completed!")
        self.logger.info(f"Best Dice: {self.best_val_dice:.4f}")
        self.logger.info(f"Best mIoU: {self.best_val_miou:.4f}")
        self.logger.info(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--feat_dim', type=int,
                        default=128, help='Feature dimension')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (binary segmentation)')
    parser.add_argument('--grid_h', type=int, default=16,
                        help='Grid height for patches')
    parser.add_argument('--grid_w', type=int, default=16,
                        help='Grid width for patches')
    parser.add_argument('--num_datasets', type=int,
                        default=4, help='Number of Kaggle datasets')
    parser.add_argument('--num_images_cap', type=int,
                        default=10000, help='Total number of images in dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/workspace/outputs/patchmoe_kaggle', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available()
                        else 'cpu', help='Device to use for training')

    args = parser.parse_args()

    cfg = PatchMoEConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        feat_dim=args.feat_dim,
        num_classes=args.num_classes,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        num_datasets=args.num_datasets,
        num_images_cap=args.num_images_cap,
        num_queries=args.grid_h * args.grid_w,
        out_dir=args.output_dir
    )

    trainer = KaggleMedicalDatasetTrainer(
        cfg, torch.device(args.device), args.output_dir)
    trainer.train()


if __name__ == '__main__':
    main()
