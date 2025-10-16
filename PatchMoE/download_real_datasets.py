#!/usr/bin/env python3
"""
実際の医用画像データセットをダウンロードするスクリプト
Kaggle APIを使用して実際のデータセットを取得
"""
import os
import zipfile
import shutil
import requests
from pathlib import Path
import json
from typing import Dict, List, Tuple
import subprocess
import sys


class RealDatasetDownloader:
    """実際の医用画像データセットのダウンローダー"""

    def __init__(self, output_dir: str = "/workspace/real_medical_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # データセット情報
        self.datasets = {
            'drive': {
                'name': 'DRIVE - Digital Retinal Images for Vessel Extraction',
                'kaggle_id': 'andrewmvd/retinal-vessel-segmentation',
                'description': 'Retinal vessel segmentation dataset',
                'expected_files': ['images', 'masks']
            },
            'stare': {
                'name': 'STARE - Structured Analysis of the Retina',
                'kaggle_id': 'andrewmvd/stare-dataset',
                'description': 'STARE retinal vessel dataset',
                'expected_files': ['images', 'masks']
            },
            'chase': {
                'name': 'CHASE_DB1 - Child Heart and Health Study',
                'kaggle_id': 'andrewmvd/chase-db1',
                'description': 'Child retinal vessel dataset',
                'expected_files': ['images', 'masks']
            }
        }

    def check_kaggle_api(self) -> bool:
        """Kaggle APIが利用可能かチェック"""
        try:
            result = subprocess.run(['kaggle', '--version'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Kaggle API found: {result.stdout.strip()}")
                return True
            else:
                print("Kaggle API not found. Please install kaggle package.")
                return False
        except FileNotFoundError:
            print("Kaggle CLI not found. Please install kaggle package.")
            return False

    def download_kaggle_dataset(self, dataset_id: str, dataset_name: str) -> bool:
        """Kaggleからデータセットをダウンロード"""
        try:
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)

            print(f"Downloading {dataset_name} from Kaggle...")

            # Kaggle APIでダウンロード
            cmd = ['kaggle', 'datasets', 'download',
                   '-d', dataset_id, '-p', str(dataset_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Successfully downloaded {dataset_name}")

                # ZIPファイルを展開
                zip_files = list(dataset_dir.glob("*.zip"))
                for zip_file in zip_files:
                    print(f"Extracting {zip_file.name}...")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    zip_file.unlink()  # ZIPファイルを削除

                return True
            else:
                print(f"Failed to download {dataset_name}: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return False

    def download_alternative_datasets(self):
        """代替データセットのダウンロード（Kaggle APIが利用できない場合）"""
        print("Kaggle API not available. Downloading alternative datasets...")

        # DRIVEデータセットの代替（公開されているサンプル）
        self.download_drive_alternative()

        # その他のデータセットも同様に実装可能

    def download_drive_alternative(self):
        """DRIVEデータセットの代替ダウンロード"""
        drive_dir = self.output_dir / "drive"
        drive_dir.mkdir(exist_ok=True)

        # サンプル画像のURL（実際のDRIVEデータセットのサンプル）
        sample_urls = [
            "https://www.dropbox.com/s/sample1.png",
            "https://www.dropbox.com/s/sample2.png",
            # 実際のURLに置き換える必要があります
        ]

        print("Note: This is a placeholder for actual DRIVE dataset download.")
        print("Please manually download the DRIVE dataset from:")
        print("https://www.isi.uu.nl/Research/Databases/DRIVE/")
        print(f"and place it in: {drive_dir}")

    def organize_dataset_structure(self, dataset_name: str):
        """データセットの構造を整理"""
        dataset_dir = self.output_dir / dataset_name

        if not dataset_dir.exists():
            print(f"Dataset {dataset_name} not found in {dataset_dir}")
            return False

        # 一般的な構造に整理
        images_dir = dataset_dir / "images"
        masks_dir = dataset_dir / "masks"

        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)

        # ファイルを適切なディレクトリに移動
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                filename = file_path.name.lower()

                # 画像ファイル
                if any(ext in filename for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
                    if any(keyword in filename for keyword in ['mask', 'gt', 'ground', 'truth']):
                        # マスクファイル
                        shutil.move(str(file_path), str(
                            masks_dir / file_path.name))
                    else:
                        # 画像ファイル
                        shutil.move(str(file_path), str(
                            images_dir / file_path.name))

        print(f"Organized {dataset_name} dataset structure")
        return True

    def create_dataset_info(self, dataset_name: str):
        """データセット情報ファイルを作成"""
        dataset_dir = self.output_dir / dataset_name
        images_dir = dataset_dir / "images"
        masks_dir = dataset_dir / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            print(f"Dataset {dataset_name} structure not found")
            return

        # 画像とマスクの数をカウント
        image_count = len(list(images_dir.glob("*")))
        mask_count = len(list(masks_dir.glob("*")))

        info = {
            "name": dataset_name,
            "description": self.datasets.get(dataset_name, {}).get("description", ""),
            "image_count": image_count,
            "mask_count": mask_count,
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "status": "ready" if image_count > 0 and mask_count > 0 else "incomplete"
        }

        # 情報ファイルを保存
        with open(dataset_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(
            f"Dataset info created for {dataset_name}: {image_count} images, {mask_count} masks")

    def download_all_datasets(self):
        """すべてのデータセットをダウンロード"""
        print("Starting real medical dataset download...")

        # Kaggle APIの確認
        if self.check_kaggle_api():
            # Kaggleからダウンロード
            for dataset_name, dataset_info in self.datasets.items():
                print(f"\n--- Downloading {dataset_name} ---")
                success = self.download_kaggle_dataset(
                    dataset_info['kaggle_id'],
                    dataset_name
                )

                if success:
                    self.organize_dataset_structure(dataset_name)
                    self.create_dataset_info(dataset_name)
        else:
            # 代替方法
            self.download_alternative_datasets()

        print(f"\nAll datasets downloaded to: {self.output_dir}")
        self.print_summary()

    def print_summary(self):
        """ダウンロード結果のサマリーを表示"""
        print("\n=== Download Summary ===")

        for dataset_name in self.datasets.keys():
            dataset_dir = self.output_dir / dataset_name
            info_file = dataset_dir / "dataset_info.json"

            if info_file.exists():
                with open(info_file, "r") as f:
                    info = json.load(f)
                print(
                    f"{dataset_name}: {info['image_count']} images, {info['mask_count']} masks - {info['status']}")
            else:
                print(f"{dataset_name}: Not downloaded")


def install_kaggle_api():
    """Kaggle APIをインストール"""
    try:
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "kaggle"], check=True)
        print("Kaggle API installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install Kaggle API")
        return False


def main():
    """メイン実行関数"""
    print("Real Medical Dataset Downloader")
    print("=" * 50)

    # Kaggle APIのインストール確認
    downloader = RealDatasetDownloader()

    if not downloader.check_kaggle_api():
        print("\nInstalling Kaggle API...")
        if install_kaggle_api():
            print("Please configure Kaggle API with your credentials:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Create API token")
            print("3. Place kaggle.json in ~/.kaggle/")
            print("4. Run this script again")
        else:
            print("Failed to install Kaggle API. Using alternative methods...")

    # データセットダウンロード
    downloader.download_all_datasets()


if __name__ == "__main__":
    main()
