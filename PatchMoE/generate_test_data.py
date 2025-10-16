#!/usr/bin/env python3
"""
実データテスト用のサンプルOCT画像を生成
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T


def generate_oct_like_image(size=(512, 512), num_layers=3):
    """OCT画像のような多層構造を持つ画像を生成"""
    img = np.zeros((size[1], size[0]), dtype=np.uint8)

    # 背景ノイズ
    noise = np.random.normal(0, 10, size).astype(np.uint8)
    img = np.clip(img + noise, 0, 255)

    # 多層構造を描画
    for i in range(num_layers):
        y_start = size[1] // (num_layers + 1) * (i + 1)
        y_end = y_start + np.random.randint(20, 40)

        # 層の境界線
        img[y_start:y_end, :] = np.random.randint(100, 200)

        # 層内の構造
        for j in range(0, size[0], 50):
            x_start = j + np.random.randint(-10, 10)
            x_end = x_start + np.random.randint(20, 40)
            if 0 <= x_start < size[0] and 0 <= x_end < size[0]:
                img[y_start:y_end, x_start:x_end] = np.random.randint(150, 255)

    return Image.fromarray(img, mode='L')


def generate_ground_truth_mask(size=(512, 512), num_regions=3):
    """対応するグランドトゥルースマスクを生成"""
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)

    # 複数の領域を描画
    for i in range(num_regions):
        # ランダムな位置とサイズ
        center_x = np.random.randint(50, size[0] - 50)
        center_y = np.random.randint(50, size[1] - 50)
        radius = np.random.randint(30, 80)

        # 円形領域
        y, x = np.ogrid[:size[1], :size[0]]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_circle] = 255

    return Image.fromarray(mask, mode='L')


def create_test_dataset(output_dir, num_images=10):
    """テスト用データセットを作成"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    for i in range(num_images):
        # OCT画像生成
        oct_img = generate_oct_like_image()
        oct_img.save(f"{output_dir}/images/oct_{i:03d}.png")

        # 対応するマスク生成
        mask = generate_ground_truth_mask()
        mask.save(f"{output_dir}/masks/mask_{i:03d}.png")

        print(f"Generated: oct_{i:03d}.png, mask_{i:03d}.png")

    print(f"Test dataset created in {output_dir}")
    print(f"Images: {output_dir}/images/")
    print(f"Masks: {output_dir}/masks/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='/workspace/test_data')
    parser.add_argument('--num_images', type=int, default=10)
    args = parser.parse_args()

    create_test_dataset(args.out, args.num_images)
