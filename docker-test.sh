#!/bin/bash
# Docker環境テストスクリプト

set -e

echo "🧪 PatchMoE Docker環境テストを開始します..."

# コンテナが起動しているかチェック
if ! docker-compose ps patchmoe-dev | grep -q "Up"; then
    echo "❌ patchmoe-devコンテナが起動していません"
    echo "まず docker-compose up -d patchmoe-dev を実行してください"
    exit 1
fi

echo "✅ patchmoe-devコンテナが起動中です"

# GPU確認テスト
echo "🎮 GPU確認テスト..."
docker-compose exec patchmoe-dev python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA is not available')
    exit(1)
"

# Python環境テスト
echo "🐍 Python環境テスト..."
docker-compose exec patchmoe-dev python -c "
import sys
print(f'Python version: {sys.version}')

# 主要ライブラリのインポートテスト
try:
    import torch
    print('✅ PyTorch imported successfully')
    import torchvision
    print('✅ TorchVision imported successfully')
    import numpy
    print('✅ NumPy imported successfully')
    import cv2
    print('✅ OpenCV imported successfully')
    import kaggle
    print('✅ Kaggle imported successfully')
    import tensorboard
    print('✅ TensorBoard imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# PatchMoEモジュールテスト
echo "🔬 PatchMoEモジュールテスト..."
docker-compose exec patchmoe-dev python -c "
import sys
sys.path.append('/workspace')

try:
    from PatchMoE import model
    print('✅ PatchMoE model imported successfully')
    
    from PatchMoE import config
    print('✅ PatchMoE config imported successfully')
    
    from PatchMoE import data
    print('✅ PatchMoE data imported successfully')
    
    from PatchMoE import losses
    print('✅ PatchMoE losses imported successfully')
    
except ImportError as e:
    print(f'❌ PatchMoE import error: {e}')
    exit(1)
"

# ファイルシステムテスト
echo "📁 ファイルシステムテスト..."
docker-compose exec patchmoe-dev bash -c "
if [ -d '/workspace/PatchMoE' ]; then
    echo '✅ PatchMoE directory exists'
else
    echo '❌ PatchMoE directory not found'
    exit 1
fi

if [ -d '/workspace/data' ]; then
    echo '✅ data directory exists'
else
    echo '❌ data directory not found'
    exit 1
fi

if [ -d '/workspace/outputs' ]; then
    echo '✅ outputs directory exists'
else
    echo '❌ outputs directory not found'
    exit 1
fi
"

# 簡単なモデル作成テスト
echo "🏗️ モデル作成テスト..."
docker-compose exec patchmoe-dev python -c "
import torch
import sys
sys.path.append('/workspace')

try:
    from PatchMoE.model import PatchMoE
    from PatchMoE.config import Config
    
    # 設定の読み込み
    config = Config()
    
    # モデルの作成
    model = PatchMoE(config)
    print('✅ PatchMoE model created successfully')
    
    # ダミー入力でのテスト
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f'✅ Model forward pass successful, output shape: {output.shape}')
        
except Exception as e:
    print(f'❌ Model test error: {e}')
    exit(1)
"

echo ""
echo "🎉 全てのテストが完了しました！"
echo ""
echo "📋 テスト結果:"
echo "✅ GPU認識: OK"
echo "✅ Python環境: OK"
echo "✅ ライブラリインポート: OK"
echo "✅ PatchMoEモジュール: OK"
echo "✅ ファイルシステム: OK"
echo "✅ モデル作成: OK"
echo ""
echo "🚀 PatchMoE Docker環境は正常に動作しています！"
echo ""
echo "次のステップ:"
echo "1. データセットのダウンロード: docker-compose exec patchmoe-dev python PatchMoE/download_real_datasets.py"
echo "2. 学習の実行: docker-compose exec patchmoe-dev python PatchMoE/train_kaggle_datasets.py"
