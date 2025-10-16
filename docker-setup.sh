#!/bin/bash
# Docker環境セットアップスクリプト

set -e

echo "🐳 PatchMoE Docker環境セットアップを開始します..."

# 必要なディレクトリの作成
echo "📁 必要なディレクトリを作成中..."
mkdir -p data outputs medical_datasets real_medical_datasets real_medical_datasets_kaggle test_data

# Kaggle認証情報の確認
echo "🔐 Kaggle認証情報を確認中..."
if [ -f "kaggle.json" ]; then
    echo "✅ kaggle.jsonが見つかりました"
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✅ Kaggle認証情報を設定しました"
else
    echo "⚠️  kaggle.jsonが見つかりません。手動で設定してください。"
fi

# Dockerイメージのビルド
echo "🔨 Dockerイメージをビルド中..."
docker-compose build patchmoe-dev

# コンテナの起動
echo "🚀 開発コンテナを起動中..."
docker-compose up -d patchmoe-dev

# GPU確認
echo "🎮 GPUの確認中..."
docker-compose exec patchmoe-dev python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA is not available')
"

echo "✅ Docker環境セットアップが完了しました！"
echo ""
echo "📋 次のステップ:"
echo "1. コンテナに入る: docker-compose exec patchmoe-dev bash"
echo "2. データセットをダウンロード: python PatchMoE/download_real_datasets.py"
echo "3. 学習を開始: python PatchMoE/train_kaggle_datasets.py"
echo ""
echo "🌐 アクセスURL:"
echo "- Jupyter Lab: http://localhost:8889"
echo "- TensorBoard: http://localhost:6007"
