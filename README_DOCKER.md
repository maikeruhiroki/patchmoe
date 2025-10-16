# PatchMoE Docker環境

## 🎯 概要
PatchMoE開発用のDocker環境です。新しいGPUのVMで一貫した開発環境を提供します。

## 🚀 クイックスタート

### 1. 環境のセットアップ
```bash
# リポジトリのクローン
git clone https://github.com/maikeruhiroki/patchmoe.git
cd patchmoe

# 自動セットアップスクリプトの実行
./docker-setup.sh
```

### 2. 手動セットアップ
```bash
# 必要なディレクトリの作成
mkdir -p data outputs medical_datasets real_medical_datasets real_medical_datasets_kaggle test_data

# Kaggle認証情報の設定
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Dockerイメージのビルド
docker-compose build patchmoe-dev

# コンテナの起動
docker-compose up -d patchmoe-dev
```

### 3. 開発環境の使用
```bash
# コンテナに入る
docker-compose exec patchmoe-dev bash

# GPUの確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# データセットのダウンロード
python PatchMoE/download_real_datasets.py

# 学習の実行
python PatchMoE/train_kaggle_datasets.py
```

## 📊 利用可能なサービス

### 開発コンテナ
```bash
# 起動
docker-compose up -d patchmoe-dev

# 入る
docker-compose exec patchmoe-dev bash
```

### Jupyter Lab
```bash
# 起動
docker-compose up -d jupyter

# アクセス
# http://localhost:8889
```

### TensorBoard
```bash
# 起動
docker-compose up -d tensorboard

# アクセス
# http://localhost:6007
```

## 🔧 トラブルシューティング

### GPU認識の問題
```bash
# NVIDIAドライバーの確認
nvidia-smi

# Docker内でのGPU確認
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

### コンテナの再ビルド
```bash
# コンテナの停止
docker-compose down

# イメージの再ビルド
docker-compose build --no-cache patchmoe-dev

# コンテナの再起動
docker-compose up -d patchmoe-dev
```

## 📁 ディレクトリ構造

```
patchmoe/
├── Dockerfile                 # Dockerイメージ定義
├── docker-compose.yml         # Docker Compose設定
├── docker-setup.sh           # 自動セットアップスクリプト
├── DOCKER_MIGRATION_GUIDE.md # 詳細な移行ガイド
├── PatchMoE/                 # プロジェクトコード
├── data/                     # データファイル（マウント）
├── outputs/                  # 学習結果（マウント）
├── medical_datasets/         # 医用画像データセット（マウント）
└── real_medical_datasets_kaggle/ # Kaggleデータセット（マウント）
```

## 💡 ベストプラクティス

1. **データの永続化**: 重要なデータはボリュームマウントを使用
2. **開発効率**: コードはホスト側で編集、コンテナは学習とテストのみ
3. **リソース管理**: 不要なコンテナは停止、定期的にイメージをクリーンアップ

## 📞 サポート

問題が発生した場合：
1. `DOCKER_MIGRATION_GUIDE.md` のトラブルシューティングセクションを確認
2. GPU要件分析レポート（`gpu_requirements_report.json`）を参照
3. 必要に応じて、より小さなバッチサイズやモデルサイズでテスト
