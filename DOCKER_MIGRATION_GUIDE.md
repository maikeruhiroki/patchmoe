# Docker環境でのPatchMoE移行ガイド

## 🎯 概要
新しいGPUのVMでDockerコンテナを使用してPatchMoE開発環境を構築する手順です。

## 📋 前提条件

### 新しいVMの要件
- **推奨VM**: ND A100 v4 (1 GPU) または ND H100 v5 (1 GPU)
- **OS**: Ubuntu 20.04 LTS または 22.04 LTS
- **GPU**: NVIDIA A100 (80GB) または H100 (80GB)
- **RAM**: 440 GiB以上

### 必要なソフトウェア
- Docker
- Docker Compose
- NVIDIA Container Toolkit

## 🚀 新しいVMでの環境構築

### 1. システムの更新
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Dockerのインストール
```bash
# Dockerのインストール
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER

# Docker Composeのインストール
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 再ログインまたはグループ変更を反映
newgrp docker
```

### 3. NVIDIA Container Toolkitのインストール
```bash
# NVIDIA Container Toolkitのインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 4. プロジェクトのクローン
```bash
# ワークスペースディレクトリの作成
mkdir -p /workspace
cd /workspace

# GitHubリポジトリのクローン
git clone https://github.com/maikeruhiroki/patchmoe.git .
```

### 5. データディレクトリの準備
```bash
# 必要なディレクトリの作成
mkdir -p data outputs medical_datasets real_medical_datasets real_medical_datasets_kaggle test_data

# Kaggle認証情報の設定
mkdir -p ~/.kaggle
# kaggle.jsonを~/.kaggle/に配置
```

### 6. Dockerイメージのビルド
```bash
# Dockerイメージのビルド
docker-compose build patchmoe-dev

# または、直接Dockerfileからビルド
docker build -t patchmoe-dev .
```

### 7. コンテナの起動
```bash
# 開発コンテナの起動
docker-compose up -d patchmoe-dev

# コンテナに入る
docker-compose exec patchmoe-dev bash
```

## 🔧 開発環境の使用

### 基本的な使用方法
```bash
# コンテナの起動
docker-compose up -d patchmoe-dev

# コンテナに入る
docker-compose exec patchmoe-dev bash

# コンテナ内でGPUの確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 基本的なテストの実行
python PatchMoE/tests/smoke_test.py
```

### Jupyter Labの使用
```bash
# Jupyter Labコンテナの起動
docker-compose up -d jupyter

# ブラウザでアクセス
# http://localhost:8889
```

### TensorBoardの使用
```bash
# TensorBoardコンテナの起動
docker-compose up -d tensorboard

# ブラウザでアクセス
# http://localhost:6007
```

## 📊 データセットの再構築

### 1. Kaggleデータセットのダウンロード
```bash
# コンテナ内で実行
docker-compose exec patchmoe-dev bash

# Kaggleデータセットのダウンロード
python PatchMoE/download_real_datasets.py
```

### 2. データセットの確認
```bash
# データセットの確認
ls -la real_medical_datasets_kaggle/
ls -la medical_datasets/
```

## 🧪 学習の実行

### 1. 基本的な学習テスト
```bash
# コンテナ内で実行
docker-compose exec patchmoe-dev bash

# 軽量U-Netの学習テスト
python PatchMoE/train_lightweight_unet.py

# Kaggleデータセットでの学習
python PatchMoE/train_kaggle_datasets.py
```

### 2. GPU要件の確認
```bash
# GPU要件分析の実行
python gpu_requirements_analysis.py
```

## 🔄 開発ワークフロー

### 1. コードの編集
```bash
# ホスト側でコードを編集
# コンテナ内のファイルは自動的に同期される
```

### 2. 学習の実行
```bash
# コンテナ内で学習を実行
docker-compose exec patchmoe-dev python PatchMoE/train_kaggle_datasets.py
```

### 3. 結果の確認
```bash
# ホスト側で結果を確認
ls -la outputs/
```

## 🛠️ トラブルシューティング

### GPU認識の問題
```bash
# NVIDIAドライバーの確認
nvidia-smi

# Docker内でのGPU確認
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

### メモリ不足の問題
```bash
# バッチサイズの調整
# PatchMoE/config.py で batch_size を小さくする

# または、より大きなVMに変更
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
/workspace/
├── Dockerfile                 # Dockerイメージ定義
├── docker-compose.yml         # Docker Compose設定
├── DOCKER_MIGRATION_GUIDE.md  # このガイド
├── PatchMoE/                  # プロジェクトコード
├── data/                      # データファイル（マウント）
├── outputs/                   # 学習結果（マウント）
├── medical_datasets/          # 医用画像データセット（マウント）
└── real_medical_datasets_kaggle/ # Kaggleデータセット（マウント）
```

## 💡 ベストプラクティス

### 1. データの永続化
- 重要なデータはボリュームマウントを使用
- 学習結果は`outputs/`ディレクトリに保存

### 2. 開発効率の向上
- コードはホスト側で編集
- コンテナは学習とテストのみに使用

### 3. リソース管理
- 不要なコンテナは停止
- 定期的にイメージをクリーンアップ

```bash
# 不要なイメージの削除
docker system prune -a

# 特定のコンテナの停止
docker-compose stop jupyter tensorboard
```

## 🎉 移行完了

移行が完了したら：
1. 新しい環境でPatchMoEの学習を開始
2. より大きなバッチサイズで実験
3. より複雑なモデルアーキテクチャのテスト
4. 本格的な研究開発の開始

---

**注意**: このガイドは推奨手順です。実際の環境に応じて調整してください。
