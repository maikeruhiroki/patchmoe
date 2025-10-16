# PatchMoE環境移行ガイド

## 🎯 概要
現在のAzure Standard NV12ads A10 v5環境から、より高性能なGPU環境（推奨：ND A100 v4）への移行手順です。

## 📋 移行前の準備（現在のVM）

### 1. GitHubリポジトリへのプッシュ
```bash
# 現在のworkspaceで実行
cd /workspace

# GitHub認証の設定（Personal Access Tokenを使用）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# GitHubにプッシュ（認証が必要）
git push -u origin main
```

**注意**: GitHubのPersonal Access Tokenが必要です。以下の手順で取得してください：
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" → "repo"権限を選択
3. 生成されたトークンをコピーして保存

### 2. 重要なファイルのバックアップ
以下のファイルは手動でバックアップしてください：

```bash
# Kaggle認証情報（機密）
cp kaggle.json ~/kaggle_backup.json

# 学習済みモデル（必要に応じて）
cp -r outputs/ ~/outputs_backup/

# データセット（必要に応じて）
cp -r real_medical_datasets_kaggle/ ~/datasets_backup/
```

## 🚀 新しいVMでの環境構築

### 1. 新しいAzure VMの作成
推奨スペック：
- **VM Type**: ND A100 v4 (1 GPU)
- **GPU**: NVIDIA A100 (80GB VRAM)
- **CPU**: 6 vCPU
- **RAM**: 440 GiB
- **OS**: Ubuntu 20.04 LTS または 22.04 LTS

### 2. 基本的な環境セットアップ
```bash
# システムの更新
sudo apt update && sudo apt upgrade -y

# 必要なパッケージのインストール
sudo apt install -y git curl wget build-essential python3-pip python3-dev

# Python環境の設定
python3 -m pip install --upgrade pip
```

### 3. CUDA環境のセットアップ
```bash
# NVIDIAドライバーの確認
nvidia-smi

# CUDA Toolkitのインストール（必要に応じて）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repository-ubuntu2004-12-1-local_12.1.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2004-12-1-local_12.1.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repository-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### 4. プロジェクトのクローン
```bash
# ワークスペースディレクトリの作成
mkdir -p /workspace
cd /workspace

# GitHubリポジトリのクローン
git clone https://github.com/maikeruhiroki/patchmoe.git .

# または、SSHを使用する場合
# git clone git@github.com:maikeruhiroki/patchmoe.git .
```

### 5. Python環境の構築
```bash
# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate

# 必要なパッケージのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn
pip install opencv-python pillow
pip install tqdm tensorboard
pip install optuna
pip install kaggle

# Tutelのインストール
cd tutel
pip install -e .
cd ..
```

### 6. データセットの再構築
```bash
# Kaggle認証情報の復元
cp ~/kaggle_backup.json kaggle.json
chmod 600 kaggle.json

# Kaggle APIの設定
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/

# データセットのダウンロード
python3 PatchMoE/download_real_datasets.py
```

### 7. 環境の検証
```bash
# GPUの確認
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 基本的なテストの実行
python3 PatchMoE/tests/smoke_test.py
```

## 🔧 トラブルシューティング

### GPU認識の問題
```bash
# NVIDIAドライバーの再インストール
sudo apt purge nvidia-*
sudo apt autoremove
sudo apt install nvidia-driver-525
sudo reboot
```

### CUDA関連のエラー
```bash
# CUDA環境変数の設定
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### メモリ不足の対処
```bash
# バッチサイズの調整
# PatchMoE/config.py で batch_size を小さくする
# または、gradient accumulation を使用
```

## 📊 移行後の確認事項

### 1. パフォーマンステスト
```bash
# GPU要件分析の実行
python3 gpu_requirements_analysis.py

# 簡単な学習テスト
python3 PatchMoE/train_lightweight_unet.py
```

### 2. メモリ使用量の確認
```bash
# リアルタイム監視
watch -n 1 nvidia-smi
```

### 3. 学習の再開
```bash
# 以前の学習結果の復元（必要に応じて）
cp -r ~/outputs_backup/ outputs/

# 新しい学習の開始
python3 PatchMoE/train_kaggle_datasets.py
```

## 💰 コスト最適化

### 1. スポットインスタンスの活用
- Azureスポットインスタンスを使用して最大90%のコスト削減
- 開発時のみVMを起動

### 2. 自動スケーリング
```bash
# VMの自動停止スクリプト
echo '#!/bin/bash
# アイドル時間の監視
while true; do
    if [ $(who | wc -l) -eq 0 ]; then
        sleep 300  # 5分待機
        if [ $(who | wc -l) -eq 0 ]; then
            sudo shutdown -h now
        fi
    fi
    sleep 60
done' > auto_shutdown.sh
chmod +x auto_shutdown.sh
```

## 📞 サポート

移行中に問題が発生した場合：
1. このガイドのトラブルシューティングセクションを確認
2. GPU要件分析レポート（`gpu_requirements_report.json`）を参照
3. 必要に応じて、より小さなバッチサイズやモデルサイズでテスト

## 🎉 移行完了

移行が完了したら：
1. 新しい環境でPatchMoEの学習を開始
2. より大きなバッチサイズで実験
3. より複雑なモデルアーキテクチャのテスト
4. 本格的な研究開発の開始

---

**注意**: このガイドは推奨手順です。実際の環境に応じて調整してください。
