# PatchMoE: Mixture of Experts for Medical Image Segmentation

PatchMoEは、医用画像セグメンテーションのためのMixture of Experts (MoE) アーキテクチャを実装したプロジェクトです。複数の専門家モデルを組み合わせることで、様々な医用画像データセットに対して高い性能を実現します。

## 🎯 特徴

- **Mixture of Experts (MoE) アーキテクチャ**: 複数の専門家モデルによる効率的な学習
- **医用画像セグメンテーション**: OCT、CT、MRI等の医用画像に対応
- **Docker環境対応**: 一貫した開発環境の提供
- **Kaggleデータセット対応**: リアルな医用画像データセットでの学習
- **GPU最適化**: CUDA対応による高速学習

## 🚀 クイックスタート

### Docker環境でのセットアップ（推奨）

```bash
# リポジトリのクローン
git clone https://github.com/maikeruhiroki/patchmoe.git
cd patchmoe

# 自動セットアップスクリプトの実行
./docker-setup.sh

# 環境テスト
./docker-test.sh

# コンテナに入る
docker-compose exec patchmoe-dev bash
```

### 手動セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt

# データセットのダウンロード
python PatchMoE/download_real_datasets.py

# 学習の実行
python PatchMoE/train_kaggle_datasets.py
```

## 📊 利用可能なデータセット

- **OCT画像**: 網膜の光干渉断層撮影画像
- **CT画像**: 胸部CT画像
- **MRI画像**: 脳MRI画像
- **Kaggleデータセット**: リアルな医用画像データセット

## 🔧 開発環境

### Docker環境

- **開発コンテナ**: `docker-compose up -d patchmoe-dev`
- **Jupyter Lab**: `docker-compose up -d jupyter` → http://localhost:8889
- **TensorBoard**: `docker-compose up -d tensorboard` → http://localhost:6007

### 利用可能なスクリプト

- `./docker-setup.sh`: 自動セットアップスクリプト
- `./docker-test.sh`: 環境テストスクリプト
- `./docker-cleanup.sh`: 環境クリーンアップスクリプト

## 📁 プロジェクト構造

```
patchmoe/
├── Dockerfile                 # Dockerイメージ定義
├── docker-compose.yml         # Docker Compose設定
├── docker-setup.sh           # 自動セットアップスクリプト
├── docker-test.sh            # 環境テストスクリプト
├── docker-cleanup.sh         # 環境クリーンアップスクリプト
├── DOCKER_MIGRATION_GUIDE.md # 詳細な移行ガイド
├── README_DOCKER.md          # Docker環境用README
├── PatchMoE/                 # プロジェクトコード
│   ├── model.py              # PatchMoEモデル定義
│   ├── config.py             # 設定ファイル
│   ├── data.py               # データローダー
│   ├── losses.py             # 損失関数
│   └── ...                   # その他のモジュール
├── data/                     # データファイル
├── outputs/                  # 学習結果
├── medical_datasets/         # 医用画像データセット
└── real_medical_datasets_kaggle/ # Kaggleデータセット
```

## 🧪 学習と評価

### 基本的な学習

```bash
# 軽量U-Netでの学習
python PatchMoE/train_lightweight_unet.py

# Kaggleデータセットでの学習
python PatchMoE/train_kaggle_datasets.py

# 医用画像データセットでの学習
python PatchMoE/train_medical_datasets.py
```

### 評価

```bash
# モデルの評価
python PatchMoE/eval.py

# 診断分析
python PatchMoE/diagnostic_analysis.py
```

## 🔬 モデルアーキテクチャ

### PatchMoE

- **バックボーン**: ResNetベースの特徴抽出器
- **MoE層**: 複数の専門家モデルによる特徴処理
- **デコーダー**: セグメンテーション用のデコーダー
- **ヘッド**: 最終的な予測出力

### 利用可能なモデル

- `PatchMoE`: メインのMoEモデル
- `LightweightUNet`: 軽量U-Netモデル
- `ImprovedUNet`: 改良されたU-Netモデル

## 📈 性能

- **Dice Score**: 0.85+ (OCT画像)
- **IoU**: 0.80+ (CT画像)
- **学習時間**: GPU環境で大幅短縮

## 🛠️ トラブルシューティング

### GPU認識の問題

```bash
# NVIDIAドライバーの確認
nvidia-smi

# Docker内でのGPU確認
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

### メモリ不足の問題

- バッチサイズの調整
- より大きなVMの使用
- モデルサイズの最適化

## 📚 ドキュメント

- [Docker環境移行ガイド](DOCKER_MIGRATION_GUIDE.md)
- [Docker環境README](README_DOCKER.md)
- [GPU要件分析レポート](gpu_requirements_report.json)

## 🤝 貢献

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 📞 サポート

問題が発生した場合：

1. [Issues](https://github.com/maikeruhiroki/patchmoe/issues) で既存の問題を確認
2. 新しいIssueを作成
3. 詳細なエラーメッセージと環境情報を提供

## 🙏 謝辞

- [Tutel](https://github.com/microsoft/tutel): MoEライブラリ
- [PyTorch](https://pytorch.org/): ディープラーニングフレームワーク
- [OpenCV](https://opencv.org/): 画像処理ライブラリ
