# バックアップチェックリスト

## 🔐 機密情報（必須バックアップ）
- [ ] `kaggle.json` - Kaggle API認証情報
- [ ] 学習済みモデル（`outputs/`ディレクトリ）
- [ ] 実験結果（`gpu_requirements_report.json`など）

## 📊 重要なデータファイル
- [ ] `real_medical_datasets_kaggle/` - 実際の医用画像データセット
- [ ] `medical_datasets/` - ダミーデータセット
- [ ] `test_data/` - テストデータ
- [ ] `outputs/` - 学習結果とチェックポイント

## 💻 設定ファイル
- [ ] Python環境のrequirements.txt（作成が必要）
- [ ] CUDA環境設定
- [ ] システム設定

## 📝 手動バックアップコマンド
```bash
# 機密情報のバックアップ
cp kaggle.json ~/kaggle_backup.json

# 学習結果のバックアップ
cp -r outputs/ ~/outputs_backup/

# データセットのバックアップ
cp -r real_medical_datasets_kaggle/ ~/datasets_backup/
cp -r medical_datasets/ ~/medical_datasets_backup/

# 設定ファイルのバックアップ
cp -r ~/.kaggle/ ~/kaggle_config_backup/
```

## 🚀 新しいVMでの復元コマンド
```bash
# 機密情報の復元
cp ~/kaggle_backup.json kaggle.json
chmod 600 kaggle.json

# 学習結果の復元
cp -r ~/outputs_backup/ outputs/

# データセットの復元
cp -r ~/datasets_backup/ real_medical_datasets_kaggle/
cp -r ~/medical_datasets_backup/ medical_datasets/

# Kaggle設定の復元
cp -r ~/kaggle_config_backup/ ~/.kaggle/
```
