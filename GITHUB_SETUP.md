# GitHubリポジトリ設定ガイド

## 🔐 GitHub認証の設定

### 方法1: Personal Access Token（推奨）

1. **GitHubでPersonal Access Tokenを作成**
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - "Generate new token (classic)" をクリック
   - 以下の権限を選択：
     - `repo` (Full control of private repositories)
     - `workflow` (Update GitHub Action workflows)
   - トークンをコピーして安全な場所に保存

2. **Git認証の設定**
   ```bash
   # Git認証情報の設定
   git config --global credential.helper store
   
   # または、一時的に認証情報を設定
   git config --global user.name "maikeruhiroki"
   git config --global user.email "your.email@example.com"
   ```

3. **GitHubにプッシュ**
   ```bash
   cd /workspace
   git push -u origin main
   # Username: maikeruhiroki
   # Password: 上記で作成したPersonal Access Token
   ```

4. **認証が失敗する場合の対処法**
   ```bash
   # 認証情報をクリア
   git config --global --unset credential.helper
   
   # 手動でURLにトークンを含める（一時的）
   git remote set-url origin https://maikeruhiroki:YOUR_TOKEN@github.com/maikeruhiroki/patchmoe.git
   git push -u origin main
   ```

### 方法2: SSH Key（セキュア）

1. **SSH Keyの生成**
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   # ファイル名はデフォルトのまま（Enter）
   # パスフレーズは設定推奨
   ```

2. **公開鍵をGitHubに追加**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   # 出力された内容をコピー
   ```
   - GitHub → Settings → SSH and GPG keys → New SSH key
   - コピーした公開鍵を貼り付け

3. **SSH接続のテスト**
   ```bash
   ssh -T git@github.com
   ```

4. **リモートURLをSSHに変更**
   ```bash
   git remote set-url origin git@github.com:maikeruhiroki/patchmoe.git
   git push -u origin main
   ```

## 🚀 リポジトリの確認

プッシュが成功したら、以下のURLでリポジトリを確認できます：
https://github.com/maikeruhiroki/patchmoe

## 📋 次のステップ

1. **新しいVMでのクローン**
   ```bash
   git clone https://github.com/maikeruhiroki/patchmoe.git
   # または SSH を使用
   git clone git@github.com:maikeruhiroki/patchmoe.git
   ```

2. **移行ガイドの実行**
   - `MIGRATION_GUIDE.md` を参照
   - `BACKUP_CHECKLIST.md` でバックアップを確認

## ⚠️ 注意事項

- `kaggle.json` などの機密情報はGitHubにプッシュされません（.gitignoreで除外）
- 大きなデータファイルも除外されています
- 新しいVMでは手動でデータセットを再構築する必要があります
