#!/bin/bash
# Docker環境クリーンアップスクリプト

echo "🧹 PatchMoE Docker環境クリーンアップを開始します..."

# 確認メッセージ
read -p "全てのDockerコンテナとイメージを削除しますか？ (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ クリーンアップをキャンセルしました"
    exit 1
fi

# コンテナの停止と削除
echo "🛑 コンテナの停止中..."
docker-compose down --remove-orphans

# PatchMoE関連のコンテナを削除
echo "🗑️ PatchMoE関連コンテナの削除中..."
docker container rm -f patchmoe-development patchmoe-jupyter patchmoe-tensorboard 2>/dev/null || true

# PatchMoE関連のイメージを削除
echo "🗑️ PatchMoE関連イメージの削除中..."
docker image rm -f patchmoe_patchmoe-dev patchmoe_jupyter patchmoe_tensorboard 2>/dev/null || true

# 使用されていないイメージの削除
echo "🗑️ 使用されていないイメージの削除中..."
docker image prune -f

# 使用されていないボリュームの削除
echo "🗑️ 使用されていないボリュームの削除中..."
docker volume prune -f

# 使用されていないネットワークの削除
echo "🗑️ 使用されていないネットワークの削除中..."
docker network prune -f

# システム全体のクリーンアップ（オプション）
read -p "システム全体のクリーンアップも実行しますか？ (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️ システム全体のクリーンアップ中..."
    docker system prune -a -f --volumes
fi

echo ""
echo "✅ Docker環境クリーンアップが完了しました！"
echo ""
echo "📋 削除されたもの:"
echo "✅ PatchMoE関連コンテナ"
echo "✅ PatchMoE関連イメージ"
echo "✅ 使用されていないイメージ"
echo "✅ 使用されていないボリューム"
echo "✅ 使用されていないネットワーク"
echo ""
echo "🔄 環境を再構築するには:"
echo "./docker-setup.sh"
