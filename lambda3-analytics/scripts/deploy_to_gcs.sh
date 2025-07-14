#!/bin/bash
# Lambda³ GCSデプロイスクリプト - 更新版

set -e

# 正しいバケット名
BUCKET="yourbacketname"

echo "=================================================="
echo "Lambda³ Code Deployment to GCS"
echo "=================================================="
echo "Bucket: gs://$BUCKET"
echo ""

# スクリプトのディレクトリを基準にする
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# バケットのルートディレクトリにアップロード
echo "Uploading Lambda³ modules to bucket root..."

# コアモジュール
echo "  - lambda3_zeroshot_tensor_field.py"
gsutil cp src/core/lambda3_zeroshot_tensor_field.py gs://$BUCKET/

echo "  - lambda3_regime_aware_extension.py"
gsutil cp src/core/lambda3_regime_aware_extension.py gs://$BUCKET/

# クラウドモジュール
echo "  - lambda3_cloud_parallel.py"
gsutil cp src/cloud/lambda3_cloud_parallel.py gs://$BUCKET/

echo "  - lambda3_gcp_ultimate.py"
gsutil cp src/cloud/lambda3_gcp_ultimate.py gs://$BUCKET/

echo "  - lambda3_cloud_worker.py"
gsutil cp src/cloud/lambda3_cloud_worker.py gs://$BUCKET/

echo "  - lambda3_result_aggregator.py"
gsutil cp src/cloud/lambda3_result_aggregator.py gs://$BUCKET/

# 実行権限を設定（パブリック読み取り）
echo ""
echo "Setting permissions..."
gsutil acl ch -u AllUsers:R gs://$BUCKET/*.py

# アップロードされたファイルを確認
echo ""
echo "Verifying uploaded files:"
gsutil ls -l gs://$BUCKET/*.py

echo ""
echo "✅ Deployment complete!"
echo ""
echo "All Lambda³ modules are now available in gs://$BUCKET/"
echo ""
echo "Next steps:"
echo "1. Run test: python test_gcp_simple.py"
echo "2. Or launch full analysis: python scripts/launch_ultimate_analysis.py"
