#!/bin/bash
# 完全版GCSデプロイスクリプト

BUCKET="lambda3-ultimate-results"

echo "Deploying Lambda³ modules to GCS..."

# スクリプトのディレクトリを基準にする
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# コアモジュール
gsutil cp src/core/lambda3_zeroshot_tensor_field.py gs://$BUCKET/
gsutil cp src/cloud/lambda3_cloud_parallel.py gs://$BUCKET/
gsutil cp src/core/lambda3_regime_aware_extension.py gs://$BUCKET/

# GCP拡張
gsutil cp src/cloud/lambda3_gcp_ultimate.py gs://$BUCKET/
gsutil cp src/cloud/lambda3_cloud_worker.py gs://$BUCKET/
gsutil cp src/cloud/lambda3_result_aggregator.py gs://$BUCKET/

# 実行権限
gsutil acl ch -u AllUsers:R gs://$BUCKET/*.py

echo "✓ All modules deployed successfully!"
