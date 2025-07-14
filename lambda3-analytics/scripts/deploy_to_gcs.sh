#!/bin/bash
# 完全版GCSデプロイスクリプト

BUCKET="lambda3-ultimate-results"

echo "Deploying Lambda³ modules to GCS..."

# コアモジュール
gsutil cp lambda3_zeroshot_tensor_field.py gs://$BUCKET/
gsutil cp lambda3_cloud_parallel.py gs://$BUCKET/
gsutil cp lambda3_regime_aware_extension.py gs://$BUCKET/

# GCP拡張
gsutil cp lambda3_gcp_ultimate.py gs://$BUCKET/
gsutil cp lambda3_cloud_worker.py gs://$BUCKET/
gsutil cp lambda3_result_aggregator.py gs://$BUCKET/

# 実行権限
gsutil acl ch -u AllUsers:R gs://$BUCKET/*.py

echo "✓ All modules deployed successfully!"