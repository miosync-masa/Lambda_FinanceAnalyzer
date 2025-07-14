#!/bin/bash
# Lambda³ コードをGCSにデプロイ

set -e

# プロジェクトとバケットの設定
PROJECT_ID="massive-journal-428603-k7"
BUCKET="lambda3-massive-journal"

echo "=================================================="
echo "Lambda³ Code Deployment to GCS"
echo "=================================================="
echo "Project: $PROJECT_ID"
echo "Bucket: gs://$BUCKET"
echo ""

# スクリプトのディレクトリを基準にする
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# コアモジュールをアップロード
echo "Uploading core modules..."
gsutil cp src/core/lambda3_zeroshot_tensor_field.py gs://$BUCKET/code/
gsutil cp src/core/lambda3_regime_aware_extension.py gs://$BUCKET/code/

# クラウドモジュールをアップロード
echo "Uploading cloud modules..."
gsutil cp src/cloud/lambda3_cloud_parallel.py gs://$BUCKET/code/
gsutil cp src/cloud/lambda3_gcp_ultimate.py gs://$BUCKET/code/
gsutil cp src/cloud/lambda3_cloud_worker.py gs://$BUCKET/code/
gsutil cp src/cloud/lambda3_result_aggregator.py gs://$BUCKET/code/

# __init__.pyファイルもアップロード（モジュール認識用）
echo "Uploading __init__.py files..."
echo "" > /tmp/__init__.py
gsutil cp /tmp/__init__.py gs://$BUCKET/code/core/__init__.py
gsutil cp /tmp/__init__.py gs://$BUCKET/code/cloud/__init__.py

# requirements.txtを生成してアップロード
echo "Creating requirements.txt..."
cat > /tmp/requirements.txt << EOF
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
pymc>=5.7.0
arviz>=0.16.0
numba>=0.57.0
google-cloud-storage>=2.10.0
google-cloud-batch>=0.11.0
google-cloud-compute>=1.14.0
EOF

gsutil cp /tmp/requirements.txt gs://$BUCKET/code/requirements.txt

# 実行権限を設定（パブリック読み取り）
echo "Setting permissions..."
gsutil acl ch -u AllUsers:R gs://$BUCKET/code/*.py
gsutil acl ch -u AllUsers:R gs://$BUCKET/code/requirements.txt

# アップロードされたファイルを確認
echo ""
echo "Uploaded files:"
gsutil ls -l gs://$BUCKET/code/

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next step: Run the test with updated code"
echo "python test_gcp_simple.py"
