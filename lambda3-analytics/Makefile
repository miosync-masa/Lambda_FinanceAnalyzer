# Lambda³ Analytics Framework - Makefile
# 物理学者向けの簡単なコマンド集

.PHONY: help install test run docker-build docker-run gcp-setup clean

# デフォルトは help
.DEFAULT_GOAL := help

help: ## このヘルプを表示
	@echo "Lambda³ Analytics Framework - 簡単コマンド集"
	@echo ""
	@echo "使い方: make [コマンド]"
	@echo ""
	@echo "利用可能なコマンド:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## ローカルインストール（開発用）
	pip install -e ".[dev]"

install-gcp: ## GCP機能も含めてインストール
	pip install -e ".[gcp,dev]"

test: ## テストを実行
	pytest tests/ -v

test-quick: ## 高速テスト（並列実行なし）
	pytest tests/ -v -k "not parallel and not slow"

run-demo: ## ローカルデモを実行
	python scripts/run_local_demo.py

docker-build: ## Dockerイメージをビルド
	docker-compose build

docker-run: ## Dockerでデモを実行
	docker-compose run lambda3-main python scripts/run_local_demo.py

docker-notebook: ## Jupyter Notebookを起動
	docker-compose up lambda3-notebook

gcp-setup: ## GCPプロジェクトをセットアップ
	python scripts/setup_gcp_project.py

gcp-deploy: ## コードをGCSにデプロイ
	./scripts/deploy_to_gcs.sh

cloud-run: ## クラウド実行（小規模）
	python scripts/launch_ultimate_analysis.py

clean: ## 一時ファイルを削除
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf logs/*.log

# データ準備
prepare-data: ## サンプルデータをダウンロード
	python -c "from src.core.lambda3_zeroshot_tensor_field import fetch_financial_data; fetch_financial_data()"

# 結果確認
show-results: ## 最新の結果を表示
	@if [ -f "results/demo/synthetic_results.pkl" ]; then \
		echo "最新の結果:"; \
		ls -la results/demo/; \
	else \
		echo "まだ結果がありません。'make run-demo' を実行してください。"; \
	fi

# 環境確認
check-env: ## 環境設定を確認
	@echo "Python: $$(python --version)"
	@echo "pip: $$(pip --version)"
	@echo ""
	@echo "Lambda³ 依存関係:"
	@pip list | grep -E "(numpy|pandas|pymc|numba)"
	@echo ""
	@if [ -f ".env" ]; then \
		echo "環境変数ファイル: .env (存在)"; \
	else \
		echo "環境変数ファイル: .env (未作成)"; \
		echo "  → 'cp docker/.env.example docker/.env' で作成してください"; \
	fi
