# Lambda³ クイックスタートガイド

## 5分で始めるLambda³

### 1. 環境構築（初回のみ）

```bash
# リポジトリをクローン
git clone https://github.com/miosync-masa/Lambda_FinanceAnalyzer/lambda3-analytics.git
cd lambda3-analytics

# Dockerを使う場合（推奨）
make docker-build
```

### 2. すぐに試す

#### 方法A: Docker（簡単）
```bash
# デモを実行
make docker-run

# Jupyter Notebookで探索
make docker-notebook
# → ブラウザで http://localhost:8888 を開く
```

#### 方法B: ローカル環境
```bash
# インストール
make install

# デモを実行
make run-demo
```

### 3. 自分のデータで解析

```python
from lambda3_zeroshot_tensor_field import run_lambda3_analysis

# CSVファイルから読み込み
results = run_lambda3_analysis(
    data_source="your_data.csv",
    verbose=True
)

# 結果を見る
print(results['pairwise_results']['summary'])
```

## 🌍 大規模解析（GCP）

### 初期設定
```bash
# GCPプロジェクトを設定
make gcp-setup

# 環境変数を設定
cp docker/.env.example docker/.env
# → .env ファイルを編集
```

### 実行
```bash
# 1000系列の解析を10,000インスタンスで実行！
make cloud-run
```

## 📊 よくあるコマンド

```bash
make help          # ヘルプを表示
make test          # テストを実行
make clean         # 一時ファイルを削除
make show-results  # 結果を確認
make check-env     # 環境を確認
```

## 🔍 Lambda³の基本概念

- **Λ (Lambda)**: 構造テンソル - システムの状態
- **ΔΛC**: 構造変化パルス - 重要なイベント
- **ρT**: 張力スカラー - システムのストレス
- **時間非依存**: 因果関係を仮定しない純粋な構造解析

## 💡 トラブルシューティング

### エラー: "No module named 'pymc'"
```bash
pip install pymc arviz
```

### エラー: "Docker not found"
→ [Docker Desktop](https://www.docker.com/products/docker-desktop/)をインストール

### メモリ不足
→ Docker Desktopの設定でメモリを8GB以上に増やす

## 📚 次のステップ

1. `notebooks/01_quick_start.ipynb` を開く
2. [理論解説](docs/lambda3_theory.md)を読む（準備中）
3. より大きなデータセットで試す
4. GCPで大規模並列実行に挑戦

---
質問は Issues へ: https://github.com/yourusername/lambda3-analytics/issues
