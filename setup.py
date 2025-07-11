# ==========================================================
# setup.py - Lambda³ Finance Analyzer Package Setup
# 
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 革新ポイント: JIT最適化とベイズ推定の統合パッケージ
# ==========================================================

"""
Lambda³ Finance Analyzer - 構造テンソル理論による金融時系列解析

Lambda³理論（Lambda Cubed Theory）は、時間に依存しない構造空間における
∆ΛC pulsations（構造変化パルス）を通じて、金融市場の本質的ダイナミクスを
解析する革新的な数理フレームワークです。

核心概念:
- 構造テンソル(Λ): 時系列の構造的状態表現
- 進行ベクトル(ΛF): 構造変化の方向性と強度
- 張力スカラー(ρT): 構造空間の張力度合い
- ∆ΛC pulsations: 構造変化の非時間的パルス現象

技術的革新:
- Numba JIT最適化による超高速計算
- ベイズ推定による不確実性定量化
- 階層的構造変化の自動分離
- 非対称相互作用の定量化
- リアルタイム危機検出システム
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Python バージョンチェック
if sys.version_info < (3, 8):
    raise RuntimeError("Lambda³ requires Python 3.8 or higher")

# パッケージディレクトリ
here = Path(__file__).parent.absolute()

# README読み込み
def load_readme():
    readme_path = here / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Lambda³ Finance Analyzer - Advanced Financial Time Series Analysis"

# 長い説明文
LONG_DESCRIPTION = load_readme()

# バージョン情報
VERSION = "1.0.0"

# 作者情報
AUTHOR = "Masamichi Iizumi"
AUTHOR_EMAIL = "m.iizumi@miosync.email"
MAINTAINER = "Miosync Research Team"
MAINTAINER_EMAIL = "info@miosync.email"

# パッケージメタデータ
PACKAGE_NAME = "lambda3"
PACKAGE_DESCRIPTION = "Lambda³ Theory: Advanced Financial Analysis with Structural Tensors and JIT Optimization"

# ホームページとリポジトリ
URL = "https://github.com/miosync-masa/Lambda_FinanceAnalyzer"
DOWNLOAD_URL = "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/archive/v1.0.0.tar.gz"

# 分類子（PyPI用）
CLASSIFIERS = [
    # 開発状況
    "Development Status :: 4 - Beta",
    
    # 対象ユーザー
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    
    # トピック
    "Topic :: Office/Business :: Financial",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Software Development :: Libraries :: Python Modules",
    
    # ライセンス
    "License :: OSI Approved :: MIT License",
    
    # Python バージョンサポート
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    
    # オペレーティングシステム
    "Operating System :: OS Independent",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    
    # 特殊分類
    "Environment :: Console",
    "Environment :: Jupyter",
    "Natural Language :: English",
    "Natural Language :: Japanese",
]

# キーワード
KEYWORDS = [
    "finance", "quantitative-finance", "time-series-analysis", 
    "structural-tensors", "lambda3-theory", "jit-optimization",
    "bayesian-analysis", "crisis-detection", "market-analysis",
    "numba", "statistical-modeling", "mathematical-finance",
    "algorithmic-trading", "risk-management", "econometrics"
]

# 必須依存関係（基本動作に必要）
INSTALL_REQUIRES = [
    # 数値計算基盤
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    
    # JIT最適化（核心機能）
    "numba>=0.56.0",
    
    # 基本可視化
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    
    # データ構造・ユーティリティ
    "dataclasses;python_version<'3.7'",  # Python 3.6以下用バックポート
    "typing-extensions>=4.0.0",
]

# オプション依存関係（追加機能用）
EXTRAS_REQUIRE = {
    # ベイズ分析（高度統計機能）
    "bayesian": [
        "pymc>=5.0.0",
        "arviz>=0.14.0",
        "theano-pymc>=1.1.0",
    ],
    
    # 金融データ取得
    "financial": [
        "yfinance>=0.2.0",
        "pandas-datareader>=0.10.0",
        "quandl>=3.7.0",
    ],
    
    # 高度可視化
    "visualization": [
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "bokeh>=2.4.0",
        "ipywidgets>=7.6.0",
    ],
    
    # ネットワーク分析
    "network": [
        "networkx>=2.6.0",
        "igraph>=0.9.0",
        "graph-tool;platform_system!='Windows'",  # Linux/Mac only
    ],
    
    # 機械学習拡張
    "ml": [
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "torch>=1.11.0",
    ],
    
    # 開発・テスト用
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-benchmark>=4.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
    
    # 性能ベンチマーク
    "performance": [
        "memory-profiler>=0.60.0",
        "line-profiler>=3.5.0",
        "py-spy>=0.3.0",
    ],
    
    # Jupyter環境
    "jupyter": [
        "jupyter>=1.0.0",
        "jupyterlab>=3.4.0",
        "ipython>=8.0.0",
        "notebook>=6.4.0",
    ],
}

# 完全インストール（全オプション含む）
EXTRAS_REQUIRE["complete"] = list(set(
    dep for deps_list in EXTRAS_REQUIRE.values() 
    for dep in deps_list
    if not dep.endswith("'Windows'")  # プラットフォーム依存除外
))

# 最小インストール（ベイズ分析なし）
EXTRAS_REQUIRE["minimal"] = EXTRAS_REQUIRE["financial"] + EXTRAS_REQUIRE["visualization"]

# Python バージョン要件
PYTHON_REQUIRES = ">=3.8"

# エントリーポイント（コマンドライン）
ENTRY_POINTS = {
    "console_scripts": [
        "lambda3=lambda3.cli:main",
        "lambda3-analyze=lambda3.cli:analyze_command",
        "lambda3-financial=lambda3.cli:financial_command",
        "lambda3-benchmark=lambda3.cli:benchmark_command",
    ],
}

# プロジェクトURL
PROJECT_URLS = {
    "Homepage": URL,
    "Repository": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer",
    "Bug Tracker": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/issues",
    "Discussions": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/discussions",
}

# パッケージデータ
PACKAGE_DATA = {
    "lambda3": [
        "data/*.json",
        "data/*.csv",
        "configs/*.yaml",
        "configs/*.toml",
        "templates/*.html",
        "static/css/*.css",
        "static/js/*.js",
    ],
}

# 含めるデータファイル
INCLUDE_PACKAGE_DATA = True

# Zip形式での安全インストール
ZIP_SAFE = False

# セットアップ実行
def setup_lambda3():
    """Lambda³パッケージセットアップ実行"""
    
    # 動的バージョン読み込み
    version_file = here / "lambda3" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    VERSION = line.split("=")[1].strip().strip('"').strip("'")
                    break
    
    setup(
        # 基本情報
        name=PACKAGE_NAME,
        version=VERSION,
        description=PACKAGE_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        
        # 作者情報
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        
        # URL情報
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        
        # パッケージ情報
        packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
        package_data=PACKAGE_DATA,
        include_package_data=INCLUDE_PACKAGE_DATA,
        
        # 依存関係
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        
        # メタデータ
        classifiers=CLASSIFIERS,
        keywords=" ".join(KEYWORDS),
        license="MIT",
        
        # エントリーポイント
        entry_points=ENTRY_POINTS,
        
        # セットアップ設定
        zip_safe=ZIP_SAFE,
        
        # テストスイート
        test_suite="tests",
        tests_require=EXTRAS_REQUIRE["dev"],
        
        # 追加設定
        platforms=["any"],
        
        # PyPI upload用設定
        cmdclass={},
        ext_modules=[],
        
        # カスタムコマンド
        options={
            "build_py": {
                "compile": True,
                "optimize": 1,
            },
            "install": {
                "optimize": 1,
            },
        },
    )

# バージョン表示関数
def print_version_info():
    """バージョン情報表示"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            Lambda³ Finance Analyzer                          ║
║                           Package Setup Information                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Version: {VERSION:<20} License: MIT                                ║
║ Author:  {AUTHOR:<67} ║
║ Python:  {PYTHON_REQUIRES:<67} ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 Package Components:
   • Core: {PACKAGE_NAME} (JIT-optimized structural tensor analysis)
   • Modules: {len(find_packages())} Python packages
   • Dependencies: {len(INSTALL_REQUIRES)} required, {len(EXTRAS_REQUIRE)} optional groups
   • Features: Bayesian analysis, Financial modeling, Network analysis

🚀 Installation Options:
   pip install {PACKAGE_NAME}                    # Basic installation
   pip install {PACKAGE_NAME}[complete]          # Full installation
   pip install {PACKAGE_NAME}[bayesian,financial] # Bayesian + Financial
   pip install {PACKAGE_NAME}[minimal]           # Essential features only

📊 Supported Platforms:
   • Windows (x64)    • Linux (x64/ARM)    • macOS (Intel/Apple Silicon)
   • Python 3.8+     • NumPy/SciPy Stack  • Jupyter Notebooks

🔬 Research Citation:
   Iizumi, M. (2024). Lambda³ Theory: Structural Tensor Analysis for 
   Financial Time Series. arXiv:2024.lambda3.theory
    """)

if __name__ == "__main__":
    # バージョン情報表示
    if "--version-info" in sys.argv:
        print_version_info()
        sys.exit(0)
    
    # セットアップ実行
    try:
        setup_lambda3()
        print("✅ Lambda³ Finance Analyzer setup completed successfully!")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)
