# ==========================================================
# setup.py - Lambda³ Finance Analyzer Package Setup
# 
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 現実的なパッケージ設定（実装されている機能のみ）
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

技術的特徴:
- Numba JIT最適化による高速計算
- ベイズ推定による不確実性定量化（オプション）
- 階層的構造変化の自動分離
- 非対称相互作用の定量化
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
VERSION = "1.0.0-alpha"

# 作者情報
AUTHOR = "Masamichi Iizumi"
AUTHOR_EMAIL = "m.iizumi@miosync.email"

# パッケージメタデータ
PACKAGE_NAME = "lambda3"
PACKAGE_DESCRIPTION = "Lambda³ Theory: Advanced Financial Analysis with Structural Tensors and JIT Optimization"

# ホームページとリポジトリ
URL = "https://github.com/miosync-masa/Lambda_FinanceAnalyzer"
DOWNLOAD_URL = "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/archive/v1.0.0-alpha.tar.gz"

# 分類子（現実的な内容のみ）
CLASSIFIERS = [
    # 開発状況
    "Development Status :: 3 - Alpha",
    
    # 対象ユーザー
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Science/Research",
    
    # トピック
    "Topic :: Office/Business :: Financial",
    "Topic :: Scientific/Engineering :: Mathematics",
    
    # ライセンス
    "License :: OSI Approved :: MIT License",
    
    # Python バージョンサポート
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    
    # オペレーティングシステム
    "Operating System :: OS Independent",
]

# キーワード（実装されている機能のみ）
KEYWORDS = [
    "finance", "time-series-analysis", "structural-tensors", 
    "jit-optimization", "numba", "financial-analysis"
]

# 必須依存関係（実際に必要なもののみ）
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
    
    # ユーティリティ
    "typing-extensions>=4.0.0",
]

# オプション依存関係（実装されている機能のみ）
EXTRAS_REQUIRE = {
    # ベイズ分析
    "bayesian": [
        "pymc>=5.0.0",
        "arviz>=0.14.0",
    ],
    
    # 金融データ取得
    "financial": [
        "yfinance>=0.2.0",
    ],
    
    # 高度可視化
    "visualization": [
        "plotly>=5.0.0",
        "networkx>=2.6.0",
    ],
    
    # 開発・テスト用
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
}

# 完全インストール（実装されている機能のみ）
EXTRAS_REQUIRE["complete"] = [
    dep for deps_list in EXTRAS_REQUIRE.values() 
    for dep in deps_list
]

# Python バージョン要件
PYTHON_REQUIRES = ">=3.8"

# プロジェクトURL（実際に存在するもののみ）
PROJECT_URLS = {
    "Homepage": URL,
    "Repository": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer",
    "Bug Tracker": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/issues",
    "Discussions": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/discussions",
}

# パッケージデータ（実際に存在するファイルのみ）
PACKAGE_DATA = {
    "lambda3": [
        "*.py",
        "core/*.py",
        "analysis/*.py",
        "visualization/*.py", 
        "pipelines/*.py",
    ],
}

# 含めるデータファイル
INCLUDE_PACKAGE_DATA = True

# Zip形式での安全インストール
ZIP_SAFE = False

# セットアップ実行
def setup_lambda3():
    """Lambda³パッケージセットアップ実行"""
    
    # 動的バージョン読み込み（__init__.pyが存在する場合）
    version_file = here / "lambda3" / "__init__.py"
    version = VERSION
    if version_file.exists():
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        version = line.split("=")[1].strip().strip('"').strip("'")
                        break
        except Exception:
            version = VERSION
    
    setup(
        # 基本情報
        name=PACKAGE_NAME,
        version=version,
        description=PACKAGE_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        
        # 作者情報
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        
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
        
        # セットアップ設定
        zip_safe=ZIP_SAFE,
        
        # テストスイート（基本設定のみ）
        test_suite="tests",
        tests_require=EXTRAS_REQUIRE["dev"],
        
        # 追加設定
        platforms=["any"],
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
   • Modules: lambda3.core, lambda3.analysis, lambda3.visualization, lambda3.pipelines
   • Dependencies: {len(INSTALL_REQUIRES)} required, {len(EXTRAS_REQUIRE)} optional groups
   • Features: Structural tensor analysis, JIT optimization, Financial modeling

🚀 Installation Options:
   pip install .                     # Basic installation
   pip install .[complete]           # Full installation  
   pip install .[bayesian,financial] # Bayesian + Financial
   pip install .[dev]                # Development dependencies

📊 Supported Platforms:
   • Windows (x64)    • Linux (x64/ARM)    • macOS (Intel/Apple Silicon)
   • Python 3.8+     • NumPy/SciPy Stack  • Jupyter Notebooks

🔬 Repository:
   GitHub: {URL}
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
        print(f"📦 Package installed with {len(INSTALL_REQUIRES)} core dependencies")
        print(f"🔧 Optional features: {', '.join(EXTRAS_REQUIRE.keys())}")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)
