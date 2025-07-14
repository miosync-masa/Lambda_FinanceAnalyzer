"""
Lambda³ Analytics Framework Setup
"""
from setuptools import setup, find_packages
from pathlib import Path

# READMEを読み込む
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "Lambda³ Analytics Framework - Universal Structural Tensor Field Analytics"

# バージョン情報
VERSION = "1.0.0"

# 基本的な依存関係
install_requires = [
    "numpy>=1.24.0",
    "pandas>=2.0.0", 
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pymc>=5.7.0",
    "arviz>=0.16.0",
    "numba>=0.57.0",
    "networkx>=3.1",
    "yfinance>=0.2.28",
    "pyyaml>=6.0",
]

# オプション依存関係
extras_require = {
    # GCP関連
    'gcp': [
        "google-cloud-batch>=0.11.0",
        "google-cloud-storage>=2.10.0",
        "google-cloud-compute>=1.14.0",
        "google-cloud-monitoring>=2.15.0",
        "google-auth>=2.22.0",
        "google-api-python-client>=2.95.0",
    ],
    # 開発用
    'dev': [
        "pytest>=7.3.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.2.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
    ],
    # 分散処理（オプション）
    'distributed': [
        "dask[distributed]>=2023.0.0",
        "ray[default]>=2.3.0",
    ],
    # 全部入り
    'all': [],  # 後で設定
}

# 'all'に全てのextrasを含める
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="lambda3-analytics",
    version=VERSION,
    author="Masamichi Iizumi",
    author_email="m.iizumi@miosync.email",
    description="Universal Structural Tensor Field Analytics Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miosync-masa/Lambda_FinanceAnalyzer/tree/main/lambda3-analytics",
    project_urls={
        "Documentation": "https://lambda3-analytics.readthedocs.io",
        "Source": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/tree/main/lambda3-analytics",
        "Issues": "https://github.com/miosync-masa/Lambda_FinanceAnalyzer/issues",
    },
    
    # パッケージ設定
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # パッケージデータ
    package_data={
        "lambda3": [
            "config/*.yaml",
            "data/sample/*.csv",
        ],
    },
    
    # 依存関係
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Pythonバージョン
    python_requires=">=3.10",
    
    # エントリーポイント（CLIコマンド）
    entry_points={
        "console_scripts": [
            "lambda3=lambda3.cli:main",
            "lambda3-analyze=scripts.run_local_demo:main",
            "lambda3-gcp-setup=scripts.setup_gcp_project:main",
            "lambda3-cloud=scripts.launch_ultimate_analysis:main",
        ],
    },
    
    # 分類
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    
    # キーワード
    keywords=[
        "lambda3",
        "structural-tensor",
        "time-series-analysis", 
        "bayesian-inference",
        "financial-analysis",
        "parallel-computing",
        "cloud-computing",
    ],
)
