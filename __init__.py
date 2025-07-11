# ==========================================================
# lambda3/__init__.py
# Lambda³ Package Initialization
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³ (Lambda Cubed) Analytics Package

構造テンソル(Λ)理論に基づく高度時系列解析フレームワーク

Lambda³理論の核心概念:
- 構造テンソル(Λ): 時系列の構造的状態表現
- 進行ベクトル(ΛF): 構造変化の方向性と強度
- 張力スカラー(ρT): 構造空間の張力度合い
- ∆ΛC pulsations: 構造変化の非時間的パルス現象

主要特徴:
- 非時間依存の構造空間解析
- 階層的構造変化検出
- 非対称相互作用モデリング
- ベイズ推定による定量化
- JIT最適化による高速計算
- 統合可視化システム

使用例:
>>> import lambda3 as l3
>>> 
>>> # 基本的な分析
>>> results = l3.analyze(data)
>>> 
>>> # 金融市場分析
>>> financial_results = l3.analyze_financial_markets()
>>> 
>>> # カスタム設定での分析
>>> config = l3.create_config('research')
>>> results = l3.analyze(data, config=config)
"""

__version__ = "1.0.0-alpha"
__author__ = "Mamichi Iizumi"
__email__ = "m.iizumi@miosync.com"
__license__ = "MIT"

# ==========================================================
# CORE IMPORTS
# ==========================================================

# 設定システム
from .core.config import (
    L3BaseConfig,
    L3BayesianConfig, 
    L3HierarchicalConfig,
    L3PairwiseConfig,
    L3VisualizationConfig,
    L3ComprehensiveConfig,
    create_default_config,
    create_financial_config,
    create_rapid_config,
    create_research_config,
    get_config
)

# 構造テンソル演算
from .core.structural_tensor import (
    StructuralTensorFeatures,
    StructuralTensorExtractor,
    StructuralTensorAnalyzer,
    extract_lambda3_features,
    analyze_lambda3_structure,
    extract_features_batch
)

# JIT最適化関数（高度ユーザー向け）
from .core.jit_functions import (
    calculate_diff_and_threshold,
    detect_structural_jumps,
    calculate_tension_scalar,
    detect_hierarchical_jumps,
    calculate_sync_profile
)

# ==========================================================
# ANALYSIS MODULES
# ==========================================================

# 階層分析
from .analysis.hierarchical import (
    HierarchicalAnalyzer,
    HierarchicalSeparationResults,
    analyze_hierarchical_structure,
    compare_multiple_hierarchies
)

# ペアワイズ分析
from .analysis.pairwise import (
    PairwiseAnalyzer,
    PairwiseInteractionResults,
    analyze_pairwise_interaction,
    compare_all_pairs
)

# ==========================================================
# VISUALIZATION MODULES
# ==========================================================

# 基底可視化
from .visualization.base import (
    Lambda3BaseVisualizer,
    TimeSeriesVisualizer,
    InteractionVisualizer,
    HierarchicalVisualizer,
    apply_lambda3_style,
    get_lambda3_colors
)

# ==========================================================
# PIPELINE MODULES
# ==========================================================

# 包括パイプライン
from .pipelines.comprehensive import (
    Lambda3ComprehensivePipeline,
    Lambda3ComprehensiveResults,
    run_lambda3_analysis,
    create_analysis_report
)

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def analyze(data, 
           config=None, 
           analysis_type='comprehensive',
           series_names=None,
           enable_visualization=True):
    """
    Lambda³分析のメイン便利関数
    
    Args:
        data: 入力データ（辞書、配列、ファイルパス）
        config: 設定オブジェクト
        analysis_type: 'comprehensive', 'financial', 'rapid', 'research'
        series_names: 系列名リスト
        enable_visualization: 可視化有効フラグ
        
    Returns:
        Lambda3ComprehensiveResults: 分析結果
    """
    # データ形式の正規化
    if isinstance(data, (list, tuple)):
        if series_names is None:
            series_names = [f"Series_{i+1}" for i in range(len(data))]
        data_dict = {name: series for name, series in zip(series_names, data)}
    elif hasattr(data, 'shape') and len(data.shape) == 1:
        # 単一系列の場合
        series_name = series_names[0] if series_names else "Series_1"
        data_dict = {series_name: data}
    elif isinstance(data, dict):
        data_dict = data
    else:
        # ファイルパスまたはその他
        data_dict = data
    
    # 設定の生成
    if config is None:
        if analysis_type == 'financial':
            config = create_financial_config()
        elif analysis_type == 'rapid':
            config = create_rapid_config()
        elif analysis_type == 'research':
            config = create_research_config()
        else:
            config = create_default_config()
    
    # パイプライン実行
    return run_lambda3_analysis(
        data_dict, 
        config=config, 
        analysis_type=analysis_type
    )

def analyze_financial_markets(tickers=None, 
                            start_date="2022-01-01", 
                            end_date="2024-12-31",
                            enable_crisis_detection=True):
    """
    金融市場分析の便利関数
    
    Args:
        tickers: ティッカー辞書
        start_date, end_date: 分析期間
        enable_crisis_detection: 危機検出有効フラグ
        
    Returns:
        Lambda3ComprehensiveResults: 金融分析結果
    """
    config = create_financial_config()
    pipeline = Lambda3ComprehensivePipeline(config)
    
    return pipeline.run_financial_analysis(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        enable_crisis_detection=enable_crisis_detection
    )

def extract_features(data, 
                    config=None, 
                    feature_level='comprehensive',
                    series_name="Series"):
    """
    構造テンソル特徴抽出の便利関数
    
    Args:
        data: 入力データ
        config: 設定オブジェクト
        feature_level: 'basic', 'hierarchical', 'comprehensive'
        series_name: 系列名
        
    Returns:
        StructuralTensorFeatures: 特徴量オブジェクト
    """
    if config is None:
        config = L3BaseConfig()
    
    return extract_lambda3_features(
        data, 
        config=config, 
        series_name=series_name,
        feature_level=feature_level
    )

def create_config(config_type='default'):
    """
    設定オブジェクト作成の便利関数
    
    Args:
        config_type: 'default', 'financial', 'rapid', 'research'
        
    Returns:
        L3ComprehensiveConfig: 設定オブジェクト
    """
    return get_config(config_type)

def apply_style(style_name='lambda3_default'):
    """
    Lambda³可視化スタイル適用
    
    Args:
        style_name: スタイル名
    """
    apply_lambda3_style(style_name)

# ==========================================================
# PACKAGE METADATA
# ==========================================================

__all__ = [
    # Core classes
    'L3BaseConfig',
    'L3ComprehensiveConfig', 
    'StructuralTensorFeatures',
    'StructuralTensorExtractor',
    'StructuralTensorAnalyzer',
    
    # Analysis classes
    'HierarchicalAnalyzer',
    'HierarchicalSeparationResults',
    'PairwiseAnalyzer', 
    'PairwiseInteractionResults',
    
    # Visualization classes
    'Lambda3BaseVisualizer',
    'TimeSeriesVisualizer',
    'InteractionVisualizer',
    'HierarchicalVisualizer',
    
    # Pipeline classes
    'Lambda3ComprehensivePipeline',
    'Lambda3ComprehensiveResults',
    
    # Main convenience functions
    'analyze',
    'analyze_financial_markets',
    'extract_features',
    'create_config',
    'apply_style',
    
    # Factory functions
    'create_default_config',
    'create_financial_config',
    'create_rapid_config', 
    'create_research_config',
    'get_config',
    
    # High-level analysis functions
    'run_lambda3_analysis',
    'create_analysis_report',
    'analyze_hierarchical_structure',
    'analyze_pairwise_interaction',
    'compare_all_pairs',
    
    # Utility functions
    'extract_lambda3_features',
    'analyze_lambda3_structure',
    'extract_features_batch',
    'apply_lambda3_style',
    'get_lambda3_colors'
]

# ==========================================================
# PACKAGE INFORMATION
# ==========================================================

def get_package_info():
    """パッケージ情報取得"""
    return {
        'name': 'lambda3',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'description': 'Lambda³ Theory: Structural Tensor Analytics for Time Series',
        'core_concepts': [
            'Structural Tensors (Λ)',
            'Progression Vectors (ΛF)',
            'Tension Scalars (ρT)', 
            'Delta-Lambda-C Pulsations (∆ΛC)'
        ],
        'key_features': [
            'Non-temporal structural space analysis',
            'Hierarchical structure change detection',
            'Asymmetric interaction modeling',
            'Bayesian tensor regression',
            'JIT-optimized computations',
            'Integrated visualization system'
        ],
        'available_configs': [
            'default - General purpose analysis',
            'financial - Financial market specialized',
            'rapid - Fast analysis with reduced precision',
            'research - High precision for academic research'
        ]
    }

def print_welcome():
    """ウェルカムメッセージ表示"""
    info = get_package_info()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            Lambda³ Analytics Package                         ║
║                         {info['description']}                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Version: {info['version']:<20} Author: {info['author']:<30} ║
║ License: {info['license']:<67} ║
╚══════════════════════════════════════════════════════════════════════════════╝

核心概念:
  • 構造テンソル(Λ): 時系列の構造的状態表現
  • 進行ベクトル(ΛF): 構造変化の方向性と強度  
  • 張力スカラー(ρT): 構造空間の張力度合い
  • ∆ΛC pulsations: 構造変化の非時間的パルス現象

クイックスタート:
  import lambda3 as l3
  
  # 基本分析
  results = l3.analyze(your_data)
  
  # 金融市場分析
  financial_results = l3.analyze_financial_markets()
  
  # カスタム設定
  config = l3.create_config('research')
  results = l3.analyze(your_data, config=config)

詳細情報: l3.get_package_info()
    """)

# ==========================================================
# INITIALIZATION CHECKS
# ==========================================================

def _check_dependencies():
    """依存関係チェック"""
    import warnings
    
    try:
        import numpy
    except ImportError:
        raise ImportError("NumPy is required for Lambda³ analytics")
    
    try:
        import numba
    except ImportError:
        warnings.warn("Numba not available. JIT optimization will be disabled.")
    
    try:
        import matplotlib
    except ImportError:
        warnings.warn("Matplotlib not available. Visualization will be limited.")
    
    try:
        import pymc
        import arviz
    except ImportError:
        warnings.warn("PyMC/ArviZ not available. Bayesian analysis will be disabled.")

def _setup_package():
    """パッケージセットアップ"""
    # 依存関係チェック
    _check_dependencies()
    
    # デフォルトスタイル適用
    try:
        apply_lambda3_style('lambda3_default')
    except:
        pass  # スタイル適用失敗は無視

# パッケージ初期化時のセットアップ実行
_setup_package()

# ==========================================================
# VERSION COMPATIBILITY
# ==========================================================

def check_version_compatibility():
    """バージョン互換性チェック"""
    import sys
    
    if sys.version_info < (3, 8):
        raise RuntimeError("Lambda³ requires Python 3.8 or higher")
    
    # NumPy バージョンチェック
    try:
        import numpy as np
        if tuple(map(int, np.__version__.split('.')[:2])) < (1, 21):
            warnings.warn("NumPy 1.21+ recommended for optimal performance")
    except:
        pass

# バージョン互換性チェック実行
check_version_compatibility()

# ==========================================================
# PACKAGE READY
# ==========================================================

print(f"Lambda³ Analytics Package v{__version__} loaded successfully!")
print("Ready for structural tensor analysis.")
print("Use l3.print_welcome() for detailed information.")
