# ==========================================================
# lambda3/__init__.py (JIT Compatible Version)
# Lambda³ Package Initialization
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 修正点: JIT最適化関数との完全互換性確保
# ==========================================================

"""
Lambda³ (Lambda Cubed) Analytics Package - JIT最適化版

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
- JIT最適化による高速計算（修正版）
- 統合可視化システム

使用例:
>>> import lambda3 as l3
>>> 
>>> # 基本的な分析（JIT最適化）
>>> results = l3.analyze(data)
>>> 
>>> # 金融市場分析
>>> financial_results = l3.analyze_financial_markets()
>>> 
>>> # カスタム設定での分析
>>> config = l3.create_config('research')
>>> results = l3.analyze(data, config=config)
>>> 
>>> # JIT最適化設定の確認
>>> print(f"JIT enabled: {results.config.base.jit_config.enable_jit}")
"""

__version__ = "1.0.0-alpha"
__author__ = "Mamichi Iizumi"
__email__ = "m.iizumi@miosync.com"
__license__ = "MIT"

import warnings
import sys
from typing import Union, Dict, Any, List, Optional
import numpy as np

# ==========================================================
# IMPORT SAFETY CHECKS - JIT互換性確保
# ==========================================================

def _check_jit_compatibility():
    """JIT互換性事前チェック"""
    try:
        import numba
        # Numbaバージョンチェック
        numba_version = tuple(map(int, numba.__version__.split('.')[:2]))
        if numba_version < (0, 56):
            warnings.warn("Numba 0.56+ recommended for optimal JIT performance")
        return True
    except ImportError:
        warnings.warn("Numba not available. JIT optimization will be disabled.")
        return False

def _validate_numpy_for_jit():
    """NumPy JIT互換性検証"""
    try:
        import numpy as np
        # NumPy dtype互換性チェック
        test_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        if test_array.dtype != np.float64:
            warnings.warn("NumPy dtype compatibility issue detected")
        return True
    except Exception as e:
        warnings.warn(f"NumPy JIT compatibility check failed: {e}")
        return False

# JIT互換性事前チェック実行
_JIT_AVAILABLE = _check_jit_compatibility()
_NUMPY_JIT_COMPATIBLE = _validate_numpy_for_jit()

# ==========================================================
# CORE IMPORTS - 修正版JIT対応
# ==========================================================

# 設定システム（JIT統合版）
try:
    from .core.config import (
        L3BaseConfig,
        L3JITConfig,  # 新規追加
        L3BayesianConfig, 
        L3HierarchicalConfig,
        L3PairwiseConfig,
        L3SynchronizationConfig,
        L3VisualizationConfig,
        L3ComprehensiveConfig,
        create_default_config,
        create_financial_config,
        create_rapid_config,
        create_research_config,
        get_config,
        validate_config  # 新規追加
    )
    _CONFIG_IMPORT_SUCCESS = True
except ImportError as e:
    warnings.warn(f"Configuration system import failed: {e}")
    _CONFIG_IMPORT_SUCCESS = False

# JIT最適化関数（修正版）
try:
    from .core.jit_functions import (
        # 修正版JIT関数群
        calculate_diff_and_threshold_fixed,
        detect_structural_jumps_fixed,
        calculate_tension_scalar_fixed,
        calculate_local_statistics_fixed,
        detect_hierarchical_jumps_fixed,
        classify_hierarchical_events_fixed,
        calculate_sync_rate_at_lag_fixed,
        calculate_sync_profile_fixed,
        detect_phase_coupling_fixed,
        # 統合特徴抽出
        extract_lambda3_features_jit,
        # ユーティリティ関数
        safe_divide_fixed,
        normalize_array_fixed,
        moving_average_fixed,
        exponential_smoothing_fixed,
        # テスト関数
        test_jit_functions_fixed,
        benchmark_performance_fixed
    )
    
    # レガシー互換性のためのエイリアス
    calculate_diff_and_threshold = calculate_diff_and_threshold_fixed
    detect_structural_jumps = detect_structural_jumps_fixed
    calculate_tension_scalar = calculate_tension_scalar_fixed
    detect_hierarchical_jumps = detect_hierarchical_jumps_fixed
    calculate_sync_profile = calculate_sync_profile_fixed
    
    _JIT_FUNCTIONS_IMPORT_SUCCESS = True
except ImportError as e:
    warnings.warn(f"JIT functions import failed: {e}")
    _JIT_FUNCTIONS_IMPORT_SUCCESS = False

# 構造テンソル演算（JIT統合版）
try:
    from .core.structural_tensor import (
        StructuralTensorFeatures,
        StructuralTensorExtractor,
        extract_lambda3_features
    )
    _STRUCTURAL_TENSOR_IMPORT_SUCCESS = True
except ImportError as e:
    warnings.warn(f"Structural tensor module import failed: {e}")
    _STRUCTURAL_TENSOR_IMPORT_SUCCESS = False

# ==========================================================
# ANALYSIS MODULES - 条件付きインポート
# ==========================================================

# 階層分析
try:
    from .analysis.hierarchical import (
        HierarchicalAnalyzer,
        HierarchicalSeparationResults,
        analyze_hierarchical_structure,
        compare_multiple_hierarchies
    )
    _HIERARCHICAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Hierarchical analysis not available: {e}")
    _HIERARCHICAL_ANALYSIS_AVAILABLE = False

# ペアワイズ分析
try:
    from .analysis.pairwise import (
        PairwiseAnalyzer,
        PairwiseInteractionResults,
        analyze_pairwise_interaction,
        compare_all_pairs
    )
    _PAIRWISE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Pairwise analysis not available: {e}")
    _PAIRWISE_ANALYSIS_AVAILABLE = False

# ==========================================================
# VISUALIZATION MODULES - 条件付きインポート
# ==========================================================

try:
    from .visualization.base import (
        Lambda3BaseVisualizer,
        TimeSeriesVisualizer,
        InteractionVisualizer,
        HierarchicalVisualizer,
        apply_lambda3_style,
        get_lambda3_colors
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Visualization modules not available: {e}")
    _VISUALIZATION_AVAILABLE = False

# ==========================================================
# PIPELINE MODULES - 条件付きインポート
# ==========================================================

try:
    from .pipelines.comprehensive import (
        Lambda3ComprehensivePipeline,
        Lambda3ComprehensiveResults,
        run_lambda3_analysis,
        create_analysis_report
    )
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Pipeline modules not available: {e}")
    _PIPELINE_AVAILABLE = False

# ==========================================================
# CONVENIENCE FUNCTIONS - JIT最適化版
# ==========================================================

def analyze(data, 
           config=None, 
           analysis_type='comprehensive',
           series_names=None,
           enable_visualization=True,
           force_jit=None):
    """
    Lambda³分析のメイン便利関数（JIT最適化版）
    
    Args:
        data: 入力データ（辞書、配列、ファイルパス）
        config: 設定オブジェクト
        analysis_type: 'comprehensive', 'financial', 'rapid', 'research'
        series_names: 系列名リスト
        enable_visualization: 可視化有効フラグ
        force_jit: JIT強制有効/無効（None=自動判定）
        
    Returns:
        Lambda3ComprehensiveResults or StructuralTensorFeatures: 分析結果
    """
    # JIT可用性チェック
    if force_jit is not None:
        use_jit = force_jit and _JIT_AVAILABLE
    else:
        use_jit = _JIT_AVAILABLE
    
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
    
    # JIT設定の調整
    if hasattr(config, 'base') and hasattr(config.base, 'jit_config'):
        config.base.jit_config.enable_jit = use_jit
    
    # パイプライン実行
    if _PIPELINE_AVAILABLE and len(data_dict) > 1:
        # 複数系列の場合は包括パイプライン使用
        return run_lambda3_analysis(
            data_dict, 
            config=config, 
            analysis_type=analysis_type
        )
    else:
        # 単一系列の場合は直接特徴抽出
        series_name, series_data = next(iter(data_dict.items()))
        return extract_features(
            series_data,
            config=config.base if hasattr(config, 'base') else config,
            series_name=series_name
        )

def analyze_financial_markets(tickers=None, 
                            start_date="2022-01-01", 
                            end_date="2024-12-31",
                            enable_crisis_detection=True,
                            use_jit_optimization=True):
    """
    金融市場分析の便利関数（JIT最適化版）
    
    Args:
        tickers: ティッカー辞書
        start_date, end_date: 分析期間
        enable_crisis_detection: 危機検出有効フラグ
        use_jit_optimization: JIT最適化使用フラグ
        
    Returns:
        Lambda3ComprehensiveResults: 金融分析結果
    """
    if not _PIPELINE_AVAILABLE:
        raise ImportError("Pipeline modules not available for financial analysis")
    
    config = create_financial_config()
    
    # JIT最適化設定
    if hasattr(config.base, 'jit_config'):
        config.base.jit_config.enable_jit = use_jit_optimization and _JIT_AVAILABLE
    
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
                    series_name="Series",
                    use_jit=None):
    """
    構造テンソル特徴抽出の便利関数（JIT最適化版）
    
    Args:
        data: 入力データ
        config: 設定オブジェクト
        feature_level: 'basic', 'hierarchical', 'comprehensive'
        series_name: 系列名
        use_jit: JIT使用フラグ（None=自動判定）
        
    Returns:
        StructuralTensorFeatures: 特徴量オブジェクト
    """
    # JIT使用判定
    if use_jit is None:
        use_jit = _JIT_AVAILABLE and _JIT_FUNCTIONS_IMPORT_SUCCESS
    
    # 設定準備
    if config is None:
        config = L3BaseConfig() if _CONFIG_IMPORT_SUCCESS else None
    
    if config and hasattr(config, 'jit_config'):
        config.jit_config.enable_jit = use_jit
    
    # データ前処理
    data = np.asarray(data, dtype=np.float64)
    
    if use_jit and _JIT_FUNCTIONS_IMPORT_SUCCESS:
        # JIT最適化版を使用
        try:
            if feature_level == 'basic':
                # 基本特徴量のみ
                diff, threshold = calculate_diff_and_threshold_fixed(
                    data, config.delta_percentile if config else 95.0
                )
                delta_pos, delta_neg = detect_structural_jumps_fixed(diff, threshold)
                rho_t = calculate_tension_scalar_fixed(
                    data, config.window if config else 10
                )
                
                # StructuralTensorFeaturesオブジェクト構築
                if _STRUCTURAL_TENSOR_IMPORT_SUCCESS:
                    features = StructuralTensorFeatures(
                        data=data,
                        series_name=series_name,
                        delta_LambdaC_pos=delta_pos,
                        delta_LambdaC_neg=delta_neg,
                        rho_T=rho_t
                    )
                    return features
                else:
                    # 辞書形式で返却
                    return {
                        'data': data,
                        'series_name': series_name,
                        'delta_pos': delta_pos,
                        'delta_neg': delta_neg,
                        'rho_T': rho_t
                    }
            
            elif feature_level in ['hierarchical', 'comprehensive']:
                # 階層的特徴量抽出
                features_tuple = extract_lambda3_features_jit(
                    data,
                    window=config.window if config else 10,
                    local_window=config.local_window if config else 5,
                    global_window=config.global_window if config else 30,
                    delta_percentile=config.delta_percentile if config else 95.0,
                    local_percentile=config.local_threshold_percentile if config else 90.0,
                    global_percentile=config.global_threshold_percentile if config else 95.0
                )
                
                delta_pos, delta_neg, rho_t, local_pos, local_neg, global_pos, global_neg = features_tuple
                
                if _STRUCTURAL_TENSOR_IMPORT_SUCCESS:
                    features = StructuralTensorFeatures(
                        data=data,
                        series_name=series_name,
                        delta_LambdaC_pos=delta_pos,
                        delta_LambdaC_neg=delta_neg,
                        rho_T=rho_t,
                        local_pos=local_pos,
                        local_neg=local_neg,
                        global_pos=global_pos,
                        global_neg=global_neg
                    )
                    return features
                else:
                    return {
                        'data': data,
                        'series_name': series_name,
                        'delta_pos': delta_pos,
                        'delta_neg': delta_neg,
                        'rho_T': rho_t,
                        'local_pos': local_pos,
                        'local_neg': local_neg,
                        'global_pos': global_pos,
                        'global_neg': global_neg
                    }
        
        except Exception as e:
            warnings.warn(f"JIT feature extraction failed: {e}. Falling back to standard extraction.")
            use_jit = False
    
    # 標準版へのフォールバック
    if _STRUCTURAL_TENSOR_IMPORT_SUCCESS:
        return extract_lambda3_features(
            data, 
            config=config, 
            series_name=series_name,
            feature_level=feature_level
        )
    else:
        raise ImportError("Neither JIT nor standard feature extraction available")

def create_config(config_type='default', enable_jit=None):
    """
    設定オブジェクト作成の便利関数（JIT対応版）
    
    Args:
        config_type: 'default', 'financial', 'rapid', 'research'
        enable_jit: JIT有効化フラグ（None=自動判定）
        
    Returns:
        L3ComprehensiveConfig: 設定オブジェクト
    """
    if not _CONFIG_IMPORT_SUCCESS:
        raise ImportError("Configuration system not available")
    
    config = get_config(config_type)
    
    # JIT設定の調整
    if enable_jit is not None:
        if hasattr(config.base, 'jit_config'):
            config.base.jit_config.enable_jit = enable_jit and _JIT_AVAILABLE
    elif not _JIT_AVAILABLE:
        if hasattr(config.base, 'jit_config'):
            config.base.jit_config.enable_jit = False
    
    return config

def apply_style(style_name='lambda3_default'):
    """
    Lambda³可視化スタイル適用
    
    Args:
        style_name: スタイル名
    """
    if _VISUALIZATION_AVAILABLE:
        apply_lambda3_style(style_name)
    else:
        warnings.warn("Visualization modules not available")

def run_jit_benchmark():
    """JIT性能ベンチマーク実行"""
    if _JIT_FUNCTIONS_IMPORT_SUCCESS:
        print("🚀 Lambda³ JIT Performance Benchmark")
        print("=" * 50)
        return benchmark_performance_fixed()
    else:
        print("❌ JIT functions not available for benchmarking")
        return None

def test_jit_functions():
    """JIT関数テスト実行"""
    if _JIT_FUNCTIONS_IMPORT_SUCCESS:
        print("🧪 Lambda³ JIT Functions Test")
        print("=" * 50)
        return test_jit_functions_fixed()
    else:
        print("❌ JIT functions not available for testing")
        return False

# ==========================================================
# PACKAGE METADATA - 拡張版
# ==========================================================

__all__ = [
    # Core classes
    'L3BaseConfig',
    'L3JITConfig',  # 新規追加
    'L3ComprehensiveConfig', 
    'StructuralTensorFeatures',
    'StructuralTensorExtractor',
    
    # JIT Functions (修正版)
    'calculate_diff_and_threshold_fixed',
    'detect_structural_jumps_fixed',
    'calculate_tension_scalar_fixed',
    'detect_hierarchical_jumps_fixed',
    'extract_lambda3_features_jit',
    
    # Legacy aliases
    'calculate_diff_and_threshold',
    'detect_structural_jumps', 
    'calculate_tension_scalar',
    'detect_hierarchical_jumps',
    
    # Main convenience functions
    'analyze',
    'analyze_financial_markets',
    'extract_features',
    'create_config',
    'apply_style',
    
    # JIT specific functions
    'run_jit_benchmark',
    'test_jit_functions',
    
    # Factory functions
    'create_default_config',
    'create_financial_config',
    'create_rapid_config', 
    'create_research_config',
    'get_config',
]

# 条件付き __all__ 拡張
if _HIERARCHICAL_ANALYSIS_AVAILABLE:
    __all__.extend([
        'HierarchicalAnalyzer',
        'HierarchicalSeparationResults',
        'analyze_hierarchical_structure',
        'compare_multiple_hierarchies'
    ])

if _PAIRWISE_ANALYSIS_AVAILABLE:
    __all__.extend([
        'PairwiseAnalyzer', 
        'PairwiseInteractionResults',
        'analyze_pairwise_interaction',
        'compare_all_pairs'
    ])

if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        'Lambda3BaseVisualizer',
        'TimeSeriesVisualizer',
        'InteractionVisualizer',
        'HierarchicalVisualizer',
        'apply_lambda3_style',
        'get_lambda3_colors'
    ])

if _PIPELINE_AVAILABLE:
    __all__.extend([
        'Lambda3ComprehensivePipeline',
        'Lambda3ComprehensiveResults',
        'run_lambda3_analysis',
        'create_analysis_report'
    ])

# ==========================================================
# PACKAGE INFORMATION - 拡張版
# ==========================================================

def get_package_info():
    """パッケージ情報取得（JIT情報含む）"""
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
            'JIT-optimized computations (Fixed)',
            'Integrated visualization system'
        ],
        'available_configs': [
            'default - General purpose analysis',
            'financial - Financial market specialized',
            'rapid - Fast analysis with reduced precision',
            'research - High precision for academic research'
        ],
        'jit_status': {
            'jit_available': _JIT_AVAILABLE,
            'jit_functions_loaded': _JIT_FUNCTIONS_IMPORT_SUCCESS,
            'numpy_jit_compatible': _NUMPY_JIT_COMPATIBLE
        },
        'module_status': {
            'config_system': _CONFIG_IMPORT_SUCCESS,
            'structural_tensor': _STRUCTURAL_TENSOR_IMPORT_SUCCESS,
            'hierarchical_analysis': _HIERARCHICAL_ANALYSIS_AVAILABLE,
            'pairwise_analysis': _PAIRWISE_ANALYSIS_AVAILABLE,
            'visualization': _VISUALIZATION_AVAILABLE,
            'pipeline': _PIPELINE_AVAILABLE
        }
    }

def print_welcome():
    """ウェルカムメッセージ表示（JIT情報含む）"""
    info = get_package_info()
    
    # JIT状態の表示マーク
    jit_mark = "🚀" if info['jit_status']['jit_available'] else "⚠️"
    jit_status = "ENABLED" if info['jit_status']['jit_available'] else "DISABLED"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            Lambda³ Analytics Package                         ║
║                         {info['description']}                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Version: {info['version']:<20} Author: {info['author']:<30} ║
║ License: {info['license']:<67} ║
║ JIT Optimization: {jit_mark} {jit_status:<52} ║
╚══════════════════════════════════════════════════════════════════════════════╝

核心概念:
  • 構造テンソル(Λ): 時系列の構造的状態表現
  • 進行ベクトル(ΛF): 構造変化の方向性と強度  
  • 張力スカラー(ρT): 構造空間の張力度合い
  • ∆ΛC pulsations: 構造変化の非時間的パルス現象

クイックスタート:
  import lambda3 as l3
  
  # 基本分析（JIT最適化）
  results = l3.analyze(your_data)
  
  # 金融市場分析
  financial_results = l3.analyze_financial_markets()
  
  # JIT性能テスト
  l3.test_jit_functions()
  l3.run_jit_benchmark()
  
  # カスタム設定
  config = l3.create_config('research', enable_jit=True)
  results = l3.analyze(your_data, config=config)

JIT最適化状態:
  • Numba使用可能: {'✅' if info['jit_status']['jit_available'] else '❌'}
  • JIT関数読み込み: {'✅' if info['jit_status']['jit_functions_loaded'] else '❌'}
  • NumPy互換性: {'✅' if info['jit_status']['numpy_jit_compatible'] else '❌'}

詳細情報: l3.get_package_info()
    """)

def print_system_status():
    """システム状態詳細表示"""
    info = get_package_info()
    
    print("\n📊 Lambda³ システム状態詳細")
    print("=" * 50)
    
    print("🔧 モジュール状態:")
    for module, status in info['module_status'].items():
        mark = "✅" if status else "❌"
        print(f"   {mark} {module}")
    
    print("\n⚡ JIT最適化状態:")
    for component, status in info['jit_status'].items():
        mark = "✅" if status else "❌"
        print(f"   {mark} {component}")
    
    if info['jit_status']['jit_available']:
        print("\n🚀 JIT最適化が有効です - 高速計算が利用可能")
        print("   テスト実行: l3.test_jit_functions()")
        print("   ベンチマーク: l3.run_jit_benchmark()")
    else:
        print("\n⚠️  JIT最適化が無効です")
        print("   Numbaのインストールを推奨: pip install numba")

# ==========================================================
# INITIALIZATION CHECKS - 拡張版
# ==========================================================

def _check_dependencies():
    """依存関係チェック（JIT対応版）"""
    import warnings
    
    try:
        import numpy
        if not _NUMPY_JIT_COMPATIBLE:
            warnings.warn("NumPy JIT compatibility issues detected")
    except ImportError:
        raise ImportError("NumPy is required for Lambda³ analytics")
    
    if not _JIT_AVAILABLE:
        warnings.warn("Numba not available. JIT optimization will be disabled.")
        warnings.warn("For optimal performance, install: pip install numba")
    
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
    """パッケージセットアップ（JIT対応版）"""
    # 依存関係チェック
    _check_dependencies()
    
    # JIT関数テスト（起動時）
    if _JIT_FUNCTIONS_IMPORT_SUCCESS:
        try:
            # 簡易JITテスト
            test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            _ = calculate_diff_and_threshold_fixed(test_data, 95.0)
        except Exception as e:
            warnings.warn(f"JIT functions startup test failed: {e}")
    
    # デフォルトスタイル適用
    if _VISUALIZATION_AVAILABLE:
        try:
            apply_lambda3_style('lambda3_default')
        except:
            pass  # スタイル適用失敗は無視

# パッケージ初期化時のセットアップ実行
_setup_package()

# ==========================================================
# VERSION COMPATIBILITY - 拡張版
# ==========================================================

def check_version_compatibility():
    """バージョン互換性チェック（JIT対応版）"""
    import sys
    
    if sys.version_info < (3, 8):
        raise RuntimeError("Lambda³ requires Python 3.8 or higher")
    
    # NumPy バージョンチェック
    try:
        import numpy as np
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        if numpy_version < (1, 21):
            warnings.warn("NumPy 1.21+ recommended for optimal JIT performance")
    except:
        pass
    
    # Numba バージョンチェック
    if _JIT_AVAILABLE:
        try:
            import numba
            numba_version = tuple(map(int, numba.__version__.split('.')[:2]))
            if numba_version < (0, 56):
                warnings.warn("Numba 0.56+ recommended for optimal JIT performance")
        except:
            pass

# バージョン互換性チェック実行
check_version_compatibility()

# ==========================================================
# PACKAGE READY - 拡張版
# ==========================================================

def _display_startup_message():
    """起動メッセージ表示"""
    jit_status = "JIT-Optimized" if _JIT_AVAILABLE else "Standard"
    module_count = sum([
        _CONFIG_IMPORT_SUCCESS,
        _STRUCTURAL_TENSOR_IMPORT_SUCCESS,
        _JIT_FUNCTIONS_IMPORT_SUCCESS,
        _HIERARCHICAL_ANALYSIS_AVAILABLE,
        _PAIRWISE_ANALYSIS_AVAILABLE,
        _VISUALIZATION_AVAILABLE,
        _PIPELINE_AVAILABLE
    ])
    
    print(f"Lambda³ Analytics Package v{__version__} loaded successfully!")
    print(f"Status: {jit_status} | Modules: {module_count}/7 available")
    print("Ready for structural tensor analysis.")
    print("Use l3.print_welcome() for detailed information.")
    
    if not _JIT_AVAILABLE:
        print("💡 Tip: Install numba for JIT optimization: pip install numba")

# 起動メッセージ表示
_display_startup_message()

# ==========================================================
# TESTING AND VALIDATION INTERFACE
# ==========================================================

def validate_installation():
    """インストール状態検証"""
    print("🔍 Lambda³ Installation Validation")
    print("=" * 50)
    
    results = {
        'core_modules': _CONFIG_IMPORT_SUCCESS and _STRUCTURAL_TENSOR_IMPORT_SUCCESS,
        'jit_optimization': _JIT_AVAILABLE and _JIT_FUNCTIONS_IMPORT_SUCCESS,
        'analysis_modules': _HIERARCHICAL_ANALYSIS_AVAILABLE and _PAIRWISE_ANALYSIS_AVAILABLE,
        'visualization': _VISUALIZATION_AVAILABLE,
        'pipeline': _PIPELINE_AVAILABLE
    }
    
    for component, status in results.items():
        mark = "✅" if status else "❌"
        print(f"{mark} {component}")
    
    # 簡易機能テスト
    if results['core_modules']:
        try:
            test_data = np.random.randn(100)
            config = create_config('default')
            features = extract_features(test_data, config=config)
            print("✅ Basic functionality test passed")
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
    
    overall_status = sum(results.values()) >= 3  # 最低3つのコンポーネントが必要
    
    print(f"\n{'✅ Installation OK' if overall_status else '❌ Installation Issues Detected'}")
    return overall_status

# パッケージ準備完了
if __name__ == "__main__":
    print_welcome()
    print_system_status()
    validate_installation()
