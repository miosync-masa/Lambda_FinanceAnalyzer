# ================================================================
# 1. 修正済み lambda3/__init__.py
# ================================================================

# lambda3/__init__.py
"""
Lambda³ (Lambda Cubed) Analytics Package - 修正版

構造テンソル(Λ)理論に基づく高度時系列解析フレームワーク

修正点:
- インポート順序の最適化
- 循環依存の解消
- エラーハンドリングの改善
- JIT互換性の確保
"""

__version__ = "1.0.0"
__author__ = "Mamichi Iizumi"
__email__ = "m.iizumi@miosync.email"
__license__ = "MIT"

import warnings
import sys
from typing import Union, Dict, Any, List, Optional
import numpy as np

# ================================================================
# 段階的インポート戦略（循環依存回避）
# ================================================================

# Stage 1: 基本モジュールのインポート
_IMPORT_ERRORS = []

# バージョン情報
try:
    from .version import __version__ as _pkg_version
    __version__ = _pkg_version
except ImportError:
    pass  # デフォルトバージョンを使用

# Stage 2: 核心モジュール（依存関係なし）
try:
    from .core.config import (
        L3BaseConfig,
        L3JITConfig, 
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
        validate_config
    )
    _CONFIG_AVAILABLE = True
except ImportError as e:
    _CONFIG_AVAILABLE = False
    _IMPORT_ERRORS.append(f"Config module: {e}")
    warnings.warn(f"Configuration system not available: {e}")

# Stage 3: JIT最適化関数（核心機能）
try:
    from .core.jit_functions import (
        calculate_diff_and_threshold_fixed,
        detect_structural_jumps_fixed,
        calculate_tension_scalar_fixed,
        calculate_local_statistics_fixed,
        detect_hierarchical_jumps_fixed,
        classify_hierarchical_events_fixed,
        calculate_sync_rate_at_lag_fixed,
        calculate_sync_profile_fixed,
        detect_phase_coupling_fixed,
        extract_lambda3_features_jit,
        safe_divide_fixed,
        normalize_array_fixed,
        moving_average_fixed,
        exponential_smoothing_fixed,
        test_jit_functions_fixed,
        benchmark_performance_fixed
    )
    _JIT_FUNCTIONS_AVAILABLE = True
    
    # レガシー互換性
    calculate_diff_and_threshold = calculate_diff_and_threshold_fixed
    detect_structural_jumps = detect_structural_jumps_fixed
    calculate_tension_scalar = calculate_tension_scalar_fixed
    detect_hierarchical_jumps = detect_hierarchical_jumps_fixed
    calculate_sync_profile = calculate_sync_profile_fixed
    test_jit_functions = test_jit_functions_fixed
    run_jit_benchmark = benchmark_performance_fixed
    
except ImportError as e:
    _JIT_FUNCTIONS_AVAILABLE = False
    _IMPORT_ERRORS.append(f"JIT functions: {e}")
    warnings.warn(f"JIT functions not available: {e}")

# Stage 4: 構造テンソル演算
try:
    from .core.structural_tensor import (
        StructuralTensorFeatures,
        StructuralTensorExtractor,
        extract_lambda3_features
    )
    _STRUCTURAL_TENSOR_AVAILABLE = True
except ImportError as e:
    _STRUCTURAL_TENSOR_AVAILABLE = False
    _IMPORT_ERRORS.append(f"Structural tensor: {e}")
    warnings.warn(f"Structural tensor module not available: {e}")

# Stage 5: 分析モジュール（コア依存）
try:
    from .analysis.hierarchical import (
        HierarchicalAnalyzer,
        HierarchicalSeparationResults,
        analyze_hierarchical_structure,
        compare_multiple_hierarchies
    )
    _HIERARCHICAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    _HIERARCHICAL_ANALYSIS_AVAILABLE = False
    _IMPORT_ERRORS.append(f"Hierarchical analysis: {e}")
    warnings.warn(f"Hierarchical analysis not available: {e}")

try:
    from .analysis.pairwise import (
        PairwiseAnalyzer,
        PairwiseInteractionResults,
        analyze_pairwise_interaction,
        compare_all_pairs
    )
    _PAIRWISE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    _PAIRWISE_ANALYSIS_AVAILABLE = False
    _IMPORT_ERRORS.append(f"Pairwise analysis: {e}")
    warnings.warn(f"Pairwise analysis not available: {e}")

# Stage 6: 可視化モジュール（分析依存）
try:
    from .visualization.base import (
        Lambda3BaseVisualizer,
        TimeSeriesVisualizer,
        InteractionVisualizer,
        HierarchicalVisualizer,
        apply_lambda3_style,
        get_lambda3_colors,
        create_lambda3_visualizer
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError as e:
    _VISUALIZATION_AVAILABLE = False
    _IMPORT_ERRORS.append(f"Visualization: {e}")
    warnings.warn(f"Visualization modules not available: {e}")

# Stage 7: パイプライン（全モジュール依存）
try:
    from .pipelines.comprehensive import (
        Lambda3ComprehensivePipeline,
        Lambda3ComprehensiveResults,
        run_lambda3_analysis,
        create_analysis_report
    )
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    _PIPELINE_AVAILABLE = False
    _IMPORT_ERRORS.append(f"Pipeline: {e}")
    warnings.warn(f"Pipeline modules not available: {e}")

# ================================================================
# 便利関数の定義（安全な実装）
# ================================================================

def analyze(data, 
           config=None, 
           analysis_type='comprehensive',
           series_names=None,
           enable_visualization=True,
           force_jit=None):
    """
    Lambda³分析のメイン便利関数（安全版）
    """
    if not _STRUCTURAL_TENSOR_AVAILABLE:
        raise ImportError("Structural tensor module required for analysis")
    
    # データ形式の正規化
    if isinstance(data, (list, tuple)):
        if series_names is None:
            series_names = [f"Series_{i+1}" for i in range(len(data))]
        data_dict = {name: series for name, series in zip(series_names, data)}
    elif hasattr(data, 'shape') and len(data.shape) == 1:
        series_name = series_names[0] if series_names else "Series_1"
        data_dict = {series_name: data}
    elif isinstance(data, dict):
        data_dict = data
    else:
        data_dict = data
    
    # 設定の生成
    if config is None and _CONFIG_AVAILABLE:
        if analysis_type == 'financial':
            config = create_financial_config()
        elif analysis_type == 'rapid':
            config = create_rapid_config()
        elif analysis_type == 'research':
            config = create_research_config()
        else:
            config = create_default_config()
    
    # JIT設定の調整
    if config and hasattr(config, 'base') and hasattr(config.base, 'jit_config'):
        jit_available = _JIT_FUNCTIONS_AVAILABLE
        if force_jit is not None:
            config.base.jit_config.enable_jit = force_jit and jit_available
        else:
            config.base.jit_config.enable_jit = jit_available
    
    # 解析実行
    if _PIPELINE_AVAILABLE and len(data_dict) > 1:
        return run_lambda3_analysis(data_dict, config=config, analysis_type=analysis_type)
    else:
        series_name, series_data = next(iter(data_dict.items()))
        return extract_features(
            series_data,
            config=config.base if hasattr(config, 'base') else config,
            series_name=series_name
        )

def extract_features(data, 
                    config=None, 
                    feature_level='comprehensive',
                    series_name="Series",
                    use_jit=None):
    """
    構造テンソル特徴抽出の便利関数（安全版）
    """
    if not _STRUCTURAL_TENSOR_AVAILABLE:
        raise ImportError("Structural tensor module required for feature extraction")
    
    # JIT使用判定
    if use_jit is None:
        use_jit = _JIT_FUNCTIONS_AVAILABLE
    
    # 設定準備
    if config is None and _CONFIG_AVAILABLE:
        config = L3BaseConfig()
    
    if config and hasattr(config, 'jit_config'):
        config.jit_config.enable_jit = use_jit
    
    # 特徴抽出実行
    return extract_lambda3_features(
        data, 
        config=config, 
        series_name=series_name,
        feature_level=feature_level
    )

def create_config(config_type='default', enable_jit=None):
    """
    設定オブジェクト作成の便利関数（安全版）
    """
    if not _CONFIG_AVAILABLE:
        raise ImportError("Configuration system not available")
    
    config = get_config(config_type)
    
    if enable_jit is not None and hasattr(config.base, 'jit_config'):
        config.base.jit_config.enable_jit = enable_jit and _JIT_FUNCTIONS_AVAILABLE
    
    return config

def apply_style(style_name='lambda3_default'):
    """
    Lambda³可視化スタイル適用（安全版）
    """
    if _VISUALIZATION_AVAILABLE:
        apply_lambda3_style(style_name)
    else:
        warnings.warn("Visualization modules not available")

# ================================================================
# パッケージ情報とステータス
# ================================================================

def get_package_info():
    """パッケージ情報取得（修正版）"""
    return {
        'name': 'lambda3',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'description': 'Lambda³ Theory: Structural Tensor Analytics for Time Series',
        'module_status': {
            'config_system': _CONFIG_AVAILABLE,
            'jit_functions': _JIT_FUNCTIONS_AVAILABLE,
            'structural_tensor': _STRUCTURAL_TENSOR_AVAILABLE,
            'hierarchical_analysis': _HIERARCHICAL_ANALYSIS_AVAILABLE,
            'pairwise_analysis': _PAIRWISE_ANALYSIS_AVAILABLE,
            'visualization': _VISUALIZATION_AVAILABLE,
            'pipeline': _PIPELINE_AVAILABLE
        },
        'jit_status': {
            'jit_available': _JIT_FUNCTIONS_AVAILABLE,
            'jit_functions_loaded': _JIT_FUNCTIONS_AVAILABLE
        },
        'import_errors': _IMPORT_ERRORS
    }

def print_welcome():
    """ウェルカムメッセージ表示（修正版）"""
    info = get_package_info()
    
    jit_mark = "🚀" if info['jit_status']['jit_available'] else "⚠️"
    jit_status = "ENABLED" if info['jit_status']['jit_available'] else "DISABLED"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            Lambda³ Analytics Package                         ║
║                         Structural Tensor Time Series Analysis               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Version: {info['version']:<20} Author: {info['author']:<30} ║
║ License: {info['license']:<67} ║
║ JIT Optimization: {jit_mark} {jit_status:<52} ║
╚══════════════════════════════════════════════════════════════════════════════╝

モジュール状態:
""")
    
    for module, available in info['module_status'].items():
        status_mark = "✅" if available else "❌"
        print(f"  {status_mark} {module}")
    
    if info['import_errors']:
        print(f"\n⚠️  インポートエラー ({len(info['import_errors'])}):")
        for error in info['import_errors'][:3]:  # 最初の3つのみ表示
            print(f"  • {error}")
    
    print(f"""
クイックスタート:
  import lambda3 as l3
  
  # 基本分析
  results = l3.analyze(your_data)
  
  # 設定作成
  config = l3.create_config('financial')
  
  # パッケージ状態確認
  l3.print_system_status()
    """)

def print_system_status():
    """システム状態詳細表示（修正版）"""
    info = get_package_info()
    
    print("\n📊 Lambda³ システム状態")
    print("=" * 50)
    
    # モジュール状態
    available_count = sum(info['module_status'].values())
    total_count = len(info['module_status'])
    print(f"🔧 モジュール状態: {available_count}/{total_count} 利用可能")
    
    for module, status in info['module_status'].items():
        mark = "✅" if status else "❌"
        print(f"   {mark} {module}")
    
    # JIT状態
    print(f"\n⚡ JIT最適化: {'有効' if info['jit_status']['jit_available'] else '無効'}")
    
    # エラー情報
    if info['import_errors']:
        print(f"\n❌ インポートエラー詳細:")
        for i, error in enumerate(info['import_errors'], 1):
            print(f"   {i}. {error}")
    
    # 推奨事項
    if available_count < total_count:
        print(f"\n💡 推奨事項:")
        print(f"   • 依存関係の再インストール: pip install -r requirements.txt")
        print(f"   • パッケージの再インストール: pip install -e .")
        if not info['jit_status']['jit_available']:
            print(f"   • JIT最適化: pip install numba")

def validate_installation():
    """インストール検証（修正版）"""
    info = get_package_info()
    
    print("🔍 Lambda³ インストール検証")
    print("=" * 40)
    
    available_modules = sum(info['module_status'].values())
    total_modules = len(info['module_status'])
    success_rate = available_modules / total_modules
    
    if success_rate >= 0.8:
        print("✅ インストール状態: 優秀")
        print("🚀 全機能が利用可能です")
    elif success_rate >= 0.6:
        print("⚠️  インストール状態: 良好")
        print("📊 主要機能が利用可能です")
    elif success_rate >= 0.4:
        print("⚠️  インストール状態: 部分的")
        print("🔧 一部機能に制限があります")
    else:
        print("❌ インストール状態: 問題あり")
        print("🔧 修復が必要です")
    
    return success_rate >= 0.6

# ================================================================
# __all__ 定義（利用可能なもののみ）
# ================================================================

__all__ = [
    # バージョン情報
    '__version__', '__author__', '__license__',
    
    # 便利関数（常に利用可能）
    'analyze', 'extract_features', 'create_config', 'apply_style',
    'get_package_info', 'print_welcome', 'print_system_status', 'validate_installation'
]

# 条件付きで追加
if _CONFIG_AVAILABLE:
    __all__.extend([
        'L3BaseConfig', 'L3JITConfig', 'L3ComprehensiveConfig',
        'create_default_config', 'create_financial_config', 
        'create_rapid_config', 'create_research_config', 'get_config'
    ])

if _JIT_FUNCTIONS_AVAILABLE:
    __all__.extend([
        'test_jit_functions', 'run_jit_benchmark',
        'calculate_diff_and_threshold', 'detect_structural_jumps',
        'calculate_tension_scalar'
    ])

if _STRUCTURAL_TENSOR_AVAILABLE:
    __all__.extend([
        'StructuralTensorFeatures', 'StructuralTensorExtractor'
    ])

if _HIERARCHICAL_ANALYSIS_AVAILABLE:
    __all__.extend([
        'HierarchicalAnalyzer', 'HierarchicalSeparationResults'
    ])

if _PAIRWISE_ANALYSIS_AVAILABLE:
    __all__.extend([
        'PairwiseAnalyzer', 'PairwiseInteractionResults'
    ])

if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        'Lambda3BaseVisualizer', 'TimeSeriesVisualizer', 
        'InteractionVisualizer', 'HierarchicalVisualizer'
    ])

if _PIPELINE_AVAILABLE:
    __all__.extend([
        'Lambda3ComprehensivePipeline', 'Lambda3ComprehensiveResults',
        'run_lambda3_analysis', 'create_analysis_report'
    ])

# ================================================================
# 初期化チェック
# ================================================================

# 最小限の機能確認
if not _CONFIG_AVAILABLE and not _STRUCTURAL_TENSOR_AVAILABLE:
    warnings.warn(
        "Lambda³ critical modules unavailable. "
        "Please check installation: pip install -e ."
    )

# JIT最適化状況通知
if not _JIT_FUNCTIONS_AVAILABLE:
    warnings.warn(
        "JIT optimization unavailable. Install numba for better performance: "
        "pip install numba"
    )

# 成功メッセージ（条件付き）
available_count = sum([
    _CONFIG_AVAILABLE, _STRUCTURAL_TENSOR_AVAILABLE, 
    _JIT_FUNCTIONS_AVAILABLE, _HIERARCHICAL_ANALYSIS_AVAILABLE,
    _PAIRWISE_ANALYSIS_AVAILABLE, _VISUALIZATION_AVAILABLE, 
    _PIPELINE_AVAILABLE
])

if available_count >= 4:  # 最低4つのモジュールが利用可能
    print(f"Lambda³ v{__version__} loaded successfully! ({available_count}/7 modules available)")
    if _JIT_FUNCTIONS_AVAILABLE:
        print("⚡ JIT optimization ready for high-performance computing")
elif available_count >= 2:
    print(f"Lambda³ v{__version__} partially loaded ({available_count}/7 modules)")
    print("Use l3.print_system_status() for details")
else:
    print(f"Lambda³ v{__version__} limited functionality ({available_count}/7 modules)")
    print("Installation repair recommended")
