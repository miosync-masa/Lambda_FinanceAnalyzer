# ==========================================================
# lambda3/__init__.py (修正版)
# Lambda³ Main Module - 循環インポート完全解決版
# ==========================================================

"""
Lambda³ (Lambda Cubed) Analytics Package - 修正版

構造テンソル(Λ)理論に基づく高度時系列解析フレームワーク

修正点:
- 段階的インポート戦略による循環依存解消
- Protocol準拠による型安全性確保  
- JIT最適化の段階的有効化
- エラー耐性の向上
- 依存関係の明確化

Author: Masamichi Iizumi (Miosync, Inc.)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Masamichi Iizumi"
__email__ = "m.iizumi@miosync.email"
__license__ = "MIT"
__description__ = "Lambda³ Theory: Structural Tensor Analytics for Time Series"

import warnings
import sys
from typing import Union, Dict, Any, List, Optional
import numpy as np

# ==========================================================
# 段階的インポート戦略（循環依存回避）
# ==========================================================

# インポートエラー記録
_IMPORT_ERRORS = []
_MODULE_STATUS = {}

print("🚀 Lambda³ Analytics Framework - Initializing...")

# ==========================================================
# Stage 1: 共通型定義（最優先）
# ==========================================================

try:
    from .core.types import (
        StructuralTensorProtocol,
        HierarchicalResultProtocol,
        PairwiseResultProtocol,
        ComprehensiveResultProtocol,
        ConfigProtocol,
        AnalyzerProtocol,
        AnalysisMode,
        FeatureLevel,
        QualityLevel,
        Lambda3Error,
        StructuralTensorError,
        is_structural_tensor_compatible,
        ensure_float_array,
        ensure_series_name
    )
    _MODULE_STATUS['types'] = True
    print("   ✅ Types system loaded")
except ImportError as e:
    _MODULE_STATUS['types'] = False
    _IMPORT_ERRORS.append(f"Types: {e}")
    warnings.warn(f"Lambda³ types not available: {e}")
    # フォールバック型定義
    StructuralTensorProtocol = Any
    HierarchicalResultProtocol = Any
    PairwiseResultProtocol = Any

# ==========================================================  
# Stage 2: 核心設定システム
# ==========================================================

try:
    from .core.config import (
        L3BaseConfig,
        L3JITConfig,
        L3BayesianConfig,
        L3HierarchicalConfig,
        L3PairwiseConfig,
        L3VisualizationConfig,
        L3ComprehensiveConfig,
        create_default_config,
        create_financial_config,
        create_rapid_config,
        create_research_config,
        get_config,
        validate_config
    )
    _MODULE_STATUS['config'] = True
    print("   ✅ Configuration system loaded")
except ImportError as e:
    _MODULE_STATUS['config'] = False
    _IMPORT_ERRORS.append(f"Config: {e}")
    warnings.warn(f"Configuration system not available: {e}")

# ==========================================================
# Stage 3: JIT最適化関数（核心機能）
# ==========================================================

try:
    from .core.jit_functions import (
        test_jit_functions_fixed,
        benchmark_performance_fixed,
        extract_lambda3_features_jit,
        calculate_diff_and_threshold_fixed,
        detect_structural_jumps_fixed,
        calculate_tension_scalar_fixed,
        detect_hierarchical_jumps_fixed,
        calculate_sync_profile_fixed
    )
    _MODULE_STATUS['jit_functions'] = True
    print("   ✅ JIT optimization loaded")
    
    # 便利エイリアス
    test_jit_functions = test_jit_functions_fixed
    run_jit_benchmark = benchmark_performance_fixed
    
except ImportError as e:
    _MODULE_STATUS['jit_functions'] = False
    _IMPORT_ERRORS.append(f"JIT functions: {e}")
    warnings.warn(f"JIT functions not available: {e}")

# ==========================================================
# Stage 4: 構造テンソル演算（コア）
# ==========================================================

try:
    from .core.structural_tensor import (
        StructuralTensorFeatures,
        StructuralTensorExtractor,
        extract_lambda3_features,
        create_sample_structural_tensor
    )
    _MODULE_STATUS['structural_tensor'] = True
    print("   ✅ Structural tensor module loaded")
except ImportError as e:
    _MODULE_STATUS['structural_tensor'] = False
    _IMPORT_ERRORS.append(f"Structural tensor: {e}")
    warnings.warn(f"Structural tensor module not available: {e}")

# ==========================================================
# Stage 5: 分析モジュール（コア依存）
# ==========================================================

try:
    if _MODULE_STATUS.get('structural_tensor', False):
        from .analysis.hierarchical import (
            HierarchicalAnalyzer,
            HierarchicalSeparationResults,
            analyze_hierarchical_structure
        )
        _MODULE_STATUS['hierarchical_analysis'] = True
        print("   ✅ Hierarchical analysis loaded")
    else:
        raise ImportError("Structural tensor module required")
except ImportError as e:
    _MODULE_STATUS['hierarchical_analysis'] = False
    _IMPORT_ERRORS.append(f"Hierarchical analysis: {e}")
    warnings.warn(f"Hierarchical analysis not available: {e}")

try:
    if _MODULE_STATUS.get('structural_tensor', False):
        from .analysis.pairwise import (
            PairwiseAnalyzer,
            PairwiseInteractionResults,
            analyze_pairwise_interaction
        )
        _MODULE_STATUS['pairwise_analysis'] = True
        print("   ✅ Pairwise analysis loaded")
    else:
        raise ImportError("Structural tensor module required")
except ImportError as e:
    _MODULE_STATUS['pairwise_analysis'] = False
    _IMPORT_ERRORS.append(f"Pairwise analysis: {e}")
    warnings.warn(f"Pairwise analysis not available: {e}")

# ==========================================================
# Stage 6: 可視化モジュール（オプション）
# ==========================================================

try:
    from .visualization.base import (
        Lambda3BaseVisualizer,
        TimeSeriesVisualizer,
        InteractionVisualizer,
        HierarchicalVisualizer,
        apply_lambda3_style,
        create_lambda3_visualizer
    )
    _MODULE_STATUS['visualization'] = True
    print("   ✅ Visualization modules loaded")
except ImportError as e:
    _MODULE_STATUS['visualization'] = False
    _IMPORT_ERRORS.append(f"Visualization: {e}")
    warnings.warn(f"Visualization modules not available: {e}")

# ==========================================================
# Stage 7: パイプライン（全モジュール依存）
# ==========================================================

try:
    # 必要モジュールの確認
    required_modules = ['types', 'structural_tensor']
    if all(_MODULE_STATUS.get(mod, False) for mod in required_modules):
        from .pipelines.comprehensive import (
            Lambda3ComprehensivePipeline,
            Lambda3ComprehensiveResults,
            run_lambda3_analysis,
            create_analysis_report
        )
        _MODULE_STATUS['pipeline'] = True
        print("   ✅ Comprehensive pipeline loaded")
    else:
        missing = [mod for mod in required_modules if not _MODULE_STATUS.get(mod, False)]
        raise ImportError(f"Required modules missing: {missing}")
except ImportError as e:
    _MODULE_STATUS['pipeline'] = False
    _IMPORT_ERRORS.append(f"Pipeline: {e}")
    warnings.warn(f"Comprehensive pipeline not available: {e}")

# ==========================================================
# 便利関数（安全版）
# ==========================================================

def analyze(data: Union[Dict[str, np.ndarray], np.ndarray], 
           analysis_type: str = 'comprehensive',
           config: Optional[Any] = None,
           **kwargs) -> Any:
    """
    Lambda³解析の便利関数（安全版）
    
    Args:
        data: 入力データ（辞書またはnumpy配列）
        analysis_type: 解析タイプ ('structural', 'hierarchical', 'pairwise', 'comprehensive', 'financial')
        config: 設定オブジェクト
        **kwargs: 追加パラメータ
        
    Returns:
        解析結果オブジェクト
    """
    if not _MODULE_STATUS.get('structural_tensor', False):
        raise ImportError("Structural tensor module required for analysis")
    
    # データの前処理
    if isinstance(data, np.ndarray):
        data_dict = {'Series': data}
    elif isinstance(data, dict):
        data_dict = data
    else:
        try:
            data_dict = {'Series': np.asarray(data)}
        except Exception as e:
            raise ValueError(f"Cannot convert data to suitable format: {e}")
    
    # 分析タイプ別実行
    if analysis_type == 'structural':
        # 構造テンソル特徴量抽出のみ
        results = {}
        for name, series_data in data_dict.items():
            features = extract_lambda3_features(series_data, series_name=name, config=config)
            results[name] = features
        return results
        
    elif analysis_type == 'hierarchical':
        # 階層分析
        if not _MODULE_STATUS.get('hierarchical_analysis', False):
            raise ImportError("Hierarchical analysis module not available")
        
        results = {}
        for name, series_data in data_dict.items():
            features = extract_lambda3_features(series_data, series_name=name, 
                                              feature_level='comprehensive', config=config)
            analyzer = HierarchicalAnalyzer(config=config)
            hierarchy_result = analyzer.analyze_hierarchical_separation(features)
            results[name] = hierarchy_result
        return results
        
    elif analysis_type == 'pairwise':
        # ペアワイズ分析
        if not _MODULE_STATUS.get('pairwise_analysis', False):
            raise ImportError("Pairwise analysis module not available")
        
        if len(data_dict) < 2:
            raise ValueError("Pairwise analysis requires at least 2 series")
        
        series_names = list(data_dict.keys())
        features_dict = {}
        
        # 特徴量抽出
        for name, series_data in data_dict.items():
            features_dict[name] = extract_lambda3_features(series_data, series_name=name, config=config)
        
        # ペアワイズ分析実行
        analyzer = PairwiseAnalyzer(config=config)
        result = analyzer.analyze_asymmetric_interaction(
            features_dict[series_names[0]], 
            features_dict[series_names[1]]
        )
        return result
        
    elif analysis_type in ['comprehensive', 'financial']:
        # 包括分析
        if not _MODULE_STATUS.get('pipeline', False):
            raise ImportError("Comprehensive pipeline not available")
        
        # 設定準備
        if config is None:
            if analysis_type == 'financial':
                config = create_financial_config() if _MODULE_STATUS.get('config', False) else None
            else:
                config = create_default_config() if _MODULE_STATUS.get('config', False) else None
        
        # パイプライン実行
        pipeline = Lambda3ComprehensivePipeline(config=config)
        return pipeline.run_analysis(data_dict, **kwargs)
    
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

def extract_features(data: Union[np.ndarray, List[float]], 
                    series_name: str = "Series",
                    feature_level: str = 'standard',
                    config: Optional[Any] = None,
                    use_jit: Optional[bool] = None) -> Any:
    """
    構造テンソル特徴抽出の便利関数（安全版）
    """
    if not _MODULE_STATUS.get('structural_tensor', False):
        raise ImportError("Structural tensor module required for feature extraction")
    
    return extract_lambda3_features(
        data, 
        config=config,
        series_name=series_name,
        feature_level=feature_level,
        use_jit=use_jit
    )

def create_config(config_type: str = 'default', **kwargs) -> Any:
    """
    設定オブジェクト作成の便利関数（安全版）
    
    Args:
        config_type: 設定タイプ ('default', 'financial', 'rapid', 'research')
        **kwargs: 追加設定パラメータ
        
    Returns:
        設定オブジェクト
    """
    if not _MODULE_STATUS.get('config', False):
        raise ImportError("Configuration system not available")
    
    config_factories = {
        'default': create_default_config,
        'financial': create_financial_config,
        'rapid': create_rapid_config,
        'research': create_research_config
    }
    
    if config_type not in config_factories:
        raise ValueError(f"Unknown config type: {config_type}")
    
    config = config_factories[config_type]()
    
    # 追加パラメータの適用
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config, 'base') and hasattr(config.base, key):
            setattr(config.base, key, value)
    
    return config

def apply_style(style_name: str = 'lambda3_default'):
    """
    Lambda³可視化スタイル適用（安全版）
    """
    if _MODULE_STATUS.get('visualization', False):
        apply_lambda3_style(style_name)
    else:
        warnings.warn("Visualization modules not available")

def create_sample_data(n_series: int = 3, 
                      n_points: int = 200,
                      series_names: Optional[List[str]] = None,
                      random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    サンプルデータ生成の便利関数
    
    Args:
        n_series: 系列数
        n_points: データポイント数
        series_names: 系列名リスト（省略時は自動生成）
        random_seed: 乱数シード
        
    Returns:
        Dict[str, np.ndarray]: サンプルデータ辞書
    """
    np.random.seed(random_seed)
    
    if series_names is None:
        series_names = [f"Series_{i+1}" for i in range(n_series)]
    elif len(series_names) != n_series:
        raise ValueError(f"series_names length ({len(series_names)}) != n_series ({n_series})")
    
    data_dict = {}
    
    for i, name in enumerate(series_names):
        # 基本トレンド
        trend = np.cumsum(np.random.randn(n_points) * 0.02)
        
        # 構造変化ジャンプ
        n_jumps = max(1, n_points // 30)
        jump_positions = np.random.choice(n_points, size=n_jumps, replace=False)
        jumps = np.zeros(n_points)
        jumps[jump_positions] = np.random.normal(0, 0.5, n_jumps)
        
        # 張力変動
        tension_base = 0.3 + 0.2 * np.sin(2 * np.pi * np.arange(n_points) / 50)
        tension_noise = np.random.normal(0, 0.1, n_points)
        
        # 最終データ
        data = 100 + trend + np.cumsum(jumps) + tension_base + tension_noise
        data_dict[name] = data.astype(np.float64)
    
    return data_dict

# ==========================================================
# パッケージ情報とステータス
# ==========================================================

def get_package_info() -> Dict[str, Any]:
    """パッケージ情報取得（修正版）"""
    return {
        'name': 'lambda3',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'description': __description__,
        'module_status': _MODULE_STATUS.copy(),
        'import_errors': _IMPORT_ERRORS.copy(),
        'jit_available': _MODULE_STATUS.get('jit_functions', False),
        'bayesian_available': False,  # 後で実装確認
        'visualization_available': _MODULE_STATUS.get('visualization', False),
        'total_modules': len(_MODULE_STATUS),
        'available_modules': sum(_MODULE_STATUS.values()),
        'availability_ratio': sum(_MODULE_STATUS.values()) / len(_MODULE_STATUS) if _MODULE_STATUS else 0.0
    }

def print_system_status():
    """システム状態表示"""
    info = get_package_info()
    
    print(f"\n🔍 Lambda³ System Status")
    print("=" * 40)
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Available modules: {info['available_modules']}/{info['total_modules']} ({info['availability_ratio']:.1%})")
    
    print(f"\n📊 Module Status:")
    for module, status in info['module_status'].items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {module}")
    
    if info['import_errors']:
        print(f"\n⚠️  Import Errors:")
        for error in info['import_errors']:
            print(f"  • {error}")
    
    print(f"\n🚀 Core Features:")
    print(f"  • JIT Optimization: {'✅' if info['jit_available'] else '❌'}")
    print(f"  • Visualization: {'✅' if info['visualization_available'] else '❌'}")
    print(f"  • Bayesian Analysis: {'✅' if info['bayesian_available'] else '❌'}")

def validate_installation() -> bool:
    """インストール検証（修正版）"""
    info = get_package_info()
    
    print("🔍 Lambda³ Installation Validation")
    print("=" * 40)
    
    # 必須モジュールの確認
    required_modules = ['types', 'structural_tensor']
    required_available = all(info['module_status'].get(mod, False) for mod in required_modules)
    
    if not required_available:
        print("❌ Critical modules missing - installation incomplete")
        missing = [mod for mod in required_modules if not info['module_status'].get(mod, False)]
        print(f"   Missing: {', '.join(missing)}")
        return False
    
    # 可用性評価
    if info['availability_ratio'] >= 0.8:
        print("✅ Installation status: Excellent")
        print("🚀 All major features available")
    elif info['availability_ratio'] >= 0.6:
        print("⚠️  Installation status: Good")
        print("📊 Core features available")
    elif info['availability_ratio'] >= 0.4:
        print("⚠️  Installation status: Partial")
        print("🔧 Some features limited")
    else:
        print("❌ Installation status: Poor")
        print("🔧 Repair needed")
        return False
    
    return True

def print_welcome():
    """ウェルカムメッセージ表示"""
    print("\n" + "=" * 60)
    print("🌟 Welcome to Lambda³ Analytics Framework! 🌟")
    print("=" * 60)
    print("📚 Lambda³ Theory: Structural Tensor Analytics for Time Series")
    print(f"🔬 Version: {__version__}")
    print(f"👨‍💻 Author: {__author__}")
    print("📖 License: MIT")
    
    print(f"\n🎯 Quick Start:")
    print("   import lambda3 as l3")
    print("   data = l3.create_sample_data()")
    print("   results = l3.analyze(data)")
    
    print(f"\n🔧 System Check:")
    info = get_package_info()
    print(f"   Modules: {info['available_modules']}/{info['total_modules']} available")
    print(f"   JIT: {'✅' if info['jit_available'] else '❌'}")
    print(f"   Status: {'✅ Ready' if info['availability_ratio'] >= 0.6 else '⚠️ Limited'}")
    
    if info['availability_ratio'] < 0.6:
        print(f"\n💡 Recommendations:")
        print("   • pip install -r requirements.txt")
        print("   • pip install -e .")
        if not info['jit_available']:
            print("   • pip install numba  # for JIT optimization")

# ==========================================================
# __all__ 定義（利用可能なもののみ）
# ==========================================================

__all__ = [
    # バージョン情報
    '__version__', '__author__', '__license__', '__description__',
    
    # 便利関数（常に利用可能）
    'analyze', 'extract_features', 'create_config', 'apply_style', 'create_sample_data',
    'get_package_info', 'print_system_status', 'validate_installation', 'print_welcome'
]

# 条件付きで追加
if _MODULE_STATUS.get('types', False):
    __all__.extend([
        'StructuralTensorProtocol', 'HierarchicalResultProtocol', 'PairwiseResultProtocol',
        'AnalysisMode', 'FeatureLevel', 'QualityLevel',
        'Lambda3Error', 'StructuralTensorError'
    ])

if _MODULE_STATUS.get('config', False):
    __all__.extend([
        'L3BaseConfig', 'L3ComprehensiveConfig',
        'create_default_config', 'create_financial_config', 
        'create_rapid_config', 'create_research_config'
    ])

if _MODULE_STATUS.get('jit_functions', False):
    __all__.extend([
        'test_jit_functions', 'run_jit_benchmark'
    ])

if _MODULE_STATUS.get('structural_tensor', False):
    __all__.extend([
        'StructuralTensorFeatures', 'StructuralTensorExtractor',
        'create_sample_structural_tensor'
    ])

if _MODULE_STATUS.get('hierarchical_analysis', False):
    __all__.extend([
        'HierarchicalAnalyzer', 'HierarchicalSeparationResults'
    ])

if _MODULE_STATUS.get('pairwise_analysis', False):
    __all__.extend([
        'PairwiseAnalyzer', 'PairwiseInteractionResults'
    ])

if _MODULE_STATUS.get('visualization', False):
    __all__.extend([
        'Lambda3BaseVisualizer', 'TimeSeriesVisualizer', 
        'InteractionVisualizer', 'HierarchicalVisualizer'
    ])

if _MODULE_STATUS.get('pipeline', False):
    __all__.extend([
        'Lambda3ComprehensivePipeline', 'Lambda3ComprehensiveResults',
        'run_lambda3_analysis', 'create_analysis_report'
    ])

# ==========================================================
# 初期化完了メッセージ
# ==========================================================

def _finalize_initialization():
    """初期化完了処理"""
    info = get_package_info()
    
    if info['availability_ratio'] >= 0.6:
        status_emoji = "✅"
        status_text = "Ready"
    elif info['availability_ratio'] >= 0.4:
        status_emoji = "⚠️"
        status_text = "Partial"
    else:
        status_emoji = "❌"
        status_text = "Limited"
    
    print(f"   {status_emoji} Lambda³ initialized: {status_text} ({info['available_modules']}/{info['total_modules']} modules)")
    
    if info['jit_available']:
        print("   ⚡ JIT optimization enabled")
    
    if info['import_errors'] and len(info['import_errors']) <= 3:
        print(f"   ⚠️  {len(info['import_errors'])} minor issues detected")

# 初期化完了
_finalize_initialization()

# ==========================================================
# 最小限の機能確認
# ==========================================================

if not _MODULE_STATUS.get('types', False) and not _MODULE_STATUS.get('structural_tensor', False):
    warnings.warn(
        "Lambda³ critical modules unavailable. "
        "Please check installation: pip install -e ."
    )

# ==========================================================
# デバッグ用ヘルパー
# ==========================================================

def debug_import_issues():
    """インポート問題のデバッグ情報表示"""
    print("🐛 Lambda³ Import Debug Information")
    print("=" * 50)
    
    print("Python Version:", sys.version)
    print("Lambda³ Version:", __version__)
    
    print(f"\nModule Status:")
    for module, status in _MODULE_STATUS.items():
        print(f"  {module}: {'✅' if status else '❌'}")
    
    if _IMPORT_ERRORS:
        print(f"\nImport Errors:")
        for i, error in enumerate(_IMPORT_ERRORS, 1):
            print(f"  {i}. {error}")
    
    print(f"\nSystem Path:")
    for i, path in enumerate(sys.path[:5], 1):
        print(f"  {i}. {path}")
    
    print(f"\nRecommended Actions:")
    if not _MODULE_STATUS.get('jit_functions', False):
        print("  • Install JIT: pip install numba")
    if _MODULE_STATUS.get('config', False) and not _MODULE_STATUS.get('structural_tensor', False):
        print("  • Check core modules: pip install -e . --force-reinstall")
    if len(_IMPORT_ERRORS) > 5:
        print("  • Clean install: pip uninstall lambda3 && pip install -e .")

# 条件付きでデバッグ関数をエクスポート
if len(_IMPORT_ERRORS) > 0:
    __all__.append('debug_import_issues')
