# ==========================================================
# lambda3/__init__.py
# Lambda³ Finance Analyzer Package Initialization (修正版)
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³ Finance Analyzer - 構造テンソル金融分析

時間非依存の構造空間における金融時系列分析のための
革新的な数学的フレームワーク。
"""

__version__ = "0.2.0"
__author__ = "Masamichi Iizumi"
__license__ = "MIT"

import warnings
from typing import Dict, Any, Optional, Union, List
import numpy as np

# モジュール状態追跡
_MODULE_STATUS = {}
_IMPORT_ERRORS = []

print("Lambda³ Finance Analyzer v" + __version__)
print("Initializing core modules...")

# ==========================================================
# Stage 1: Configuration System
# ==========================================================

try:
    from .core.config import (
        L3Config,
        L3FinancialConfig,
        L3ResearchConfig,
        L3RapidConfig,
        L3ConfigFactory,
        create_config,
        DEFAULT_CONFIG,
        FINANCIAL_CONFIG,
        RESEARCH_CONFIG,
        RAPID_CONFIG
    )
    _MODULE_STATUS['config'] = True
    print("   ✅ Configuration system loaded")
except ImportError as e:
    _MODULE_STATUS['config'] = False
    _IMPORT_ERRORS.append(f"Config: {e}")
    warnings.warn(f"Configuration module not available: {e}")

# ==========================================================
# Stage 2: JIT Functions (Core Computational Engine)
# ==========================================================

try:
    from .core.jit_functions import (
        calculate_diff_and_threshold,
        detect_jumps,
        calculate_local_std,
        calculate_rho_t,
        detect_local_global_jumps,
        sync_rate_at_lag,
        calculate_sync_profile_jit,
        calc_lambda3_features_v2,
        detect_phase_coupling,
        test_jit_functions,
        run_jit_benchmark
    )
    _MODULE_STATUS['jit_functions'] = True
    JIT_FUNCTIONS_AVAILABLE = True
    print("   ✅ JIT optimization loaded")
except ImportError as e:
    _MODULE_STATUS['jit_functions'] = False
    JIT_FUNCTIONS_AVAILABLE = False
    _IMPORT_ERRORS.append(f"JIT functions: {e}")
    warnings.warn(f"JIT functions not available: {e}")

# ==========================================================
# Stage 3: Structural Tensor (Core Theory Implementation)
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
# Stage 4: Analysis Modules
# ==========================================================

# Hierarchical Analysis
try:
    if _MODULE_STATUS.get('structural_tensor', False):
        from .analysis.hierarchical import (
            HierarchicalAnalyzer,
            HierarchicalSeparationResults,
            complete_hierarchical_analysis,
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

# Pairwise Analysis
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
# Stage 5: Financial Analysis (Optional)
# ==========================================================

try:
    from .financial import (
        analyze_financial_markets,
        detect_financial_crises,
        FinancialAnalysisResults
    )
    _MODULE_STATUS['financial'] = True
    print("   ✅ Financial analysis loaded")
except ImportError as e:
    _MODULE_STATUS['financial'] = False
    _IMPORT_ERRORS.append(f"Financial: {e}")
    # Financial is optional, so no warning

# ==========================================================
# Stage 6: Data Acquisition (Optional)
# ==========================================================

try:
    from .data.acquisition import (
        DataAcquisitionConfig,
        fetch_market_data,
        load_financial_data
    )
    _MODULE_STATUS['data_acquisition'] = True
    print("   ✅ Data acquisition loaded")
except ImportError as e:
    _MODULE_STATUS['data_acquisition'] = False
    _IMPORT_ERRORS.append(f"Data acquisition: {e}")
    # Data acquisition is optional

# ==========================================================
# Stage 7: Visualization (Optional)
# ==========================================================

try:
    from .visualization import (
        plot_structural_tensor_features,
        plot_hierarchical_separation,
        plot_pairwise_interaction,
        create_analysis_dashboard
    )
    _MODULE_STATUS['visualization'] = True
    print("   ✅ Visualization loaded")
except ImportError as e:
    _MODULE_STATUS['visualization'] = False
    _IMPORT_ERRORS.append(f"Visualization: {e}")
    # Visualization is optional

# ==========================================================
# High-Level API
# ==========================================================

def analyze(
    data: Union[np.ndarray, Dict[str, np.ndarray], List[float]],
    analysis_type: str = 'comprehensive',
    config: Optional[L3Config] = None,
    **kwargs
) -> Any:
    """
    Lambda³分析の統一インターフェース
    
    Args:
        data: 分析対象データ（単一系列または複数系列）
        analysis_type: 分析タイプ
            - 'structural': 構造テンソル特徴量のみ
            - 'hierarchical': 階層的構造分析
            - 'pairwise': ペアワイズ非対称分析
            - 'comprehensive': 包括的分析
            - 'financial': 金融市場分析
            - 'rapid': 高速分析
        config: 設定オブジェクト
        **kwargs: 追加パラメータ
        
    Returns:
        分析結果オブジェクト
    """
    if not _MODULE_STATUS.get('structural_tensor', False):
        raise ImportError("Structural tensor module required for analysis")
    
    # デフォルト設定
    if config is None:
        if analysis_type == 'financial':
            config = FINANCIAL_CONFIG if 'config' in _MODULE_STATUS else L3Config()
        elif analysis_type == 'rapid':
            config = RAPID_CONFIG if 'config' in _MODULE_STATUS else L3Config()
        else:
            config = DEFAULT_CONFIG if 'config' in _MODULE_STATUS else L3Config()
    
    # データ準備
    if isinstance(data, (list, np.ndarray)):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data_dict = {'Series': data}
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # 分析実行
    if analysis_type == 'structural':
        return _analyze_structural(data_dict, config, **kwargs)
    elif analysis_type == 'hierarchical':
        return _analyze_hierarchical(data_dict, config, **kwargs)
    elif analysis_type == 'pairwise':
        return _analyze_pairwise(data_dict, config, **kwargs)
    elif analysis_type == 'comprehensive':
        return _analyze_comprehensive(data_dict, config, **kwargs)
    elif analysis_type == 'financial':
        return _analyze_financial(data_dict, config, **kwargs)
    elif analysis_type == 'rapid':
        return _analyze_rapid(data_dict, config, **kwargs)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

def _analyze_structural(data_dict: Dict[str, np.ndarray], config: L3Config, **kwargs) -> Dict[str, Any]:
    """構造テンソル特徴量抽出"""
    results = {}
    extractor = StructuralTensorExtractor(config)
    
    for name, series_data in data_dict.items():
        features = extractor.extract_features(
            series_data,
            series_name=name,
            feature_level=kwargs.get('feature_level', 'basic')
        )
        results[name] = features
    
    return results

def _analyze_hierarchical(data_dict: Dict[str, np.ndarray], config: L3Config, **kwargs) -> Dict[str, Any]:
    """階層的構造分析"""
    if not _MODULE_STATUS.get('hierarchical_analysis', False):
        raise ImportError("Hierarchical analysis module not available")
    
    return complete_hierarchical_analysis(data_dict, config)

def _analyze_pairwise(data_dict: Dict[str, np.ndarray], config: L3Config, **kwargs) -> Any:
    """ペアワイズ分析"""
    if not _MODULE_STATUS.get('pairwise_analysis', False):
        raise ImportError("Pairwise analysis module not available")
    
    if len(data_dict) < 2:
        raise ValueError("Pairwise analysis requires at least 2 series")
    
    series_names = list(data_dict.keys())[:2]
    
    # 特徴量抽出
    extractor = StructuralTensorExtractor(config)
    features = {}
    for name in series_names:
        features[name] = extractor.extract_features(
            data_dict[name],
            series_name=name,
            feature_level='basic'
        )
    
    # ペアワイズ分析
    analyzer = PairwiseAnalyzer(config)
    return analyzer.analyze_asymmetric_interaction(
        features[series_names[0]],
        features[series_names[1]]
    )

def _analyze_comprehensive(data_dict: Dict[str, np.ndarray], config: L3Config, **kwargs) -> Dict[str, Any]:
    """包括的分析"""
    results = {
        'structural': _analyze_structural(data_dict, config, feature_level='comprehensive')
    }
    
    if _MODULE_STATUS.get('hierarchical_analysis', False):
        results['hierarchical'] = _analyze_hierarchical(data_dict, config)
    
    if _MODULE_STATUS.get('pairwise_analysis', False) and len(data_dict) >= 2:
        results['pairwise'] = _analyze_pairwise(data_dict, config)
    
    return results

def _analyze_financial(data_dict: Dict[str, np.ndarray], config: L3Config, **kwargs) -> Any:
    """金融市場分析"""
    if not _MODULE_STATUS.get('financial', False):
        # フォールバック: 包括的分析
        return _analyze_comprehensive(data_dict, config, **kwargs)
    
    return analyze_financial_markets(data_dict, config=config, **kwargs)

def _analyze_rapid(data_dict: Dict[str, np.ndarray], config: L3Config, **kwargs) -> Dict[str, Any]:
    """高速分析（基本特徴量のみ）"""
    return _analyze_structural(data_dict, config, feature_level='basic')

# ==========================================================
# Utility Functions
# ==========================================================

def get_module_status() -> Dict[str, bool]:
    """モジュール状態を取得"""
    return _MODULE_STATUS.copy()

def get_import_errors() -> List[str]:
    """インポートエラーを取得"""
    return _IMPORT_ERRORS.copy()

def create_sample_data(n_series: int = 2, n_points: int = 100, seed: int = 42) -> Dict[str, np.ndarray]:
    """サンプルデータ生成"""
    np.random.seed(seed)
    data = {}
    
    for i in range(n_series):
        series_name = f"Series_{chr(65 + i)}"  # A, B, C, ...
        data[series_name] = create_sample_structural_tensor(n_points, seed + i)
    
    return data

# ==========================================================
# Module Summary
# ==========================================================

print("\nModule Status Summary:")
for module, status in _MODULE_STATUS.items():
    status_symbol = "✅" if status else "❌"
    print(f"   {status_symbol} {module}")

if _IMPORT_ERRORS:
    print("\n⚠️  Some modules failed to load. Run with verbose=True for details.")

print(f"\nLambda³ Finance Analyzer initialized successfully!")
print(f"JIT optimization: {'Enabled' if JIT_FUNCTIONS_AVAILABLE else 'Disabled'}")

# ==========================================================
# Module Exports
# ==========================================================

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Main API
    'analyze',
    
    # Configuration
    'L3Config',
    'L3FinancialConfig',
    'L3ResearchConfig',
    'L3RapidConfig',
    'create_config',
    'DEFAULT_CONFIG',
    'FINANCIAL_CONFIG',
    
    # Core classes
    'StructuralTensorFeatures',
    'StructuralTensorExtractor',
    'HierarchicalAnalyzer',
    'PairwiseAnalyzer',
    
    # Results classes
    'HierarchicalSeparationResults',
    'PairwiseInteractionResults',
    
    # Analysis functions
    'extract_lambda3_features',
    'analyze_hierarchical_structure',
    'analyze_pairwise_interaction',
    'complete_hierarchical_analysis',
    
    # JIT functions
    'test_jit_functions',
    'run_jit_benchmark',
    
    # Utilities
    'get_module_status',
    'get_import_errors',
    'create_sample_data',
    'create_sample_structural_tensor',
    
    # Constants
    'JIT_FUNCTIONS_AVAILABLE'
]
