# ================================================================
# lambda3/utils/helpers.py - 修正版
# ================================================================

"""
Lambda³ Helper Functions

Lambda³理論の実装を支援する便利関数群
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# ================================================================
# BASE HELPERS FROM __init__.py
# ================================================================

# 数値計算
def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """安全な除算（ゼロ除算回避）"""
    if isinstance(denominator, np.ndarray):
        result = np.full_like(denominator, default, dtype=float)
        mask = np.abs(denominator) > 1e-12
        result[mask] = numerator[mask] / denominator[mask] if isinstance(numerator, np.ndarray) else numerator / denominator[mask]
        return result
    else:
        return numerator / denominator if abs(denominator) > 1e-12 else default

def robust_normalize(data: np.ndarray, 
                    method: str = 'zscore',
                    axis: Optional[int] = None) -> np.ndarray:
    """ロバスト正規化"""
    if method == 'zscore':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return safe_divide(data - mean, std)
    
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return safe_divide(data - min_val, max_val - min_val)
    
    elif method == 'robust':
        median = np.median(data, axis=axis, keepdims=True)
        mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
        return safe_divide(data - median, mad * 1.4826)
    
    elif method == 'quantile':
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        median = np.median(data, axis=axis, keepdims=True)
        iqr = q75 - q25
        return safe_divide(data - median, iqr)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# ================================================================
# EXTENDED HELPERS
# ================================================================

def quick_lambda3_analysis(data: Union[np.ndarray, Dict[str, np.ndarray]], 
                          print_summary: bool = True) -> Dict[str, Any]:
    """
    Lambda³クイック分析
    
    基本的な構造テンソル分析を簡単に実行
    """
    try:
        # Lambda³インポート（循環インポート回避）
        from .. import extract_features, analyze
        
        if isinstance(data, np.ndarray):
            features = extract_features(data, series_name="QuickAnalysis")
            
            summary = {
                'series_name': features.series_name,
                'data_points': len(features.data),
                'positive_changes': int(np.sum(features.delta_LambdaC_pos)),
                'negative_changes': int(np.sum(features.delta_LambdaC_neg)),
                'average_tension': float(np.mean(features.rho_T)),
                'max_tension': float(np.max(features.rho_T))
            }
        else:
            results = analyze(data, analysis_type='rapid')
            summary = results.get_analysis_summary()
        
        if print_summary:
            print("Lambda³ Quick Analysis Results:")
            print("=" * 40)
            for key, value in summary.items():
                print(f"  {key}: {value}")
        
        return summary
        
    except ImportError:
        warnings.warn("Lambda³ core modules not available for quick analysis")
        return {'error': 'Lambda³ modules not available'}

def estimate_analysis_time(data_size: int, analysis_type: str = 'comprehensive') -> Dict[str, float]:
    """
    分析時間推定
    
    データサイズと分析タイプに基づいて実行時間を推定
    """
    # 基準性能（points per second）
    base_rates = {
        'basic': 10000,      # 基本特徴抽出
        'hierarchical': 5000, # 階層分析
        'pairwise': 2000,    # ペアワイズ分析
        'comprehensive': 1000 # 包括分析
    }
    
    # JIT最適化による高速化係数
    jit_speedup = 10
    
    try:
        # JIT利用可能性確認
        from .. import get_package_info
        package_info = get_package_info()
        jit_available = package_info['jit_status']['jit_available']
    except:
        jit_available = False
    
    base_rate = base_rates.get(analysis_type, base_rates['comprehensive'])
    
    if jit_available:
        effective_rate = base_rate * jit_speedup
    else:
        effective_rate = base_rate
    
    estimated_time = data_size / effective_rate
    
    return {
        'estimated_seconds': estimated_time,
        'estimated_minutes': estimated_time / 60,
        'data_size': data_size,
        'analysis_type': analysis_type,
        'jit_available': jit_available,
        'processing_rate': effective_rate
    }

def create_lambda3_report_template() -> str:
    """Lambda³分析レポートテンプレート生成"""
    template = """
# Lambda³ Analysis Report

## Executive Summary
- **Analysis Date**: {analysis_date}
- **Data Series**: {series_count} series
- **Analysis Type**: {analysis_type}
- **JIT Optimization**: {jit_status}

## Key Findings

### Structural Tensor Analysis
- **∆ΛC Positive Changes**: {positive_changes}
- **∆ΛC Negative Changes**: {negative_changes}
- **Average Tension (ρT)**: {average_tension:.4f}
- **Maximum Tension**: {max_tension:.4f}

### Hierarchical Analysis
- **Escalation Strength**: {escalation_strength:.4f}
- **Deescalation Strength**: {deescalation_strength:.4f}
- **Hierarchy Correlation**: {hierarchy_correlation:.4f}

### Pairwise Interactions
- **Strongest Coupling**: {strongest_coupling:.4f}
- **Most Asymmetric Pair**: {most_asymmetric_pair}
- **Network Density**: {network_density:.4f}

## Performance Metrics
- **Execution Time**: {execution_time:.2f} seconds
- **Processing Rate**: {processing_rate:.0f} points/sec
- **Memory Usage**: {memory_usage:.2f} MB

## Recommendations
{recommendations}

---
*Generated by Lambda³ Finance Analyzer*
*Theory: Non-temporal structural tensor analysis*
    """
    
    return template

# ================================================================
# EXPORTS - 正しく初期化
# ================================================================

__all__ = [
    # 基本ヘルパー
    'safe_divide',
    'robust_normalize',
    # 拡張ヘルパー
    'quick_lambda3_analysis',
    'estimate_analysis_time', 
    'create_lambda3_report_template'
]
