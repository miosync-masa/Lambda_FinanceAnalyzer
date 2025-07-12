
# ================================================================
# lambda3/utils/__init__.py
# ================================================================

"""
Lambda³ Utilities Module

Lambda³理論のためのユーティリティ関数群:
- 数値計算ヘルパー
- データ変換ユーティリティ
- 性能測定ツール
- デバッグ支援機能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import warnings
from functools import wraps
import inspect

# ================================================================
# 数値計算ユーティリティ
# ================================================================

def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    安全な除算（ゼロ除算回避）
    
    Args:
        numerator: 分子
        denominator: 分母
        default: ゼロ除算時のデフォルト値
        
    Returns:
        除算結果
    """
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
    """
    ロバスト正規化
    
    Args:
        data: 入力データ
        method: 'zscore', 'minmax', 'robust', 'quantile'
        axis: 正規化軸
        
    Returns:
        正規化されたデータ
    """
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
        return safe_divide(data - median, mad * 1.4826)  # MAD to std conversion
    
    elif method == 'quantile':
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        median = np.median(data, axis=axis, keepdims=True)
        iqr = q75 - q25
        return safe_divide(data - median, iqr)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_rolling_statistics(data: np.ndarray, 
                                window: int,
                                statistics: List[str] = None) -> Dict[str, np.ndarray]:
    """
    ローリング統計量計算
    
    Args:
        data: 入力データ
        window: 窓サイズ
        statistics: 計算する統計量リスト
        
    Returns:
        統計量辞書
    """
    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max', 'median']
    
    n = len(data)
    results = {}
    
    for stat in statistics:
        results[stat] = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            
            if stat == 'mean':
                results[stat][i] = np.mean(window_data)
            elif stat == 'std':
                results[stat][i] = np.std(window_data)
            elif stat == 'min':
                results[stat][i] = np.min(window_data)
            elif stat == 'max':
                results[stat][i] = np.max(window_data)
            elif stat == 'median':
                results[stat][i] = np.median(window_data)
            elif stat == 'skew':
                from scipy.stats import skew
                results[stat][i] = skew(window_data)
            elif stat == 'kurtosis':
                from scipy.stats import kurtosis
                results[stat][i] = kurtosis(window_data)
    
    return results

# ================================================================
# データ変換ユーティリティ
# ================================================================

def ensure_numpy_array(data: Any, dtype: np.dtype = np.float64) -> np.ndarray:
    """データをNumPy配列に確実に変換"""
    if isinstance(data, np.ndarray):
        return data.astype(dtype)
    elif isinstance(data, (list, tuple)):
        return np.array(data, dtype=dtype)
    elif isinstance(data, pd.Series):
        return data.values.astype(dtype)
    elif hasattr(data, '__array__'):
        return np.array(data, dtype=dtype)
    else:
        raise ValueError(f"Cannot convert {type(data)} to numpy array")

def dict_to_dataframe(data_dict: Dict[str, np.ndarray], 
                     index: Optional[pd.Index] = None) -> pd.DataFrame:
    """辞書をDataFrameに変換"""
    # データ長の統一
    lengths = [len(series) for series in data_dict.values()]
    min_length = min(lengths)
    
    if len(set(lengths)) > 1:
        warnings.warn(f"Series have different lengths. Truncating to {min_length}")
        data_dict = {name: series[:min_length] for name, series in data_dict.items()}
    
    df = pd.DataFrame(data_dict, index=index)
    return df

def standardize_time_series_format(data: Any) -> Tuple[np.ndarray, List[str]]:
    """
    時系列データを標準形式に変換
    
    Returns:
        data_array: 2D配列 (n_points, n_series)
        series_names: 系列名リスト
    """
    if isinstance(data, dict):
        series_names = list(data.keys())
        data_arrays = [ensure_numpy_array(series) for series in data.values()]
        
        # 長さ統一
        min_length = min(len(arr) for arr in data_arrays)
        data_arrays = [arr[:min_length] for arr in data_arrays]
        
        data_array = np.column_stack(data_arrays)
        
    elif isinstance(data, pd.DataFrame):
        series_names = data.columns.tolist()
        data_array = data.values
        
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_array = data.reshape(-1, 1)
            series_names = ['Series_1']
        else:
            data_array = data
            series_names = [f'Series_{i+1}' for i in range(data_array.shape[1])]
    
    else:
        # 単一系列として処理
        data_array = ensure_numpy_array(data).reshape(-1, 1)
        series_names = ['Series_1']
    
    return data_array, series_names

# ================================================================
# 性能測定ツール
# ================================================================

class Timer:
    """高精度タイマー"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name} completed in {self.elapsed_time:.4f} seconds")

def performance_monitor(func: Callable) -> Callable:
    """関数の性能監視デコレーター"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            end_memory = get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # 性能ログ
            print(f"Function: {func.__name__}")
            print(f"  Execution time: {execution_time:.4f}s")
            print(f"  Memory delta: {memory_delta:.2f}MB")
            print(f"  Success: {success}")
            if error:
                print(f"  Error: {error}")
        
        return result
    
    return wrapper

def get_memory_usage() -> float:
    """現在のメモリ使用量取得（MB）"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except:
            return 0.0

# ================================================================
# デバッグ支援機能
# ================================================================

def debug_array_info(arr: np.ndarray, name: str = "Array") -> str:
    """配列デバッグ情報生成"""
    info_lines = [
        f"{name} Information:",
        f"  Shape: {arr.shape}",
        f"  Dtype: {arr.dtype}",
        f"  Size: {arr.size}",
        f"  Memory: {arr.nbytes / 1024:.2f} KB"
    ]
    
    if arr.size > 0:
        info_lines.extend([
            f"  Min: {np.min(arr):.6f}",
            f"  Max: {np.max(arr):.6f}",
            f"  Mean: {np.mean(arr):.6f}",
            f"  Std: {np.std(arr):.6f}",
            f"  NaN count: {np.sum(np.isnan(arr))}",
            f"  Inf count: {np.sum(np.isinf(arr))}",
            f"  Zero count: {np.sum(arr == 0)}"
        ])
    
    return "\n".join(info_lines)

def validate_function_inputs(func: Callable) -> Callable:
    """関数入力検証デコレーター"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 関数シグネチャ取得
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # 型ヒントに基づく検証
        for param_name, param_value in bound_args.arguments.items():
            param_info = sig.parameters[param_name]
            
            if param_info.annotation != inspect.Parameter.empty:
                expected_type = param_info.annotation
                
                # NumPy配列の検証
                if expected_type == np.ndarray and not isinstance(param_value, np.ndarray):
                    warnings.warn(f"Parameter '{param_name}' expected numpy.ndarray, got {type(param_value)}")
                
                # 数値型の検証
                elif expected_type in [int, float] and not isinstance(param_value, (int, float)):
                    warnings.warn(f"Parameter '{param_name}' expected {expected_type.__name__}, got {type(param_value)}")
        
        return func(*args, **kwargs)
    
    return wrapper

def create_debug_checkpoint(data: Any, 
                          checkpoint_name: str,
                          save_to_file: bool = False) -> None:
    """デバッグチェックポイント作成"""
    print(f"\n=== DEBUG CHECKPOINT: {checkpoint_name} ===")
    
    if isinstance(data, np.ndarray):
        print(debug_array_info(data, checkpoint_name))
    elif isinstance(data, dict):
        print(f"Dictionary with {len(data)} keys:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print(f"Type: {type(data)}")
        print(f"Value: {data}")
    
    print("=" * (20 + len(checkpoint_name)))
    
    if save_to_file:
        filename = f"debug_{checkpoint_name}_{int(time.time())}.txt"
        with open(filename, 'w') as f:
            f.write(f"Debug checkpoint: {checkpoint_name}\n")
            f.write(f"Timestamp: {time.ctime()}\n")
            f.write(f"Data info: {data}\n")

# ================================================================
# パッケージ初期化
# ================================================================

# ヘルパー関数のエクスポート
__all__ = [
    # 数値計算
    'safe_divide',
    'robust_normalize', 
    'calculate_rolling_statistics',
    
    # データ変換
    'ensure_numpy_array',
    'dict_to_dataframe',
    'standardize_time_series_format',
    
    # 性能測定
    'Timer',
    'performance_monitor',
    'get_memory_usage',
    
    # デバッグ支援
    'debug_array_info',
    'validate_function_inputs',
    'create_debug_checkpoint'
]
