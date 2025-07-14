# ==========================================================
# Lambda³ Data Loading and Preprocessing Utilities
# ==========================================================

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import warnings
import pickle
import json

@dataclass
class SeriesMetadata:
    """Metadata for a time series"""
    name: str
    length: int
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    frequency: Optional[str] = None
    missing_count: int = 0
    data_type: str = "float64"
    source: Optional[str] = None

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_series: int
    total_points: int
    missing_ratio: float
    length_consistency: bool
    date_alignment: bool
    issues: List[str]
    recommendations: List[str]

# =========================
# DATA LOADING FUNCTIONS
# =========================

def load_csv_data(
    filepath: Union[str, Path],
    time_column: Optional[str] = None,
    value_columns: Optional[List[str]] = None,
    parse_dates: bool = True,
    index_col: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load time series data from CSV file
    
    Parameters:
    -----------
    filepath : str or Path
        Path to CSV file
    time_column : str, optional
        Name of time/date column
    value_columns : list, optional
        List of columns to load as series
    parse_dates : bool
        Whether to parse date columns
    index_col : str, optional
        Column to use as index
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary of series name to numpy array
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data
    df = pd.read_csv(
        filepath,
        parse_dates=parse_dates,
        index_col=index_col
    )
    
    print(f"Loaded CSV: {filepath.name}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
    
    # Sort by time if specified
    if time_column and time_column in df.columns:
        df = df.sort_values(by=time_column)
    elif df.index.name and pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.sort_index()
    
    # Select value columns
    if value_columns is None:
        # Auto-detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_column and time_column in numeric_cols:
            numeric_cols.remove(time_column)
        value_columns = numeric_cols
    
    # Convert to dictionary
    series_dict = {}
    for col in value_columns:
        if col in df.columns:
            data = df[col].values.astype(np.float64)
            # Handle missing values
            if pd.isna(data).any():
                data = handle_missing_data(data)
            series_dict[col] = data
        else:
            warnings.warn(f"Column '{col}' not found in CSV")
    
    return series_dict

def load_parquet_data(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Load time series data from Parquet file"""
    filepath = Path(filepath)
    
    df = pd.read_parquet(filepath, columns=columns)
    
    # Convert to series dict
    series_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series_dict[col] = df[col].values.astype(np.float64)
    
    return series_dict

def load_excel_data(
    filepath: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Load time series data from Excel file"""
    filepath = Path(filepath)
    
    df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    # Convert to series dict
    series_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series_dict[col] = df[col].values.astype(np.float64)
    
    return series_dict

def load_multiple_files(
    file_pattern: str,
    loader_func: callable = load_csv_data,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Load multiple files matching a pattern
    
    Parameters:
    -----------
    file_pattern : str
        Glob pattern for files (e.g., "data/*.csv")
    loader_func : callable
        Function to load individual files
    **kwargs
        Additional arguments for loader function
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Combined dictionary of all series
    """
    from glob import glob
    
    all_series = {}
    files = sorted(glob(file_pattern))
    
    print(f"Found {len(files)} files matching '{file_pattern}'")
    
    for filepath in files:
        try:
            series_dict = loader_func(filepath, **kwargs)
            # Add filename prefix to avoid collisions
            file_prefix = Path(filepath).stem
            for name, data in series_dict.items():
                key = f"{file_prefix}_{name}"
                all_series[key] = data
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
    
    return all_series

# =========================
# DATA VALIDATION
# =========================

def validate_series_data(
    series_dict: Dict[str, np.ndarray],
    min_length: int = 10,
    max_missing_ratio: float = 0.5
) -> DataQualityReport:
    """
    Validate time series data quality
    
    Parameters:
    -----------
    series_dict : dict
        Dictionary of series
    min_length : int
        Minimum acceptable series length
    max_missing_ratio : float
        Maximum acceptable missing data ratio
        
    Returns:
    --------
    DataQualityReport
        Detailed quality assessment
    """
    issues = []
    recommendations = []
    
    # Check series count
    n_series = len(series_dict)
    if n_series == 0:
        issues.append("No series found")
        recommendations.append("Check data loading parameters")
    
    # Check lengths
    lengths = [len(data) for data in series_dict.values()]
    length_consistent = len(set(lengths)) == 1
    
    if not length_consistent:
        issues.append(f"Inconsistent series lengths: {min(lengths)} to {max(lengths)}")
        recommendations.append("Consider aligning series with align_series()")
    
    # Check for short series
    short_series = [name for name, data in series_dict.items() if len(data) < min_length]
    if short_series:
        issues.append(f"{len(short_series)} series shorter than {min_length} points")
        recommendations.append(f"Remove short series: {short_series[:3]}")
    
    # Check missing data
    total_points = sum(lengths)
    missing_counts = []
    
    for name, data in series_dict.items():
        n_missing = np.isnan(data).sum()
        missing_counts.append(n_missing)
        
        missing_ratio = n_missing / len(data)
        if missing_ratio > max_missing_ratio:
            issues.append(f"{name}: {missing_ratio:.1%} missing")
    
    total_missing = sum(missing_counts)
    overall_missing_ratio = total_missing / total_points if total_points > 0 else 0
    
    if overall_missing_ratio > 0.1:
        recommendations.append("Consider imputation with handle_missing_data()")
    
    return DataQualityReport(
        total_series=n_series,
        total_points=total_points,
        missing_ratio=overall_missing_ratio,
        length_consistency=length_consistent,
        date_alignment=True,  # Simplified
        issues=issues,
        recommendations=recommendations
    )

def check_data_quality(
    series_dict: Dict[str, np.ndarray],
    verbose: bool = True
) -> bool:
    """Quick data quality check with printed report"""
    report = validate_series_data(series_dict)
    
    if verbose:
        print("Data Quality Report")
        print("="*50)
        print(f"Series count: {report.total_series}")
        print(f"Total points: {report.total_points:,}")
        print(f"Missing ratio: {report.missing_ratio:.1%}")
        print(f"Length consistency: {'✓' if report.length_consistency else '✗'}")
        
        if report.issues:
            print("\nIssues:")
            for issue in report.issues:
                print(f"  - {issue}")
        
        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")
    
    return len(report.issues) == 0

# =========================
# DATA PREPROCESSING
# =========================

def handle_missing_data(
    data: np.ndarray,
    method: str = 'interpolate',
    limit: Optional[int] = None
) -> np.ndarray:
    """
    Handle missing values in time series
    
    Parameters:
    -----------
    data : np.ndarray
        Time series with potential NaN values
    method : str
        'interpolate', 'forward_fill', 'backward_fill', 'mean', 'zero'
    limit : int, optional
        Maximum number of consecutive NaNs to fill
        
    Returns:
    --------
    np.ndarray
        Series with missing values handled
    """
    if not np.isnan(data).any():
        return data
    
    series = pd.Series(data)
    
    if method == 'interpolate':
        series = series.interpolate(method='linear', limit=limit)
    elif method == 'forward_fill':
        series = series.fillna(method='ffill', limit=limit)
    elif method == 'backward_fill':
        series = series.fillna(method='bfill', limit=limit)
    elif method == 'mean':
        series = series.fillna(series.mean())
    elif method == 'zero':
        series = series.fillna(0)
    
    # Handle any remaining NaNs
    series = series.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return series.values

def align_series(
    series_dict: Dict[str, np.ndarray],
    method: str = 'truncate'
) -> Dict[str, np.ndarray]:
    """
    Align series to same length
    
    Parameters:
    -----------
    series_dict : dict
        Dictionary of series
    method : str
        'truncate' (shortest), 'pad' (longest), 'interpolate'
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Aligned series
    """
    lengths = [len(data) for data in series_dict.values()]
    
    if method == 'truncate':
        target_length = min(lengths)
        aligned = {
            name: data[:target_length] 
            for name, data in series_dict.items()
        }
    
    elif method == 'pad':
        target_length = max(lengths)
        aligned = {}
        for name, data in series_dict.items():
            if len(data) < target_length:
                # Pad with last value
                pad_length = target_length - len(data)
                padded = np.pad(data, (0, pad_length), mode='edge')
                aligned[name] = padded
            else:
                aligned[name] = data
    
    elif method == 'interpolate':
        target_length = int(np.median(lengths))
        aligned = {}
        for name, data in series_dict.items():
            if len(data) != target_length:
                # Resample to target length
                old_x = np.linspace(0, 1, len(data))
                new_x = np.linspace(0, 1, target_length)
                resampled = np.interp(new_x, old_x, data)
                aligned[name] = resampled
            else:
                aligned[name] = data
    
    return aligned

def resample_series(
    data: np.ndarray,
    source_freq: str,
    target_freq: str,
    method: str = 'mean'
) -> np.ndarray:
    """Resample time series to different frequency"""
    # Create pandas series with frequency
    dates = pd.date_range(start='2020-01-01', periods=len(data), freq=source_freq)
    series = pd.Series(data, index=dates)
    
    # Resample
    resampled = series.resample(target_freq)
    
    if method == 'mean':
        result = resampled.mean()
    elif method == 'sum':
        result = resampled.sum()
    elif method == 'last':
        result = resampled.last()
    elif method == 'first':
        result = resampled.first()
    
    return result.values

def preprocess_series(
    series_dict: Dict[str, np.ndarray],
    align: bool = True,
    handle_missing: bool = True,
    normalize: bool = False,
    detrend: bool = False
) -> Dict[str, np.ndarray]:
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    series_dict : dict
        Raw series data
    align : bool
        Whether to align series lengths
    handle_missing : bool
        Whether to handle missing values
    normalize : bool
        Whether to normalize series
    detrend : bool
        Whether to remove linear trend
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Preprocessed series
    """
    processed = series_dict.copy()
    
    # Handle missing values
    if handle_missing:
        processed = {
            name: handle_missing_data(data)
            for name, data in processed.items()
        }
    
    # Align series
    if align:
        processed = align_series(processed)
    
    # Detrend
    if detrend:
        from scipy import signal
        processed = {
            name: signal.detrend(data)
            for name, data in processed.items()
        }
    
    # Normalize
    if normalize:
        processed = {
            name: (data - np.mean(data)) / (np.std(data) + 1e-8)
            for name, data in processed.items()
        }
    
    return processed

# =========================
# DATA GENERATION
# =========================

def generate_synthetic_data(
    n_series: int = 5,
    n_points: int = 500,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Generate synthetic time series for testing"""
    if seed is not None:
        np.random.seed(seed)
    
    data = {}
    
    for i in range(n_series):
        # Different types of series
        if i % 5 == 0:
            # Trend + noise
            series = np.linspace(0, 10, n_points) + np.random.randn(n_points)
        elif i % 5 == 1:
            # Oscillatory
            t = np.linspace(0, 10, n_points)
            series = np.sin(2 * np.pi * t) + 0.5 * np.random.randn(n_points)
        elif i % 5 == 2:
            # Random walk
            series = np.cumsum(np.random.randn(n_points) * 0.1)
        elif i % 5 == 3:
            # Step changes
            series = np.zeros(n_points)
            for step in range(0, n_points, n_points // 5):
                series[step:] += np.random.randn()
        else:
            # AR(1) process
            series = np.zeros(n_points)
            phi = 0.8
            for t in range(1, n_points):
                series[t] = phi * series[t-1] + np.random.randn()
        
        data[f"Series_{i+1:02d}"] = series
    
    return data

def generate_structural_jumps(
    data: np.ndarray,
    n_jumps: int = 5,
    jump_size_range: Tuple[float, float] = (1.0, 3.0)
) -> np.ndarray:
    """Add structural jumps to time series"""
    data_with_jumps = data.copy()
    
    # Random jump times
    jump_times = np.random.choice(len(data), n_jumps, replace=False)
    jump_times.sort()
    
    for t in jump_times:
        jump_size = np.random.uniform(*jump_size_range)
        jump_sign = np.random.choice([-1, 1])
        data_with_jumps[t:] += jump_sign * jump_size
    
    return data_with_jumps

def generate_regime_switching_data(
    n_points: int = 600,
    regimes: List[Dict[str, float]] = None,
    transition_prob: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate regime-switching time series
    
    Returns:
    --------
    data : np.ndarray
        Time series data
    regime_labels : np.ndarray
        Regime labels for each point
    """
    if regimes is None:
        regimes = [
            {'mean': 0.1, 'std': 0.5},   # Bull
            {'mean': 0.0, 'std': 0.3},   # Neutral
            {'mean': -0.1, 'std': 0.7}   # Bear
        ]
    
    n_regimes = len(regimes)
    regime_labels = np.zeros(n_points, dtype=int)
    data = np.zeros(n_points)
    
    # Start in random regime
    current_regime = np.random.randint(n_regimes)
    
    for t in range(n_points):
        # Generate data point
        regime = regimes[current_regime]
        data[t] = np.random.normal(regime['mean'], regime['std'])
        regime_labels[t] = current_regime
        
        # Possibly switch regime
        if np.random.rand() < transition_prob:
            current_regime = np.random.randint(n_regimes)
    
    # Convert to cumulative for price-like series
    data = np.cumsum(data)
    
    return data, regime_labels

# =========================
# DATA EXPORT
# =========================

def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'pickle'
) -> None:
    """Save Lambda³ analysis results"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    elif format == 'json':
        # Convert numpy arrays to lists for JSON
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(v) for v in obj]
            return obj
        
        json_results = convert_arrays(results)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {filepath}")

def export_to_csv(
    series_dict: Dict[str, np.ndarray],
    filepath: Union[str, Path],
    include_index: bool = True
) -> None:
    """Export time series to CSV"""
    df = pd.DataFrame(series_dict)
    
    if include_index:
        df.index.name = 'index'
    
    df.to_csv(filepath, index=include_index)
    print(f"Exported {len(series_dict)} series to: {filepath}")

def export_to_parquet(
    series_dict: Dict[str, np.ndarray],
    filepath: Union[str, Path]
) -> None:
    """Export time series to Parquet format"""
    df = pd.DataFrame(series_dict)
    df.to_parquet(filepath, engine='pyarrow')
    print(f"Exported {len(series_dict)} series to: {filepath}")
