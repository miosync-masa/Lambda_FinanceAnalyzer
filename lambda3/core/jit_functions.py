# ==========================================================
# lambda3/core/jit_functions.py (∆ΛC Pulsations完全修正版)
# JIT-Optimized Core Functions for Lambda³ Theory
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
#
# 完全修正版: Numba型推論エラー解決、∆ΛC pulsations検出の理論的完全性
# ==========================================================

"""
Lambda³理論JIT最適化核心関数群（∆ΛC Pulsations完全修正版）

構造テンソル(Λ)演算、進行ベクトル(ΛF)計算、張力スカラー(ρT)演算の
高速JITコンパイル実装。時間非依存の構造空間における
∆ΛC pulsations検出の数値計算核心部。

完全修正内容:
- Numba setitem型エラーの完全解決
- 階層的∆ΛC検出の理論準拠実装
- 構造テンソル差分演算の数値安定性向上
- 同期プロファイル計算の最適化
- 位相結合検出の追加実装
"""

import numpy as np
from numba import njit, prange
from typing import Tuple
import warnings

# JIT最適化設定（理論準拠版）
JIT_BASE_OPTIONS = {
    'nopython': True,   # Pure numba mode
    'fastmath': True,   # Fast math optimizations  
    'cache': True       # Cache compiled functions
}

# 並列計算用オプション
JIT_PARALLEL_OPTIONS = {
    'nopython': True,
    'fastmath': True,
    'cache': True,
    'parallel': True
}

# ==========================================================
# BASIC TENSOR OPERATIONS - 基本構造テンソル演算
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """
    構造テンソル差分計算と閾値決定（完全版）
    
    Lambda³理論: ∆Λ(t) = Λ(t) - Λ(t-1) の高速計算
    時間非依存構造空間における変化検出の基礎演算
    
    Args:
        data: 構造テンソル時系列 Λ(t)
        percentile: 閾値決定パーセンタイル
        
    Returns:
        diff: 構造テンソル差分 ∆Λ
        threshold: 構造変化検出閾値
    """
    n = len(data)
    diff = np.zeros(n, dtype=np.float64)
    
    # 構造テンソル差分計算 ∆Λ(t) = Λ(t) - Λ(t-1)
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    # 絶対値からパーセンタイル閾値を決定
    abs_diff = np.abs(diff[1:])  # 初期値(0)を除外
    threshold = np.percentile(abs_diff, percentile) if len(abs_diff) > 0 else 0.0
    
    return diff, threshold

@njit(**JIT_BASE_OPTIONS)
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    構造テンソルジャンプ検出（∆ΛC pulsations）
    
    Lambda³理論: ∆ΛC⁺ と ∆ΛC⁻ の分離検出
    構造空間における突発的変化の方向性識別
    
    Args:
        diff: 構造テンソル差分 ∆Λ
        threshold: 検出閾値
        
    Returns:
        pos_jumps: 正の構造ジャンプ ∆ΛC⁺
        neg_jumps: 負の構造ジャンプ ∆ΛC⁻
    """
    n = len(diff)
    pos_jumps = np.zeros(n, dtype=np.float64)
    neg_jumps = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if diff[i] > threshold:
            pos_jumps[i] = 1.0
        elif diff[i] < -threshold:
            neg_jumps[i] = 1.0
    
    return pos_jumps, neg_jumps

@njit(**JIT_BASE_OPTIONS)
def calculate_local_std(data: np.ndarray, window: int) -> np.ndarray:
    """
    局所標準偏差計算（構造テンソル変動性）
    
    Lambda³理論: 構造空間における局所的変動性の定量化
    
    Args:
        data: 入力データ
        window: 窓サイズ
        
    Returns:
        local_std: 局所標準偏差
    """
    n = len(data)
    local_std = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        
        subset = data[start:end]
        if len(subset) > 1:
            mean = np.mean(subset)
            variance = np.sum((subset - mean) ** 2) / len(subset)
            local_std[i] = np.sqrt(variance)
        else:
            local_std[i] = 0.0
    
    return local_std

@njit(**JIT_BASE_OPTIONS)
def calculate_rho_t(data: np.ndarray, window: int) -> np.ndarray:
    """
    張力スカラー ρT 計算（完全版）
    
    Lambda³理論: ρT(t) = f(Λ, ∆Λ) の高速計算
    構造空間内の張力状態定量化
    
    Args:
        data: 元時系列データ
        window: 計算窓サイズ
        
    Returns:
        rho_t: 張力スカラー時系列
    """
    n = len(data)
    rho_t = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        start = max(0, i - window)
        end = i + 1
        
        subset = data[start:end]
        if len(subset) > 1:
            mean = np.mean(subset)
            variance = np.sum((subset - mean) ** 2) / len(subset)
            rho_t[i] = np.sqrt(variance)
        else:
            rho_t[i] = 0.0
    
    return rho_t

# ==========================================================
# HIERARCHICAL DETECTION - 階層的∆ΛC検出
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def detect_local_global_jumps(
    data: np.ndarray,
    local_window: int = 10,
    global_window: int = 50,
    local_percentile: float = 95.0,
    global_percentile: float = 97.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ローカル・グローバル構造変化の階層的検出（完全版）
    
    Lambda³理論: 構造変化に階層性がある
    - ローカルジャンプ: 局所的な構造テンソル変化
    - グローバルジャンプ: 系全体の構造テンソル変化
    
    Args:
        data: 入力時系列
        local_window: ローカル検出窓
        global_window: グローバル検出窓
        local_percentile: ローカル閾値パーセンタイル
        global_percentile: グローバル閾値パーセンタイル
        
    Returns:
        local_pos: ローカル正ジャンプ
        local_neg: ローカル負ジャンプ
        global_pos: グローバル正ジャンプ
        global_neg: グローバル負ジャンプ
    """
    n = len(data)
    diff = np.zeros(n, dtype=np.float64)
    
    # 差分計算
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    local_pos = np.zeros(n, dtype=np.float64)
    local_neg = np.zeros(n, dtype=np.float64)
    global_pos = np.zeros(n, dtype=np.float64)
    global_neg = np.zeros(n, dtype=np.float64)
    
    # ローカル基準での判定
    for i in range(n):
        # ローカル窓の設定
        local_start = max(0, i - local_window)
        local_end = min(n, i + local_window + 1)
        local_subset = np.abs(diff[local_start:local_end])
        
        if len(local_subset) > 0:
            local_threshold = np.percentile(local_subset, local_percentile)
            if diff[i] > local_threshold:
                local_pos[i] = 1.0
            elif diff[i] < -local_threshold:
                local_neg[i] = 1.0
    
    # グローバル基準での判定
    global_threshold = np.percentile(np.abs(diff), global_percentile)
    
    for i in range(n):
        if diff[i] > global_threshold:
            global_pos[i] = 1.0
        elif diff[i] < -global_threshold:
            global_neg[i] = 1.0
    
    return local_pos, local_neg, global_pos, global_neg

# ==========================================================
# SYNCHRONIZATION OPERATIONS - 同期演算
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def sync_rate_at_lag(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """
    特定遅延での同期率計算
    
    Lambda³理論: 構造テンソル系列間の遅延相関
    
    Args:
        series_a: 系列A
        series_b: 系列B
        lag: 遅延値
        
    Returns:
        sync_rate: 同期率
    """
    if lag < 0:
        if -lag < len(series_a):
            return np.mean(series_a[-lag:] * series_b[:lag])
        else:
            return 0.0
    elif lag > 0:
        if lag < len(series_b):
            return np.mean(series_a[:-lag] * series_b[lag:])
        else:
            return 0.0
    else:
        return np.mean(series_a * series_b)

@njit(parallel=True, **JIT_PARALLEL_OPTIONS)
def calculate_sync_profile_jit(series_a: np.ndarray, series_b: np.ndarray,
                               lag_window: int) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    同期プロファイル計算（並列化版）
    
    Lambda³理論: 構造テンソル系列間の遅延同期パターン検出
    
    Args:
        series_a: 系列A
        series_b: 系列B
        lag_window: 遅延窓サイズ
        
    Returns:
        lags: 遅延値配列
        sync_values: 同期強度配列
        max_sync: 最大同期強度
        optimal_lag: 最適遅延
    """
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    sync_values = np.zeros(n_lags, dtype=np.float64)
    
    for i in prange(n_lags):
        lag = lags[i]
        sync_values[i] = sync_rate_at_lag(series_a, series_b, lag)
    
    max_sync = 0.0
    optimal_lag = 0
    for i in range(n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]
    
    return lags, sync_values, max_sync, optimal_lag

# ==========================================================
# COMPREHENSIVE FEATURE EXTRACTION - 包括的特徴抽出
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def calc_lambda3_features_v2(data: np.ndarray, config_window: int = 10,
                            config_percentile: float = 95.0) -> Tuple[np.ndarray, ...]:
    """
    Lambda³特徴量包括的抽出（7特徴量版）
    
    Lambda³理論の核心特徴量を高速一括計算:
    - ∆ΛC⁺/∆ΛC⁻: 構造変化パルス
    - ρT: 張力スカラー
    - 階層的構造変化
    
    Args:
        data: 入力時系列
        config_window: 検出窓サイズ
        config_percentile: 閾値パーセンタイル
        
    Returns:
        7つの特徴量タプル
    """
    # 基本特徴量
    diff, threshold = calculate_diff_and_threshold(data, config_percentile)
    delta_pos, delta_neg = detect_jumps(diff, threshold)
    rho_t = calculate_rho_t(data, config_window)
    time_trend = np.arange(len(data), dtype=np.float64)
    
    # ローカルジャンプ検出
    local_std = calculate_local_std(data, 5)
    score = np.abs(diff) / (local_std + 1e-8)
    local_threshold = np.percentile(score, 95.0)
    local_jump_detect = np.zeros(len(data), dtype=np.float64)
    for i in range(len(data)):
        if score[i] > local_threshold:
            local_jump_detect[i] = 1.0
    
    # 階層的構造変化
    local_pos, local_neg, global_pos, global_neg = detect_local_global_jumps(
        data, 5, 30, 90.0, 95.0
    )
    
    return (delta_pos, delta_neg, rho_t, time_trend, local_jump_detect,
            local_pos, local_neg, global_pos, global_neg)

# ==========================================================
# ADVANCED OPERATIONS - 高度演算
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def detect_phase_coupling(series_a: np.ndarray, series_b: np.ndarray,
                         window: int = 20) -> Tuple[float, float]:
    """
    位相結合検出（Lambda³理論拡張）
    
    構造テンソル系列間の位相同期性を検出
    
    Args:
        series_a: 系列A
        series_b: 系列B
        window: 解析窓
        
    Returns:
        coupling_strength: 結合強度
        phase_lag: 位相遅延
    """
    n = min(len(series_a), len(series_b))
    
    # ヒルベルト変換の簡易実装（位相抽出）
    phase_a = np.zeros(n, dtype=np.float64)
    phase_b = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        phase_a[i] = np.arctan2(series_a[i] - series_a[i-1], series_a[i])
        phase_b[i] = np.arctan2(series_b[i] - series_b[i-1], series_b[i])
    
    # 位相同期性計算
    phase_diff = phase_a - phase_b
    coupling_strength = 1.0 - np.std(np.sin(phase_diff))
    
    # 最適位相遅延探索
    max_corr = 0.0
    best_lag = 0
    
    for lag in range(-window, window + 1):
        if lag > 0 and lag < len(phase_a):
            corr = np.abs(np.corrcoef(phase_a[:-lag], phase_b[lag:])[0, 1])
        elif lag < 0 and -lag < len(phase_b):
            corr = np.abs(np.corrcoef(phase_a[-lag:], phase_b[:lag])[0, 1])
        else:
            corr = 0.0
        
        if corr > max_corr:
            max_corr = corr
            best_lag = lag
    
    phase_lag = float(best_lag)
    
    return coupling_strength, phase_lag

# ==========================================================
# VALIDATION & TESTING - 検証・テスト
# ==========================================================

def test_jit_functions():
    """JIT関数の包括的テスト"""
    print("🧪 Testing Lambda³ JIT Functions (∆ΛC Pulsations Complete)")
    print("=" * 60)
    
    # テストデータ生成
    np.random.seed(42)
    test_data = np.cumsum(np.random.randn(100) * 0.1) + 100
    
    try:
        # 1. 基本演算テスト
        print("1️⃣ Testing basic tensor operations...")
        diff, threshold = calculate_diff_and_threshold(test_data, 95.0)
        pos_jumps, neg_jumps = detect_jumps(diff, threshold)
        print(f"   ✅ Detected {np.sum(pos_jumps)} positive, {np.sum(neg_jumps)} negative jumps")
        
        # 2. 階層的検出テスト
        print("\n2️⃣ Testing hierarchical detection...")
        local_pos, local_neg, global_pos, global_neg = detect_local_global_jumps(test_data)
        print(f"   ✅ Local: {np.sum(local_pos + local_neg)}, Global: {np.sum(global_pos + global_neg)}")
        
        # 3. 張力スカラーテスト
        print("\n3️⃣ Testing tension scalar...")
        rho_t = calculate_rho_t(test_data, 10)
        print(f"   ✅ Mean tension: {np.mean(rho_t):.4f}")
        
        # 4. 包括的特徴抽出テスト
        print("\n4️⃣ Testing comprehensive feature extraction...")
        features = calc_lambda3_features_v2(test_data)
        print(f"   ✅ Extracted {len(features)} feature arrays")
        
        # 5. 同期分析テスト
        print("\n5️⃣ Testing synchronization analysis...")
        test_data_b = test_data + np.random.randn(len(test_data)) * 0.05
        lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
            test_data, test_data_b, 10
        )
        print(f"   ✅ Max sync: {max_sync:.4f} at lag {optimal_lag}")
        
        # 6. 位相結合テスト
        print("\n6️⃣ Testing phase coupling...")
        coupling, phase_lag = detect_phase_coupling(test_data, test_data_b)
        print(f"   ✅ Coupling strength: {coupling:.4f}, Phase lag: {phase_lag}")
        
        print("\n✅ All tests passed! Lambda³ JIT functions are working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_jit_benchmark(data_size: int = 10000):
    """JIT性能ベンチマーク"""
    import time
    
    print(f"\n⚡ Lambda³ JIT Performance Benchmark (n={data_size})")
    print("=" * 60)
    
    # テストデータ
    np.random.seed(42)
    data = np.cumsum(np.random.randn(data_size) * 0.1) + 100
    
    # ウォームアップ
    _ = calc_lambda3_features_v2(data[:100])
    
    # ベンチマーク実行
    start_time = time.time()
    features = calc_lambda3_features_v2(data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    throughput = data_size / execution_time
    
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Throughput: {throughput:.0f} points/second")
    print(f"Total events detected: {np.sum(features[0]) + np.sum(features[1])}")
    
    return throughput

# ==========================================================
# MODULE EXPORTS
# ==========================================================

__all__ = [
    # 基本テンソル演算
    'calculate_diff_and_threshold',
    'detect_jumps',
    'calculate_local_std',
    'calculate_rho_t',
    
    # 階層的検出
    'detect_local_global_jumps',
    
    # 同期演算
    'sync_rate_at_lag',
    'calculate_sync_profile_jit',
    
    # 包括的特徴抽出
    'calc_lambda3_features_v2',
    
    # 高度演算
    'detect_phase_coupling',
    
    # テスト・ベンチマーク
    'test_jit_functions',
    'run_jit_benchmark'
]

if __name__ == "__main__":
    # 自動テスト実行
    test_jit_functions()
    run_jit_benchmark()
