# ==========================================================
# lambda3/core/jit_functions.py
# JIT-Optimized Core Functions for Lambda³ Theory (修正版)
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論JIT最適化核心関数群（統一修正版）

構造テンソル(Λ)演算、進行ベクトル(ΛF)計算、張力スカラー(ρT)演算の
高速JITコンパイル実装。時間非依存の構造空間における
∆ΛC pulsations検出の数値計算核心部。

修正内容:
- 冗長なエイリアス関数の削除
- float64型の徹底使用
- 理論準拠のパラメータ初期値
"""

import numpy as np
from numba import njit, prange
from typing import Tuple
import warnings

# JIT最適化設定（理論準拠版）
JIT_BASE_OPTIONS = {
    'nopython': True,
    'fastmath': True,
    'cache': True
}

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
def calculate_diff_and_threshold(data: np.ndarray, percentile: float = 97.0) -> Tuple[np.ndarray, float]:
    """
    構造テンソル差分計算と閾値決定
    
    Lambda³理論: ∆Λ(t) = Λ(t) - Λ(t-1) の高速計算
    """
    n = len(data)
    diff = np.zeros(n, dtype=np.float64)  # 明示的にfloat64
    
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    abs_diff = np.abs(diff[1:])  # 初期値を除外
    threshold = np.percentile(abs_diff, percentile) if len(abs_diff) > 0 else 0.0
    
    return diff, threshold

@njit(**JIT_BASE_OPTIONS)
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    構造テンソルジャンプ検出（∆ΛC pulsations）
    """
    n = len(diff)
    pos_jumps = np.zeros(n, dtype=np.float64)  # int32からfloat64に変更
    neg_jumps = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if diff[i] > threshold:
            pos_jumps[i] = 1.0
        elif diff[i] < -threshold:
            neg_jumps[i] = 1.0
    
    return pos_jumps, neg_jumps

@njit(**JIT_BASE_OPTIONS)
def calculate_rho_t(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    張力スカラー ρT 計算
    
    Lambda³理論: ρT(t) = f(Λ, ∆Λ) の高速計算
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
    
    return rho_t

@njit(**JIT_BASE_OPTIONS)
def calculate_local_std(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    局所標準偏差計算（構造テンソル変動性）
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
    
    return local_std

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
    ローカル・グローバル構造変化の階層的検出
    
    Lambda³理論: 構造変化には階層性がある
    - ローカル: 短期的な構造テンソル変化（高頻度∆ΛC）
    - グローバル: 長期的な構造テンソル変化（低頻度∆ΛC）
    
    Returns:
        local_pos, local_neg, global_pos, global_neg (全てfloat64)
    """
    n = len(data)
    diff = np.zeros(n, dtype=np.float64)
    diff[0] = 0
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    # 初期化（float64保証）
    local_pos = np.zeros(n, dtype=np.float64)
    local_neg = np.zeros(n, dtype=np.float64)
    global_pos = np.zeros(n, dtype=np.float64)
    global_neg = np.zeros(n, dtype=np.float64)
    
    # ローカル基準での判定（適応的閾値）
    for i in range(n):
        local_start = max(0, i - local_window)
        local_end = min(n, i + local_window + 1)
        local_subset = np.abs(diff[local_start:local_end])
        
        if len(local_subset) > 0:
            local_threshold = np.percentile(local_subset, local_percentile)
            if diff[i] > local_threshold:
                local_pos[i] = 1.0
            elif diff[i] < -local_threshold:
                local_neg[i] = 1.0
    
    # グローバル基準での判定（固定閾値）
    global_threshold = np.percentile(np.abs(diff), global_percentile)
    
    for i in range(n):
        if diff[i] > global_threshold:
            global_pos[i] = 1.0
        elif diff[i] < -global_threshold:
            global_neg[i] = 1.0
    
    return local_pos, local_neg, global_pos, global_neg

# ==========================================================
# SYNCHRONIZATION - 同期演算
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def sync_rate_at_lag(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """
    特定ラグでの同期率計算
    """
    series_a = series_a.astype(np.float64)
    series_b = series_b.astype(np.float64)
    
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
def calculate_sync_profile_jit(
    series_a: np.ndarray, 
    series_b: np.ndarray,
    lag_window: int = 10
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    同期プロファイル計算（並列化）
    """
    series_a = series_a.astype(np.float64)
    series_b = series_b.astype(np.float64)
    
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1, dtype=np.int32)
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
    
    return lags.astype(np.float64), sync_values, max_sync, optimal_lag

# ==========================================================
# COMPREHENSIVE FEATURE EXTRACTION - 包括的特徴抽出
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def calc_lambda3_features_v2(
    data: np.ndarray,
    window: int = 10,
    percentile: float = 97.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lambda³特徴量抽出（基本版）
    """
    data = data.astype(np.float64)
    
    # 基本特徴量
    diff, threshold = calculate_diff_and_threshold(data, percentile)
    delta_pos, delta_neg = detect_jumps(diff, threshold)
    rho_t = calculate_rho_t(data, window)
    time_trend = np.arange(len(data), dtype=np.float64)
    
    # ローカルジャンプ検出
    local_std = calculate_local_std(data, window // 2)
    score = np.abs(diff) / (local_std + 1e-8)
    local_threshold = np.percentile(score, 95.0)
    local_jump_detect = (score > local_threshold).astype(np.float64)
    
    return delta_pos, delta_neg, rho_t, time_trend, local_jump_detect

# ==========================================================
# PHASE COUPLING - 位相結合検出
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def detect_phase_coupling(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int = 20
) -> Tuple[float, float]:
    """
    位相結合検出（Lambda³理論拡張）
    """
    series_a = series_a.astype(np.float64)
    series_b = series_b.astype(np.float64)
    
    n = min(len(series_a), len(series_b))
    phase_diff = np.zeros(n - window, dtype=np.float64)
    
    for i in range(window, n):
        # 簡易ヒルベルト変換近似
        a_segment = series_a[i-window:i]
        b_segment = series_b[i-window:i]
        
        # 位相推定（簡易版）
        a_phase = np.arctan2(np.mean(a_segment[1:] - a_segment[:-1]), 
                            np.mean(a_segment))
        b_phase = np.arctan2(np.mean(b_segment[1:] - b_segment[:-1]), 
                            np.mean(b_segment))
        
        phase_diff[i-window] = a_phase - b_phase
    
    # 位相結合強度
    coupling_strength = 1.0 - np.std(phase_diff) / np.pi
    mean_phase_lag = np.mean(phase_diff)
    
    return max(0.0, coupling_strength), mean_phase_lag

# ==========================================================
# TESTING & VALIDATION - テスト・検証
# ==========================================================

def test_jit_functions():
    """JIT関数の包括的テスト"""
    print("🧪 Testing Lambda³ JIT Functions")
    print("=" * 60)
    
    # テストデータ生成
    np.random.seed(42)
    test_data = np.cumsum(np.random.randn(100) * 0.1) + 100
    
    try:
        # 1. 基本演算テスト
        print("1️⃣ Testing basic tensor operations...")
        diff, threshold = calculate_diff_and_threshold(test_data)
        pos_jumps, neg_jumps = detect_jumps(diff, threshold)
        print(f"   ✅ Detected {np.sum(pos_jumps):.0f} positive, {np.sum(neg_jumps):.0f} negative jumps")
        
        # 2. 階層的検出テスト
        print("\n2️⃣ Testing hierarchical detection...")
        local_pos, local_neg, global_pos, global_neg = detect_local_global_jumps(
            test_data,
            local_window=5,
            global_window=20,
            local_percentile=85.0,
            global_percentile=92.5
        )
        print(f"   ✅ Local: {np.sum(local_pos + local_neg):.0f} events")
        print(f"   ✅ Global: {np.sum(global_pos + global_neg):.0f} events")
        print(f"   ✅ Local positive: {np.sum(local_pos):.0f}, negative: {np.sum(local_neg):.0f}")
        print(f"   ✅ Global positive: {np.sum(global_pos):.0f}, negative: {np.sum(global_neg):.0f}")
        
        # 3. 張力スカラーテスト
        print("\n3️⃣ Testing tension scalar...")
        rho_t = calculate_rho_t(test_data)
        print(f"   ✅ Mean tension: {np.mean(rho_t):.4f}")
        
        # 4. 同期演算テスト
        print("\n4️⃣ Testing synchronization...")
        test_data_b = np.cumsum(np.random.randn(100) * 0.1) + 100
        lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(test_data, test_data_b)
        print(f"   ✅ Max sync: {max_sync:.4f} at lag {optimal_lag}")
        
        print("\n✅ All tests passed!")
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
    print(f"Total events detected: {np.sum(features[0]) + np.sum(features[1]):.0f}")
    
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
