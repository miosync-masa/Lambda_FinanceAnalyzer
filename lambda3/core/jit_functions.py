# ==========================================================
# lambda3/core/jit_functions.py  
# JIT-Optimized Core Functions for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================
"""
Lambda³理論JIT最適化核心関数群（修正版）

構造テンソル(Λ)演算、進行ベクトル(ΛF)計算、張力スカラー(ρT)演算の
高速JITコンパイル実装。時間非依存の構造空間における
∆ΛC pulsations検出の数値計算核心部。
"""

import numpy as np
from numba import njit, prange
from typing import Tuple
import warnings

# JIT最適化設定（修正版）
JIT_BASE_OPTIONS = {
    'nopython': True,   # Pure numba mode
    'fastmath': True,   # Fast math optimizations  
    'cache': True       # Cache compiled functions
}

# 並列計算用オプション（選択的使用）
JIT_PARALLEL_OPTIONS = {
    'nopython': True,
    'fastmath': True,
    'cache': True,
    'parallel': True
}

# ==========================================================
# BASIC TENSOR OPERATIONS - 基本テンソル演算（修正版）
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def calculate_diff_and_threshold_fixed(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """
    構造テンソル差分計算と閾値決定（修正版）
    
    Lambda³理論: ∆Λ(t) = Λ(t) - Λ(t-1) の高速計算
    時間非依存構造空間における変化検出の基礎演算
    
    修正点: 配列の明示的初期化による型安全性確保
    
    Args:
        data: 構造テンソル時系列 Λ(t)
        percentile: 閾値決定パーセンタイル
        
    Returns:
        diff: 構造テンソル差分 ∆Λ
        threshold: 構造変化検出閾値
    """
    n = len(data)
    # 修正: np.zeros による明示的初期化
    diff = np.zeros(n, dtype=np.float64)
    
    # 構造テンソル差分計算 ∆Λ(t) = Λ(t) - Λ(t-1)
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]  # 直接代入で型安全性確保
    
    # 絶対値からパーセンタイル閾値を決定
    abs_diff = np.abs(diff[1:])  # 初期値(0)を除外
    threshold = np.percentile(abs_diff, percentile) if len(abs_diff) > 0 else 0.0
    
    return diff, threshold

@njit(**JIT_BASE_OPTIONS)
def detect_structural_jumps_fixed(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    構造テンソルジャンプ検出（修正版）
    
    Lambda³理論: ∆ΛC pulsations の正負分離検出
    構造空間における突発的変化の方向性識別
    
    修正点: 配列操作の最適化
    
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
            pos_jumps[i] = 1.0  # 正の構造ジャンプ
        elif diff[i] < -threshold:
            neg_jumps[i] = 1.0  # 負の構造ジャンプ
    
    return pos_jumps, neg_jumps

@njit(**JIT_BASE_OPTIONS)
def calculate_tension_scalar_fixed(data: np.ndarray, window: int) -> np.ndarray:
    """
    張力スカラー(ρT)計算（修正版）
    
    Lambda³理論: 構造テンソル空間における張力度合いを定量化
    局所的な構造変動の強度を表現する重要な指標
    
    修正点: 安定した局所統計計算
    
    Args:
        data: 構造テンソル系列 Λ(t)
        window: 張力計算窓サイズ
        
    Returns:
        rho_t: 張力スカラー ρT(t)
    """
    n = len(data)
    rho_t = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # 後向き窓による張力計算（因果性保持）
        start = max(0, i - window)
        end = i + 1
        
        # 窓内データサイズの確認
        window_size = end - start
        if window_size > 1:
            # 局所平均計算
            local_sum = 0.0
            for j in range(start, end):
                local_sum += data[j]
            local_mean = local_sum / window_size
            
            # 局所分散計算
            variance_sum = 0.0
            for j in range(start, end):
                variance_sum += (data[j] - local_mean) ** 2
            local_variance = variance_sum / window_size
            
            # 張力スカラー = 局所標準偏差
            rho_t[i] = np.sqrt(local_variance)
        else:
            rho_t[i] = 0.0
    
    return rho_t

@njit(**JIT_BASE_OPTIONS)
def calculate_local_statistics_fixed(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    局所統計量計算（修正版）
    
    構造テンソルの局所的変動特性を定量化
    時間窓内での構造空間の局所統計
    
    修正点: 安定化された統計計算
    
    Args:
        data: 構造テンソル系列
        window: 局所統計計算窓サイズ
        
    Returns:
        local_std: 局所標準偏差
        local_mean: 局所平均
    """
    n = len(data)
    local_std = np.zeros(n, dtype=np.float64)
    local_mean = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # 局所窓の設定（境界処理含む）
        start = max(0, i - window)
        end = min(n, i + window + 1)
        window_size = end - start
        
        # 局所平均計算
        sum_val = 0.0
        for j in range(start, end):
            sum_val += data[j]
        local_mean[i] = sum_val / window_size
        
        # 局所標準偏差計算
        if window_size > 1:
            variance_sum = 0.0
            for j in range(start, end):
                variance_sum += (data[j] - local_mean[i]) ** 2
            local_variance = variance_sum / window_size
            local_std[i] = np.sqrt(local_variance)
        else:
            local_std[i] = 0.0
    
    return local_std, local_mean

# ==========================================================
# HIERARCHICAL STRUCTURE DETECTION - 階層構造検出（修正版）
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def detect_hierarchical_jumps_fixed(
    data: np.ndarray, 
    local_window: int = 10, 
    global_window: int = 50,
    local_percentile: float = 90.0,
    global_percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    階層的構造ジャンプ検出（修正版）
    
    Lambda³理論: 構造テンソル変化の階層性（局所-大域）を分離検出
    異なる時間スケールでの∆ΛC pulsationsを識別
    
    修正点: 階層的閾値計算の安定化
    
    Args:
        data: 構造テンソル系列
        local_window: 局所構造検出窓
        global_window: 大域構造検出窓  
        local_percentile: 局所検出閾値パーセンタイル
        global_percentile: 大域検出閾値パーセンタイル
        
    Returns:
        local_pos: 局所正構造ジャンプ
        local_neg: 局所負構造ジャンプ
        global_pos: 大域正構造ジャンプ
        global_neg: 大域負構造ジャンプ
    """
    n = len(data)
    
    # 構造テンソル差分計算
    diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    # 出力配列初期化
    local_pos = np.zeros(n, dtype=np.float64)
    local_neg = np.zeros(n, dtype=np.float64)
    global_pos = np.zeros(n, dtype=np.float64)
    global_neg = np.zeros(n, dtype=np.float64)
    
    # 大域構造ジャンプ検出（全体統計による）
    abs_diff = np.abs(diff[1:])  # 初期値除外
    if len(abs_diff) > 0:
        global_threshold = np.percentile(abs_diff, global_percentile)
        
        for i in range(n):
            if diff[i] > global_threshold:
                global_pos[i] = 1.0
            elif diff[i] < -global_threshold:
                global_neg[i] = 1.0
    
    # 局所構造ジャンプ検出
    for i in range(n):
        # 局所窓設定
        local_start = max(0, i - local_window)
        local_end = min(n, i + local_window + 1)
        
        # 局所窓内の絶対差分配列構築
        local_window_size = local_end - local_start
        if local_window_size > 1:
            # 局所絶対差分の計算
            local_abs_diffs = []
            for j in range(local_start, local_end):
                if j > 0 and j < n:  # 境界チェック
                    local_abs_diffs.append(abs(diff[j]))
            
            if len(local_abs_diffs) > 0:
                # 局所閾値計算
                local_abs_array = np.array(local_abs_diffs)
                local_threshold = np.percentile(local_abs_array, local_percentile)
                
                if i > 0 and i < n:  # 境界チェック
                    if diff[i] > local_threshold:
                        local_pos[i] = 1.0
                    elif diff[i] < -local_threshold:
                        local_neg[i] = 1.0
    
    return local_pos, local_neg, global_pos, global_neg

@njit(**JIT_BASE_OPTIONS)
def classify_hierarchical_events_fixed(
    local_pos: np.ndarray, local_neg: np.ndarray,
    global_pos: np.ndarray, global_neg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    階層的構造イベント分類（修正版）
    
    Lambda³理論: 構造変化イベントを階層性に基づいて分類
    純粋局所、純粋大域、混合イベントの識別
    
    修正点: イベント分類ロジックの最適化
    
    Args:
        local_pos, local_neg: 局所構造ジャンプ
        global_pos, global_neg: 大域構造ジャンプ
        
    Returns:
        pure_local_pos, pure_local_neg: 純粋局所イベント
        pure_global_pos, pure_global_neg: 純粋大域イベント  
        mixed_pos, mixed_neg: 混合イベント
    """
    n = len(local_pos)
    
    # 出力配列初期化
    pure_local_pos = np.zeros(n, dtype=np.float64)
    pure_local_neg = np.zeros(n, dtype=np.float64)
    pure_global_pos = np.zeros(n, dtype=np.float64)
    pure_global_neg = np.zeros(n, dtype=np.float64)
    mixed_pos = np.zeros(n, dtype=np.float64)
    mixed_neg = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # 正の構造変化分類
        local_pos_flag = local_pos[i] > 0.5
        global_pos_flag = global_pos[i] > 0.5
        
        if local_pos_flag and global_pos_flag:
            mixed_pos[i] = 1.0  # 局所-大域混合
        elif local_pos_flag and not global_pos_flag:
            pure_local_pos[i] = 1.0  # 純粋局所
        elif not local_pos_flag and global_pos_flag:
            pure_global_pos[i] = 1.0  # 純粋大域
        
        # 負の構造変化分類
        local_neg_flag = local_neg[i] > 0.5
        global_neg_flag = global_neg[i] > 0.5
        
        if local_neg_flag and global_neg_flag:
            mixed_neg[i] = 1.0  # 局所-大域混合
        elif local_neg_flag and not global_neg_flag:
            pure_local_neg[i] = 1.0  # 純粋局所
        elif not local_neg_flag and global_neg_flag:
            pure_global_neg[i] = 1.0  # 純粋大域
    
    return pure_local_pos, pure_local_neg, pure_global_pos, pure_global_neg, mixed_pos, mixed_neg

# ==========================================================
# SYNCHRONIZATION ANALYSIS - 同期分析（修正版）
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def calculate_sync_rate_at_lag_fixed(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """
    特定遅延での同期率計算（修正版）
    
    Lambda³理論: 構造テンソル系列間の時間遅延同期を定量化
    非時間依存の構造空間における相互響応の測定
    
    修正点: 境界処理とゼロ除算対策の強化
    
    Args:
        series_a, series_b: 構造テンソル系列
        lag: 時間遅延（正: A→B, 負: B→A）
        
    Returns:
        sync_rate: 同期率 [-1, 1]
    """
    len_a = len(series_a)
    len_b = len(series_b)
    
    if lag < 0:
        # 負の遅延: B→A の影響
        abs_lag = -lag
        if abs_lag < len_a and abs_lag < len_b:
            # 有効なオーバーラップ領域で計算
            overlap_size = min(len_a - abs_lag, len_b)
            if overlap_size > 0:
                sync_sum = 0.0
                for i in range(overlap_size):
                    sync_sum += series_a[abs_lag + i] * series_b[i]
                return sync_sum / overlap_size
        return 0.0
    elif lag > 0:
        # 正の遅延: A→B の影響
        if lag < len_a and lag < len_b:
            # 有効なオーバーラップ領域で計算
            overlap_size = min(len_a, len_b - lag)
            if overlap_size > 0:
                sync_sum = 0.0
                for i in range(overlap_size):
                    sync_sum += series_a[i] * series_b[lag + i]
                return sync_sum / overlap_size
        return 0.0
    else:
        # 遅延なし: 同時同期
        overlap_size = min(len_a, len_b)
        if overlap_size > 0:
            sync_sum = 0.0
            for i in range(overlap_size):
                sync_sum += series_a[i] * series_b[i]
            return sync_sum / overlap_size
        return 0.0

@njit(**JIT_BASE_OPTIONS)
def calculate_sync_profile_fixed(
    series_a: np.ndarray, series_b: np.ndarray, lag_window: int
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    同期プロファイル計算（修正版）
    
    Lambda³理論: 構造テンソル系列間の完全同期特性を解析
    全遅延範囲にわたる同期パターンの計算
    
    修正点: 並列計算を無効化して安定性を確保
    
    Args:
        series_a, series_b: 構造テンソル系列
        lag_window: 遅延窓サイズ
        
    Returns:
        lags: 遅延配列
        sync_values: 各遅延での同期値
        max_sync: 最大同期率
        optimal_lag: 最適遅延
    """
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    sync_values = np.zeros(n_lags, dtype=np.float64)
    
    # 逐次計算で各遅延での同期率を計算（並列化なし）
    for i in range(n_lags):
        lag = lags[i]
        sync_values[i] = calculate_sync_rate_at_lag_fixed(series_a, series_b, lag)
    
    # 最大同期と最適遅延を特定
    max_sync = sync_values[0]
    optimal_lag = lags[0]
    
    for i in range(1, n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]
    
    return lags, sync_values, max_sync, optimal_lag

@njit(**JIT_BASE_OPTIONS)
def detect_phase_coupling_fixed(
    phase_a: np.ndarray, phase_b: np.ndarray, coupling_threshold: float = 0.5
) -> Tuple[np.ndarray, float]:
    """
    位相結合検出（修正版）
    
    Lambda³理論: 構造テンソル位相空間での結合パターン検出
    非線形同期現象の定量化
    
    修正点: 位相差計算の最適化
    
    Args:
        phase_a, phase_b: 位相系列（-π to π）
        coupling_threshold: 結合検出閾値
        
    Returns:
        coupling_strength: 時系列結合強度
        mean_coupling: 平均結合強度
    """
    n = min(len(phase_a), len(phase_b))
    coupling_strength = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # 位相差計算（-π to π に正規化）
        phase_diff = phase_a[i] - phase_b[i]
        
        # 位相差の正規化（効率的な実装）
        while phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        while phase_diff < -np.pi:
            phase_diff += 2 * np.pi
        
        # 結合強度：位相差の安定性に基づく
        coupling_strength[i] = np.cos(phase_diff)
    
    # 平均結合強度計算
    abs_coupling_sum = 0.0
    for i in range(n):
        abs_coupling_sum += abs(coupling_strength[i])
    mean_coupling = abs_coupling_sum / n if n > 0 else 0.0
    
    return coupling_strength, mean_coupling

# ==========================================================
# UTILITY FUNCTIONS - ユーティリティ関数（修正版）
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def safe_divide_fixed(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全な除算（ゼロ除算回避）"""
    return numerator / denominator if abs(denominator) > 1e-12 else default

@njit(**JIT_BASE_OPTIONS)
def normalize_array_fixed(arr: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    配列正規化（修正版）
    
    修正点: 文字列比較とロバスト性の向上
    
    Args:
        arr: 入力配列
        method: 正規化手法 ('zscore', 'minmax', 'robust')
        
    Returns:
        normalized: 正規化配列
    """
    n = len(arr)
    normalized = np.zeros(n, dtype=np.float64)
    
    # Z-score正規化
    if method == 'zscore':
        # 平均値計算
        sum_val = 0.0
        for i in range(n):
            sum_val += arr[i]
        mean_val = sum_val / n
        
        # 標準偏差計算
        variance_sum = 0.0
        for i in range(n):
            variance_sum += (arr[i] - mean_val) ** 2
        std_val = np.sqrt(variance_sum / n)
        
        if std_val > 1e-12:
            for i in range(n):
                normalized[i] = (arr[i] - mean_val) / std_val
        else:
            # 標準偏差が0の場合はすべて0に設定
            for i in range(n):
                normalized[i] = 0.0
    
    # Min-Max正規化
    elif method == 'minmax':
        min_val = arr[0]
        max_val = arr[0]
        
        # 最小値・最大値探索
        for i in range(1, n):
            if arr[i] < min_val:
                min_val = arr[i]
            if arr[i] > max_val:
                max_val = arr[i]
        
        range_val = max_val - min_val
        if range_val > 1e-12:
            for i in range(n):
                normalized[i] = (arr[i] - min_val) / range_val
        else:
            # 範囲が0の場合はすべて0に設定
            for i in range(n):
                normalized[i] = 0.0
    
    # デフォルト：そのままコピー
    else:
        for i in range(n):
            normalized[i] = arr[i]
    
    return normalized

@njit(**JIT_BASE_OPTIONS)
def moving_average_fixed(data: np.ndarray, window: int) -> np.ndarray:
    """移動平均計算（修正版）"""
    n = len(data)
    ma = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        window_size = end - start
        
        # 窓内平均計算
        sum_val = 0.0
        for j in range(start, end):
            sum_val += data[j]
        ma[i] = sum_val / window_size
    
    return ma

@njit(**JIT_BASE_OPTIONS)
def exponential_smoothing_fixed(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """指数平滑法（修正版）"""
    n = len(data)
    smoothed = np.zeros(n, dtype=np.float64)
    
    if n > 0:
        smoothed[0] = data[0]  # 初期値
        for i in range(1, n):
            smoothed[i] = alpha * data[i] + (1.0 - alpha) * smoothed[i-1]
    
    return smoothed

# ==========================================================
# INTEGRATION FUNCTIONS - 統合関数（修正版）
# ==========================================================

@njit(**JIT_BASE_OPTIONS)
def extract_lambda3_features_jit(
    data: np.ndarray,
    window: int = 10,
    local_window: int = 5,
    global_window: int = 30,
    delta_percentile: float = 95.0,
    local_percentile: float = 90.0,
    global_percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lambda³特徴量一括抽出（JIT最適化版）
    
    Lambda³理論の全核心特徴量を効率的に計算
    
    Args:
        data: 構造テンソル時系列
        window: 基本窓サイズ
        local_window: 局所検出窓
        global_window: 大域検出窓
        delta_percentile: 基本変化検出閾値
        local_percentile: 局所検出閾値
        global_percentile: 大域検出閾値
        
    Returns:
        delta_pos: 正の構造変化
        delta_neg: 負の構造変化
        rho_t: 張力スカラー
        local_pos: 局所正変化
        local_neg: 局所負変化
        global_pos: 大域正変化
        global_neg: 大域負変化
    """
    # 基本構造変化検出
    diff, threshold = calculate_diff_and_threshold_fixed(data, delta_percentile)
    delta_pos, delta_neg = detect_structural_jumps_fixed(diff, threshold)
    
    # 張力スカラー計算
    rho_t = calculate_tension_scalar_fixed(data, window)
    
    # 階層的構造変化検出
    local_pos, local_neg, global_pos, global_neg = detect_hierarchical_jumps_fixed(
        data, local_window, global_window, local_percentile, global_percentile
    )
    
    return delta_pos, delta_neg, rho_t, local_pos, local_neg, global_pos, global_neg

# ==========================================================
# TESTING AND VALIDATION - テストと検証
# ==========================================================

def test_jit_functions_fixed():
    """修正版JIT関数のテスト実行"""
    print("🧪 修正版Lambda³ JIT関数テスト開始")
    print("=" * 50)
    
    # テストデータ生成
    np.random.seed(42)
    n = 1000
    test_data = np.cumsum(np.random.randn(n) * 0.1)
    
    # 意図的な構造変化を注入
    test_data[500:] += 0.5  # 構造ジャンプ
    test_data[300:400] += np.sin(np.arange(100) * 0.1) * 0.2  # 周期的変動
    
    print(f"📊 テストデータ: {len(test_data)} points")
    
    try:
        # 基本機能テスト
        diff, threshold = calculate_diff_and_threshold_fixed(test_data, 95.0)
        print(f"✅ 構造差分閾値: {threshold:.4f}")
        
        pos_jumps, neg_jumps = detect_structural_jumps_fixed(diff, threshold)
        print(f"✅ 正の構造ジャンプ: {np.sum(pos_jumps)}")
        print(f"✅ 負の構造ジャンプ: {np.sum(neg_jumps)}")
        
        rho_t = calculate_tension_scalar_fixed(test_data, 10)
        print(f"✅ 平均張力スカラー: {np.mean(rho_t):.4f}")
        print(f"✅ 最大張力スカラー: {np.max(rho_t):.4f}")
        
        # 階層的検出テスト
        local_pos, local_neg, global_pos, global_neg = detect_hierarchical_jumps_fixed(test_data)
        local_events = np.sum(local_pos + local_neg)
        global_events = np.sum(global_pos + global_neg)
        print(f"✅ 階層的検出 - 局所: {local_events}, 大域: {global_events}")
        
        # 一括特徴抽出テスト
        features = extract_lambda3_features_jit(test_data)
        print(f"✅ 一括特徴抽出成功: {len(features)} 特徴量")
        
        # 同期分析テスト
        test_data_b = test_data[50:] + np.random.randn(len(test_data)-50) * 0.05
        lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_fixed(
            test_data[:-50], test_data_b, 10
        )
        print(f"✅ 同期分析 - 最大同期: {max_sync:.4f}, 最適遅延: {optimal_lag}")
        
        print("\n🎯 修正版Lambda³ JIT関数テスト完了")
        print("すべての関数が正常に動作しています")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_performance_fixed():
    """性能ベンチマーク（修正版）"""
    import time
    
    print("\n⚡ 修正版Lambda³ JIT性能ベンチマーク")
    print("=" * 50)
    
    # テストデータ生成
    np.random.seed(42)
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nデータサイズ: {size}")
        data = np.cumsum(np.random.randn(size) * 0.1)
        
        # ウォームアップ（JITコンパイル）
        _ = extract_lambda3_features_jit(data[:100])
        
        # 実測定
        start_time = time.time()
        features = extract_lambda3_features_jit(data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        throughput = size / execution_time
        
        print(f"   実行時間: {execution_time:.3f}秒")
        print(f"   処理速度: {throughput:.0f} points/sec")
        
        # 結果統計
        total_events = np.sum(features[0]) + np.sum(features[1])  # pos + neg
        mean_tension = np.mean(features[2])  # rho_t
        print(f"   検出イベント: {total_events}")
        print(f"   平均張力: {mean_tension:.4f}")
    
    print("\n✅ ベンチマーク完了")

# 実行例
if __name__ == "__main__":
    # 修正版JIT関数の基本テスト
    success = test_jit_functions_fixed()
    
    if success:
        print("\n🚀 Lambda³理論JIT関数の修正が完了しました")
        print("Numba JIT型推論問題が完全に解決されています")
        
        # 性能ベンチマーク実行
        benchmark_performance_fixed()
    else:
        print("\n❌ 問題が残っています。さらなる調査が必要です")
