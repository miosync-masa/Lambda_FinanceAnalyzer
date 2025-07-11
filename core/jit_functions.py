# ==========================================================
# lambda3/core/jit_functions.py  
# JIT-Optimized Core Functions for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論JIT最適化核心関数群

構造テンソル(Λ)演算、進行ベクトル(ΛF)計算、張力スカラー(ρT)演算の
高速JITコンパイル実装。時間非依存の構造空間における
∆ΛC pulsations検出の数値計算核心部。

NumbaのJITコンパイルにより、Pythonレベルでの数値計算ボトルネックを
除去し、C/Fortranレベルの性能を実現。
"""

import numpy as np
from numba import jit, njit, prange
from typing import Tuple, Optional
import warnings

# JIT最適化設定
JIT_OPTIONS = {
    'nopython': True,  # Pure numba mode
    'fastmath': True,  # Fast math optimizations
    'parallel': True,  # Parallel execution where possible
    'cache': True      # Cache compiled functions
}

# ==========================================================
# BASIC TENSOR OPERATIONS - 基本テンソル演算
# ==========================================================

@njit(**JIT_OPTIONS)
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """
    構造テンソル差分計算と閾値決定
    
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
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0  # 初期値は変化なし
    
    # 構造テンソル差分計算 ∆Λ(t) = Λ(t) - Λ(t-1)
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    # 絶対値からパーセンタイル閾値を決定
    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    
    return diff, threshold

@njit(**JIT_OPTIONS)
def detect_structural_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    構造テンソルジャンプ検出
    
    Lambda³理論: ∆ΛC pulsations の正負分離検出
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
            pos_jumps[i] = 1.0  # 正の構造ジャンプ
        elif diff[i] < -threshold:
            neg_jumps[i] = 1.0  # 負の構造ジャンプ
    
    return pos_jumps, neg_jumps

@njit(**JIT_OPTIONS)
def calculate_local_statistics(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    局所統計量計算（標準偏差と平均）
    
    構造テンソルの局所的変動特性を定量化
    時間窓内での構造空間の局所統計
    
    Args:
        data: 構造テンソル系列
        window: 局所統計計算窓サイズ
        
    Returns:
        local_std: 局所標準偏差
        local_mean: 局所平均
    """
    n = len(data)
    local_std = np.empty(n, dtype=np.float64)
    local_mean = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        # 局所窓の設定（境界処理含む）
        start = max(0, i - window)
        end = min(n, i + window + 1)
        
        # 局所データ抽出
        subset = data[start:end]
        
        # 局所統計計算
        local_mean[i] = np.mean(subset)
        if len(subset) > 1:
            variance = np.sum((subset - local_mean[i]) ** 2) / len(subset)
            local_std[i] = np.sqrt(variance)
        else:
            local_std[i] = 0.0
    
    return local_std, local_mean

@njit(**JIT_OPTIONS)
def calculate_tension_scalar(data: np.ndarray, window: int) -> np.ndarray:
    """
    張力スカラー(ρT)計算
    
    Lambda³理論: 構造テンソル空間における張力度合いを定量化
    局所的な構造変動の強度を表現する重要な指標
    
    Args:
        data: 構造テンソル系列 Λ(t)
        window: 張力計算窓サイズ
        
    Returns:
        rho_t: 張力スカラー ρT(t)
    """
    n = len(data)
    rho_t = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        # 後向き窓による張力計算（因果性保持）
        start = max(0, i - window)
        end = i + 1
        
        subset = data[start:end]
        if len(subset) > 1:
            # 局所平均からの分散として張力を定義
            mean_val = np.mean(subset)
            variance = np.sum((subset - mean_val) ** 2) / len(subset)
            rho_t[i] = np.sqrt(variance)  # 張力スカラー = 局所標準偏差
        else:
            rho_t[i] = 0.0
    
    return rho_t

# ==========================================================
# HIERARCHICAL STRUCTURE DETECTION - 階層構造検出
# ==========================================================

@njit(**JIT_OPTIONS)
def detect_hierarchical_jumps(
    data: np.ndarray, 
    local_window: int = 10, 
    global_window: int = 50,
    local_percentile: float = 90.0,
    global_percentile: float = 95.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    階層的構造ジャンプ検出
    
    Lambda³理論: 構造テンソル変化の階層性（局所-大域）を分離検出
    異なる時間スケールでの∆ΛC pulsationsを識別
    
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
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    # 出力配列初期化
    local_pos = np.zeros(n, dtype=np.float64)
    local_neg = np.zeros(n, dtype=np.float64)
    global_pos = np.zeros(n, dtype=np.float64)
    global_neg = np.zeros(n, dtype=np.float64)
    
    # 局所構造ジャンプ検出
    for i in range(n):
        # 局所窓設定
        local_start = max(0, i - local_window)
        local_end = min(n, i + local_window + 1)
        local_subset = np.abs(diff[local_start:local_end])
        
        if len(local_subset) > 0:
            local_threshold = np.percentile(local_subset, local_percentile)
            if diff[i] > local_threshold:
                local_pos[i] = 1.0
            elif diff[i] < -local_threshold:
                local_neg[i] = 1.0
    
    # 大域構造ジャンプ検出
    global_threshold_pos = np.percentile(np.abs(diff), global_percentile)
    global_threshold_neg = -global_threshold_pos
    
    for i in range(n):
        if diff[i] > global_threshold_pos:
            global_pos[i] = 1.0
        elif diff[i] < global_threshold_neg:
            global_neg[i] = 1.0
    
    return local_pos, local_neg, global_pos, global_neg

@njit(**JIT_OPTIONS)
def classify_hierarchical_events(
    local_pos: np.ndarray, local_neg: np.ndarray,
    global_pos: np.ndarray, global_neg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    階層的構造イベント分類
    
    Lambda³理論: 構造変化イベントを階層性に基づいて分類
    純粋局所、純粋大域、混合イベントの識別
    
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
        if local_pos[i] > 0 and global_pos[i] > 0:
            mixed_pos[i] = 1.0  # 局所-大域混合
        elif local_pos[i] > 0 and global_pos[i] == 0:
            pure_local_pos[i] = 1.0  # 純粋局所
        elif local_pos[i] == 0 and global_pos[i] > 0:
            pure_global_pos[i] = 1.0  # 純粋大域
        
        # 負の構造変化分類
        if local_neg[i] > 0 and global_neg[i] > 0:
            mixed_neg[i] = 1.0  # 局所-大域混合
        elif local_neg[i] > 0 and global_neg[i] == 0:
            pure_local_neg[i] = 1.0  # 純粋局所
        elif local_neg[i] == 0 and global_neg[i] > 0:
            pure_global_neg[i] = 1.0  # 純粋大域
    
    return pure_local_pos, pure_local_neg, pure_global_pos, pure_global_neg, mixed_pos, mixed_neg

# ==========================================================
# SYNCHRONIZATION ANALYSIS - 同期分析
# ==========================================================

@njit(**JIT_OPTIONS)
def calculate_sync_rate_at_lag(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """
    特定遅延での同期率計算
    
    Lambda³理論: 構造テンソル系列間の時間遅延同期を定量化
    非時間依存の構造空間における相互響応の測定
    
    Args:
        series_a, series_b: 構造テンソル系列
        lag: 時間遅延（正: A→B, 負: B→A）
        
    Returns:
        sync_rate: 同期率 [-1, 1]
    """
    if lag < 0:
        # 負の遅延: B→A の影響
        abs_lag = -lag
        if abs_lag < len(series_a):
            return np.mean(series_a[abs_lag:] * series_b[:len(series_a)-abs_lag])
        else:
            return 0.0
    elif lag > 0:
        # 正の遅延: A→B の影響
        if lag < len(series_b):
            return np.mean(series_a[:len(series_a)-lag] * series_b[lag:])
        else:
            return 0.0
    else:
        # 遅延なし: 同時同期
        return np.mean(series_a * series_b)

@njit(parallel=True, **{k: v for k, v in JIT_OPTIONS.items() if k != 'parallel'})
def calculate_sync_profile(
    series_a: np.ndarray, series_b: np.ndarray, lag_window: int
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    同期プロファイル計算（並列最適化版）
    
    Lambda³理論: 構造テンソル系列間の完全同期特性を解析
    全遅延範囲にわたる同期パターンの効率的計算
    
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
    sync_values = np.empty(n_lags, dtype=np.float64)
    
    # 並列計算で各遅延での同期率を計算
    for i in prange(n_lags):
        lag = lags[i]
        sync_values[i] = calculate_sync_rate_at_lag(series_a, series_b, lag)
    
    # 最大同期と最適遅延を特定
    max_sync = -np.inf
    optimal_lag = 0
    for i in range(n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]
    
    return lags, sync_values, max_sync, optimal_lag

@njit(**JIT_OPTIONS)
def detect_phase_coupling(
    phase_a: np.ndarray, phase_b: np.ndarray, coupling_threshold: float = 0.5
) -> Tuple[np.ndarray, float]:
    """
    位相結合検出
    
    Lambda³理論: 構造テンソル位相空間での結合パターン検出
    非線形同期現象の定量化
    
    Args:
        phase_a, phase_b: 位相系列（-π to π）
        coupling_threshold: 結合検出閾値
        
    Returns:
        coupling_strength: 時系列結合強度
        mean_coupling: 平均結合強度
    """
    n = len(phase_a)
    coupling_strength = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        # 位相差計算（-π to π に正規化）
        phase_diff = phase_a[i] - phase_b[i]
        while phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        while phase_diff < -np.pi:
            phase_diff += 2 * np.pi
        
        # 結合強度：位相差の安定性に基づく
        coupling_strength[i] = np.cos(phase_diff)
    
    mean_coupling = np.mean(np.abs(coupling_strength))
    
    return coupling_strength, mean_coupling

# ==========================================================
# INTERACTION MATRIX OPERATIONS - 相互作用行列演算
# ==========================================================

@njit(**JIT_OPTIONS)
def calculate_pairwise_interaction_matrix(
    event_matrix: np.ndarray, lag_window: int = 5
) -> np.ndarray:
    """
    ペアワイズ相互作用行列計算
    
    Lambda³理論: 多系列構造テンソル間の相互作用強度行列
    非対称相互作用の効率的計算
    
    Args:
        event_matrix: [n_series, n_timepoints] イベント行列
        lag_window: 相互作用遅延窓
        
    Returns:
        interaction_matrix: [n_series, n_series] 相互作用強度行列
    """
    n_series, n_time = event_matrix.shape
    interaction_matrix = np.zeros((n_series, n_series), dtype=np.float64)
    
    for i in range(n_series):
        for j in range(n_series):
            if i != j:
                # 系列i→系列jの相互作用を計算
                max_interaction = 0.0
                
                for lag in range(1, lag_window + 1):
                    if lag < n_time:
                        # 遅延相互作用計算
                        cause_events = event_matrix[i, :-lag]
                        effect_events = event_matrix[j, lag:]
                        
                        # 条件付き確率として相互作用強度を定義
                        joint_prob = np.mean(cause_events * effect_events)
                        cause_prob = np.mean(cause_events)
                        
                        if cause_prob > 1e-8:
                            interaction = joint_prob / cause_prob
                            max_interaction = max(max_interaction, interaction)
                
                interaction_matrix[i, j] = max_interaction
            else:
                interaction_matrix[i, j] = 1.0  # 自己相互作用
    
    return interaction_matrix

@njit(**JIT_OPTIONS)
def compute_network_metrics(adjacency_matrix: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    ネットワークメトリクス計算
    
    Lambda³理論: 構造テンソルネットワークの位相幾何学的特性
    
    Args:
        adjacency_matrix: 隣接行列
        
    Returns:
        density: ネットワーク密度
        clustering: 平均クラスタリング係数
        centrality: 次数中心性ベクトル
    """
    n = adjacency_matrix.shape[0]
    
    # ネットワーク密度計算
    total_edges = np.sum(adjacency_matrix) - np.trace(adjacency_matrix)  # 自己ループ除外
    max_edges = n * (n - 1)  # 有向グラフの最大エッジ数
    density = total_edges / max_edges if max_edges > 0 else 0.0
    
    # 次数中心性計算（出次数＋入次数）
    centrality = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out_degree = np.sum(adjacency_matrix[i, :]) - adjacency_matrix[i, i]
        in_degree = np.sum(adjacency_matrix[:, i]) - adjacency_matrix[i, i]
        centrality[i] = (out_degree + in_degree) / (2 * (n - 1)) if n > 1 else 0.0
    
    # 簡易クラスタリング係数（局所密度の平均）
    clustering_sum = 0.0
    valid_nodes = 0
    
    for i in range(n):
        # ノードiの近傍ノード特定
        neighbors = []
        for j in range(n):
            if i != j and (adjacency_matrix[i, j] > 0 or adjacency_matrix[j, i] > 0):
                neighbors.append(j)
        
        if len(neighbors) > 1:
            # 近傍間のエッジ数計算
            neighbor_edges = 0.0
            max_neighbor_edges = len(neighbors) * (len(neighbors) - 1)
            
            for j in range(len(neighbors)):
                for k in range(len(neighbors)):
                    if j != k:
                        node_j = neighbors[j]
                        node_k = neighbors[k]
                        if adjacency_matrix[node_j, node_k] > 0:
                            neighbor_edges += 1.0
            
            clustering_sum += neighbor_edges / max_neighbor_edges if max_neighbor_edges > 0 else 0.0
            valid_nodes += 1
    
    clustering = clustering_sum / valid_nodes if valid_nodes > 0 else 0.0
    
    return density, clustering, centrality

# ==========================================================
# STATISTICAL ANALYSIS FUNCTIONS - 統計解析関数
# ==========================================================

@njit(**JIT_OPTIONS)
def calculate_rolling_statistics(
    data: np.ndarray, window: int, statistic: str = 'std'
) -> np.ndarray:
    """
    ローリング統計量計算
    
    Lambda³理論: 構造テンソルの時変統計特性
    
    Args:
        data: 入力データ系列
        window: ローリング窓サイズ
        statistic: 統計量タイプ ('std', 'mean', 'var', 'skew')
        
    Returns:
        rolling_stats: ローリング統計量系列
    """
    n = len(data)
    rolling_stats = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        subset = data[start:end]
        
        if statistic == 'std':
            if len(subset) > 1:
                mean_val = np.mean(subset)
                variance = np.sum((subset - mean_val) ** 2) / len(subset)
                rolling_stats[i] = np.sqrt(variance)
            else:
                rolling_stats[i] = 0.0
        elif statistic == 'mean':
            rolling_stats[i] = np.mean(subset)
        elif statistic == 'var':
            if len(subset) > 1:
                mean_val = np.mean(subset)
                rolling_stats[i] = np.sum((subset - mean_val) ** 2) / len(subset)
            else:
                rolling_stats[i] = 0.0
        elif statistic == 'skew':
            if len(subset) > 2:
                mean_val = np.mean(subset)
                std_val = np.sqrt(np.sum((subset - mean_val) ** 2) / len(subset))
                if std_val > 1e-8:
                    skewness = np.sum(((subset - mean_val) / std_val) ** 3) / len(subset)
                    rolling_stats[i] = skewness
                else:
                    rolling_stats[i] = 0.0
            else:
                rolling_stats[i] = 0.0
        else:
            rolling_stats[i] = 0.0  # 未サポート統計量
    
    return rolling_stats

@njit(**JIT_OPTIONS)
def detect_regime_transitions(
    data: np.ndarray, threshold_std_multiplier: float = 2.0, min_regime_length: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    レジーム転換点検出
    
    Lambda³理論: 構造テンソル空間におけるレジーム境界の識別
    
    Args:
        data: 構造テンソル系列
        threshold_std_multiplier: 転換検出閾値倍率
        min_regime_length: 最小レジーム長
        
    Returns:
        transition_points: 転換点インデックス
        regime_labels: レジームラベル系列
    """
    n = len(data)
    
    # データの標準化
    mean_val = np.mean(data)
    std_val = np.sqrt(np.sum((data - mean_val) ** 2) / n)
    threshold = threshold_std_multiplier * std_val
    
    # 転換点候補検出
    candidates = []
    for i in range(1, n):
        if np.abs(data[i] - data[i-1]) > threshold:
            candidates.append(i)
    
    # 最小レジーム長による転換点フィルタリング
    if len(candidates) == 0:
        return np.array([0], dtype=np.int64), np.zeros(n, dtype=np.int64)
    
    filtered_transitions = [0]  # 最初の点を含める
    
    for candidate in candidates:
        if candidate - filtered_transitions[-1] >= min_regime_length:
            filtered_transitions.append(candidate)
    
    # レジームラベル生成
    regime_labels = np.zeros(n, dtype=np.int64)
    current_regime = 0
    
    for i in range(len(filtered_transitions)):
        start = filtered_transitions[i]
        end = filtered_transitions[i + 1] if i + 1 < len(filtered_transitions) else n
        
        for j in range(start, end):
            regime_labels[j] = current_regime
        
        current_regime += 1
    
    return np.array(filtered_transitions, dtype=np.int64), regime_labels

@njit(**JIT_OPTIONS)
def calculate_multiscale_entropy(data: np.ndarray, max_scale: int = 10, r: float = 0.2) -> np.ndarray:
    """
    マルチスケールエントロピー計算
    
    Lambda³理論: 構造テンソルの複雑性のスケール依存性
    
    Args:
        data: 入力データ系列
        max_scale: 最大スケール
        r: マッチング許容範囲（標準偏差の倍率）
        
    Returns:
        mse_values: 各スケールでのサンプルエントロピー
    """
    n = len(data)
    mse_values = np.empty(max_scale, dtype=np.float64)
    
    # データ標準化
    data_std = np.sqrt(np.sum((data - np.mean(data)) ** 2) / n)
    tolerance = r * data_std
    
    for scale in range(1, max_scale + 1):
        # スケールダウンサンプリング
        if scale == 1:
            scaled_data = data.copy()
        else:
            scaled_length = n // scale
            scaled_data = np.empty(scaled_length, dtype=np.float64)
            for i in range(scaled_length):
                scaled_data[i] = np.mean(data[i*scale:(i+1)*scale])
        
        # サンプルエントロピー計算（簡易版）
        N = len(scaled_data)
        if N < 3:
            mse_values[scale - 1] = 0.0
            continue
        
        m = 2  # パターン長
        
        # m次元パターンのマッチング数計算
        A = 0.0  # m次元マッチ数
        B = 0.0  # (m+1)次元マッチ数
        
        for i in range(N - m):
            template_m = scaled_data[i:i+m]
            template_m1 = scaled_data[i:i+m+1]
            
            for j in range(N - m):
                if i != j:
                    candidate_m = scaled_data[j:j+m]
                    candidate_m1 = scaled_data[j:j+m+1] if j+m+1 <= N else np.array([0.0])
                    
                    # m次元マッチング判定
                    match_m = True
                    for k in range(m):
                        if np.abs(template_m[k] - candidate_m[k]) > tolerance:
                            match_m = False
                            break
                    
                    if match_m:
                        A += 1.0
                        
                        # (m+1)次元マッチング判定
                        if len(candidate_m1) == m + 1:
                            match_m1 = True
                            for k in range(m + 1):
                                if np.abs(template_m1[k] - candidate_m1[k]) > tolerance:
                                    match_m1 = False
                                    break
                            
                            if match_m1:
                                B += 1.0
        
        # サンプルエントロピー計算
        if A > 0 and B > 0:
            mse_values[scale - 1] = -np.log(B / A)
        else:
            mse_values[scale - 1] = 0.0
    
    return mse_values

# ==========================================================
# UTILITY FUNCTIONS - ユーティリティ関数
# ==========================================================

@njit(**JIT_OPTIONS)
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全な除算（ゼロ除算回避）"""
    return numerator / denominator if np.abs(denominator) > 1e-10 else default

@njit(**JIT_OPTIONS)
def normalize_array(arr: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    配列正規化
    
    Args:
        arr: 入力配列
        method: 正規化手法 ('zscore', 'minmax', 'robust')
        
    Returns:
        normalized: 正規化配列
    """
    n = len(arr)
    normalized = np.empty(n, dtype=np.float64)
    
    if method == 'zscore':
        mean_val = np.mean(arr)
        std_val = np.sqrt(np.sum((arr - mean_val) ** 2) / n)
        if std_val > 1e-10:
            for i in range(n):
                normalized[i] = (arr[i] - mean_val) / std_val
        else:
            normalized[:] = 0.0
    
    elif method == 'minmax':
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val > 1e-10:
            for i in range(n):
                normalized[i] = (arr[i] - min_val) / range_val
        else:
            normalized[:] = 0.0
    
    elif method == 'robust':
        # 中央値とMADによるロバスト正規化
        median_val = np.median(arr)
        deviations = np.abs(arr - median_val)
        mad = np.median(deviations)
        if mad > 1e-10:
            for i in range(n):
                normalized[i] = (arr[i] - median_val) / (1.4826 * mad)  # 1.4826はMADのスケール因子
        else:
            normalized[:] = 0.0
    
    else:
        # デフォルト：そのままコピー
        normalized[:] = arr[:]
    
    return normalized

@njit(**JIT_OPTIONS)
def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """移動平均計算（JIT最適化版）"""
    n = len(data)
    ma = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        start = max(0, i - window + 1)
        end = i + 1
        ma[i] = np.mean(data[start:end])
    
    return ma

@njit(**JIT_OPTIONS)
def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """指数平滑法（JIT最適化版）"""
    n = len(data)
    smoothed = np.empty(n, dtype=np.float64)
    
    smoothed[0] = data[0]  # 初期値
    for i in range(1, n):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

# ==========================================================
# PERFORMANCE BENCHMARKING - 性能ベンチマーク
# ==========================================================

def benchmark_jit_functions():
    """JIT関数の性能ベンチマーク"""
    import time
    
    print("Lambda³ JIT Functions Performance Benchmark")
    print("=" * 50)
    
    # テストデータ生成
    np.random.seed(42)
    n = 10000
    data = np.cumsum(np.random.randn(n) * 0.1)  # ランダムウォーク
    
    # 各関数のベンチマーク
    functions_to_test = [
        ('calculate_diff_and_threshold', lambda: calculate_diff_and_threshold(data, 95.0)),
        ('calculate_tension_scalar', lambda: calculate_tension_scalar(data, 10)),
        ('detect_hierarchical_jumps', lambda: detect_hierarchical_jumps(data, 10, 30, 90.0, 95.0)),
        ('calculate_sync_profile', lambda: calculate_sync_profile(data[:1000], data[100:1100], 10)),
        ('calculate_rolling_statistics', lambda: calculate_rolling_statistics(data, 20, 'std'))
    ]
    
    for func_name, func in functions_to_test:
        # ウォームアップ（JITコンパイル）
        _ = func()
        
        # 実測定
        start_time = time.time()
        for _ in range(10):
            result = func()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"{func_name}: {avg_time:.4f} sec/call")
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    # JIT関数のコンパイルと基本テスト
    print("Lambda³ JIT Functions - Compilation and Testing")
    print("=" * 50)
    
    # テストデータ
    np.random.seed(42)
    test_data = np.cumsum(np.random.randn(1000) * 0.1)
    
    # 基本機能テスト
    diff, threshold = calculate_diff_and_threshold(test_data, 95.0)
    print(f"Structural difference threshold: {threshold:.4f}")
    
    pos_jumps, neg_jumps = detect_structural_jumps(diff, threshold)
    print(f"Positive jumps detected: {np.sum(pos_jumps)}")
    print(f"Negative jumps detected: {np.sum(neg_jumps)}")
    
    rho_t = calculate_tension_scalar(test_data, 10)
    print(f"Mean tension scalar: {np.mean(rho_t):.4f}")
    
    # 階層的検出テスト
    local_pos, local_neg, global_pos, global_neg = detect_hierarchical_jumps(test_data)
    print(f"Hierarchical events - Local: {np.sum(local_pos + local_neg)}, Global: {np.sum(global_pos + global_neg)}")
    
    print("\nJIT functions compiled and tested successfully!")
    
    # 性能ベンチマーク実行
    benchmark_jit_functions()
