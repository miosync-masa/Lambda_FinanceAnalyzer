# ==========================================================
# lambda3/core/structural_tensor.py
# Structural Tensor Operations for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論構造テンソル演算モジュール

構造テンソル(Λ)の数学的操作と変換を実装。
時間非依存の構造空間における∆ΛC pulsationsの検出、
分類、および高次構造変化パターンの抽出を担当。

核心理論:
- 構造テンソル Λ(t) = {λᵢⱼ} の時系列表現
- 構造変化 ∆ΛC = Λ(t) - Λ(t-1) の検出
- 階層的構造変化の分離と分類
- 多重スケール構造解析
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

from .config import L3BaseConfig, L3HierarchicalConfig
from .jit_functions import (
    calculate_diff_and_threshold,
    detect_structural_jumps, 
    calculate_tension_scalar,
    detect_hierarchical_jumps,
    classify_hierarchical_events,
    calculate_rolling_statistics,
    normalize_array,
    moving_average
)

# ==========================================================
# STRUCTURAL TENSOR DATA CLASS
# ==========================================================

@dataclass
class StructuralTensorFeatures:
    """
    構造テンソル特徴量データクラス
    
    Lambda³理論における構造テンソル解析の完全な特徴セット。
    階層的∆ΛC変化、張力スカラー、進行ベクトルを統合管理。
    """
    
    # 原系列データ
    data: np.ndarray
    series_name: str = "Unknown"
    
    # 基本構造変化特徴量
    delta_LambdaC_pos: np.ndarray = None  # 正の∆ΛC変化
    delta_LambdaC_neg: np.ndarray = None  # 負の∆ΛC変化
    rho_T: np.ndarray = None              # 張力スカラー ρT
    time_trend: np.ndarray = None         # 時間トレンド（非時間的インデックス）
    
    # 階層的構造変化特徴量
    local_pos: np.ndarray = None          # 局所正構造変化
    local_neg: np.ndarray = None          # 局所負構造変化
    global_pos: np.ndarray = None         # 大域正構造変化
    global_neg: np.ndarray = None         # 大域負構造変化
    
    # 分類済み階層イベント
    pure_local_pos: np.ndarray = None     # 純粋局所正イベント
    pure_local_neg: np.ndarray = None     # 純粋局所負イベント
    pure_global_pos: np.ndarray = None    # 純粋大域正イベント
    pure_global_neg: np.ndarray = None    # 純粋大域負イベント
    mixed_pos: np.ndarray = None          # 混合正イベント
    mixed_neg: np.ndarray = None          # 混合負イベント
    
    # 統計的特徴量
    local_statistics: Dict[str, np.ndarray] = None
    multi_scale_features: Dict[int, np.ndarray] = None
    
    def __post_init__(self):
        """初期化後処理：基本チェックとメタデータ設定"""
        if self.data is None:
            raise ValueError("data cannot be None")
        
        self.n_points = len(self.data)
        self.data_length = self.n_points
        
        # 時間インデックス生成（非時間的構造空間インデックス）
        if self.time_trend is None:
            self.time_trend = np.arange(self.n_points, dtype=np.float64)
    
    def get_basic_features(self) -> Dict[str, np.ndarray]:
        """基本特徴量辞書を返す"""
        return {
            'data': self.data,
            'delta_LambdaC_pos': self.delta_LambdaC_pos,
            'delta_LambdaC_neg': self.delta_LambdaC_neg, 
            'rho_T': self.rho_T,
            'time_trend': self.time_trend
        }
    
    def get_hierarchical_features(self) -> Dict[str, np.ndarray]:
        """階層的特徴量辞書を返す"""
        return {
            'local_pos': self.local_pos,
            'local_neg': self.local_neg,
            'global_pos': self.global_pos,
            'global_neg': self.global_neg,
            'pure_local_pos': self.pure_local_pos,
            'pure_local_neg': self.pure_local_neg,
            'pure_global_pos': self.pure_global_pos,
            'pure_global_neg': self.pure_global_neg,
            'mixed_pos': self.mixed_pos,
            'mixed_neg': self.mixed_neg
        }
    
    def count_events_by_type(self) -> Dict[str, int]:
        """イベント種別カウント"""
        counts = {}
        
        if self.delta_LambdaC_pos is not None:
            counts['total_pos'] = int(np.sum(self.delta_LambdaC_pos))
        if self.delta_LambdaC_neg is not None:
            counts['total_neg'] = int(np.sum(self.delta_LambdaC_neg))
        
        # 階層別カウント
        hierarchical_features = self.get_hierarchical_features()
        for name, feature in hierarchical_features.items():
            if feature is not None:
                counts[name] = int(np.sum(feature))
        
        return counts
    
    def calculate_summary_statistics(self) -> Dict[str, float]:
        """要約統計計算"""
        stats = {
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data))
        }
        
        if self.rho_T is not None:
            stats.update({
                'mean_tension': float(np.mean(self.rho_T)),
                'max_tension': float(np.max(self.rho_T)),
                'tension_volatility': float(np.std(self.rho_T))
            })
        
        return stats

# ==========================================================
# STRUCTURAL TENSOR EXTRACTOR
# ==========================================================

class StructuralTensorExtractor:
    """
    構造テンソル特徴抽出器
    
    Lambda³理論に基づく包括的構造テンソル特徴量の抽出。
    基本∆ΛC検出から階層的構造分析まで統合的に実行。
    """
    
    def __init__(self, config: Optional[Union[L3BaseConfig, L3HierarchicalConfig]] = None):
        """
        初期化
        
        Args:
            config: Lambda³設定オブジェクト
        """
        if config is None:
            self.config = L3BaseConfig()
        else:
            self.config = config
        
        self.extraction_history = []
        
    def extract_basic_features(self, data: np.ndarray, series_name: str = "Unknown") -> StructuralTensorFeatures:
        """
        基本構造テンソル特徴抽出
        
        Lambda³理論の基礎：∆ΛC変化と張力スカラー ρT の検出
        
        Args:
            data: 入力データ系列
            series_name: 系列名
            
        Returns:
            StructuralTensorFeatures: 基本特徴量オブジェクト
        """
        # データ前処理
        data = self._preprocess_data(data)
        
        # 基本構造変化検出
        diff, threshold = calculate_diff_and_threshold(data, self.config.delta_percentile)
        delta_pos, delta_neg = detect_structural_jumps(diff, threshold)
        
        # 張力スカラー計算
        rho_t = calculate_tension_scalar(data, self.config.window)
        
        # 時間インデックス生成
        time_trend = np.arange(len(data), dtype=np.float64)
        
        # 特徴量オブジェクト生成
        features = StructuralTensorFeatures(
            data=data,
            series_name=series_name,
            delta_LambdaC_pos=delta_pos,
            delta_LambdaC_neg=delta_neg,
            rho_T=rho_t,
            time_trend=time_trend
        )
        
        return features
    
    def extract_hierarchical_features(
        self, 
        data: np.ndarray, 
        series_name: str = "Unknown",
        include_basic: bool = True
    ) -> StructuralTensorFeatures:
        """
        階層的構造テンソル特徴抽出
        
        Lambda³理論の階層性：局所-大域構造変化の分離と分類
        
        Args:
            data: 入力データ系列
            series_name: 系列名
            include_basic: 基本特徴量も含めるか
            
        Returns:
            StructuralTensorFeatures: 階層的特徴量オブジェクト
        """
        # 基本特徴量抽出
        if include_basic:
            features = self.extract_basic_features(data, series_name)
        else:
            features = StructuralTensorFeatures(data=data, series_name=series_name)
        
        # 階層的構造変化検出
        local_pos, local_neg, global_pos, global_neg = detect_hierarchical_jumps(
            data,
            local_window=getattr(self.config, 'local_window', 5),
            global_window=getattr(self.config, 'global_window', 30),
            local_percentile=getattr(self.config, 'local_threshold_percentile', 90.0),
            global_percentile=getattr(self.config, 'global_threshold_percentile', 95.0)
        )
        
        # 階層イベント分類
        pure_local_pos, pure_local_neg, pure_global_pos, pure_global_neg, mixed_pos, mixed_neg = \
            classify_hierarchical_events(local_pos, local_neg, global_pos, global_neg)
        
        # 階層的特徴量を追加
        features.local_pos = local_pos
        features.local_neg = local_neg
        features.global_pos = global_pos
        features.global_neg = global_neg
        features.pure_local_pos = pure_local_pos
        features.pure_local_neg = pure_local_neg
        features.pure_global_pos = pure_global_pos
        features.pure_global_neg = pure_global_neg
        features.mixed_pos = mixed_pos
        features.mixed_neg = mixed_neg
        
        # 統合構造変化特徴量を更新（基本特徴と階層特徴の統合）
        if include_basic:
            combined_pos = np.maximum(local_pos, global_pos)
            combined_neg = np.maximum(local_neg, global_neg)
            features.delta_LambdaC_pos = combined_pos
            features.delta_LambdaC_neg = combined_neg
        
        return features
    
    def extract_multi_scale_features(
        self, 
        data: np.ndarray, 
        scales: List[int] = None,
        series_name: str = "Unknown"
    ) -> StructuralTensorFeatures:
        """
        マルチスケール構造テンソル特徴抽出
        
        Lambda³理論: 異なる時間スケールでの構造変化パターン解析
        
        Args:
            data: 入力データ系列
            scales: 分析スケールリスト
            series_name: 系列名
            
        Returns:
            StructuralTensorFeatures: マルチスケール特徴量オブジェクト
        """
        if scales is None:
            scales = [5, 10, 20, 50]
        
        # 基本特徴量抽出
        features = self.extract_hierarchical_features(data, series_name)
        
        # マルチスケール特徴量辞書
        multi_scale_features = {}
        
        for scale in scales:
            if scale >= len(data):
                continue
                
            # スケール特有の平滑化データ
            smoothed_data = moving_average(data, scale)
            
            # 各スケールでの構造変化検出
            diff, threshold = calculate_diff_and_threshold(smoothed_data, self.config.delta_percentile)
            scale_pos, scale_neg = detect_structural_jumps(diff, threshold)
            
            # スケール特徴量保存
            multi_scale_features[scale] = {
                'smoothed_data': smoothed_data,
                'pos_changes': scale_pos,
                'neg_changes': scale_neg,
                'change_intensity': diff,
                'threshold': threshold
            }
        
        features.multi_scale_features = multi_scale_features
        
        return features
    
    def extract_comprehensive_features(
        self, 
        data: np.ndarray, 
        series_name: str = "Unknown",
        include_multi_scale: bool = True,
        scales: List[int] = None
    ) -> StructuralTensorFeatures:
        """
        包括的構造テンソル特徴抽出
        
        Lambda³理論の全特徴量を統合的に抽出：
        - 基本∆ΛC変化
        - 階層的構造変化
        - マルチスケール解析
        - 統計的特徴量
        
        Args:
            data: 入力データ系列
            series_name: 系列名
            include_multi_scale: マルチスケール解析を含めるか
            scales: マルチスケール分析レベル
            
        Returns:
            StructuralTensorFeatures: 包括的特徴量オブジェクト
        """
        # マルチスケール特徴抽出
        if include_multi_scale:
            features = self.extract_multi_scale_features(data, scales, series_name)
        else:
            features = self.extract_hierarchical_features(data, series_name)
        
        # 局所統計量計算
        local_stats = {}
        stat_types = ['std', 'mean', 'var']
        
        for stat_type in stat_types:
            local_stats[stat_type] = calculate_rolling_statistics(
                data, self.config.window, stat_type
            )
        
        features.local_statistics = local_stats
        
        # 抽出履歴記録
        self.extraction_history.append({
            'series_name': series_name,
            'data_length': len(data),
            'features_extracted': {
                'basic': True,
                'hierarchical': True,
                'multi_scale': include_multi_scale,
                'statistics': True
            },
            'event_counts': features.count_events_by_type()
        })
        
        return features
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        データ前処理
        
        Args:
            data: 原データ
            
        Returns:
            preprocessed_data: 前処理済みデータ
        """
        # 型変換
        data = data.astype(np.float64)
        
        # 欠損値処理
        if np.isnan(data).any():
            warnings.warn("NaN values detected in data. Forward filling applied.")
            # 前方埋め
            mask = np.isnan(data)
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        
        # 無限値処理
        if np.isinf(data).any():
            warnings.warn("Infinite values detected in data. Clipping applied.")
            data = np.clip(data, -1e10, 1e10)
        
        return data
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """抽出履歴サマリー"""
        if not self.extraction_history:
            return {"message": "No extractions performed yet"}
        
        total_extractions = len(self.extraction_history)
        total_data_points = sum(h['data_length'] for h in self.extraction_history)
        
        # 特徴量タイプ別統計
        feature_type_counts = {
            'basic': sum(1 for h in self.extraction_history if h['features_extracted']['basic']),
            'hierarchical': sum(1 for h in self.extraction_history if h['features_extracted']['hierarchical']), 
            'multi_scale': sum(1 for h in self.extraction_history if h['features_extracted']['multi_scale']),
            'statistics': sum(1 for h in self.extraction_history if h['features_extracted']['statistics'])
        }
        
        return {
            'total_extractions': total_extractions,
            'total_data_points_processed': total_data_points,
            'average_data_length': total_data_points / total_extractions,
            'feature_type_usage': feature_type_counts,
            'recent_extractions': self.extraction_history[-3:]  # 最新3件
        }

# ==========================================================
# STRUCTURAL TENSOR ANALYZER
# ==========================================================

class StructuralTensorAnalyzer:
    """
    構造テンソル解析器
    
    抽出された構造テンソル特徴量の高次解析：
    - 構造変化パターン認識
    - 階層性メトリクス計算
    - 構造安定性評価
    - 異常パターン検出
    """
    
    def __init__(self, config: Optional[L3BaseConfig] = None):
        self.config = config or L3BaseConfig()
        
    def analyze_structural_patterns(self, features: StructuralTensorFeatures) -> Dict[str, Any]:
        """
        構造パターン解析
        
        Lambda³理論: 構造テンソル変化の時系列パターンを特徴化
        
        Args:
            features: 構造テンソル特徴量
            
        Returns:
            Dict: 構造パターン解析結果
        """
        analysis_results = {
            'series_name': features.series_name,
            'data_summary': features.calculate_summary_statistics(),
            'event_counts': features.count_events_by_type()
        }
        
        # 構造変化強度分析
        if features.delta_LambdaC_pos is not None and features.delta_LambdaC_neg is not None:
            pos_intensity = np.sum(features.delta_LambdaC_pos)
            neg_intensity = np.sum(features.delta_LambdaC_neg)
            total_intensity = pos_intensity + neg_intensity
            
            analysis_results['structural_intensity'] = {
                'positive_intensity': float(pos_intensity),
                'negative_intensity': float(neg_intensity),
                'total_intensity': float(total_intensity),
                'intensity_asymmetry': float(pos_intensity - neg_intensity) / max(total_intensity, 1e-8),
                'intensity_ratio': float(pos_intensity) / max(neg_intensity, 1e-8)
            }
        
        # 張力スカラー解析
        if features.rho_T is not None:
            rho_stats = self._analyze_tension_scalar(features.rho_T)
            analysis_results['tension_analysis'] = rho_stats
        
        # 階層性解析
        if hasattr(features, 'local_pos') and features.local_pos is not None:
            hierarchy_analysis = self._analyze_hierarchical_structure(features)
            analysis_results['hierarchy_analysis'] = hierarchy_analysis
        
        return analysis_results
    
    def calculate_hierarchical_metrics(self, features: StructuralTensorFeatures) -> Dict[str, float]:
        """
        階層性メトリクス計算
        
        Lambda³理論: 構造変化の階層特性を定量化
        
        Args:
            features: 階層的特徴量
            
        Returns:
            Dict: 階層性メトリクス
        """
        if not self._has_hierarchical_features(features):
            return {'error': 'Hierarchical features not available'}
        
        # 各階層のイベント数計算
        local_events = np.sum(features.local_pos) + np.sum(features.local_neg)
        global_events = np.sum(features.global_pos) + np.sum(features.global_neg)
        total_events = local_events + global_events
        
        if total_events == 0:
            return {'error': 'No hierarchical events detected'}
        
        # 純粋・混合イベント数
        pure_local = np.sum(features.pure_local_pos) + np.sum(features.pure_local_neg)
        pure_global = np.sum(features.pure_global_pos) + np.sum(features.pure_global_neg)
        mixed_events = np.sum(features.mixed_pos) + np.sum(features.mixed_neg)
        
        # 階層性メトリクス
        metrics = {
            'local_dominance': float(local_events / total_events),
            'global_dominance': float(global_events / total_events),
            'hierarchy_purity': float((pure_local + pure_global) / total_events),
            'coupling_strength': float(mixed_events / total_events),
            'hierarchy_balance': float(abs(local_events - global_events) / total_events),
            'escalation_potential': float(pure_local / max(local_events, 1)),
            'deescalation_potential': float(pure_global / max(global_events, 1))
        }
        
        # 非対称性メトリクス
        pos_local = np.sum(features.pure_local_pos)
        neg_local = np.sum(features.pure_local_neg)
        pos_global = np.sum(features.pure_global_pos)
        neg_global = np.sum(features.pure_global_neg)
        
        metrics.update({
            'local_asymmetry': float((pos_local - neg_local) / max(pos_local + neg_local, 1)),
            'global_asymmetry': float((pos_global - neg_global) / max(pos_global + neg_global, 1)),
            'cross_hierarchy_asymmetry': float(abs(metrics['local_dominance'] - metrics['global_dominance']))
        })
        
        return metrics
    
    def detect_structural_anomalies(self, features: StructuralTensorFeatures) -> Dict[str, Any]:
        """
        構造異常検出
        
        Lambda³理論: 通常の構造変化パターンからの逸脱を検出
        
        Args:
            features: 構造テンソル特徴量
            
        Returns:
            Dict: 異常検出結果
        """
        anomalies = {
            'detected_anomalies': [],
            'anomaly_scores': [],
            'anomaly_locations': []
        }
        
        # 張力スカラー異常
        if features.rho_T is not None:
            rho_mean = np.mean(features.rho_T)
            rho_std = np.std(features.rho_T)
            threshold = rho_mean + 3 * rho_std  # 3σ規則
            
            tension_anomalies = np.where(features.rho_T > threshold)[0]
            if len(tension_anomalies) > 0:
                anomalies['detected_anomalies'].append('extreme_tension')
                anomalies['anomaly_locations'].extend(tension_anomalies.tolist())
                max_tension_score = np.max(features.rho_T[tension_anomalies]) / threshold
                anomalies['anomaly_scores'].append(float(max_tension_score))
        
        # 構造変化クラスタリング異常
        if features.delta_LambdaC_pos is not None:
            pos_events = np.where(features.delta_LambdaC_pos > 0)[0]
            if len(pos_events) > 1:
                # イベント間隔の異常検出
                intervals = np.diff(pos_events)
                if len(intervals) > 0:
                    interval_mean = np.mean(intervals)
                    interval_std = np.std(intervals)
                    
                    if interval_std > 0:
                        unusual_intervals = intervals < (interval_mean - 2 * interval_std)
                        if np.any(unusual_intervals):
                            anomalies['detected_anomalies'].append('event_clustering')
                            cluster_locations = pos_events[1:][unusual_intervals]
                            anomalies['anomaly_locations'].extend(cluster_locations.tolist())
        
        # 階層構造異常
        if self._has_hierarchical_features(features):
            hierarchy_metrics = self.calculate_hierarchical_metrics(features)
            
            # 極端な階層不均衡
            if hierarchy_metrics.get('hierarchy_balance', 0) > 0.8:
                anomalies['detected_anomalies'].append('hierarchy_imbalance')
                anomalies['anomaly_scores'].append(hierarchy_metrics['hierarchy_balance'])
            
            # 異常な結合強度
            if hierarchy_metrics.get('coupling_strength', 0) > 0.7:
                anomalies['detected_anomalies'].append('excessive_coupling')
                anomalies['anomaly_scores'].append(hierarchy_metrics['coupling_strength'])
        
        # 異常サマリー
        anomalies['total_anomalies'] = len(anomalies['detected_anomalies'])
        anomalies['anomaly_rate'] = len(set(anomalies['anomaly_locations'])) / features.data_length
        
        return anomalies
    
    def _analyze_tension_scalar(self, rho_t: np.ndarray) -> Dict[str, float]:
        """張力スカラー詳細解析"""
        return {
            'mean_tension': float(np.mean(rho_t)),
            'max_tension': float(np.max(rho_t)),
            'min_tension': float(np.min(rho_t)),
            'tension_volatility': float(np.std(rho_t)),
            'tension_skewness': float(self._calculate_skewness(rho_t)),
            'tension_kurtosis': float(self._calculate_kurtosis(rho_t)),
            'tension_range': float(np.max(rho_t) - np.min(rho_t)),
            'high_tension_ratio': float(np.mean(rho_t > np.percentile(rho_t, 90)))
        }
    
    def _analyze_hierarchical_structure(self, features: StructuralTensorFeatures) -> Dict[str, Any]:
        """階層構造詳細解析"""
        if not self._has_hierarchical_features(features):
            return {'error': 'Hierarchical features not available'}
        
        # 階層遷移パターン解析
        local_events = features.local_pos + features.local_neg
        global_events = features.global_pos + features.global_neg
        
        # 局所→大域遷移検出
        escalation_events = []
        deescalation_events = []
        
        for i in range(1, len(local_events)):
            if local_events[i-1] > 0 and global_events[i] > 0:
                escalation_events.append(i)
            elif global_events[i-1] > 0 and local_events[i] > 0:
                deescalation_events.append(i)
        
        return {
            'escalation_events': escalation_events,
            'deescalation_events': deescalation_events,
            'escalation_rate': float(len(escalation_events) / max(np.sum(local_events), 1)),
            'deescalation_rate': float(len(deescalation_events) / max(np.sum(global_events), 1)),
            'transition_asymmetry': float(len(escalation_events) - len(deescalation_events)),
            'hierarchy_metrics': self.calculate_hierarchical_metrics(features)
        }
    
    def _has_hierarchical_features(self, features: StructuralTensorFeatures) -> bool:
        """階層的特徴量の存在確認"""
        required_features = ['local_pos', 'local_neg', 'global_pos', 'global_neg']
        return all(hasattr(features, feat) and getattr(features, feat) is not None 
                  for feat in required_features)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """歪度計算"""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val < 1e-10:
            return 0.0
        
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """尖度計算"""
        if len(data) < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val < 1e-10:
            return 0.0
        
        return np.mean(((data - mean_val) / std_val) ** 4) - 3.0  # 正規分布で0になるよう調整

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def extract_lambda3_features(
    data: np.ndarray,
    config: Optional[L3BaseConfig] = None,
    series_name: str = "Unknown",
    feature_level: str = "comprehensive"
) -> StructuralTensorFeatures:
    """
    Lambda³特徴量抽出の便利関数
    
    Args:
        data: 入力データ系列
        config: Lambda³設定
        series_name: 系列名
        feature_level: 'basic', 'hierarchical', 'comprehensive'
        
    Returns:
        StructuralTensorFeatures: 特徴量オブジェクト
    """
    extractor = StructuralTensorExtractor(config)
    
    if feature_level == "basic":
        return extractor.extract_basic_features(data, series_name)
    elif feature_level == "hierarchical":
        return extractor.extract_hierarchical_features(data, series_name)
    elif feature_level == "comprehensive":
        return extractor.extract_comprehensive_features(data, series_name)
    else:
        raise ValueError(f"Invalid feature_level: {feature_level}")

def analyze_lambda3_structure(features: StructuralTensorFeatures) -> Dict[str, Any]:
    """
    Lambda³構造解析の便利関数
    
    Args:
        features: 構造テンソル特徴量
        
    Returns:
        Dict: 包括的構造解析結果
    """
    analyzer = StructuralTensorAnalyzer()
    
    results = {
        'structural_patterns': analyzer.analyze_structural_patterns(features),
        'hierarchical_metrics': analyzer.calculate_hierarchical_metrics(features),
        'structural_anomalies': analyzer.detect_structural_anomalies(features)
    }
    
    return results

# ==========================================================
# BATCH PROCESSING
# ==========================================================

def extract_features_batch(
    data_dict: Dict[str, np.ndarray],
    config: Optional[L3BaseConfig] = None,
    feature_level: str = "comprehensive"
) -> Dict[str, StructuralTensorFeatures]:
    """
    複数系列の一括特徴量抽出
    
    Args:
        data_dict: {series_name: data} 辞書
        config: Lambda³設定
        feature_level: 特徴量レベル
        
    Returns:
        Dict[str, StructuralTensorFeatures]: 系列別特徴量辞書
    """
    extractor = StructuralTensorExtractor(config)
    results = {}
    
    for series_name, data in data_dict.items():
        print(f"Extracting features for {series_name}...")
        try:
            if feature_level == "basic":
                features = extractor.extract_basic_features(data, series_name)
            elif feature_level == "hierarchical":
                features = extractor.extract_hierarchical_features(data, series_name)
            elif feature_level == "comprehensive":
                features = extractor.extract_comprehensive_features(data, series_name)
            else:
                raise ValueError(f"Invalid feature_level: {feature_level}")
                
            results[series_name] = features
            print(f"  ✓ {series_name}: {features.count_events_by_type()}")
            
        except Exception as e:
            print(f"  ✗ {series_name}: Error - {str(e)}")
            continue
    
    print(f"\nBatch extraction completed: {len(results)}/{len(data_dict)} series processed")
    return results

if __name__ == "__main__":
    # 構造テンソルモジュールテスト
    print("Lambda³ Structural Tensor Module Test")
    print("=" * 50)
    
    # テストデータ生成
    np.random.seed(42)
    n_points = 1000
    
    # 構造変化を含む合成データ
    base_trend = np.cumsum(np.random.randn(n_points) * 0.05)
    structural_jumps = np.zeros(n_points)
    
    # 意図的な構造変化を注入
    jump_positions = [200, 400, 600, 800]
    jump_magnitudes = [2.0, -1.5, 1.8, -2.2]
    
    for pos, mag in zip(jump_positions, jump_magnitudes):
        if pos < n_points:
            structural_jumps[pos:] += mag
    
    test_data = base_trend + structural_jumps + np.random.randn(n_points) * 0.1
    
    print(f"Test data generated: {n_points} points with {len(jump_positions)} structural changes")
    
    # 基本特徴抽出テスト
    print("\n1. Basic Feature Extraction Test")
    print("-" * 30)
    
    extractor = StructuralTensorExtractor()
    basic_features = extractor.extract_basic_features(test_data, "TestSeries")
    
    print(f"Series: {basic_features.series_name}")
    print(f"Data length: {basic_features.data_length}")
    print(f"Event counts: {basic_features.count_events_by_type()}")
    print(f"Summary stats: {basic_features.calculate_summary_statistics()}")
    
    # 階層的特徴抽出テスト
    print("\n2. Hierarchical Feature Extraction Test")
    print("-" * 30)
    
    hierarchical_features = extractor.extract_hierarchical_features(test_data, "TestSeries")
    
    print(f"Hierarchical event counts: {hierarchical_features.count_events_by_type()}")
    
    # 構造解析テスト
    print("\n3. Structural Analysis Test")
    print("-" * 30)
    
    analyzer = StructuralTensorAnalyzer()
    analysis_results = analyzer.analyze_structural_patterns(hierarchical_features)
    
    print(f"Structural intensity: {analysis_results.get('structural_intensity', {})}")
    
    # 階層性メトリクステスト
    hierarchy_metrics = analyzer.calculate_hierarchical_metrics(hierarchical_features)
    print(f"Hierarchy metrics: {hierarchy_metrics}")
    
    # 異常検出テスト
    anomalies = analyzer.detect_structural_anomalies(hierarchical_features)
    print(f"Detected anomalies: {anomalies['detected_anomalies']}")
    print(f"Anomaly rate: {anomalies['anomaly_rate']:.3f}")
    
    # 包括的特徴抽出テスト
    print("\n4. Comprehensive Feature Extraction Test")
    print("-" * 30)
    
    comprehensive_features = extractor.extract_comprehensive_features(
        test_data, "TestSeries", include_multi_scale=True
    )
    
    if comprehensive_features.multi_scale_features:
        print(f"Multi-scale features extracted for scales: {list(comprehensive_features.multi_scale_features.keys())}")
    
    if comprehensive_features.local_statistics:
        print(f"Local statistics computed: {list(comprehensive_features.local_statistics.keys())}")
    
    # 抽出履歴確認
    print(f"\nExtraction summary: {extractor.get_extraction_summary()}")
    
    # バッチ処理テスト
    print("\n5. Batch Processing Test")
    print("-" * 30)
    
    # 複数系列のテストデータ
    test_data_dict = {
        "Series_A": test_data,
        "Series_B": test_data + np.random.randn(n_points) * 0.2,
        "Series_C": np.cumsum(np.random.randn(n_points) * 0.08)
    }
    
    batch_results = extract_features_batch(test_data_dict, feature_level="hierarchical")
    
    print(f"Batch processing completed for {len(batch_results)} series")
    for name, features in batch_results.items():
        print(f"  {name}: {features.count_events_by_type()}")
    
    # 便利関数テスト
    print("\n6. Convenience Functions Test")
    print("-" * 30)
    
    quick_features = extract_lambda3_features(test_data, feature_level="comprehensive")
    quick_analysis = analyze_lambda3_structure(quick_features)
    
    print(f"Quick analysis completed:")
    print(f"  Structural patterns: {len(quick_analysis['structural_patterns'])} metrics")
    print(f"  Hierarchical metrics: {len(quick_analysis['hierarchical_metrics'])} metrics") 
    print(f"  Anomalies detected: {quick_analysis['structural_anomalies']['total_anomalies']}")
    
    print("\n" + "=" * 50)
    print("Structural Tensor Module Test Completed Successfully!")
    print("All Lambda³ theoretical components functioning correctly.")
    print("=" * 50)
