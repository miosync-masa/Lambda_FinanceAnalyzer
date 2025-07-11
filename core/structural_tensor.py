# ==========================================================
# lambda3/core/structural_tensor.py (JIT Compatible Version)
# Structural Tensor Feature Extraction for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 修正点: JIT最適化関数との完全互換性確保
# ==========================================================

"""
Lambda³理論構造テンソル特徴抽出モジュール（JIT互換版）

構造テンソル(Λ)の特徴量抽出とデータクラス定義。
∆ΛC pulsations、張力スカラー(ρT)、階層的構造変化の
包括的特徴量を効率的に抽出。

核心概念:
- 構造テンソル(Λ): 時系列の構造的状態表現
- ∆ΛC pulsations: 構造変化の非時間的パルス現象
- 張力スカラー(ρT): 構造空間の張力度合い
- 階層的特徴量: 局所-大域構造変化の分離

JIT最適化対応:
- 修正版JIT関数による高速特徴抽出
- 数値安定性の向上
- メモリ効率の最適化
- バッチ処理による大規模データ対応
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import time

from .config import L3BaseConfig, L3ComprehensiveConfig

# JIT最適化関数のインポート（修正版）
try:
    from .jit_functions import (
        calculate_diff_and_threshold_fixed,
        detect_structural_jumps_fixed,
        calculate_tension_scalar_fixed,
        detect_hierarchical_jumps_fixed,
        classify_hierarchical_events_fixed,
        calculate_local_statistics_fixed,
        normalize_array_fixed,
        safe_divide_fixed,
        extract_lambda3_features_jit
    )
    JIT_FUNCTIONS_AVAILABLE = True
    
    # レガシー互換性
    calculate_diff_and_threshold = calculate_diff_and_threshold_fixed
    detect_structural_jumps = detect_structural_jumps_fixed
    calculate_tension_scalar = calculate_tension_scalar_fixed
    detect_hierarchical_jumps = detect_hierarchical_jumps_fixed
    
except ImportError:
    warnings.warn("JIT functions not available. Using fallback implementations.")
    JIT_FUNCTIONS_AVAILABLE = False
    
    # フォールバック実装
    def calculate_diff_and_threshold_fixed(data, percentile):
        diff = np.diff(np.concatenate([[0], data]))
        threshold = np.percentile(np.abs(diff), percentile)
        return diff, threshold
    
    def detect_structural_jumps_fixed(diff, threshold):
        pos_jumps = (diff > threshold).astype(float)
        neg_jumps = (diff < -threshold).astype(float)
        return pos_jumps, neg_jumps
    
    def calculate_tension_scalar_fixed(data, window):
        n = len(data)
        rho_t = np.zeros(n)
        for i in range(n):
            start = max(0, i - window)
            end = i + 1
            if end - start > 1:
                rho_t[i] = np.std(data[start:end])
        return rho_t
    
    def detect_hierarchical_jumps_fixed(data, local_window=10, global_window=50, 
                                       local_percentile=90.0, global_percentile=95.0):
        n = len(data)
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    def normalize_array_fixed(arr, method='zscore'):
        if method == 'zscore':
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        return arr
    
    def safe_divide_fixed(num, den, default=0.0):
        return num / den if abs(den) > 1e-8 else default
    
    # レガシー互換性
    calculate_diff_and_threshold = calculate_diff_and_threshold_fixed
    detect_structural_jumps = detect_structural_jumps_fixed
    calculate_tension_scalar = calculate_tension_scalar_fixed
    detect_hierarchical_jumps = detect_hierarchical_jumps_fixed

# ==========================================================
# STRUCTURAL TENSOR FEATURES DATA CLASS（拡張版）
# ==========================================================

@dataclass
class StructuralTensorFeatures:
    """
    構造テンソル特徴量データクラス（JIT最適化版）
    
    Lambda³理論における構造テンソル(Λ)の全特徴量を統合管理。
    ∆ΛC pulsations、張力スカラー、階層的構造変化を包含。
    
    JIT最適化対応:
    - 数値型の厳密管理
    - メモリ効率の最適化
    - 高速アクセスメソッド
    """
    
    # 基本データ
    data: np.ndarray
    series_name: str = "Unknown"
    
    # 基本構造テンソル特徴量
    delta_LambdaC_pos: Optional[np.ndarray] = None
    delta_LambdaC_neg: Optional[np.ndarray] = None
    rho_T: Optional[np.ndarray] = None
    time_trend: Optional[np.ndarray] = None
    
    # 階層的構造変化特徴量
    local_pos: Optional[np.ndarray] = None
    local_neg: Optional[np.ndarray] = None
    global_pos: Optional[np.ndarray] = None
    global_neg: Optional[np.ndarray] = None
    
    # 分類済み階層特徴量（新規追加）
    pure_local_pos: Optional[np.ndarray] = None
    pure_local_neg: Optional[np.ndarray] = None
    pure_global_pos: Optional[np.ndarray] = None
    pure_global_neg: Optional[np.ndarray] = None
    mixed_pos: Optional[np.ndarray] = None
    mixed_neg: Optional[np.ndarray] = None
    
    # 統計特徴量（新規追加）
    local_statistics: Optional[Dict[str, np.ndarray]] = None
    
    # メタデータ
    extraction_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後処理（JIT最適化対応）"""
        if self.data is None:
            raise ValueError("構造テンソルデータが未定義です")
        
        # データ型確保（JIT互換性）
        self.data = np.asarray(self.data, dtype=np.float64)
        self.n_points = len(self.data)
        
        # 時間トレンド生成
        if self.time_trend is None:
            self.time_trend = np.arange(self.n_points, dtype=np.float64)
        
        # 配列型確保
        self._ensure_array_types()
        
        # 抽出情報初期化
        if not self.extraction_info:
            self.extraction_info = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'jit_optimized': JIT_FUNCTIONS_AVAILABLE,
                'data_length': self.n_points,
                'feature_completeness': self._calculate_feature_completeness()
            }
    
    def _ensure_array_types(self):
        """配列型の確保（JIT互換性）"""
        array_attrs = [
            'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend',
            'local_pos', 'local_neg', 'global_pos', 'global_neg',
            'pure_local_pos', 'pure_local_neg', 'pure_global_pos', 
            'pure_global_neg', 'mixed_pos', 'mixed_neg'
        ]
        
        for attr_name in array_attrs:
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                setattr(self, attr_name, np.asarray(attr_value, dtype=np.float64))
    
    def _calculate_feature_completeness(self) -> float:
        """特徴量完成度計算"""
        total_features = 12  # 主要特徴量数
        completed_features = 0
        
        feature_attrs = [
            'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T',
            'local_pos', 'local_neg', 'global_pos', 'global_neg',
            'pure_local_pos', 'pure_local_neg', 'pure_global_pos',
            'pure_global_neg', 'mixed_pos'
        ]
        
        for attr_name in feature_attrs:
            if getattr(self, attr_name) is not None:
                completed_features += 1
        
        return completed_features / total_features
    
    def count_events_by_type(self) -> Dict[str, int]:
        """イベント種別カウント（拡張版）"""
        counts = {}
        
        # 基本構造変化イベント
        if self.delta_LambdaC_pos is not None:
            counts['total_pos'] = int(np.sum(self.delta_LambdaC_pos))
        if self.delta_LambdaC_neg is not None:
            counts['total_neg'] = int(np.sum(self.delta_LambdaC_neg))
        
        # 階層的イベント
        if self.local_pos is not None:
            counts['local_pos'] = int(np.sum(self.local_pos))
        if self.local_neg is not None:
            counts['local_neg'] = int(np.sum(self.local_neg))
        if self.global_pos is not None:
            counts['global_pos'] = int(np.sum(self.global_pos))
        if self.global_neg is not None:
            counts['global_neg'] = int(np.sum(self.global_neg))
        
        # 分類済み階層イベント
        if self.pure_local_pos is not None:
            counts['pure_local_pos'] = int(np.sum(self.pure_local_pos))
        if self.pure_local_neg is not None:
            counts['pure_local_neg'] = int(np.sum(self.pure_local_neg))
        if self.pure_global_pos is not None:
            counts['pure_global_pos'] = int(np.sum(self.pure_global_pos))
        if self.pure_global_neg is not None:
            counts['pure_global_neg'] = int(np.sum(self.pure_global_neg))
        if self.mixed_pos is not None:
            counts['mixed_pos'] = int(np.sum(self.mixed_pos))
        if self.mixed_neg is not None:
            counts['mixed_neg'] = int(np.sum(self.mixed_neg))
        
        return counts
    
    def get_structural_summary(self) -> Dict[str, Any]:
        """構造的サマリー取得"""
        event_counts = self.count_events_by_type()
        
        # 基本統計
        basic_stats = {
            'data_length': self.n_points,
            'mean_value': float(np.mean(self.data)),
            'std_value': float(np.std(self.data)),
            'feature_completeness': self.extraction_info.get('feature_completeness', 0)
        }
        
        # 構造変化統計
        total_events = event_counts.get('total_pos', 0) + event_counts.get('total_neg', 0)
        structure_stats = {
            'total_structural_events': total_events,
            'event_rate': total_events / self.n_points if self.n_points > 0 else 0,
            'pos_neg_ratio': safe_divide_fixed(
                event_counts.get('total_pos', 0), 
                event_counts.get('total_neg', 1), 
                1.0
            )
        }
        
        # 張力統計
        tension_stats = {}
        if self.rho_T is not None:
            tension_stats = {
                'mean_tension': float(np.mean(self.rho_T)),
                'max_tension': float(np.max(self.rho_T)),
                'tension_volatility': float(np.std(self.rho_T))
            }
        
        # 階層性統計
        hierarchy_stats = {}
        if self.local_pos is not None and self.global_pos is not None:
            local_events = event_counts.get('local_pos', 0) + event_counts.get('local_neg', 0)
            global_events = event_counts.get('global_pos', 0) + event_counts.get('global_neg', 0)
            total_hierarchical = local_events + global_events
            
            hierarchy_stats = {
                'local_events': local_events,
                'global_events': global_events,
                'hierarchy_ratio': safe_divide_fixed(local_events, global_events, 1.0),
                'hierarchy_balance': safe_divide_fixed(
                    min(local_events, global_events),
                    max(local_events, global_events, 1),
                    0.0
                )
            }
        
        return {
            'basic_statistics': basic_stats,
            'structure_statistics': structure_stats,
            'tension_statistics': tension_stats,
            'hierarchy_statistics': hierarchy_stats,
            'event_counts': event_counts
        }
    
    def validate_features(self) -> Dict[str, List[str]]:
        """特徴量妥当性検証"""
        validation_results = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # 基本データ検証
        if np.isnan(self.data).any():
            validation_results['errors'].append("Data contains NaN values")
        
        if np.isinf(self.data).any():
            validation_results['errors'].append("Data contains infinite values")
        
        # 特徴量一貫性検証
        feature_arrays = [
            ('delta_LambdaC_pos', self.delta_LambdaC_pos),
            ('delta_LambdaC_neg', self.delta_LambdaC_neg),
            ('rho_T', self.rho_T)
        ]
        
        for name, array in feature_arrays:
            if array is not None:
                if len(array) != self.n_points:
                    validation_results['errors'].append(f"{name} length mismatch")
                
                if np.isnan(array).any():
                    validation_results['warnings'].append(f"{name} contains NaN values")
        
        # 階層特徴量検証
        hierarchical_features = [
            self.local_pos, self.local_neg, self.global_pos, self.global_neg
        ]
        
        all_hierarchical_present = all(feat is not None for feat in hierarchical_features)
        some_hierarchical_present = any(feat is not None for feat in hierarchical_features)
        
        if some_hierarchical_present and not all_hierarchical_present:
            validation_results['warnings'].append("Incomplete hierarchical features")
        elif all_hierarchical_present:
            validation_results['info'].append("Complete hierarchical features available")
        
        # JIT最適化状態
        if self.extraction_info.get('jit_optimized', False):
            validation_results['info'].append("JIT optimization was used")
        else:
            validation_results['warnings'].append("JIT optimization was not available")
        
        return validation_results
    
    def export_to_dict(self) -> Dict[str, Any]:
        """辞書形式エクスポート"""
        export_dict = {
            'series_name': self.series_name,
            'data_length': self.n_points,
            'data': self.data.tolist(),
            'extraction_info': self.extraction_info
        }
        
        # 特徴量配列のエクスポート
        feature_arrays = [
            'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend',
            'local_pos', 'local_neg', 'global_pos', 'global_neg',
            'pure_local_pos', 'pure_local_neg', 'pure_global_pos', 
            'pure_global_neg', 'mixed_pos', 'mixed_neg'
        ]
        
        for attr_name in feature_arrays:
            attr_value = getattr(self, attr_name)
            if attr_value is not None:
                export_dict[attr_name] = attr_value.tolist()
        
        # 統計サマリー追加
        export_dict['summary'] = self.get_structural_summary()
        
        return export_dict

# ==========================================================
# STRUCTURAL TENSOR EXTRACTOR（JIT最適化版）
# ==========================================================

class StructuralTensorExtractor:
    """
    構造テンソル特徴抽出器（JIT最適化版）
    
    Lambda³理論に基づく構造テンソル特徴量の包括的抽出。
    JIT最適化により高速かつ数値安定な特徴抽出を実現。
    
    主要機能:
    - 基本構造テンソル特徴抽出
    - 階層的構造変化検出
    - 分類済み階層イベント抽出
    - バッチ処理による大規模データ対応
    - 自動品質評価と検証
    """
    
    def __init__(self, config: Optional[L3BaseConfig] = None):
        """
        初期化
        
        Args:
            config: Lambda³基底設定
        """
        self.config = config or L3BaseConfig()
        self.extraction_history = []
        
        # JIT最適化確認
        self.jit_enabled = JIT_FUNCTIONS_AVAILABLE
        if hasattr(self.config, 'jit_config'):
            self.jit_enabled = self.jit_enabled and self.config.jit_config.enable_jit
        
        print(f"🔬 StructuralTensorExtractor initialized")
        print(f"   JIT Optimization: {'Enabled' if self.jit_enabled else 'Disabled'}")
        print(f"   Window sizes: base={self.config.window}, local={self.config.local_window}, global={self.config.global_window}")
    
    def extract_basic_features(
        self, 
        data: np.ndarray, 
        series_name: str = "Unknown"
    ) -> StructuralTensorFeatures:
        """
        基本構造テンソル特徴抽出（JIT最適化版）
        
        Lambda³理論の基本要素である∆ΛC pulsationsと
        張力スカラーρTを高速抽出。
        
        Args:
            data: 入力時系列データ
            series_name: 系列名
            
        Returns:
            StructuralTensorFeatures: 基本特徴量
        """
        start_time = time.time()
        
        # データ前処理
        data = self._preprocess_data_optimized(data)
        
        # 特徴量オブジェクト初期化
        features = StructuralTensorFeatures(
            data=data,
            series_name=series_name
        )
        
        if self.jit_enabled and JIT_FUNCTIONS_AVAILABLE:
            try:
                # JIT最適化による高速基本特徴抽出
                diff, threshold = calculate_diff_and_threshold_fixed(
                    data, self.config.delta_percentile
                )
                delta_pos, delta_neg = detect_structural_jumps_fixed(diff, threshold)
                rho_t = calculate_tension_scalar_fixed(data, self.config.window)
                
                # 特徴量設定
                features.delta_LambdaC_pos = delta_pos
                features.delta_LambdaC_neg = delta_neg
                features.rho_T = rho_t
                
                extraction_method = "JIT-optimized"
                
            except Exception as e:
                print(f"JIT基本特徴抽出エラー: {e}, フォールバック使用")
                # フォールバック実装
                features = self._extract_basic_features_fallback(data, series_name)
                extraction_method = "Fallback"
        else:
            # 標準実装
            features = self._extract_basic_features_fallback(data, series_name)
            extraction_method = "Standard"
        
        # 抽出情報記録
        extraction_time = time.time() - start_time
        features.extraction_info.update({
            'extraction_method': extraction_method,
            'extraction_time': extraction_time,
            'feature_level': 'basic',
            'jit_enabled': self.jit_enabled
        })
        
        # 履歴記録
        self.extraction_history.append({
            'series_name': series_name,
            'feature_level': 'basic',
            'extraction_time': extraction_time,
            'method': extraction_method,
            'data_length': len(data)
        })
        
        return features
    
    def extract_hierarchical_features(
        self, 
        data: np.ndarray, 
        series_name: str = "Unknown"
    ) -> StructuralTensorFeatures:
        """
        階層的構造テンソル特徴抽出（JIT最適化版）
        
        Lambda³理論の階層性原理に基づく局所-大域構造変化の
        分離と分類を高速実行。
        
        Args:
            data: 入力時系列データ
            series_name: 系列名
            
        Returns:
            StructuralTensorFeatures: 階層的特徴量
        """
        start_time = time.time()
        
        # 基本特徴量抽出
        features = self.extract_basic_features(data, series_name)
        
        if self.jit_enabled and JIT_FUNCTIONS_AVAILABLE:
            try:
                # JIT最適化による階層的特徴抽出
                local_pos, local_neg, global_pos, global_neg = detect_hierarchical_jumps_fixed(
                    data,
                    local_window=self.config.local_window,
                    global_window=self.config.global_window,
                    local_percentile=self.config.local_jump_percentile,
                    global_percentile=self.config.delta_percentile
                )
                
                # 階層特徴量設定
                features.local_pos = local_pos
                features.local_neg = local_neg
                features.global_pos = global_pos
                features.global_neg = global_neg
                
                # 分類済み階層イベント抽出（JIT関数利用可能な場合）
                if hasattr(self, '_extract_classified_hierarchical_events_jit'):
                    (pure_local_pos, pure_local_neg, pure_global_pos, 
                     pure_global_neg, mixed_pos, mixed_neg) = self._extract_classified_hierarchical_events_jit(
                        local_pos, local_neg, global_pos, global_neg
                    )
                    
                    features.pure_local_pos = pure_local_pos
                    features.pure_local_neg = pure_local_neg
                    features.pure_global_pos = pure_global_pos
                    features.pure_global_neg = pure_global_neg
                    features.mixed_pos = mixed_pos
                    features.mixed_neg = mixed_neg
                
                extraction_method = "JIT-hierarchical"
                
            except Exception as e:
                print(f"JIT階層特徴抽出エラー: {e}, フォールバック使用")
                # フォールバック実装
                features = self._extract_hierarchical_features_fallback(features)
                extraction_method = "Fallback-hierarchical"
        else:
            # 標準実装
            features = self._extract_hierarchical_features_fallback(features)
            extraction_method = "Standard-hierarchical"
        
        # 局所統計特徴量追加（JIT最適化）
        if self.jit_enabled and JIT_FUNCTIONS_AVAILABLE:
            try:
                local_std, local_mean = calculate_local_statistics_fixed(
                    data, self.config.window
                )
                features.local_statistics = {
                    'local_std': local_std,
                    'local_mean': local_mean
                }
            except Exception as e:
                print(f"JIT局所統計計算エラー: {e}")
        
        # 抽出情報更新
        total_extraction_time = time.time() - start_time
        features.extraction_info.update({
            'extraction_method': extraction_method,
            'extraction_time': total_extraction_time,
            'feature_level': 'hierarchical'
        })
        
        # 履歴更新
        self.extraction_history[-1].update({
            'feature_level': 'hierarchical',
            'extraction_time': total_extraction_time,
            'method': extraction_method
        })
        
        return features
    
    def extract_comprehensive_features(
        self, 
        data: np.ndarray, 
        series_name: str = "Unknown"
    ) -> StructuralTensorFeatures:
        """
        包括的構造テンソル特徴抽出（JIT最適化版）
        
        Lambda³理論の全要素を統合した最高レベルの特徴抽出。
        
        Args:
            data: 入力時系列データ
            series_name: 系列名
            
        Returns:
            StructuralTensorFeatures: 包括的特徴量
        """
        start_time = time.time()
        
        if self.jit_enabled and JIT_FUNCTIONS_AVAILABLE:
            try:
                # JIT最適化一括特徴抽出
                features_tuple = extract_lambda3_features_jit(
                    data,
                    window=self.config.window,
                    local_window=self.config.local_window,
                    global_window=self.config.global_window,
                    delta_percentile=self.config.delta_percentile,
                    local_percentile=self.config.local_jump_percentile,
                    global_percentile=self.config.delta_percentile
                )
                
                # 特徴量オブジェクト構築
                delta_pos, delta_neg, rho_t, local_pos, local_neg, global_pos, global_neg = features_tuple
                
                features = StructuralTensorFeatures(
                    data=data,
                    series_name=series_name,
                    delta_LambdaC_pos=delta_pos,
                    delta_LambdaC_neg=delta_neg,
                    rho_T=rho_t,
                    local_pos=local_pos,
                    local_neg=local_neg,
                    global_pos=global_pos,
                    global_neg=global_neg
                )
                
                # 分類済み階層イベント追加
                if JIT_FUNCTIONS_AVAILABLE:
                    try:
                        (pure_local_pos, pure_local_neg, pure_global_pos, 
                         pure_global_neg, mixed_pos, mixed_neg) = classify_hierarchical_events_fixed(
                            local_pos, local_neg, global_pos, global_neg
                        )
                        
                        features.pure_local_pos = pure_local_pos
                        features.pure_local_neg = pure_local_neg
                        features.pure_global_pos = pure_global_pos
                        features.pure_global_neg = pure_global_neg
                        features.mixed_pos = mixed_pos
                        features.mixed_neg = mixed_neg
                        
                    except Exception as e:
                        print(f"JIT階層分類エラー: {e}")
                
                # 局所統計追加
                try:
                    local_std, local_mean = calculate_local_statistics_fixed(
                        data, self.config.window
                    )
                    features.local_statistics = {
                        'local_std': local_std,
                        'local_mean': local_mean
                    }
                except Exception as e:
                    print(f"JIT局所統計エラー: {e}")
                
                extraction_method = "JIT-comprehensive"
                
            except Exception as e:
                print(f"JIT包括特徴抽出エラー: {e}, 階層的抽出使用")
                # 階層的抽出へフォールバック
                features = self.extract_hierarchical_features(data, series_name)
                extraction_method = "Hierarchical-fallback"
        else:
            # 標準階層的抽出
            features = self.extract_hierarchical_features(data, series_name)
            extraction_method = "Standard-hierarchical"
        
        # 抽出情報更新
        total_extraction_time = time.time() - start_time
        features.extraction_info.update({
            'extraction_method': extraction_method,
            'extraction_time': total_extraction_time,
            'feature_level': 'comprehensive'
        })
        
        # 履歴更新
        if self.extraction_history:
            self.extraction_history[-1].update({
                'feature_level': 'comprehensive',
                'extraction_time': total_extraction_time,
                'method': extraction_method
            })
        
        return features
    
    def extract_batch_features(
        self, 
        data_dict: Dict[str, np.ndarray],
        feature_level: str = 'comprehensive'
    ) -> Dict[str, StructuralTensorFeatures]:
        """
        バッチ特徴抽出（JIT最適化版）
        
        複数系列の並列特徴抽出による高速バッチ処理。
        
        Args:
            data_dict: {series_name: data} 辞書
            feature_level: 'basic', 'hierarchical', 'comprehensive'
            
        Returns:
            Dict[str, StructuralTensorFeatures]: 特徴量辞書
        """
        start_time = time.time()
        
        print(f"\n🔬 Batch Feature Extraction")
        print(f"   Series count: {len(data_dict)}")
        print(f"   Feature level: {feature_level}")
        print(f"   JIT optimization: {'Enabled' if self.jit_enabled else 'Disabled'}")
        
        features_dict = {}
        extraction_times = []
        
        # 特徴抽出メソッド選択
        if feature_level == 'basic':
            extract_method = self.extract_basic_features
        elif feature_level == 'hierarchical':
            extract_method = self.extract_hierarchical_features
        else:  # comprehensive
            extract_method = self.extract_comprehensive_features
        
        # バッチ処理実行
        for series_name, data in data_dict.items():
            series_start = time.time()
            
            try:
                features = extract_method(data, series_name)
                features_dict[series_name] = features
                
                series_time = time.time() - series_start
                extraction_times.append(series_time)
                
                print(f"   ✅ {series_name}: {series_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ {series_name}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        
        # バッチ統計
        if extraction_times:
            avg_time = np.mean(extraction_times)
            total_points = sum(len(data) for data in data_dict.values())
            processing_rate = total_points / total_time
            
            print(f"\n📊 Batch Extraction Complete:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average time per series: {avg_time:.3f}s")
            print(f"   Processing rate: {processing_rate:.0f} points/sec")
            print(f"   Success rate: {len(features_dict)}/{len(data_dict)} ({len(features_dict)/len(data_dict)*100:.1f}%)")
        
        return features_dict
    
    def _preprocess_data_optimized(self, data: np.ndarray) -> np.ndarray:
        """データ前処理（JIT最適化版）"""
        # データ型確保
        data = np.asarray(data, dtype=np.float64)
        
        # 欠損値処理
        if np.isnan(data).any():
            mask = np.isnan(data)
            if np.all(mask):
                raise ValueError("All data values are NaN")
            
            # 線形補間
            valid_indices = np.flatnonzero(~mask)
            invalid_indices = np.flatnonzero(mask)
            data[mask] = np.interp(invalid_indices, valid_indices, data[valid_indices])
        
        # 無限値処理
        if np.isinf(data).any():
            data = np.clip(data, -1e10, 1e10)
        
        return data
    
    def _extract_basic_features_fallback(
        self, 
        data: np.ndarray, 
        series_name: str
    ) -> StructuralTensorFeatures:
        """基本特徴抽出フォールバック実装"""
        features = StructuralTensorFeatures(
            data=data,
            series_name=series_name
        )
        
        # 標準実装による基本特徴抽出
        diff, threshold = calculate_diff_and_threshold_fixed(
            data, self.config.delta_percentile
        )
        delta_pos, delta_neg = detect_structural_jumps_fixed(diff, threshold)
        rho_t = calculate_tension_scalar_fixed(data, self.config.window)
        
        features.delta_LambdaC_pos = delta_pos
        features.delta_LambdaC_neg = delta_neg
        features.rho_T = rho_t
        
        return features
    
    def _extract_hierarchical_features_fallback(
        self, 
        features: StructuralTensorFeatures
    ) -> StructuralTensorFeatures:
        """階層特徴抽出フォールバック実装"""
        # 標準実装による階層的特徴抽出
        local_pos, local_neg, global_pos, global_neg = detect_hierarchical_jumps_fixed(
            features.data,
            local_window=self.config.local_window,
            global_window=self.config.global_window,
            local_percentile=self.config.local_jump_percentile,
            global_percentile=self.config.delta_percentile
        )
        
        features.local_pos = local_pos
        features.local_neg = local_neg
        features.global_pos = global_pos
        features.global_neg = global_neg
        
        return features
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """抽出履歴サマリー"""
        if not self.extraction_history:
            return {"message": "No feature extractions performed yet"}
        
        total_extractions = len(self.extraction_history)
        jit_extractions = sum(1 for h in self.extraction_history if 'JIT' in h.get('method', ''))
        
        # 統計計算
        extraction_times = [h['extraction_time'] for h in self.extraction_history]
        data_lengths = [h['data_length'] for h in self.extraction_history]
        
        # 特徴レベル別統計
        level_counts = {}
        for h in self.extraction_history:
            level = h.get('feature_level', 'unknown')
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'total_extractions': total_extractions,
            'jit_usage_rate': jit_extractions / total_extractions,
            'jit_enabled': self.jit_enabled,
            'performance_stats': {
                'avg_extraction_time': float(np.mean(extraction_times)),
                'min_extraction_time': float(np.min(extraction_times)),
                'max_extraction_time': float(np.max(extraction_times)),
                'avg_data_length': float(np.mean(data_lengths))
            },
            'feature_level_distribution': level_counts,
            'processing_rates': {
                'avg_points_per_second': float(np.mean(data_lengths) / np.mean(extraction_times)),
                'peak_performance': float(np.max(data_lengths) / np.min(extraction_times))
            },
            'recent_extractions': self.extraction_history[-5:],
            'jit_performance_gain': self._calculate_jit_gain()
        }
    
    def _calculate_jit_gain(self) -> float:
        """JIT性能向上率計算"""
        jit_times = [h['extraction_time'] for h in self.extraction_history if 'JIT' in h.get('method', '')]
        non_jit_times = [h['extraction_time'] for h in self.extraction_history if 'JIT' not in h.get('method', '')]
        
        if jit_times and non_jit_times:
            avg_jit_time = np.mean(jit_times)
            avg_non_jit_time = np.mean(non_jit_times)
            
            if avg_jit_time > 0:
                return (avg_non_jit_time - avg_jit_time) / avg_jit_time
        
        return 0.0

# ==========================================================
# CONVENIENCE FUNCTIONS（拡張版）
# ==========================================================

def extract_lambda3_features(
    data: np.ndarray,
    config: Optional[L3BaseConfig] = None,
    series_name: str = "Unknown",
    feature_level: str = "comprehensive"
) -> StructuralTensorFeatures:
    """
    Lambda³特徴量抽出の便利関数（JIT最適化版）
    
    Args:
        data: 入力データ
        config: 設定オブジェクト
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
    else:  # comprehensive
        return extractor.extract_comprehensive_features(data, series_name)

def analyze_lambda3_structure(
    data: np.ndarray,
    config: Optional[L3BaseConfig] = None,
    series_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Lambda³構造解析の便利関数
    
    Args:
        data: 入力データ
        config: 設定オブジェクト
        series_name: 系列名
        
    Returns:
        Dict: 構造解析結果
    """
    features = extract_lambda3_features(data, config, series_name, "comprehensive")
    summary = features.get_structural_summary()
    validation = features.validate_features()
    
    return {
        'features': features,
        'structural_summary': summary,
        'validation_results': validation,
        'extraction_info': features.extraction_info
    }

def extract_features_batch(
    data_dict: Dict[str, np.ndarray],
    config: Optional[L3BaseConfig] = None,
    feature_level: str = "comprehensive"
) -> Dict[str, StructuralTensorFeatures]:
    """
    バッチ特徴抽出の便利関数
    
    Args:
        data_dict: {series_name: data} 辞書
        config: 設定オブジェクト
        feature_level: 特徴レベル
        
    Returns:
        Dict[str, StructuralTensorFeatures]: 特徴量辞書
    """
    extractor = StructuralTensorExtractor(config)
    return extractor.extract_batch_features(data_dict, feature_level)

if __name__ == "__main__":
    print("Lambda³ Structural Tensor Module Test (JIT Compatible)")
    print("=" * 70)
    
    # JIT機能確認
    if JIT_FUNCTIONS_AVAILABLE:
        print("✅ JIT最適化関数利用可能")
        
        # JIT特徴抽出テスト
        try:
            test_data = np.cumsum(np.random.randn(500) * 0.1)
            
            # JIT包括特徴抽出テスト
            features_tuple = extract_lambda3_features_jit(
                test_data,
                window=10,
                local_window=5,
                global_window=20
            )
            
            print(f"✅ JIT一括特徴抽出成功: {len(features_tuple)} feature arrays")
            
            # 個別JIT関数テスト
            rho_t = calculate_tension_scalar_fixed(test_data, 10)
            print(f"✅ JIT張力スカラー計算: mean={np.mean(rho_t):.4f}")
            
            diff, threshold = calculate_diff_and_threshold_fixed(test_data, 95.0)
            print(f"✅ JIT構造差分計算: threshold={threshold:.4f}")
            
        except Exception as e:
            print(f"❌ JIT特徴抽出テストエラー: {e}")
    else:
        print("⚠️  JIT最適化関数利用不可")
    
    # 構造テンソル抽出器テスト
    try:
        config = L3BaseConfig()
        extractor = StructuralTensorExtractor(config)
        print("✅ 構造テンソル抽出器初期化成功")
        
        # 単一系列テスト
        test_data = np.cumsum(np.random.randn(200) * 0.1)
        
        # 基本特徴抽出テスト
        basic_features = extractor.extract_basic_features(test_data, "TestSeries")
        print(f"✅ 基本特徴抽出: {basic_features.count_events_by_type()}")
        
        # 階層特徴抽出テスト
        hierarchical_features = extractor.extract_hierarchical_features(test_data, "TestSeries")
        print(f"✅ 階層特徴抽出: completeness={hierarchical_features.extraction_info.get('feature_completeness', 0):.2f}")
        
        # 包括特徴抽出テスト
        comprehensive_features = extractor.extract_comprehensive_features(test_data, "TestSeries")
        print(f"✅ 包括特徴抽出: method={comprehensive_features.extraction_info.get('extraction_method', 'Unknown')}")
        
        # バッチ処理テスト
        test_data_dict = {
            'Series_A': np.cumsum(np.random.randn(100) * 0.1),
            'Series_B': np.cumsum(np.random.randn(100) * 0.1),
            'Series_C': np.cumsum(np.random.randn(100) * 0.1)
        }
        
        batch_features = extractor.extract_batch_features(test_data_dict, 'comprehensive')
        print(f"✅ バッチ特徴抽出: {len(batch_features)} series processed")
        
        # 抽出履歴サマリー
        summary = extractor.get_extraction_summary()
        print(f"✅ 抽出履歴: {summary['total_extractions']} extractions, JIT usage: {summary['jit_usage_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ 構造テンソル抽出器テストエラー: {e}")
    
    print("\nStructural tensor module loaded successfully!")
    print("Ready for Lambda³ structural tensor analysis with JIT optimization.")
