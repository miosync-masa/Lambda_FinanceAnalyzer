# ==========================================================
# lambda3/analysis/pairwise.py (非対称相互作用完全修正版)
# Pairwise Asymmetric Interaction Analysis for Lambda³ Theory
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
#
# 完全修正版: 構造テンソル成分のみを使用した理論準拠実装
# ==========================================================

"""
Lambda³ペアワイズ非対称相互作用分析（完全修正版）

構造テンソル(Λ)系列間の非対称相互作用を、∆ΛC pulsationsと
張力スカラー(ρT)の変化から完全に解析。元データの直接相関を
排除し、純粋な構造空間での相互作用を定量化。

完全修正内容:
- 構造テンソル成分（∆ΛC, ∆ρT）のみを使用
- 非対称因果性の理論準拠計算
- 階層的相互作用の導入
- ベイズモデルの構造空間準拠
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
import warnings
import time
from datetime import datetime

# Bayesian analysis imports
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian analysis disabled.")

# JIT imports
try:
    from ..core.jit_functions import (
        sync_rate_at_lag,
        calculate_sync_profile_jit,
        detect_phase_coupling,
        normalize_array_fixed
    )
    JIT_FUNCTIONS_AVAILABLE = True
except ImportError:
    JIT_FUNCTIONS_AVAILABLE = False
    warnings.warn("JIT functions not available.")

# Configuration imports
try:
    from ..core.config import L3BaseConfig, L3PairwiseConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Type imports
try:
    from ..core.structural_tensor import StructuralTensorProtocol
    from ..types.core import Lambda3Error
    TYPES_AVAILABLE = True
except ImportError:
    TYPES_AVAILABLE = False
    StructuralTensorProtocol = Any
    Lambda3Error = Exception

# ==========================================================
# RESULTS DATACLASS - ペアワイズ分析結果
# ==========================================================

@dataclass
class PairwiseInteractionResults:
    """
    ペアワイズ相互作用分析結果（Lambda³理論完全準拠版）
    
    全ての指標は構造テンソル成分の変化（∆ΛC, ∆ρT）から算出。
    元データの直接相関は含まない。
    """
    
    # 系列識別子
    name_a: str = "Series_A"
    name_b: str = "Series_B"
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 構造同期指標（∆ΛCベース）
    structure_synchronization: float = 0.0     # ∆ΛC同期強度
    tension_synchronization: float = 0.0       # ∆ρT同期強度
    
    # 構造因果性指標（∆ΛCベース）
    structure_causality_a_to_b: float = 0.0    # ∆ΛC: A→B因果
    structure_causality_b_to_a: float = 0.0    # ∆ΛC: B→A因果
    
    # 張力因果性指標（∆ρTベース）
    tension_causality_a_to_b: float = 0.0      # ∆ρT: A→B因果
    tension_causality_b_to_a: float = 0.0      # ∆ρT: B→A因果
    
    # 統合指標
    asymmetry_index: float = 0.0              # 非対称性指標
    interaction_strength: float = 0.0          # 相互作用強度
    
    # データ品質
    data_overlap_length: int = 0               # データ重複長
    structure_quality: float = 0.0             # 構造品質
    
    # 詳細分析結果
    structure_sync_profile: Dict[int, float] = field(default_factory=dict)
    tension_sync_profile: Dict[int, float] = field(default_factory=dict)
    interaction_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    phase_coupling: Dict[str, float] = field(default_factory=dict)
    
    # メタデータ
    analysis_method: str = "standard"          # 分析手法
    bayesian_trace: Optional[Any] = None       # ベイズトレース
    bayesian_summary: Optional[Dict[str, Any]] = None  # ベイズサマリー
    processing_time: float = 0.0               # 処理時間
    
    # 拡張メトリクス
    asymmetry_metrics: Dict[str, float] = field(default_factory=dict)
    causality_patterns: Dict[str, Dict[int, float]] = field(default_factory=dict)
    hierarchical_interaction: Dict[str, float] = field(default_factory=dict)
    interaction_quality: Dict[str, float] = field(default_factory=dict)
    
    @property
    def synchronization_strength(self) -> float:
        """統合同期強度"""
        return (self.structure_synchronization + self.tension_synchronization) / 2
    
    @property
    def causality_a_to_b(self) -> float:
        """統合因果強度 A→B"""
        return (self.structure_causality_a_to_b + self.tension_causality_a_to_b) / 2
    
    @property
    def causality_b_to_a(self) -> float:
        """統合因果強度 B→A"""
        return (self.structure_causality_b_to_a + self.tension_causality_b_to_a) / 2
    
    def get_interaction_summary(self) -> Dict[str, float]:
        """相互作用サマリー取得"""
        return {
            'structure_sync': self.structure_synchronization,
            'tension_sync': self.tension_synchronization,
            'asymmetry': self.asymmetry_index,
            'causality_a_to_b': self.causality_a_to_b,
            'causality_b_to_a': self.causality_b_to_a,
            'quality': self.structure_quality
        }
    
    def get_dominant_direction(self) -> str:
        """優勢方向判定"""
        if self.causality_a_to_b > self.causality_b_to_a * 1.2:
            return f'{self.name_a}_to_{self.name_b}'
        elif self.causality_b_to_a > self.causality_a_to_b * 1.2:
            return f'{self.name_b}_to_{self.name_a}'
        else:
            return 'symmetric'
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        return {
            'series_names': [self.name_a, self.name_b],
            'analysis_timestamp': self.analysis_timestamp,
            'interaction_summary': self.get_interaction_summary(),
            'dominant_direction': self.get_dominant_direction(),
            'asymmetry_metrics': self.asymmetry_metrics,
            'causality_patterns': self.causality_patterns,
            'hierarchical_interaction': self.hierarchical_interaction,
            'analysis_method': self.analysis_method,
            'processing_time': self.processing_time
        }

# ==========================================================
# PAIRWISE ANALYZER - ペアワイズ分析器
# ==========================================================

class PairwiseAnalyzer:
    """
    Lambda³ペアワイズ相互作用分析器（完全準拠版）
    
    構造テンソル成分（∆ΛC, ∆ρT）のみを使用した非対称相互作用解析。
    元データの直接使用を排除し、純粋な構造空間での分析を実現。
    """
    
    def __init__(self, config: Optional[Any] = None, use_jit: Optional[bool] = None):
        """
        Args:
            config: 設定オブジェクト
            use_jit: JIT最適化使用フラグ
        """
        self.config = self._initialize_config(config)
        self.use_jit = use_jit if use_jit is not None else JIT_FUNCTIONS_AVAILABLE
        
    def _initialize_config(self, config: Optional[Any]) -> Any:
        """設定初期化"""
        if config is not None:
            return config
        
        if CONFIG_AVAILABLE:
            return L3PairwiseConfig()
        else:
            # フォールバック設定
            return type('Config', (), {
                'lag_window': 10,
                'sync_threshold': 0.3,
                'causality_lag_max': 5,
                'min_events_threshold': 5,
                'bayesian_draws': 2000,
                'bayesian_tune': 1000
            })()
    
    def analyze_asymmetric_interaction(
        self,
        features_a: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
        features_b: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
        use_bayesian: bool = False
    ) -> PairwiseInteractionResults:
        """
        非対称相互作用分析実行（Lambda³理論完全準拠）
        
        Args:
            features_a: 系列Aの構造テンソル特徴量
            features_b: 系列Bの構造テンソル特徴量
            use_bayesian: ベイズ分析使用フラグ
            
        Returns:
            PairwiseInteractionResults: ペアワイズ分析結果
        """
        start_time = time.time()
        
        # 構造テンソル成分の抽出
        components_a = self._extract_tensor_components(features_a)
        components_b = self._extract_tensor_components(features_b)
        
        print(f"\n{'='*60}")
        print(f"PAIRWISE ASYMMETRIC INTERACTION ANALYSIS")
        print(f"{components_a['name']} ⇄ {components_b['name']}")
        print(f"{'='*60}")
        
        # データ検証と前処理
        min_length = self._validate_and_align_components(components_a, components_b)
        
        # 構造変化イベント統計
        self._print_event_statistics(components_a, components_b, min_length)
        
        # 分析手法の選択と実行
        if use_bayesian and BAYESIAN_AVAILABLE:
            results = self._analyze_bayesian(components_a, components_b)
        else:
            results = self._analyze_standard(components_a, components_b)
        
        # 処理時間記録
        results.processing_time = time.time() - start_time
        
        print(f"\nAnalysis completed in {results.processing_time:.2f} seconds")
        self._print_results_summary(results)
        
        return results
    
    def _extract_tensor_components(
        self,
        features: Union[StructuralTensorProtocol, Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """構造テンソル成分の抽出"""
        components = {}
        
        # 系列名の取得
        if hasattr(features, 'series_name'):
            components['name'] = features.series_name
        elif isinstance(features, dict) and 'series_name' in features:
            components['name'] = features['series_name']
        else:
            components['name'] = 'Series'
        
        # ∆ΛC成分の取得
        if hasattr(features, 'delta_LambdaC_pos'):
            components['delta_pos'] = np.asarray(features.delta_LambdaC_pos, dtype=np.float64)
            components['delta_neg'] = np.asarray(features.delta_LambdaC_neg, dtype=np.float64)
        elif isinstance(features, dict):
            components['delta_pos'] = np.asarray(features.get('delta_LambdaC_pos', []), dtype=np.float64)
            components['delta_neg'] = np.asarray(features.get('delta_LambdaC_neg', []), dtype=np.float64)
        else:
            raise ValueError("Cannot extract ∆ΛC components from features")
        
        # ρT成分の取得
        if hasattr(features, 'rho_T'):
            components['rho_T'] = np.asarray(features.rho_T, dtype=np.float64)
        elif isinstance(features, dict):
            components['rho_T'] = np.asarray(features.get('rho_T', []), dtype=np.float64)
        else:
            raise ValueError("Cannot extract ρT component from features")
        
        # 階層的特徴の取得（オプション）
        for feat in ['local_pos', 'local_neg', 'global_pos', 'global_neg']:
            if hasattr(features, feat):
                components[feat] = np.asarray(getattr(features, feat), dtype=np.float64)
            elif isinstance(features, dict) and feat in features:
                components[feat] = np.asarray(features[feat], dtype=np.float64)
        
        return components
    
    def _validate_and_align_components(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> int:
        """成分の検証と整列"""
        # 最小長の決定
        min_length = min(
            len(components_a['delta_pos']),
            len(components_b['delta_pos'])
        )
        
        if min_length < self.config.min_events_threshold:
            raise ValueError(f"Insufficient data length: {min_length} < {self.config.min_events_threshold}")
        
        # 長さの整列
        for key in ['delta_pos', 'delta_neg', 'rho_T']:
            components_a[key] = components_a[key][:min_length]
            components_b[key] = components_b[key][:min_length]
        
        # 階層的特徴の整列（存在する場合）
        for key in ['local_pos', 'local_neg', 'global_pos', 'global_neg']:
            if key in components_a and key in components_b:
                components_a[key] = components_a[key][:min_length]
                components_b[key] = components_b[key][:min_length]
        
        return min_length
    
    def _print_event_statistics(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ):
        """イベント統計の表示"""
        events_a = np.sum(components_a['delta_pos']) + np.sum(components_a['delta_neg'])
        events_b = np.sum(components_b['delta_pos']) + np.sum(components_b['delta_neg'])
        
        print(f"\nStructural Event Statistics:")
        print(f"  {components_a['name']}: {int(events_a)} events")
        print(f"    Positive: {int(np.sum(components_a['delta_pos']))}")
        print(f"    Negative: {int(np.sum(components_a['delta_neg']))}")
        print(f"  {components_b['name']}: {int(events_b)} events")
        print(f"    Positive: {int(np.sum(components_b['delta_pos']))}")
        print(f"    Negative: {int(np.sum(components_b['delta_neg']))}")
        print(f"  Data length: {min_length}")
    
    def _analyze_standard(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> PairwiseInteractionResults:
        """標準ペアワイズ分析（Lambda³理論準拠）"""
        print("\nPerforming standard pairwise analysis...")
        
        name_a = components_a['name']
        name_b = components_b['name']
        min_length = len(components_a['delta_pos'])
        
        # 構造同期分析（∆ΛCベース）
        structure_sync = self._calculate_structure_synchronization(
            components_a, components_b
        )
        
        # 張力同期分析（∆ρTベース）
        tension_sync = self._calculate_tension_synchronization(
            components_a, components_b
        )
        
        # 構造因果性分析（∆ΛCベース）
        structure_causality = self._calculate_structure_causality(
            components_a, components_b
        )
        
        # 張力因果性分析（∆ρTベース）
        tension_causality = self._calculate_tension_causality(
            components_a, components_b
        )
        
        # 相互作用係数計算
        interaction_coeffs = self._calculate_interaction_coefficients(
            components_a, components_b
        )
        
        # 位相結合分析
        phase_coupling = self._calculate_phase_coupling(
            components_a, components_b
        )
        
        # 品質評価
        structure_quality = self._assess_structure_quality(
            components_a, components_b
        )
        
        # 拡張メトリクス
        asymmetry_metrics = self._calculate_asymmetry_metrics(
            structure_causality, tension_causality
        )
        
        causality_patterns = self._calculate_causality_patterns(
            components_a, components_b
        )
        
        hierarchical_interaction = self._calculate_hierarchical_interaction(
            components_a, components_b
        )
        
        # 統合指標
        asymmetry_index = self._calculate_asymmetry_index(
            structure_causality, tension_causality
        )
        
        interaction_strength = self._calculate_interaction_strength(
            structure_sync, tension_sync,
            structure_causality, tension_causality
        )
        
        # 結果構築
        results = PairwiseInteractionResults(
            name_a=name_a,
            name_b=name_b,
            structure_synchronization=structure_sync['strength'],
            tension_synchronization=tension_sync['strength'],
            structure_causality_a_to_b=structure_causality['a_to_b'],
            structure_causality_b_to_a=structure_causality['b_to_a'],
            tension_causality_a_to_b=tension_causality['a_to_b'],
            tension_causality_b_to_a=tension_causality['b_to_a'],
            asymmetry_index=asymmetry_index,
            interaction_strength=interaction_strength,
            data_overlap_length=min_length,
            structure_quality=structure_quality,
            structure_sync_profile=structure_sync['profile'],
            tension_sync_profile=tension_sync['profile'],
            interaction_coefficients=interaction_coeffs,
            phase_coupling=phase_coupling,
            analysis_method='standard',
            asymmetry_metrics=asymmetry_metrics,
            causality_patterns=causality_patterns,
            hierarchical_interaction=hierarchical_interaction
        )
        
        return results
    
    def _calculate_structure_synchronization(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """構造同期分析（∆ΛCベース）"""
        # 構造変化イベントの統合
        events_a = components_a['delta_pos'] + components_a['delta_neg']
        events_b = components_b['delta_pos'] + components_b['delta_neg']
        
        if self.use_jit and JIT_FUNCTIONS_AVAILABLE:
            # JIT最適化版
            lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
                events_a, events_b, self.config.lag_window
            )
            sync_profile = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
        else:
            # 純Python版
            sync_profile = {}
            max_sync = 0.0
            optimal_lag = 0
            
            for lag in range(-self.config.lag_window, self.config.lag_window + 1):
                if lag == 0:
                    sync = np.corrcoef(events_a, events_b)[0, 1]
                elif lag > 0 and lag < len(events_a):
                    sync = np.corrcoef(events_a[:-lag], events_b[lag:])[0, 1]
                elif lag < 0 and -lag < len(events_b):
                    sync = np.corrcoef(events_a[-lag:], events_b[:lag])[0, 1]
                else:
                    sync = 0.0
                
                sync_profile[lag] = abs(sync) if not np.isnan(sync) else 0.0
                
                if sync_profile[lag] > max_sync:
                    max_sync = sync_profile[lag]
                    optimal_lag = lag
        
        return {
            'strength': max_sync,
            'optimal_lag': optimal_lag,
            'profile': sync_profile
        }
    
    def _calculate_tension_synchronization(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """張力同期分析（∆ρTベース）"""
        # 張力変化率の計算
        delta_rho_a = np.diff(components_a['rho_T'], prepend=components_a['rho_T'][0])
        delta_rho_b = np.diff(components_b['rho_T'], prepend=components_b['rho_T'][0])
        
        # 正規化
        if np.std(delta_rho_a) > 0:
            delta_rho_a = delta_rho_a / np.std(delta_rho_a)
        if np.std(delta_rho_b) > 0:
            delta_rho_b = delta_rho_b / np.std(delta_rho_b)
        
        # 同期プロファイル計算（構造同期と同様）
        if self.use_jit and JIT_FUNCTIONS_AVAILABLE:
            lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
                delta_rho_a, delta_rho_b, self.config.lag_window
            )
            sync_profile = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
        else:
            sync_profile = {}
            max_sync = 0.0
            optimal_lag = 0
            
            for lag in range(-self.config.lag_window, self.config.lag_window + 1):
                if lag == 0:
                    sync = np.corrcoef(delta_rho_a, delta_rho_b)[0, 1]
                elif lag > 0 and lag < len(delta_rho_a):
                    sync = np.corrcoef(delta_rho_a[:-lag], delta_rho_b[lag:])[0, 1]
                elif lag < 0 and -lag < len(delta_rho_b):
                    sync = np.corrcoef(delta_rho_a[-lag:], delta_rho_b[:lag])[0, 1]
                else:
                    sync = 0.0
                
                sync_profile[lag] = abs(sync) if not np.isnan(sync) else 0.0
                
                if sync_profile[lag] > max_sync:
                    max_sync = sync_profile[lag]
                    optimal_lag = lag
        
        return {
            'strength': max_sync,
            'optimal_lag': optimal_lag,
            'profile': sync_profile
        }
    
    def _calculate_structure_causality(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, float]:
        """構造因果性分析（∆ΛCベース）"""
        # 正負の構造変化を個別に分析
        causality_a_to_b_pos = self._calculate_directional_causality(
            components_a['delta_pos'], components_b['delta_pos'], 'pos'
        )
        causality_a_to_b_neg = self._calculate_directional_causality(
            components_a['delta_neg'], components_b['delta_neg'], 'neg'
        )
        
        causality_b_to_a_pos = self._calculate_directional_causality(
            components_b['delta_pos'], components_a['delta_pos'], 'pos'
        )
        causality_b_to_a_neg = self._calculate_directional_causality(
            components_b['delta_neg'], components_a['delta_neg'], 'neg'
        )
        
        # 統合因果強度
        causality_a_to_b = (causality_a_to_b_pos + causality_a_to_b_neg) / 2
        causality_b_to_a = (causality_b_to_a_pos + causality_b_to_a_neg) / 2
        
        return {
            'a_to_b': causality_a_to_b,
            'b_to_a': causality_b_to_a,
            'a_to_b_pos': causality_a_to_b_pos,
            'a_to_b_neg': causality_a_to_b_neg,
            'b_to_a_pos': causality_b_to_a_pos,
            'b_to_a_neg': causality_b_to_a_neg
        }
    
    def _calculate_directional_causality(
        self,
        cause_events: np.ndarray,
        effect_events: np.ndarray,
        event_type: str
    ) -> float:
        """方向性因果強度計算"""
        max_lag = self.config.causality_lag_max
        causality_scores = []
        
        for lag in range(1, max_lag + 1):
            if lag < len(cause_events):
                # 原因イベント後の効果イベント発生確率
                cause_indices = np.where(cause_events[:-lag] > 0)[0]
                
                if len(cause_indices) > 0:
                    effect_count = 0
                    for idx in cause_indices:
                        if idx + lag < len(effect_events):
                            effect_count += effect_events[idx + lag]
                    
                    causality_prob = effect_count / len(cause_indices)
                    causality_scores.append(causality_prob)
        
        return np.mean(causality_scores) if causality_scores else 0.0
    
    def _calculate_tension_causality(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, float]:
        """張力因果性分析（∆ρTベース）"""
        # 張力変化率
        delta_rho_a = np.diff(components_a['rho_T'], prepend=components_a['rho_T'][0])
        delta_rho_b = np.diff(components_b['rho_T'], prepend=components_b['rho_T'][0])
        
        # 閾値を超える変化の検出
        threshold_a = np.percentile(np.abs(delta_rho_a), 80)
        threshold_b = np.percentile(np.abs(delta_rho_b), 80)
        
        tension_events_a = (np.abs(delta_rho_a) > threshold_a).astype(float)
        tension_events_b = (np.abs(delta_rho_b) > threshold_b).astype(float)
        
        # 方向性因果計算
        causality_a_to_b = self._calculate_directional_causality(
            tension_events_a, tension_events_b, 'tension'
        )
        causality_b_to_a = self._calculate_directional_causality(
            tension_events_b, tension_events_a, 'tension'
        )
        
        return {
            'a_to_b': causality_a_to_b,
            'b_to_a': causality_b_to_a
        }
    
    def _calculate_interaction_coefficients(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """相互作用係数計算"""
        # 自己効果
        self_effects = {
            components_a['name']: {
                'pos_persistence': self._calculate_event_persistence(components_a['delta_pos']),
                'neg_persistence': self._calculate_event_persistence(components_a['delta_neg']),
                'tension_volatility': np.std(components_a['rho_T'])
            },
            components_b['name']: {
                'pos_persistence': self._calculate_event_persistence(components_b['delta_pos']),
                'neg_persistence': self._calculate_event_persistence(components_b['delta_neg']),
                'tension_volatility': np.std(components_b['rho_T'])
            }
        }
        
        # 交差効果
        cross_effects = {
            f"{components_a['name']}_to_{components_b['name']}": {
                'structure_influence': self._calculate_structure_influence(
                    components_a, components_b
                ),
                'tension_influence': self._calculate_tension_influence(
                    components_a, components_b
                )
            },
            f"{components_b['name']}_to_{components_a['name']}": {
                'structure_influence': self._calculate_structure_influence(
                    components_b, components_a
                ),
                'tension_influence': self._calculate_tension_influence(
                    components_b, components_a
                )
            }
        }
        
        return {
            'self_effects': self_effects,
            'cross_effects': cross_effects
        }
    
    def _calculate_event_persistence(self, events: np.ndarray) -> float:
        """イベント持続性計算"""
        if np.sum(events) == 0:
            return 0.0
        
        # 連続イベントの検出
        changes = np.diff(np.concatenate([[0], events, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(starts) == 0:
            return 0.0
        
        durations = ends - starts
        return np.mean(durations)
    
    def _calculate_structure_influence(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> float:
        """構造影響度計算"""
        # ソースの構造変化がターゲットの張力に与える影響
        source_events = source['delta_pos'] + source['delta_neg']
        
        # イベント時とそれ以外での張力差
        event_tension = np.mean(target['rho_T'][source_events > 0])
        baseline_tension = np.mean(target['rho_T'][source_events == 0])
        
        if baseline_tension > 0:
            return (event_tension - baseline_tension) / baseline_tension
        else:
            return 0.0
    
    def _calculate_tension_influence(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> float:
        """張力影響度計算"""
        # ソースの高張力がターゲットの構造変化に与える影響
        high_tension_threshold = np.percentile(source['rho_T'], 80)
        high_tension_mask = source['rho_T'] > high_tension_threshold
        
        target_events = target['delta_pos'] + target['delta_neg']
        
        # 高張力時の構造変化率
        if np.sum(high_tension_mask) > 0:
            event_rate_high = np.mean(target_events[high_tension_mask])
            event_rate_normal = np.mean(target_events[~high_tension_mask])
            
            if event_rate_normal > 0:
                return (event_rate_high - event_rate_normal) / event_rate_normal
            else:
                return event_rate_high
        else:
            return 0.0
    
    def _calculate_phase_coupling(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, float]:
        """位相結合分析"""
        if self.use_jit and JIT_FUNCTIONS_AVAILABLE:
            # JIT版位相結合
            coupling_strength, phase_lag = detect_phase_coupling(
                components_a['rho_T'], components_b['rho_T']
            )
        else:
            # 簡易版
            coupling_strength = np.abs(np.corrcoef(
                components_a['rho_T'], components_b['rho_T']
            )[0, 1])
            phase_lag = 0.0
        
        return {
            'coupling_strength': coupling_strength,
            'phase_lag': phase_lag
        }
    
    def _assess_structure_quality(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> float:
        """構造品質評価"""
        # イベント密度
        events_a = np.sum(components_a['delta_pos']) + np.sum(components_a['delta_neg'])
        events_b = np.sum(components_b['delta_pos']) + np.sum(components_b['delta_neg'])
        
        event_density = (events_a + events_b) / (2 * len(components_a['delta_pos']))
        
        # 張力変動性
        tension_var_a = np.std(components_a['rho_T']) / (np.mean(components_a['rho_T']) + 1e-8)
        tension_var_b = np.std(components_b['rho_T']) / (np.mean(components_b['rho_T']) + 1e-8)
        
        tension_quality = 1 / (1 + np.exp(-2 * (tension_var_a + tension_var_b)))
        
        # 総合品質
        return event_density * 0.5 + tension_quality * 0.5
    
    def _calculate_asymmetry_metrics(
        self,
        structure_causality: Dict[str, float],
        tension_causality: Dict[str, float]
    ) -> Dict[str, float]:
        """非対称性メトリクス計算"""
        # 構造非対称性
        structure_asym = abs(structure_causality['a_to_b'] - structure_causality['b_to_a'])
        
        # 張力非対称性
        tension_asym = abs(tension_causality['a_to_b'] - tension_causality['b_to_a'])
        
        # 方向性優勢度
        if structure_causality['a_to_b'] + structure_causality['b_to_a'] > 0:
            structure_dominance = max(
                structure_causality['a_to_b'],
                structure_causality['b_to_a']
            ) / (structure_causality['a_to_b'] + structure_causality['b_to_a'])
        else:
            structure_dominance = 0.5
        
        return {
            'structure_asymmetry': structure_asym,
            'tension_asymmetry': tension_asym,
            'total_asymmetry': (structure_asym + tension_asym) / 2,
            'structure_dominance': structure_dominance
        }
    
    def _calculate_causality_patterns(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, Dict[int, float]]:
        """因果パターン分析"""
        patterns = {}
        
        # A→Bパターン
        patterns['a_to_b'] = {}
        for lag in range(1, self.config.causality_lag_max + 1):
            events_a = components_a['delta_pos'] + components_a['delta_neg']
            events_b = components_b['delta_pos'] + components_b['delta_neg']
            
            if lag < len(events_a):
                cause_mask = events_a[:-lag] > 0
                effect_prob = np.mean(events_b[lag:][cause_mask]) if np.any(cause_mask) else 0
                patterns['a_to_b'][lag] = effect_prob
        
        # B→Aパターン
        patterns['b_to_a'] = {}
        for lag in range(1, self.config.causality_lag_max + 1):
            events_a = components_a['delta_pos'] + components_a['delta_neg']
            events_b = components_b['delta_pos'] + components_b['delta_neg']
            
            if lag < len(events_b):
                cause_mask = events_b[:-lag] > 0
                effect_prob = np.mean(events_a[lag:][cause_mask]) if np.any(cause_mask) else 0
                patterns['b_to_a'][lag] = effect_prob
        
        return patterns
    
    def _calculate_hierarchical_interaction(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Dict[str, float]:
        """階層的相互作用計算"""
        # 階層的特徴が利用可能な場合
        if all(key in components_a for key in ['local_pos', 'global_pos']):
            # ローカル相互作用
            local_interaction = self._calculate_level_interaction(
                components_a['local_pos'] + components_a['local_neg'],
                components_b['local_pos'] + components_b['local_neg']
            )
            
            # グローバル相互作用
            global_interaction = self._calculate_level_interaction(
                components_a['global_pos'] + components_a['global_neg'],
                components_b['global_pos'] + components_b['global_neg']
            )
            
            # 階層間伝播
            cross_hierarchy = self._calculate_cross_hierarchy_propagation(
                components_a, components_b
            )
            
            return {
                'local_interaction': local_interaction,
                'global_interaction': global_interaction,
                'cross_hierarchy': cross_hierarchy,
                'hierarchy_coupling': abs(local_interaction - global_interaction)
            }
        else:
            # 階層的特徴が利用できない場合
            return {
                'local_interaction': 0.0,
                'global_interaction': 0.0,
                'cross_hierarchy': 0.0,
                'hierarchy_coupling': 0.0
            }
    
    def _calculate_level_interaction(
        self,
        events_a: np.ndarray,
        events_b: np.ndarray
    ) -> float:
        """階層レベル別相互作用"""
        if np.sum(events_a) > 0 and np.sum(events_b) > 0:
            return np.corrcoef(events_a, events_b)[0, 1]
        else:
            return 0.0
    
    def _calculate_cross_hierarchy_propagation(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> float:
        """階層間伝播強度"""
        # A系列のローカル→B系列のグローバル
        a_local_to_b_global = self._calculate_directional_causality(
            components_a['local_pos'] + components_a['local_neg'],
            components_b['global_pos'] + components_b['global_neg'],
            'hierarchy'
        )
        
        # B系列のローカル→A系列のグローバル
        b_local_to_a_global = self._calculate_directional_causality(
            components_b['local_pos'] + components_b['local_neg'],
            components_a['global_pos'] + components_a['global_neg'],
            'hierarchy'
        )
        
        return (a_local_to_b_global + b_local_to_a_global) / 2
    
    def _calculate_asymmetry_index(
        self,
        structure_causality: Dict[str, float],
        tension_causality: Dict[str, float]
    ) -> float:
        """統合非対称性指標"""
        # 構造非対称性
        structure_asym = abs(structure_causality['a_to_b'] - structure_causality['b_to_a'])
        
        # 張力非対称性
        tension_asym = abs(tension_causality['a_to_b'] - tension_causality['b_to_a'])
        
        # 重み付き統合
        return structure_asym * 0.6 + tension_asym * 0.4
    
    def _calculate_interaction_strength(
        self,
        structure_sync: Dict[str, Any],
        tension_sync: Dict[str, Any],
        structure_causality: Dict[str, float],
        tension_causality: Dict[str, float]
    ) -> float:
        """統合相互作用強度"""
        # 同期成分
        sync_strength = (structure_sync['strength'] + tension_sync['strength']) / 2
        
        # 因果成分
        causality_strength = (
            structure_causality['a_to_b'] + structure_causality['b_to_a'] +
            tension_causality['a_to_b'] + tension_causality['b_to_a']
        ) / 4
        
        # 統合強度
        return sync_strength * 0.5 + causality_strength * 0.5
    
    def _analyze_bayesian(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> PairwiseInteractionResults:
        """ベイズペアワイズ分析"""
        print("\nPerforming Bayesian pairwise analysis...")
        
        # 標準分析を先に実行
        standard_results = self._analyze_standard(components_a, components_b)
        
        # ベイズモデル構築と推定
        trace, model = self._fit_bayesian_pairwise_model(components_a, components_b)
        
        # ベイズ係数抽出
        bayesian_coeffs = self._extract_bayesian_coefficients(trace)
        
        # 結果の更新
        standard_results.analysis_method = 'bayesian'
        standard_results.bayesian_trace = trace
        standard_results.bayesian_summary = az.summary(trace).to_dict()
        
        # ベイズ推定値で一部更新
        if bayesian_coeffs:
            # 相互作用係数の更新
            standard_results.interaction_coefficients['bayesian'] = bayesian_coeffs
        
        return standard_results
    
    def _fit_bayesian_pairwise_model(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """ベイズペアワイズモデル（構造空間準拠）"""
        # 構造変化イベント
        events_a = components_a['delta_pos'] + components_a['delta_neg']
        events_b = components_b['delta_pos'] + components_b['delta_neg']
        
        # 張力変化率
        delta_rho_a = np.diff(components_a['rho_T'], prepend=components_a['rho_T'][0])
        delta_rho_b = np.diff(components_b['rho_T'], prepend=components_b['rho_T'][0])
        
        with pm.Model() as model:
            # === 構造相互作用パラメータ ===
            beta_struct_ab = pm.Normal('beta_struct_ab', mu=0, sigma=1)
            beta_struct_ba = pm.Normal('beta_struct_ba', mu=0, sigma=1)
            
            # === 張力相互作用パラメータ ===
            beta_tension_ab = pm.Normal('beta_tension_ab', mu=0, sigma=0.5)
            beta_tension_ba = pm.Normal('beta_tension_ba', mu=0, sigma=0.5)
            
            # === 構造-張力結合パラメータ ===
            gamma_coupling = pm.Normal('gamma_coupling', mu=0, sigma=0.5)
            
            # === 観測モデル ===
            # 構造変化の相互影響
            mu_events_a = pm.math.sigmoid(beta_struct_ba * events_b)
            mu_events_b = pm.math.sigmoid(beta_struct_ab * events_a)
            
            obs_events_a = pm.Bernoulli('obs_events_a', p=mu_events_a, observed=events_a)
            obs_events_b = pm.Bernoulli('obs_events_b', p=mu_events_b, observed=events_b)
            
            # 張力変化の相互影響
            sigma_rho = pm.HalfNormal('sigma_rho', sigma=1)
            
            mu_delta_rho_a = beta_tension_ba * delta_rho_b + gamma_coupling * events_b
            mu_delta_rho_b = beta_tension_ab * delta_rho_a + gamma_coupling * events_a
            
            obs_delta_rho_a = pm.Normal('obs_delta_rho_a', mu=mu_delta_rho_a,
                                       sigma=sigma_rho, observed=delta_rho_a)
            obs_delta_rho_b = pm.Normal('obs_delta_rho_b', mu=mu_delta_rho_b,
                                       sigma=sigma_rho, observed=delta_rho_b)
            
            # サンプリング
            trace = pm.sample(
                draws=self.config.bayesian_draws,
                tune=self.config.bayesian_tune,
                target_accept=0.9,
                return_inferencedata=True,
                cores=2,
                chains=2
            )
        
        return trace, model
    
    def _extract_bayesian_coefficients(self, trace: Any) -> Dict[str, float]:
        """ベイズ係数抽出"""
        try:
            summary = az.summary(trace)
            
            coefficients = {
                'structure_a_to_b': abs(summary.loc['beta_struct_ab', 'mean']),
                'structure_b_to_a': abs(summary.loc['beta_struct_ba', 'mean']),
                'tension_a_to_b': abs(summary.loc['beta_tension_ab', 'mean']),
                'tension_b_to_a': abs(summary.loc['beta_tension_ba', 'mean']),
                'coupling_strength': abs(summary.loc['gamma_coupling', 'mean'])
            }
            
            return coefficients
            
        except Exception as e:
            print(f"Bayesian coefficient extraction failed: {e}")
            return {}
    
    def _print_results_summary(self, results: PairwiseInteractionResults):
        """結果サマリーの表示"""
        print(f"\n{'='*40}")
        print("PAIRWISE INTERACTION SUMMARY")
        print(f"{'='*40}")
        print(f"Direction: {results.get_dominant_direction()}")
        print(f"Structure Sync: {results.structure_synchronization:.4f}")
        print(f"Tension Sync: {results.tension_synchronization:.4f}")
        print(f"Asymmetry Index: {results.asymmetry_index:.4f}")
        print(f"Interaction Strength: {results.interaction_strength:.4f}")
        
        print(f"\nCausality:")
        print(f"  {results.name_a} → {results.name_b}: {results.causality_a_to_b:.4f}")
        print(f"  {results.name_b} → {results.name_a}: {results.causality_b_to_a:.4f}")

# ==========================================================
# CONVENIENCE FUNCTIONS - 便利関数
# ==========================================================

def analyze_pairwise_interaction(
    features_a: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
    features_b: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
    config: Optional[Any] = None,
    use_bayesian: bool = False
) -> PairwiseInteractionResults:
    """
    ペアワイズ相互作用分析の便利関数
    
    Args:
        features_a: 系列Aの構造テンソル特徴量
        features_b: 系列Bの構造テンソル特徴量
        config: 設定オブジェクト
        use_bayesian: ベイズ分析使用フラグ
        
    Returns:
        PairwiseInteractionResults: ペアワイズ分析結果
    """
    analyzer = PairwiseAnalyzer(config=config)
    return analyzer.analyze_asymmetric_interaction(features_a, features_b, use_bayesian)

def analyze_multiple_pairs(
    features_dict: Dict[str, Union[StructuralTensorProtocol, Dict[str, np.ndarray]]],
    config: Optional[Any] = None,
    pairs: Optional[List[Tuple[str, str]]] = None
) -> Dict[Tuple[str, str], PairwiseInteractionResults]:
    """
    複数ペアの相互作用分析
    
    Args:
        features_dict: 系列名→特徴量の辞書
        config: 設定オブジェクト
        pairs: 分析するペアのリスト（Noneの場合は全組み合わせ）
        
    Returns:
        Dict[Tuple[str, str], PairwiseInteractionResults]: ペア→結果の辞書
    """
    analyzer = PairwiseAnalyzer(config=config)
    results = {}
    
    # ペアの決定
    if pairs is None:
        series_names = list(features_dict.keys())
        from itertools import combinations
        pairs = list(combinations(series_names, 2))
    
    # 各ペアの分析
    for name_a, name_b in pairs:
        if name_a in features_dict and name_b in features_dict:
            print(f"\nAnalyzing pair: {name_a} ⇄ {name_b}")
            results[(name_a, name_b)] = analyzer.analyze_asymmetric_interaction(
                features_dict[name_a],
                features_dict[name_b]
            )
    
    # 比較サマリー表示
    _print_pairs_comparison(results)
    
    return results

def _print_pairs_comparison(results: Dict[Tuple[str, str], PairwiseInteractionResults]):
    """ペア比較サマリー表示"""
    print(f"\n{'='*70}")
    print("PAIRWISE COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Pair':<30} {'Sync':<10} {'Asymmetry':<10} {'Direction':<20}")
    print(f"{'-'*70}")
    
    for (name_a, name_b), result in results.items():
        pair_name = f"{name_a} ⇄ {name_b}"
        sync = result.synchronization_strength
        asym = result.asymmetry_index
        direction = result.get_dominant_direction()
        
        print(f"{pair_name:<30} {sync:>9.4f} {asym:>9.4f} {direction:<20}")

# ==========================================================
# VALIDATION & TESTING - 検証・テスト
# ==========================================================

def test_pairwise_analysis():
    """ペアワイズ分析のテスト"""
    print("🧪 Testing Pairwise Analysis Implementation")
    print("=" * 60)
    
    try:
        # テストデータ生成
        np.random.seed(42)
        
        # 相互作用のあるデータ
        base_a = np.cumsum(np.random.randn(200) * 0.1)
        base_b = np.cumsum(np.random.randn(200) * 0.1)
        
        # A→Bの影響を追加
        for i in range(50, 150):
            if i % 10 == 0:
                base_a[i] += np.random.randn() * 0.5
                if i + 2 < len(base_b):
                    base_b[i+2] += np.random.randn() * 0.3
        
        # 模擬的な構造テンソル特徴量
        def create_mock_features(data, name):
            diff = np.diff(data, prepend=data[0])
            threshold = np.percentile(np.abs(diff), 90)
            
            return {
                'series_name': name,
                'delta_LambdaC_pos': (diff > threshold).astype(float),
                'delta_LambdaC_neg': (diff < -threshold).astype(float),
                'rho_T': np.array([np.std(data[max(0, i-10):i+1]) for i in range(len(data))])
            }
        
        features_a = create_mock_features(base_a, "Test_A")
        features_b = create_mock_features(base_b, "Test_B")
        
        analyzer = PairwiseAnalyzer()
        
        print("\n📊 Testing standard pairwise analysis...")
        results_std = analyzer.analyze_asymmetric_interaction(
            features_a, features_b, use_bayesian=False
        )
        print(f"✅ Standard analysis completed")
        print(f"   Dominant direction: {results_std.get_dominant_direction()}")
        print(f"   Asymmetry index: {results_std.asymmetry_index:.4f}")
        
        # ベイズ分析（利用可能な場合）
        if BAYESIAN_AVAILABLE:
            print("\n📊 Testing Bayesian pairwise analysis...")
            results_bayes = analyzer.analyze_asymmetric_interaction(
                features_a, features_b, use_bayesian=True
            )
            print(f"✅ Bayesian analysis completed")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================================
# MODULE EXPORTS
# ==========================================================

__all__ = [
    # クラス
    'PairwiseAnalyzer',
    'PairwiseInteractionResults',
    
    # 関数
    'analyze_pairwise_interaction',
    'analyze_multiple_pairs',
    
    # テスト
    'test_pairwise_analysis'
]

if __name__ == "__main__":
    # 自動テスト実行
    test_pairwise_analysis()
