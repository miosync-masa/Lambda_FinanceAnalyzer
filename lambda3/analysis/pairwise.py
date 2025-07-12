# ==========================================================
# lambda3/analysis/pairwise.py (Lambda³理論完全準拠版)
# Pairwise Interaction Analysis for Lambda³ Theory
# ==========================================================

"""
Lambda³理論ペアワイズ相互作用解析モジュール（完全準拠版）

構造テンソル(Λ)系列間の非対称相互作用を定量化し、
∆ΛC pulsationsの相互響応パターンを解析。

重要: 全ての相関・同期計算は∆ΛC（構造変化）とρT（張力スカラー）
のみで実行。元データの直接使用は厳禁。

核心概念:
- 非対称相互作用: A→B と B→A の方向別影響度
- 構造結合: 構造テンソル変化の相互同期
- 因果構造: 時間非依存の構造空間因果関係
- 張力伝播: ρT張力スカラーの系列間伝播

Author: Mamichi Iizumi (Miosync, Inc.)
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import warnings
import time

# ベイズ分析（オプション）
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian analysis will be disabled.")

# 型定義のインポート（循環回避）
try:
    from ..core.types import (
        StructuralTensorProtocol,
        PairwiseResultProtocol,
        ConfigProtocol,
        FloatArray,
        ArrayLike,
        Lambda3Error,
        ensure_float_array,
        is_structural_tensor_compatible
    )
    TYPES_AVAILABLE = True
except ImportError as e:
    TYPES_AVAILABLE = False
    warnings.warn(f"Types module not available: {e}")
    # フォールバック
    StructuralTensorProtocol = Any
    PairwiseResultProtocol = Any
    Lambda3Error = Exception

# 設定のインポート
try:
    from ..core.config import L3BaseConfig, L3PairwiseConfig, L3BayesianConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # ダミー設定
    class L3BaseConfig:
        def __init__(self):
            self.window = 10
            self.threshold_percentile = 95.0

# 構造テンソル（条件付きインポート）
if TYPE_CHECKING:
    from ..core.structural_tensor import StructuralTensorFeatures
else:
    try:
        from ..core.structural_tensor import StructuralTensorFeatures
        STRUCTURAL_TENSOR_AVAILABLE = True
    except ImportError:
        STRUCTURAL_TENSOR_AVAILABLE = False
        warnings.warn("StructuralTensorFeatures not available - using Protocol")

# JIT最適化関数のインポート（修正版）
try:
    from ..core.jit_functions import (
        calculate_sync_profile_fixed,
        calculate_sync_rate_at_lag_fixed,
        detect_phase_coupling_fixed,
        normalize_array_fixed,
        moving_average_fixed,
        exponential_smoothing_fixed,
        safe_divide_fixed
    )
    JIT_FUNCTIONS_AVAILABLE = True
    
except ImportError:
    JIT_FUNCTIONS_AVAILABLE = False
    warnings.warn("JIT functions not available. Using fallback implementations.")

# ==========================================================
# ペアワイズ分析結果データクラス（Protocol準拠）
# ==========================================================

@dataclass
class PairwiseInteractionResults:
    """
    ペアワイズ相互作用分析結果（Lambda³理論準拠版）
    
    全ての指標は構造テンソル成分（∆ΛC, ρT）から算出。
    元データの直接相関は含まない。
    """
    
    # 系列識別子
    name_a: str = "Series_A"
    name_b: str = "Series_B"
    analysis_timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    
    # 構造同期指標（∆ΛCベース）
    structure_synchronization: float = 0.0     # ∆ΛC同期強度
    tension_synchronization: float = 0.0       # ρT同期強度
    
    # 構造因果性指標（∆ΛCベース）
    structure_causality_a_to_b: float = 0.0    # ∆ΛC: A→B因果
    structure_causality_b_to_a: float = 0.0    # ∆ΛC: B→A因果
    
    # 張力因果性指標（ρTベース）
    tension_causality_a_to_b: float = 0.0      # ρT: A→B因果
    tension_causality_b_to_a: float = 0.0      # ρT: B→A因果
    
    # 統合指標
    asymmetry_index: float = 0.0              # 非対称性指標
    interaction_strength: float = 0.0          # 相互作用強度
    
    # データ品質
    data_overlap_length: int = 0               # データ重複長
    structure_quality: float = 0.0             # 構造品質
    
    # 詳細分析結果
    structure_sync_profile: Dict[str, float] = field(default_factory=dict)
    tension_sync_profile: Dict[str, float] = field(default_factory=dict)
    interaction_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    phase_coupling: Dict[str, float] = field(default_factory=dict)
    
    # メタデータ
    analysis_method: str = "standard"          # 分析手法
    bayesian_trace: Optional[Any] = None       # ベイズトレース
    processing_time: float = 0.0               # 処理時間
    
    # 拡張メトリクス
    asymmetry_metrics: Dict[str, float] = field(default_factory=dict)
    causality_patterns: Dict[str, Dict[int, float]] = field(default_factory=dict)
    hierarchical_interaction: Dict[str, float] = field(default_factory=dict)
    interaction_quality: Dict[str, float] = field(default_factory=dict)
    
    @property
    def synchronization_strength(self) -> float:
        """統合同期強度（後方互換性）"""
        return (self.structure_synchronization + self.tension_synchronization) / 2
    
    @property
    def causality_a_to_b(self) -> float:
        """統合因果強度 A→B（後方互換性）"""
        return (self.structure_causality_a_to_b + self.tension_causality_a_to_b) / 2
    
    @property
    def causality_b_to_a(self) -> float:
        """統合因果強度 B→A（後方互換性）"""
        return (self.structure_causality_b_to_a + self.tension_causality_b_to_a) / 2
    
    @property
    def correlation_quality(self) -> float:
        """相関品質（後方互換性）"""
        return self.structure_quality
    
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
        total_a_to_b = self.causality_a_to_b
        total_b_to_a = self.causality_b_to_a
        
        if total_a_to_b > total_b_to_a * 1.2:
            return 'a_to_b'
        elif total_b_to_a > total_a_to_b * 1.2:
            return 'b_to_a'
        else:
            return 'symmetric'
    
    def calculate_bidirectional_coupling(self) -> float:
        """双方向結合強度計算"""
        return (self.causality_a_to_b + self.causality_b_to_a) / 2

# ==========================================================
# ペアワイズ分析器クラス（Lambda³理論準拠版）
# ==========================================================

class PairwiseAnalyzer:
    """
    Lambda³ペアワイズ相互作用分析器（完全準拠版）
    
    全ての分析は構造テンソル成分（∆ΛC, ρT）のみで実行。
    元データの直接使用は行わない。
    """
    
    def __init__(self, config: Optional[Any] = None, use_jit: Optional[bool] = None):
        """
        Args:
            config: 設定オブジェクト
            use_jit: JIT最適化使用フラグ
        """
        # 設定の初期化
        if config is None:
            if CONFIG_AVAILABLE:
                self.config = L3BaseConfig()
            else:
                # フォールバック設定
                self.config = type('Config', (), {
                    'window': 10,
                    'threshold_percentile': 95.0,
                    'lag_window': 10,
                    'sync_threshold': 0.3
                })()
        else:
            self.config = config
        
        # JIT使用判定
        if use_jit is None:
            self.use_jit = JIT_FUNCTIONS_AVAILABLE and getattr(self.config, 'enable_jit', True)
        else:
            self.use_jit = use_jit and JIT_FUNCTIONS_AVAILABLE
        
        print(f"PairwiseAnalyzer initialized: JIT={self.use_jit}")
    
    def analyze_asymmetric_interaction(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol,
        use_bayesian: bool = False
    ) -> PairwiseInteractionResults:
        """
        非対称相互作用分析実行（Lambda³理論準拠）
        
        Args:
            features_a: 系列Aの構造テンソル特徴量
            features_b: 系列Bの構造テンソル特徴量
            use_bayesian: ベイズ分析使用フラグ
            
        Returns:
            PairwiseInteractionResults: ペアワイズ分析結果
        """
        start_time = time.time()
        
        # 入力検証
        self._validate_inputs(features_a, features_b)
        
        # 構造テンソル成分の取得
        components_a = self._extract_tensor_components(features_a)
        components_b = self._extract_tensor_components(features_b)
        
        name_a = components_a['name']
        name_b = components_b['name']
        
        print(f"Analyzing pairwise interaction: {name_a} ↔ {name_b}")
        
        try:
            if use_bayesian and BAYESIAN_AVAILABLE:
                # ベイズペアワイズ分析
                results = self._analyze_with_bayesian(components_a, components_b)
            else:
                # 標準ペアワイズ分析
                results = self._analyze_standard(components_a, components_b)
            
            # 処理時間記録
            results.processing_time = time.time() - start_time
            
            return results
            
        except Exception as e:
            raise Lambda3Error(f"Pairwise analysis failed for {name_a}-{name_b}: {e}")
    
    def _validate_inputs(self, features_a: StructuralTensorProtocol, features_b: StructuralTensorProtocol):
        """入力検証（Lambda³理論準拠）"""
        # ∆ΛC成分の存在確認
        if not self._has_structure_components(features_a):
            raise Lambda3Error("features_a lacks required ∆ΛC components")
        if not self._has_structure_components(features_b):
            raise Lambda3Error("features_b lacks required ∆ΛC components")
        
        # データ長の確認
        length_a = self._get_component_length(features_a)
        length_b = self._get_component_length(features_b)
        
        if length_a < 10 or length_b < 10:
            raise Lambda3Error("Insufficient data for pairwise analysis")
    
    def _has_structure_components(self, features: StructuralTensorProtocol) -> bool:
        """構造テンソル成分の存在確認"""
        if hasattr(features, 'delta_LambdaC_pos') and hasattr(features, 'rho_T'):
            return True
        elif isinstance(features, dict) and 'delta_LambdaC_pos' in features and 'rho_T' in features:
            return True
        return False
    
    def _get_component_length(self, features: StructuralTensorProtocol) -> int:
        """構造テンソル成分の長さ取得"""
        if hasattr(features, 'delta_LambdaC_pos') and features.delta_LambdaC_pos is not None:
            return len(features.delta_LambdaC_pos)
        elif isinstance(features, dict) and 'delta_LambdaC_pos' in features:
            return len(features['delta_LambdaC_pos'])
        return 0
    
    def _extract_tensor_components(self, features: StructuralTensorProtocol) -> Dict[str, Any]:
        """構造テンソル成分の抽出（Lambda³理論準拠）"""
        
        components = {}
        
        # 名前の取得
        if hasattr(features, 'series_name'):
            components['name'] = features.series_name
        elif isinstance(features, dict) and 'series_name' in features:
            components['name'] = features['series_name']
        else:
            components['name'] = 'Series'
        
        # ∆ΛC成分の取得
        if hasattr(features, 'delta_LambdaC_pos'):
            components['delta_pos'] = features.delta_LambdaC_pos
            components['delta_neg'] = features.delta_LambdaC_neg
        elif isinstance(features, dict):
            components['delta_pos'] = features.get('delta_LambdaC_pos', np.array([]))
            components['delta_neg'] = features.get('delta_LambdaC_neg', np.array([]))
        
        # ρT成分の取得
        if hasattr(features, 'rho_T'):
            components['rho_T'] = features.rho_T
        elif isinstance(features, dict):
            components['rho_T'] = features.get('rho_T', np.array([]))
        
        # 階層的特徴の取得（オプション）
        if hasattr(features, 'local_pos'):
            components['local_pos'] = features.local_pos
            components['local_neg'] = features.local_neg
            components['global_pos'] = features.global_pos
            components['global_neg'] = features.global_neg
        elif isinstance(features, dict):
            components['local_pos'] = features.get('local_pos', None)
            components['local_neg'] = features.get('local_neg', None)
            components['global_pos'] = features.get('global_pos', None)
            components['global_neg'] = features.get('global_neg', None)
        
        # 型変換と検証
        for key in ['delta_pos', 'delta_neg', 'rho_T']:
            if key in components and components[key] is not None:
                components[key] = np.asarray(components[key], dtype=np.float64)
        
        return components
    
    def _analyze_standard(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> PairwiseInteractionResults:
        """標準ペアワイズ分析（Lambda³理論準拠）"""
        
        name_a = components_a['name']
        name_b = components_b['name']
        
        print(f"Running standard pairwise analysis: {name_a} ↔ {name_b}")
        
        # データ長の統一
        min_length = min(
            len(components_a['delta_pos']),
            len(components_b['delta_pos'])
        )
        
        # 構造同期分析（∆ΛCベース）
        structure_sync_results = self._calculate_structure_synchronization(
            components_a, components_b, min_length
        )
        
        # 張力同期分析（ρTベース）
        tension_sync_results = self._calculate_tension_synchronization(
            components_a, components_b, min_length
        )
        
        # 構造因果性分析（∆ΛCベース）
        structure_causality = self._calculate_structure_causality(
            components_a, components_b, min_length
        )
        
        # 張力因果性分析（ρTベース）
        tension_causality = self._calculate_tension_causality(
            components_a, components_b, min_length
        )
        
        # 相互作用係数計算
        interaction_coeffs = self._calculate_interaction_coefficients(
            components_a, components_b, min_length
        )
        
        # 位相結合分析（構造テンソルベース）
        phase_coupling = self._calculate_phase_coupling(
            components_a, components_b, min_length
        )
        
        # 品質評価
        structure_quality = self._assess_structure_quality(
            components_a, components_b, min_length
        )
        
        # 拡張メトリクス計算
        asymmetry_metrics = self._calculate_asymmetry_metrics(
            structure_causality, tension_causality
        )
        
        causality_patterns = self._calculate_causality_patterns(
            components_a, components_b, min_length
        )
        
        hierarchical_interaction = self._calculate_hierarchical_interaction(
            components_a, components_b, min_length
        )
        
        interaction_quality = self._calculate_interaction_quality(
            structure_sync_results, tension_sync_results,
            structure_causality, tension_causality,
            structure_quality
        )
        
        # 統合指標の計算
        asymmetry_index = self._calculate_asymmetry_index(
            structure_causality, tension_causality
        )
        
        interaction_strength = self._calculate_interaction_strength(
            structure_sync_results, tension_sync_results,
            structure_causality, tension_causality
        )
        
        # 結果構築
        results = PairwiseInteractionResults(
            name_a=name_a,
            name_b=name_b,
            structure_synchronization=structure_sync_results['sync_strength'],
            tension_synchronization=tension_sync_results['sync_strength'],
            structure_causality_a_to_b=structure_causality['a_to_b'],
            structure_causality_b_to_a=structure_causality['b_to_a'],
            tension_causality_a_to_b=tension_causality['a_to_b'],
            tension_causality_b_to_a=tension_causality['b_to_a'],
            asymmetry_index=asymmetry_index,
            interaction_strength=interaction_strength,
            data_overlap_length=min_length,
            structure_quality=structure_quality,
            structure_sync_profile=structure_sync_results['profile'],
            tension_sync_profile=tension_sync_results['profile'],
            interaction_coefficients=interaction_coeffs,
            phase_coupling=phase_coupling,
            analysis_method='standard',
            asymmetry_metrics=asymmetry_metrics,
            causality_patterns=causality_patterns,
            hierarchical_interaction=hierarchical_interaction,
            interaction_quality=interaction_quality
        )
        
        return results
    
    def _calculate_structure_synchronization(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, Any]:
        """構造同期計算（∆ΛCベース）"""
        
        # 構造変化イベントの統合
        events_a = components_a['delta_pos'][:min_length] + components_a['delta_neg'][:min_length]
        events_b = components_b['delta_pos'][:min_length] + components_b['delta_neg'][:min_length]
        
        # 基本同期強度
        if np.sum(events_a) > 0 and np.sum(events_b) > 0:
            # 構造変化の同時発生率
            sync_strength = np.mean(events_a * events_b)
            
            # 条件付き同期率
            conditional_sync_ab = np.sum(events_a * events_b) / (np.sum(events_a) + 1e-8)
            conditional_sync_ba = np.sum(events_a * events_b) / (np.sum(events_b) + 1e-8)
        else:
            sync_strength = 0.0
            conditional_sync_ab = 0.0
            conditional_sync_ba = 0.0
        
        # 同期プロファイル計算
        sync_profile = self._calculate_sync_profile(events_a, events_b)
        
        return {
            'sync_strength': sync_strength,
            'conditional_ab': conditional_sync_ab,
            'conditional_ba': conditional_sync_ba,
            'profile': sync_profile
        }
    
    def _calculate_tension_synchronization(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, Any]:
        """張力同期計算（ρTベース）"""
        
        rho_a = components_a['rho_T'][:min_length]
        rho_b = components_b['rho_T'][:min_length]
        
        # 張力相関（正規化後）
        if len(rho_a) > 1 and len(rho_b) > 1:
            # 正規化
            rho_a_norm = (rho_a - np.mean(rho_a)) / (np.std(rho_a) + 1e-8)
            rho_b_norm = (rho_b - np.mean(rho_b)) / (np.std(rho_b) + 1e-8)
            
            # 相関計算
            sync_strength = abs(np.corrcoef(rho_a_norm, rho_b_norm)[0, 1])
            if np.isnan(sync_strength):
                sync_strength = 0.0
        else:
            sync_strength = 0.0
        
        # 同期プロファイル計算
        sync_profile = self._calculate_sync_profile(rho_a, rho_b)
        
        return {
            'sync_strength': sync_strength,
            'profile': sync_profile
        }
    
    def _calculate_structure_causality(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, float]:
        """構造因果性計算（∆ΛCベース）"""
        
        events_a = components_a['delta_pos'][:min_length] + components_a['delta_neg'][:min_length]
        events_b = components_b['delta_pos'][:min_length] + components_b['delta_neg'][:min_length]
        
        causality_a_to_b = 0.0
        causality_b_to_a = 0.0
        
        # 遅延因果性の計算
        for lag in range(1, min(10, min_length // 5)):
            if lag < min_length:
                # A(t-lag) → B(t)
                if np.sum(events_a[:-lag]) > 0:
                    joint_ab = np.sum(events_a[:-lag] * events_b[lag:])
                    marginal_a = np.sum(events_a[:-lag])
                    causality_a_to_b = max(causality_a_to_b, joint_ab / marginal_a)
                
                # B(t-lag) → A(t)
                if np.sum(events_b[:-lag]) > 0:
                    joint_ba = np.sum(events_b[:-lag] * events_a[lag:])
                    marginal_b = np.sum(events_b[:-lag])
                    causality_b_to_a = max(causality_b_to_a, joint_ba / marginal_b)
        
        return {
            'a_to_b': causality_a_to_b,
            'b_to_a': causality_b_to_a
        }
    
    def _calculate_tension_causality(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, float]:
        """張力因果性計算（ρTベース）"""
        
        rho_a = components_a['rho_T'][:min_length]
        rho_b = components_b['rho_T'][:min_length]
        
        causality_a_to_b = 0.0
        causality_b_to_a = 0.0
        
        # 遅延相関による因果性
        for lag in range(1, min(10, min_length // 5)):
            if lag < min_length:
                # A(t-lag) → B(t)
                if len(rho_a[:-lag]) > 1 and len(rho_b[lag:]) > 1:
                    corr_ab = np.corrcoef(rho_a[:-lag], rho_b[lag:])[0, 1]
                    if not np.isnan(corr_ab):
                        causality_a_to_b = max(causality_a_to_b, abs(corr_ab))
                
                # B(t-lag) → A(t)
                if len(rho_b[:-lag]) > 1 and len(rho_a[lag:]) > 1:
                    corr_ba = np.corrcoef(rho_b[:-lag], rho_a[lag:])[0, 1]
                    if not np.isnan(corr_ba):
                        causality_b_to_a = max(causality_b_to_a, abs(corr_ba))
        
        return {
            'a_to_b': causality_a_to_b,
            'b_to_a': causality_b_to_a
        }
    
    def _calculate_sync_profile(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray
    ) -> Dict[str, float]:
        """同期プロファイル計算（汎用）"""
        
        lag_window = getattr(self.config, 'lag_window', 10)
        sync_profile = {}
        max_sync = 0.0
        optimal_lag = 0
        
        for lag in range(-lag_window, lag_window + 1):
            if lag < 0:
                # B leads A
                if -lag < len(series_a):
                    a_lagged = series_a[-lag:]
                    b_current = series_b[:lag]
                    if len(a_lagged) > 1 and len(b_current) > 1:
                        sync = abs(np.corrcoef(a_lagged, b_current)[0, 1])
                        if np.isnan(sync):
                            sync = 0.0
                    else:
                        sync = 0.0
                else:
                    sync = 0.0
            elif lag > 0:
                # A leads B
                if lag < len(series_b):
                    a_current = series_a[:-lag]
                    b_lagged = series_b[lag:]
                    if len(a_current) > 1 and len(b_lagged) > 1:
                        sync = abs(np.corrcoef(a_current, b_lagged)[0, 1])
                        if np.isnan(sync):
                            sync = 0.0
                    else:
                        sync = 0.0
                else:
                    sync = 0.0
            else:
                # No lag
                if len(series_a) > 1 and len(series_b) > 1:
                    sync = abs(np.corrcoef(series_a, series_b)[0, 1])
                    if np.isnan(sync):
                        sync = 0.0
                else:
                    sync = 0.0
            
            sync_profile[lag] = sync
            
            if sync > max_sync:
                max_sync = sync
                optimal_lag = lag
        
        return {
            'max_sync': max_sync,
            'optimal_lag': optimal_lag,
            'profile': sync_profile
        }
    
    def _calculate_interaction_coefficients(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, Dict[str, float]]:
        """相互作用係数計算（Lambda³理論準拠）"""
        
        # 構造-張力相互作用
        structure_tension_interaction = self._calculate_structure_tension_interaction(
            components_a, components_b, min_length
        )
        
        # 階層間相互作用（階層的特徴が利用可能な場合）
        if components_a.get('local_pos') is not None and components_b.get('local_pos') is not None:
            hierarchical_interaction = self._calculate_hierarchical_coefficients(
                components_a, components_b, min_length
            )
        else:
            hierarchical_interaction = {}
        
        return {
            'structure_tension': structure_tension_interaction,
            'hierarchical': hierarchical_interaction
        }
    
    def _calculate_structure_tension_interaction(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, float]:
        """構造-張力相互作用計算"""
        
        # 構造変化→張力への影響
        events_a = components_a['delta_pos'][:min_length] + components_a['delta_neg'][:min_length]
        events_b = components_b['delta_pos'][:min_length] + components_b['delta_neg'][:min_length]
        rho_a = components_a['rho_T'][:min_length]
        rho_b = components_b['rho_T'][:min_length]
        
        # 構造変化時の張力応答
        structure_to_tension_a = 0.0
        structure_to_tension_b = 0.0
        
        if np.sum(events_a) > 0:
            # A構造変化時のB張力応答
            event_indices_a = np.where(events_a > 0)[0]
            for idx in event_indices_a:
                if idx + 5 < min_length:  # 5ステップ先まで見る
                    tension_response = np.mean(rho_b[idx:idx+5])
                    baseline = np.mean(rho_b[max(0, idx-5):idx]) if idx > 0 else np.mean(rho_b)
                    structure_to_tension_a += abs(tension_response - baseline)
            structure_to_tension_a /= (len(event_indices_a) + 1e-8)
        
        if np.sum(events_b) > 0:
            # B構造変化時のA張力応答
            event_indices_b = np.where(events_b > 0)[0]
            for idx in event_indices_b:
                if idx + 5 < min_length:
                    tension_response = np.mean(rho_a[idx:idx+5])
                    baseline = np.mean(rho_a[max(0, idx-5):idx]) if idx > 0 else np.mean(rho_a)
                    structure_to_tension_b += abs(tension_response - baseline)
            structure_to_tension_b /= (len(event_indices_b) + 1e-8)
        
        return {
            'structure_a_to_tension_b': structure_to_tension_a,
            'structure_b_to_tension_a': structure_to_tension_b
        }
    
    def _calculate_hierarchical_coefficients(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, float]:
        """階層間相互作用係数計算"""
        
        # 局所→大域相互作用
        local_to_global_a = 0.0
        local_to_global_b = 0.0
        
        # 局所イベント
        local_events_a = (components_a['local_pos'][:min_length] + 
                         components_a['local_neg'][:min_length])
        local_events_b = (components_b['local_pos'][:min_length] + 
                         components_b['local_neg'][:min_length])
        
        # 大域イベント
        global_events_a = (components_a['global_pos'][:min_length] + 
                          components_a['global_neg'][:min_length])
        global_events_b = (components_b['global_pos'][:min_length] + 
                          components_b['global_neg'][:min_length])
        
        # 局所Aが大域Bに与える影響
        if np.sum(local_events_a) > 0 and np.sum(global_events_b) > 0:
            local_to_global_a = np.sum(local_events_a * global_events_b) / np.sum(local_events_a)
        
        # 局所Bが大域Aに与える影響
        if np.sum(local_events_b) > 0 and np.sum(global_events_a) > 0:
            local_to_global_b = np.sum(local_events_b * global_events_a) / np.sum(local_events_b)
        
        return {
            'local_a_to_global_b': local_to_global_a,
            'local_b_to_global_a': local_to_global_b
        }
    
    def _calculate_phase_coupling(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, float]:
        """位相結合計算（構造テンソルベース）"""
        
        rho_a = components_a['rho_T'][:min_length]
        rho_b = components_b['rho_T'][:min_length]
        
        if len(rho_a) > 10 and len(rho_b) > 10:
            # 張力変化率（位相情報）
            phase_a = np.diff(rho_a)
            phase_b = np.diff(rho_b)
            
            # 位相同期
            if len(phase_a) > 1 and len(phase_b) > 1:
                coupling_strength = abs(np.corrcoef(phase_a, phase_b)[0, 1])
                if np.isnan(coupling_strength):
                    coupling_strength = 0.0
                
                # 位相遅延
                cross_corr = np.correlate(phase_a, phase_b, mode='full')
                phase_lag = np.argmax(cross_corr) - len(phase_a) + 1
            else:
                coupling_strength = 0.0
                phase_lag = 0
        else:
            coupling_strength = 0.0
            phase_lag = 0
        
        return {
            'coupling_strength': coupling_strength,
            'phase_lag': float(phase_lag)
        }
    
    def _assess_structure_quality(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> float:
        """構造品質評価"""
        
        # データ長品質
        length_quality = min(1.0, min_length / 50)
        
        # 構造変化の豊富さ
        events_a = components_a['delta_pos'][:min_length] + components_a['delta_neg'][:min_length]
        events_b = components_b['delta_pos'][:min_length] + components_b['delta_neg'][:min_length]
        
        event_density_a = np.sum(events_a) / min_length
        event_density_b = np.sum(events_b) / min_length
        
        # 適度な構造変化密度（0.05-0.2が理想的）
        density_quality_a = min(1.0, event_density_a / 0.1) if event_density_a < 0.2 else 0.5
        density_quality_b = min(1.0, event_density_b / 0.1) if event_density_b < 0.2 else 0.5
        density_quality = (density_quality_a + density_quality_b) / 2
        
        # 張力の変動性
        rho_std_a = np.std(components_a['rho_T'][:min_length])
        rho_std_b = np.std(components_b['rho_T'][:min_length])
        variability_quality = 1.0 if rho_std_a > 1e-3 and rho_std_b > 1e-3 else 0.5
        
        # 統合品質
        overall_quality = (
            length_quality * 0.3 +
            density_quality * 0.4 +
            variability_quality * 0.3
        )
        
        return overall_quality
    
    def _calculate_asymmetry_metrics(
        self,
        structure_causality: Dict[str, float],
        tension_causality: Dict[str, float]
    ) -> Dict[str, float]:
        """非対称性メトリクス計算"""
        
        # 構造非対称性
        structure_asymmetry = abs(
            structure_causality['a_to_b'] - structure_causality['b_to_a']
        )
        
        # 張力非対称性
        tension_asymmetry = abs(
            tension_causality['a_to_b'] - tension_causality['b_to_a']
        )
        
        # 統合非対称性
        total_asymmetry = (structure_asymmetry + tension_asymmetry) / 2
        
        # 優勢性
        total_a_to_b = (structure_causality['a_to_b'] + tension_causality['a_to_b']) / 2
        total_b_to_a = (structure_causality['b_to_a'] + tension_causality['b_to_a']) / 2
        
        if min(total_a_to_b, total_b_to_a) > 1e-8:
            dominance = max(total_a_to_b, total_b_to_a) / min(total_a_to_b, total_b_to_a)
        else:
            dominance = 1.0
        
        return {
            'structure_asymmetry': structure_asymmetry,
            'tension_asymmetry': tension_asymmetry,
            'total_asymmetry': total_asymmetry,
            'dominance': dominance
        }
    
    def _calculate_causality_patterns(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, Dict[int, float]]:
        """因果性パターン計算（構造テンソルベース）"""
        
        patterns = {
            'structure_a_to_b': {},
            'structure_b_to_a': {},
            'tension_a_to_b': {},
            'tension_b_to_a': {}
        }
        
        # 構造変化イベント
        events_a = components_a['delta_pos'][:min_length] + components_a['delta_neg'][:min_length]
        events_b = components_b['delta_pos'][:min_length] + components_b['delta_neg'][:min_length]
        
        # 張力
        rho_a = components_a['rho_T'][:min_length]
        rho_b = components_b['rho_T'][:min_length]
        
        # 各遅延での因果性
        for lag in range(1, min(20, min_length // 10)):
            # 構造因果性
            if lag < min_length and np.sum(events_a[:-lag]) > 0:
                patterns['structure_a_to_b'][lag] = (
                    np.sum(events_a[:-lag] * events_b[lag:]) / np.sum(events_a[:-lag])
                )
            
            if lag < min_length and np.sum(events_b[:-lag]) > 0:
                patterns['structure_b_to_a'][lag] = (
                    np.sum(events_b[:-lag] * events_a[lag:]) / np.sum(events_b[:-lag])
                )
            
            # 張力因果性
            if lag < min_length and len(rho_a[:-lag]) > 1:
                corr_ab = np.corrcoef(rho_a[:-lag], rho_b[lag:])[0, 1]
                patterns['tension_a_to_b'][lag] = abs(corr_ab) if not np.isnan(corr_ab) else 0.0
            
            if lag < min_length and len(rho_b[:-lag]) > 1:
                corr_ba = np.corrcoef(rho_b[:-lag], rho_a[lag:])[0, 1]
                patterns['tension_b_to_a'][lag] = abs(corr_ba) if not np.isnan(corr_ba) else 0.0
        
        return patterns
    
    def _calculate_hierarchical_interaction(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Dict[str, float]:
        """階層的相互作用計算"""
        
        if components_a.get('local_pos') is None or components_b.get('local_pos') is None:
            return {}
        
        # 各階層での同期強度
        local_sync = self._calculate_hierarchical_sync(
            components_a['local_pos'][:min_length] + components_a['local_neg'][:min_length],
            components_b['local_pos'][:min_length] + components_b['local_neg'][:min_length]
        )
        
        global_sync = self._calculate_hierarchical_sync(
            components_a['global_pos'][:min_length] + components_a['global_neg'][:min_length],
            components_b['global_pos'][:min_length] + components_b['global_neg'][:min_length]
        )
        
        # クロススケール相互作用
        cross_scale_a_to_b = self._calculate_cross_scale_interaction(
            components_a['local_pos'][:min_length] + components_a['local_neg'][:min_length],
            components_b['global_pos'][:min_length] + components_b['global_neg'][:min_length]
        )
        
        cross_scale_b_to_a = self._calculate_cross_scale_interaction(
            components_b['local_pos'][:min_length] + components_b['local_neg'][:min_length],
            components_a['global_pos'][:min_length] + components_a['global_neg'][:min_length]
        )
        
        return {
            'local_sync': local_sync,
            'global_sync': global_sync,
            'cross_scale_a_to_b': cross_scale_a_to_b,
            'cross_scale_b_to_a': cross_scale_b_to_a
        }
    
    def _calculate_hierarchical_sync(self, events_a: np.ndarray, events_b: np.ndarray) -> float:
        """階層的同期計算"""
        if np.sum(events_a) > 0 and np.sum(events_b) > 0:
            return np.sum(events_a * events_b) / min(np.sum(events_a), np.sum(events_b))
        return 0.0
    
    def _calculate_cross_scale_interaction(self, local_events: np.ndarray, global_events: np.ndarray) -> float:
        """クロススケール相互作用計算"""
        if np.sum(local_events) > 0 and np.sum(global_events) > 0:
            # 局所イベントが大域イベントに先行する確率
            interaction = 0.0
            for lag in range(1, min(10, len(local_events) // 10)):
                if lag < len(global_events):
                    lagged_interaction = np.sum(local_events[:-lag] * global_events[lag:])
                    interaction = max(interaction, lagged_interaction / (np.sum(local_events[:-lag]) + 1e-8))
            return interaction
        return 0.0
    
    def _calculate_interaction_quality(
        self,
        structure_sync: Dict[str, Any],
        tension_sync: Dict[str, Any],
        structure_causality: Dict[str, float],
        tension_causality: Dict[str, float],
        structure_quality: float
    ) -> Dict[str, float]:
        """相互作用品質計算"""
        
        # 同期品質
        sync_quality = (structure_sync['sync_strength'] + tension_sync['sync_strength']) / 2
        
        # 因果品質
        causality_quality = (
            (structure_causality['a_to_b'] + structure_causality['b_to_a'] +
             tension_causality['a_to_b'] + tension_causality['b_to_a']) / 4
        )
        
        # 総合品質
        overall_quality = (
            sync_quality * 0.3 +
            causality_quality * 0.3 +
            structure_quality * 0.4
        )
        
        return {
            'sync_quality': sync_quality,
            'causality_quality': causality_quality,
            'structure_quality': structure_quality,
            'overall_quality': overall_quality
        }
    
    def _calculate_asymmetry_index(
        self,
        structure_causality: Dict[str, float],
        tension_causality: Dict[str, float]
    ) -> float:
        """非対称性指標計算"""
        
        # 各成分の非対称性
        structure_asym = abs(structure_causality['a_to_b'] - structure_causality['b_to_a'])
        tension_asym = abs(tension_causality['a_to_b'] - tension_causality['b_to_a'])
        
        # 統合非対称性（重み付き平均）
        return structure_asym * 0.6 + tension_asym * 0.4
    
    def _calculate_interaction_strength(
        self,
        structure_sync: Dict[str, Any],
        tension_sync: Dict[str, Any],
        structure_causality: Dict[str, float],
        tension_causality: Dict[str, float]
    ) -> float:
        """相互作用強度計算"""
        
        # 同期成分
        sync_strength = (structure_sync['sync_strength'] + tension_sync['sync_strength']) / 2
        
        # 因果成分
        causality_strength = (
            structure_causality['a_to_b'] + structure_causality['b_to_a'] +
            tension_causality['a_to_b'] + tension_causality['b_to_a']
        ) / 4
        
        # 統合強度
        return sync_strength * 0.5 + causality_strength * 0.5
    
    def _analyze_with_bayesian(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any]
    ) -> PairwiseInteractionResults:
        """ベイズペアワイズ分析（Lambda³理論準拠）"""
        
        name_a = components_a['name']
        name_b = components_b['name']
        
        print(f"Running Bayesian pairwise analysis: {name_a} ↔ {name_b}")
        
        # データ長の統一
        min_length = min(
            len(components_a['delta_pos']),
            len(components_b['delta_pos'])
        )
        
        # ベイズモデル構築と推定
        trace, model = self._fit_bayesian_lambda3_model(components_a, components_b, min_length)
        
        # ベイズ結果から係数抽出
        bayesian_coeffs = self._extract_bayesian_coefficients(trace)
        
        # 標準分析も実行（比較用）
        standard_results = self._analyze_standard(components_a, components_b)
        
        # ベイズ結果で更新
        standard_results.analysis_method = 'bayesian'
        standard_results.bayesian_trace = trace
        
        # ベイズ係数で因果性を更新
        if 'structure_a_to_b' in bayesian_coeffs:
            standard_results.structure_causality_a_to_b = bayesian_coeffs['structure_a_to_b']
            standard_results.structure_causality_b_to_a = bayesian_coeffs['structure_b_to_a']
            standard_results.tension_causality_a_to_b = bayesian_coeffs['tension_a_to_b']
            standard_results.tension_causality_b_to_a = bayesian_coeffs['tension_b_to_a']
        
        return standard_results
    
    def _fit_bayesian_lambda3_model(
        self,
        components_a: Dict[str, Any],
        components_b: Dict[str, Any],
        min_length: int
    ) -> Tuple[Any, Any]:
        """Lambda³ベイズモデル推定"""
        
        # 構造テンソル成分の準備
        delta_pos_a = components_a['delta_pos'][:min_length]
        delta_neg_a = components_a['delta_neg'][:min_length]
        rho_a = components_a['rho_T'][:min_length]
        
        delta_pos_b = components_b['delta_pos'][:min_length]
        delta_neg_b = components_b['delta_neg'][:min_length]
        rho_b = components_b['rho_T'][:min_length]
        
        with pm.Model() as model:
            # === 構造相互作用パラメータ ===
            # A → B 構造影響
            beta_struct_ab_pos = pm.Normal('beta_struct_ab_pos', mu=0, sigma=2)
            beta_struct_ab_neg = pm.Normal('beta_struct_ab_neg', mu=0, sigma=2)
            
            # B → A 構造影響
            beta_struct_ba_pos = pm.Normal('beta_struct_ba_pos', mu=0, sigma=2)
            beta_struct_ba_neg = pm.Normal('beta_struct_ba_neg', mu=0, sigma=2)
            
            # === 張力相互作用パラメータ ===
            beta_tension_ab = pm.Normal('beta_tension_ab', mu=0, sigma=1.5)
            beta_tension_ba = pm.Normal('beta_tension_ba', mu=0, sigma=1.5)
            
            # === 構造→張力変換パラメータ ===
            gamma_struct_to_tension_a = pm.Normal('gamma_struct_to_tension_a', mu=0, sigma=1)
            gamma_struct_to_tension_b = pm.Normal('gamma_struct_to_tension_b', mu=0, sigma=1)
            
            # === 張力モデル ===
            # A系列の張力
            mu_rho_a = (
                gamma_struct_to_tension_a * (delta_pos_a + delta_neg_a) +
                beta_tension_ba * rho_b
            )
            
            # B系列の張力
            mu_rho_b = (
                gamma_struct_to_tension_b * (delta_pos_b + delta_neg_b) +
                beta_tension_ab * rho_a
            )
            
            # ノイズパラメータ
            sigma_rho_a = pm.HalfNormal('sigma_rho_a', sigma=1)
            sigma_rho_b = pm.HalfNormal('sigma_rho_b', sigma=1)
            
            # 観測モデル
            obs_rho_a = pm.Normal('obs_rho_a', mu=mu_rho_a, sigma=sigma_rho_a, observed=rho_a)
            obs_rho_b = pm.Normal('obs_rho_b', mu=mu_rho_b, sigma=sigma_rho_b, observed=rho_b)
            
            # サンプリング
            trace = pm.sample(
                draws=1000,
                tune=500,
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
                'structure_a_to_b': abs(summary.loc['beta_struct_ab_pos', 'mean'] + 
                                       summary.loc['beta_struct_ab_neg', 'mean']) / 2,
                'structure_b_to_a': abs(summary.loc['beta_struct_ba_pos', 'mean'] + 
                                       summary.loc['beta_struct_ba_neg', 'mean']) / 2,
                'tension_a_to_b': abs(summary.loc['beta_tension_ab', 'mean']),
                'tension_b_to_a': abs(summary.loc['beta_tension_ba', 'mean']),
                'struct_to_tension_a': abs(summary.loc['gamma_struct_to_tension_a', 'mean']),
                'struct_to_tension_b': abs(summary.loc['gamma_struct_to_tension_b', 'mean'])
            }
            
            return coefficients
            
        except Exception as e:
            print(f"Bayesian coefficient extraction failed: {e}")
            return {}

# ==========================================================
# 便利関数
# ==========================================================

def analyze_pairwise_interaction(
    features_a: StructuralTensorProtocol,
    features_b: StructuralTensorProtocol,
    config: Optional[Any] = None,
    use_bayesian: bool = False
) -> PairwiseInteractionResults:
    """
    ペアワイズ相互作用分析の便利関数（Lambda³理論準拠）
    
    Args:
        features_a: 系列Aの構造テンソル特徴量
        features_b: 系列Bの構造テンソル特徴量
        config: 設定オブジェクト
        use_bayesian: ベイズ分析使用フラグ
        
    Returns:
        PairwiseInteractionResults: ペアワイズ分析結果
    """
    analyzer = PairwiseAnalyzer(config=config)
    return analyzer.analyze_asymmetric_interaction(features_a, features_b, use_bayesian=use_bayesian)

# ==========================================================
# モジュール情報
# ==========================================================

__all__ = [
    'PairwiseInteractionResults',
    'PairwiseAnalyzer',
    'analyze_pairwise_interaction'
]
