# ==========================================================
# lambda3/analysis/pairwise.py (完全版)
# Pairwise Interaction Analysis for Lambda³ Theory
# ==========================================================

"""
Lambda³理論ペアワイズ相互作用解析モジュール（完全版）

構造テンソル(Λ)系列間の非対称相互作用を定量化し、
∆ΛC pulsationsの相互響応パターンを解析。

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
    
    # レガシー互換性
    calculate_sync_profile = calculate_sync_profile_fixed
    calculate_sync_rate_at_lag = calculate_sync_rate_at_lag_fixed
    
except ImportError:
    JIT_FUNCTIONS_AVAILABLE = False
    warnings.warn("JIT functions not available. Using fallback implementations.")
    
    # フォールバック実装
    def calculate_sync_profile_fixed(series_a, series_b, lag_window):
        lags = np.arange(-lag_window, lag_window + 1)
        sync_values = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            if lag == 0:
                if len(series_a) > 1 and len(series_b) > 1:
                    sync_values[i] = np.corrcoef(series_a, series_b)[0, 1]
            elif lag > 0 and lag < len(series_a):
                if len(series_a[:-lag]) > 1 and len(series_b[lag:]) > 1:
                    sync_values[i] = np.corrcoef(series_a[:-lag], series_b[lag:])[0, 1]
            else:
                abs_lag = -lag
                if abs_lag < len(series_b) and len(series_a[abs_lag:]) > 1:
                    sync_values[i] = np.corrcoef(series_a[abs_lag:], series_b[:-abs_lag])[0, 1]
        
        # NaN値を0に置換
        sync_values = np.nan_to_num(sync_values, nan=0.0)
        max_idx = np.argmax(np.abs(sync_values))
        return lags, sync_values, sync_values[max_idx], lags[max_idx]
    
    def normalize_array_fixed(arr, method='zscore'):
        if method == 'zscore':
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        return arr
    
    def safe_divide_fixed(num, den, default=0.0):
        return num / den if abs(den) > 1e-8 else default
    
    def detect_phase_coupling_fixed(series_a, series_b):
        if len(series_a) > 10 and len(series_b) > 10:
            phase_a = np.diff(series_a)
            phase_b = np.diff(series_b)
            min_len = min(len(phase_a), len(phase_b))
            phase_a = phase_a[:min_len]
            phase_b = phase_b[:min_len]
            if min_len > 1:
                coupling = abs(np.corrcoef(phase_a, phase_b)[0, 1])
                if np.isnan(coupling):
                    coupling = 0.0
                cross_corr = np.correlate(phase_a, phase_b, mode='full')
                lag = np.argmax(cross_corr) - len(phase_a) + 1
                return coupling, lag
        return 0.0, 0.0

# ==========================================================
# ペアワイズ分析結果データクラス（Protocol準拠）
# ==========================================================

@dataclass
class PairwiseInteractionResults:
    """
    ペアワイズ相互作用分析結果（完全版）
    
    PairwiseResultProtocolに準拠し、循環インポートを回避した
    ペアワイズ分析結果の具体実装。
    """
    
    # 系列識別子
    name_a: str = "Series_A"
    name_b: str = "Series_B"
    analysis_timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    
    # 同期性指標
    synchronization_strength: float = 0.0      # 同期強度
    structure_synchronization: float = 0.0     # 構造変化同期
    
    # 因果性指標
    causality_a_to_b: float = 0.0             # A→B因果強度
    causality_b_to_a: float = 0.0             # B→A因果強度
    asymmetry_index: float = 0.0              # 非対称性指標
    
    # データ品質
    data_overlap_length: int = 0               # データ重複長
    correlation_quality: float = 0.0          # 相関品質
    
    # 詳細分析結果
    synchronization_profile: Dict[str, float] = field(default_factory=dict)
    interaction_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    phase_coupling: Dict[str, float] = field(default_factory=dict)
    
    # メタデータ
    analysis_method: str = "standard"          # 分析手法
    bayesian_trace: Optional[Any] = None       # ベイズトレース
    processing_time: float = 0.0               # 処理時間
    
    # 拡張メトリクス
    asymmetry_metrics: Dict[str, float] = field(default_factory=dict)
    causality_patterns: Dict[str, Dict[int, float]] = field(default_factory=dict)
    prediction_metrics: Dict[str, float] = field(default_factory=dict)
    interaction_quality: Dict[str, float] = field(default_factory=dict)
    
    def get_interaction_summary(self) -> Dict[str, float]:
        """相互作用サマリー取得"""
        return {
            'synchronization': self.synchronization_strength,
            'asymmetry': self.asymmetry_index,
            'causality_a_to_b': self.causality_a_to_b,
            'causality_b_to_a': self.causality_b_to_a,
            'quality': self.correlation_quality
        }
    
    def get_dominant_direction(self) -> str:
        """優勢方向判定"""
        if self.causality_a_to_b > self.causality_b_to_a * 1.2:
            return 'a_to_b'
        elif self.causality_b_to_a > self.causality_a_to_b * 1.2:
            return 'b_to_a'
        else:
            return 'symmetric'
    
    def get_sync_strength(self) -> float:
        """同期強度取得（互換性）"""
        return self.synchronization_strength
    
    def get_asymmetry_strength(self) -> float:
        """非対称性強度取得（互換性）"""
        return self.asymmetry_index
    
    def calculate_bidirectional_coupling(self) -> float:
        """双方向結合強度計算"""
        return (self.causality_a_to_b + self.causality_b_to_a) / 2
    
    def get_asymmetry_score(self) -> float:
        """非対称性総合スコア取得"""
        return self.asymmetry_metrics.get('total_asymmetry', self.asymmetry_index)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            'name_a': self.name_a,
            'name_b': self.name_b,
            'analysis_timestamp': self.analysis_timestamp,
            'synchronization_strength': self.synchronization_strength,
            'structure_synchronization': self.structure_synchronization,
            'causality_a_to_b': self.causality_a_to_b,
            'causality_b_to_a': self.causality_b_to_a,
            'asymmetry_index': self.asymmetry_index,
            'data_overlap_length': self.data_overlap_length,
            'correlation_quality': self.correlation_quality,
            'dominant_direction': self.get_dominant_direction(),
            'analysis_method': self.analysis_method,
            'processing_time': self.processing_time,
            'bidirectional_coupling': self.calculate_bidirectional_coupling()
        }

# ==========================================================
# ペアワイズ分析器クラス（完全版）
# ==========================================================

class PairwiseAnalyzer:
    """
    Lambda³ペアワイズ相互作用分析器（完全版）
    
    Protocol準拠による型安全性を確保し、循環インポートを回避した
    ペアワイズ相互作用分析のメインエンジン。
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
        非対称相互作用分析実行
        
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
        
        # データ取得
        data_a, name_a = self._extract_data_from_features(features_a)
        data_b, name_b = self._extract_data_from_features(features_b)
        
        print(f"Analyzing pairwise interaction: {name_a} ↔ {name_b}")
        
        try:
            if use_bayesian and BAYESIAN_AVAILABLE:
                # ベイズペアワイズ分析
                results = self._analyze_with_bayesian(features_a, features_b, data_a, data_b, name_a, name_b)
            else:
                # 標準ペアワイズ分析
                results = self._analyze_standard(features_a, features_b, data_a, data_b, name_a, name_b)
            
            # 処理時間記録
            results.processing_time = time.time() - start_time
            
            return results
            
        except Exception as e:
            raise Lambda3Error(f"Pairwise analysis failed for {name_a}-{name_b}: {e}")
    
    def _validate_inputs(self, features_a: StructuralTensorProtocol, features_b: StructuralTensorProtocol):
        """入力検証"""
        if TYPES_AVAILABLE:
            if not is_structural_tensor_compatible(features_a):
                raise Lambda3Error("features_a is not compatible with StructuralTensorProtocol")
            if not is_structural_tensor_compatible(features_b):
                raise Lambda3Error("features_b is not compatible with StructuralTensorProtocol")
        
        # データの存在確認
        data_a = self._get_data_length(features_a)
        data_b = self._get_data_length(features_b)
        
        if data_a < 10 or data_b < 10:
            raise Lambda3Error("Insufficient data for pairwise analysis")
    
    def _get_data_length(self, features: StructuralTensorProtocol) -> int:
        """データ長取得"""
        if hasattr(features, 'data'):
            return len(features.data)
        elif hasattr(features, '__getitem__') and 'data' in features:
            return len(features['data'])
        elif hasattr(features, '__len__'):
            return len(features)
        else:
            return 0
    
    def _extract_data_from_features(self, features: StructuralTensorProtocol) -> Tuple[np.ndarray, str]:
        """特徴量からデータ抽出"""
        
        # データの取得
        if hasattr(features, 'data'):
            data = features.data
            series_name = getattr(features, 'series_name', 'Series')
        elif hasattr(features, '__getitem__') and 'data' in features:
            data = features['data']
            series_name = features.get('series_name', 'Series')
        else:
            # フォールバック：features自体をデータとして扱う
            data = np.asarray(features)
            series_name = 'Series'
        
        # 型安全性確保
        if TYPES_AVAILABLE:
            data = ensure_float_array(data)
        else:
            data = np.asarray(data, dtype=np.float64)
        
        return data, series_name
    
    def _analyze_standard(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol,
        data_a: np.ndarray,
        data_b: np.ndarray,
        name_a: str,
        name_b: str
    ) -> PairwiseInteractionResults:
        """標準ペアワイズ分析"""
        
        print(f"Running standard pairwise analysis: {name_a} ↔ {name_b}")
        
        # データ長の統一
        min_length = min(len(data_a), len(data_b))
        data_a_aligned = data_a[:min_length]
        data_b_aligned = data_b[:min_length]
        
        # 同期性分析
        sync_results = self._calculate_synchronization(data_a_aligned, data_b_aligned, features_a, features_b)
        
        # 因果性分析
        causality_results = self._calculate_causality(data_a_aligned, data_b_aligned, features_a, features_b)
        
        # 相互作用係数計算
        interaction_coeffs = self._calculate_interaction_coefficients(features_a, features_b)
        
        # 位相結合分析
        phase_coupling = self._calculate_phase_coupling(data_a_aligned, data_b_aligned)
        
        # 品質評価
        correlation_quality = self._assess_correlation_quality(data_a_aligned, data_b_aligned)
        
        # 拡張メトリクス計算
        asymmetry_metrics = self._calculate_asymmetry_metrics(causality_results)
        causality_patterns = self._calculate_causality_patterns(data_a_aligned, data_b_aligned)
        prediction_metrics = self._calculate_prediction_metrics(data_a_aligned, data_b_aligned)
        interaction_quality = self._calculate_interaction_quality(sync_results, causality_results, correlation_quality)
        
        # 結果構築
        results = PairwiseInteractionResults(
            name_a=name_a,
            name_b=name_b,
            synchronization_strength=sync_results['sync_strength'],
            structure_synchronization=sync_results['structure_sync'],
            causality_a_to_b=causality_results['a_to_b'],
            causality_b_to_a=causality_results['b_to_a'],
            asymmetry_index=abs(causality_results['a_to_b'] - causality_results['b_to_a']),
            data_overlap_length=min_length,
            correlation_quality=correlation_quality,
            synchronization_profile=sync_results['profile'],
            interaction_coefficients=interaction_coeffs,
            phase_coupling=phase_coupling,
            analysis_method='standard',
            asymmetry_metrics=asymmetry_metrics,
            causality_patterns=causality_patterns,
            prediction_metrics=prediction_metrics,
            interaction_quality=interaction_quality
        )
        
        return results
    
    def _analyze_with_bayesian(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol,
        data_a: np.ndarray,
        data_b: np.ndarray,
        name_a: str,
        name_b: str
    ) -> PairwiseInteractionResults:
        """ベイズペアワイズ分析"""
        
        print(f"Running Bayesian pairwise analysis: {name_a} ↔ {name_b}")
        
        # データ長の統一
        min_length = min(len(data_a), len(data_b))
        data_a_aligned = data_a[:min_length]
        data_b_aligned = data_b[:min_length]
        
        # ベイズモデル構築と推定
        trace, model = self._fit_bayesian_pairwise_model(data_a_aligned, data_b_aligned, features_a, features_b)
        
        # ベイズ結果から係数抽出
        bayesian_coeffs = self._extract_bayesian_pairwise_coefficients(trace)
        
        # 標準分析も実行（比較用）
        sync_results = self._calculate_synchronization(data_a_aligned, data_b_aligned, features_a, features_b)
        phase_coupling = self._calculate_phase_coupling(data_a_aligned, data_b_aligned)
        correlation_quality = self._assess_correlation_quality(data_a_aligned, data_b_aligned)
        
        # 拡張メトリクス計算
        asymmetry_metrics = self._calculate_asymmetry_metrics(bayesian_coeffs)
        causality_patterns = self._calculate_causality_patterns(data_a_aligned, data_b_aligned)
        prediction_metrics = self._calculate_prediction_metrics(data_a_aligned, data_b_aligned)
        interaction_quality = self._calculate_interaction_quality(sync_results, bayesian_coeffs, correlation_quality)
        
        # 結果構築
        results = PairwiseInteractionResults(
            name_a=name_a,
            name_b=name_b,
            synchronization_strength=sync_results['sync_strength'],
            structure_synchronization=sync_results['structure_sync'],
            causality_a_to_b=bayesian_coeffs.get('causality_a_to_b', 0.0),
            causality_b_to_a=bayesian_coeffs.get('causality_b_to_a', 0.0),
            asymmetry_index=bayesian_coeffs.get('asymmetry', 0.0),
            data_overlap_length=min_length,
            correlation_quality=correlation_quality,
            synchronization_profile=sync_results['profile'],
            interaction_coefficients={'bayesian': bayesian_coeffs},
            phase_coupling=phase_coupling,
            analysis_method='bayesian',
            bayesian_trace=trace,
            asymmetry_metrics=asymmetry_metrics,
            causality_patterns=causality_patterns,
            prediction_metrics=prediction_metrics,
            interaction_quality=interaction_quality
        )
        
        return results
    
    def _calculate_synchronization(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> Dict[str, Any]:
        """同期性計算（修正版）"""
        
        # 基本同期強度（差分ベースに変更）
        if len(data_a) > 1 and len(data_b) > 1:
            # 構造テンソル差分での相関計算（トレンド除去）
            diff_a = np.diff(data_a)
            diff_b = np.diff(data_b)
            
            if len(diff_a) > 0 and len(diff_b) > 0:
                correlation = np.corrcoef(diff_a, diff_b)[0, 1]
                sync_strength = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                sync_strength = 0.0
        else:
            sync_strength = 0.0
        
        # 構造変化同期（これは既に正しい）
        structure_sync = self._calculate_structure_synchronization(features_a, features_b)
        
        # 同期プロファイル（これも差分ベースに）
        if self.use_jit and JIT_FUNCTIONS_AVAILABLE:
            sync_profile = self._calculate_sync_profile_jit(diff_a, diff_b)
        else:
            sync_profile = self._calculate_sync_profile_python(diff_a, diff_b)
        
        return {
            'sync_strength': sync_strength,
            'structure_sync': structure_sync,
            'profile': sync_profile
        }
    
    def _calculate_structure_synchronization(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> float:
        """構造変化同期計算"""
        
        # 構造変化イベントの取得
        events_a = self._get_structure_events(features_a)
        events_b = self._get_structure_events(features_b)
        
        # 長さの統一
        min_length = min(len(events_a), len(events_b))
        if min_length > 1:
            events_a = events_a[:min_length]
            events_b = events_b[:min_length]
            
            # 構造変化同期率
            if np.sum(events_a) > 0 and np.sum(events_b) > 0:
                sync_rate = np.mean(events_a * events_b)
            else:
                sync_rate = 0.0
        else:
            sync_rate = 0.0
        
        return sync_rate
    
    def _get_structure_events(self, features: StructuralTensorProtocol) -> np.ndarray:
        """構造変化イベント取得"""
        
        if hasattr(features, 'delta_LambdaC_pos') and hasattr(features, 'delta_LambdaC_neg'):
            pos_events = features.delta_LambdaC_pos if features.delta_LambdaC_pos is not None else np.array([])
            neg_events = features.delta_LambdaC_neg if features.delta_LambdaC_neg is not None else np.array([])
            
            if len(pos_events) > 0 and len(neg_events) > 0:
                return pos_events + neg_events
            elif len(pos_events) > 0:
                return pos_events
            elif len(neg_events) > 0:
                return neg_events
            else:
                return np.array([])
                
        elif isinstance(features, dict):
            pos_events = features.get('delta_LambdaC_pos', np.array([]))
            neg_events = features.get('delta_LambdaC_neg', np.array([]))
            
            if len(pos_events) > 0 and len(neg_events) > 0:
                return pos_events + neg_events
            elif len(pos_events) > 0:
                return pos_events
            else:
                return np.array([])
        
        return np.array([])
    
    def _calculate_sync_profile_jit(self, data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, float]:
        """JIT同期プロファイル計算"""
        
        try:
            lag_window = getattr(self.config, 'lag_window', 10)
            
            # JIT関数で同期プロファイル計算
            lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_fixed(
                data_a, data_b, lag_window
            )
            
            return {
                'max_sync': float(max_sync),
                'optimal_lag': int(optimal_lag),
                'profile': {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
            }
            
        except Exception as e:
            print(f"JIT sync profile calculation failed: {e}, falling back to Python")
            return self._calculate_sync_profile_python(data_a, data_b)
    
    def _calculate_sync_profile_python(self, data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, float]:
        """Python同期プロファイル計算"""
        
        lag_window = getattr(self.config, 'lag_window', 10)
        sync_profile = {}
        max_sync = 0.0
        optimal_lag = 0
        
        for lag in range(-lag_window, lag_window + 1):
            if lag < 0:
                if -lag < len(data_a):
                    sync = np.corrcoef(data_a[-lag:], data_b[:lag])[0, 1] if len(data_a[-lag:]) > 1 else 0.0
                else:
                    sync = 0.0
            elif lag > 0:
                if lag < len(data_b):
                    sync = np.corrcoef(data_a[:-lag], data_b[lag:])[0, 1] if len(data_a[:-lag]) > 1 else 0.0
                else:
                    sync = 0.0
            else:
                sync = np.corrcoef(data_a, data_b)[0, 1] if len(data_a) > 1 else 0.0
            
            if np.isnan(sync):
                sync = 0.0
            
            sync_profile[lag] = abs(sync)
            
            if abs(sync) > max_sync:
                max_sync = abs(sync)
                optimal_lag = lag
        
        return {
            'max_sync': max_sync,
            'optimal_lag': optimal_lag,
            'profile': sync_profile
        }
    
    def _calculate_causality(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> Dict[str, float]:
        """因果性計算"""
        
        # 基本遅延相関による因果性
        causality_a_to_b = self._calculate_lagged_correlation(data_a, data_b, direction='forward')
        causality_b_to_a = self._calculate_lagged_correlation(data_b, data_a, direction='forward')
        
        # 構造変化因果性
        structure_causality = self._calculate_structure_causality(features_a, features_b)
        
        # 統合因果性
        integrated_a_to_b = (causality_a_to_b + structure_causality['a_to_b']) / 2
        integrated_b_to_a = (causality_b_to_a + structure_causality['b_to_a']) / 2
        
        return {
            'a_to_b': integrated_a_to_b,
            'b_to_a': integrated_b_to_a
        }
    
    def _calculate_lagged_correlation(self, series_cause: np.ndarray, series_effect: np.ndarray, direction: str = 'forward') -> float:
        """遅延相関計算"""
        
        max_causality = 0.0
        
        for lag in range(1, min(10, len(series_cause) // 10)):
            if direction == 'forward' and lag < len(series_cause):
                cause_past = series_cause[:-lag]
                effect_future = series_effect[lag:]
                
                if len(cause_past) > 1 and len(effect_future) > 1:
                    correlation = np.corrcoef(cause_past, effect_future)[0, 1]
                    if not np.isnan(correlation):
                        max_causality = max(max_causality, abs(correlation))
        
        return max_causality
    
    def _calculate_structure_causality(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> Dict[str, float]:
        """構造変化因果性計算"""
        
        events_a = self._get_structure_events(features_a)
        events_b = self._get_structure_events(features_b)
        
        # 長さの統一
        min_length = min(len(events_a), len(events_b))
        if min_length > 10:
            events_a = events_a[:min_length]
            events_b = events_b[:min_length]
            
            # A→B構造因果性
            causality_a_to_b = 0.0
            causality_b_to_a = 0.0
            
            for lag in range(1, min(5, min_length // 5)):
                if lag < min_length:
                    # A(t-lag) → B(t)
                    joint_ab = np.mean(events_a[:-lag] * events_b[lag:])
                    marginal_a = np.mean(events_a[:-lag])
                    if marginal_a > 0:
                        causality_a_to_b = max(causality_a_to_b, joint_ab / marginal_a)
                    
                    # B(t-lag) → A(t)
                    joint_ba = np.mean(events_b[:-lag] * events_a[lag:])
                    marginal_b = np.mean(events_b[:-lag])
                    if marginal_b > 0:
                        causality_b_to_a = max(causality_b_to_a, joint_ba / marginal_b)
        else:
            causality_a_to_b = 0.0
            causality_b_to_a = 0.0
        
        return {
            'a_to_b': causality_a_to_b,
            'b_to_a': causality_b_to_a
        }
    
    def _calculate_interaction_coefficients(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> Dict[str, Dict[str, float]]:
        """相互作用係数計算"""
        
        # 張力スカラー相互作用
        tension_interaction = self._calculate_tension_interaction(features_a, features_b)
        
        # 構造変化相互作用
        structure_interaction = self._calculate_structure_interaction(features_a, features_b)
        
        return {
            'tension': tension_interaction,
            'structure': structure_interaction
        }
    
    def _calculate_tension_interaction(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> Dict[str, float]:
        """張力スカラー相互作用計算"""
        
        # 張力データ取得
        rho_a = self._get_tension_data(features_a)
        rho_b = self._get_tension_data(features_b)
        
        if len(rho_a) > 0 and len(rho_b) > 0:
            min_length = min(len(rho_a), len(rho_b))
            rho_a = rho_a[:min_length]
            rho_b = rho_b[:min_length]
            
            # 張力相関
            if min_length > 1:
                tension_correlation = np.corrcoef(rho_a, rho_b)[0, 1]
                if np.isnan(tension_correlation):
                    tension_correlation = 0.0
            else:
                tension_correlation = 0.0
            
            # 張力因果性
            tension_causality_ab = self._calculate_lagged_correlation(rho_a, rho_b)
            tension_causality_ba = self._calculate_lagged_correlation(rho_b, rho_a)
        else:
            tension_correlation = 0.0
            tension_causality_ab = 0.0
            tension_causality_ba = 0.0
        
        return {
            'correlation': tension_correlation,
            'causality_a_to_b': tension_causality_ab,
            'causality_b_to_a': tension_causality_ba
        }
    
    def _get_tension_data(self, features: StructuralTensorProtocol) -> np.ndarray:
        """張力データ取得"""
        
        if hasattr(features, 'rho_T') and features.rho_T is not None:
            return features.rho_T
        elif isinstance(features, dict) and 'rho_T' in features:
            return features['rho_T'] if features['rho_T'] is not None else np.array([])
        else:
            return np.array([])
    
    def _calculate_structure_interaction(
        self,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> Dict[str, float]:
        """構造変化相互作用計算"""
        
        events_a = self._get_structure_events(features_a)
        events_b = self._get_structure_events(features_b)
        
        if len(events_a) > 0 and len(events_b) > 0:
            min_length = min(len(events_a), len(events_b))
            events_a = events_a[:min_length]
            events_b = events_b[:min_length]
            
            # 構造変化同時発生率
            joint_events = np.mean(events_a * events_b)
            
            # 構造変化因果効果
            causality_ab = self._calculate_lagged_correlation(events_a, events_b)
            causality_ba = self._calculate_lagged_correlation(events_b, events_a)
        else:
            joint_events = 0.0
            causality_ab = 0.0
            causality_ba = 0.0
        
        return {
            'joint_events': joint_events,
            'causality_a_to_b': causality_ab,
            'causality_b_to_a': causality_ba
        }
    
    def _calculate_phase_coupling(self, data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, float]:
        """位相結合計算"""
        
        if self.use_jit and JIT_FUNCTIONS_AVAILABLE:
            return self._calculate_phase_coupling_jit(data_a, data_b)
        else:
            return self._calculate_phase_coupling_python(data_a, data_b)
    
    def _calculate_phase_coupling_jit(self, data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, float]:
        """JIT位相結合計算"""
        
        try:
            # JIT関数で位相結合検出
            coupling_strength, phase_lag = detect_phase_coupling_fixed(data_a, data_b)
            
            return {
                'coupling_strength': float(coupling_strength),
                'phase_lag': float(phase_lag)
            }
            
        except Exception as e:
            print(f"JIT phase coupling calculation failed: {e}, falling back to Python")
            return self._calculate_phase_coupling_python(data_a, data_b)
    
    def _calculate_phase_coupling_python(self, data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, float]:
        """Python位相結合計算（簡易版）"""
        
        # 簡易位相結合（相関ベース）
        if len(data_a) > 10 and len(data_b) > 10:
            # 差分ベースの位相情報
            phase_a = np.diff(data_a)
            phase_b = np.diff(data_b)
            
            min_length = min(len(phase_a), len(phase_b))
            phase_a = phase_a[:min_length]
            phase_b = phase_b[:min_length]
            
            if min_length > 1:
                coupling_strength = abs(np.corrcoef(phase_a, phase_b)[0, 1])
                if np.isnan(coupling_strength):
                    coupling_strength = 0.0
                
                # 位相遅延（クロス相関ピーク位置）
                cross_corr = np.correlate(phase_a, phase_b, mode='full')
                phase_lag = np.argmax(cross_corr) - len(phase_a) + 1
            else:
                coupling_strength = 0.0
                phase_lag = 0.0
        else:
            coupling_strength = 0.0
            phase_lag = 0.0
        
        return {
            'coupling_strength': coupling_strength,
            'phase_lag': float(phase_lag)
        }
    
    def _assess_correlation_quality(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """相関品質評価"""
        
        # データ長による品質
        length_quality = min(1.0, len(data_a) / 50)  # 50点以上で満点
        
        # 分散による品質
        var_a = np.var(data_a)
        var_b = np.var(data_b)
        variance_quality = 1.0 if var_a > 1e-8 and var_b > 1e-8 else 0.5
        
        # 有限値による品質
        finite_quality_a = np.sum(np.isfinite(data_a)) / len(data_a)
        finite_quality_b = np.sum(np.isfinite(data_b)) / len(data_b)
        finite_quality = (finite_quality_a + finite_quality_b) / 2
        
        # 統合品質
        overall_quality = (length_quality * 0.3 + variance_quality * 0.3 + finite_quality * 0.4)
        
        return overall_quality
    
    def _calculate_asymmetry_metrics(self, causality_results: Dict[str, float]) -> Dict[str, float]:
        """非対称性メトリクス計算"""
        
        a_to_b = causality_results.get('a_to_b', 0.0)
        b_to_a = causality_results.get('b_to_a', 0.0)
        
        # 基本非対称性
        basic_asymmetry = abs(a_to_b - b_to_a)
        
        # 正規化非対称性
        total_strength = a_to_b + b_to_a
        normalized_asymmetry = basic_asymmetry / (total_strength + 1e-8)
        
        # 優勢性指標
        dominance = max(a_to_b, b_to_a) / (min(a_to_b, b_to_a) + 1e-8)
        
        return {
            'basic_asymmetry': basic_asymmetry,
            'normalized_asymmetry': normalized_asymmetry,
            'dominance': dominance,
            'total_asymmetry': normalized_asymmetry
        }
    
    def _calculate_causality_patterns(self, data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, Dict[int, float]]:
        """因果性パターン計算"""
        
        patterns = {'a_to_b': {}, 'b_to_a': {}}
        
        for lag in range(1, min(20, len(data_a) // 10)):
            if lag < len(data_a):
                # A→B因果性
                cause_a = data_a[:-lag]
                effect_b = data_b[lag:]
                if len(cause_a) > 1:
                    corr_ab = np.corrcoef(cause_a, effect_b)[0, 1]
                    patterns['a_to_b'][lag] = abs(corr_ab) if not np.isnan(corr_ab) else 0.0
                
                # B→A因果性
                cause_b = data_b[:-lag]
                effect_a = data_a[lag:]
                if len(cause_b) > 1:
                    corr_ba = np.corrcoef(cause_b, effect_a)[0, 1]
                    patterns['b_to_a'][lag] = abs(corr_ba) if not np.isnan(corr_ba) else 0.0
        
        return patterns
    
    def _calculate_prediction_metrics(self, data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, float]:
        """予測性能メトリクス計算"""
        
        if len(data_a) < 20 or len(data_b) < 20:
            return {'prediction_accuracy_a': 0.0, 'prediction_accuracy_b': 0.0}
        
        # 簡易予測精度（直線回帰ベース）
        mid_point = len(data_a) // 2
        
        # Aの予測精度
        train_a = data_a[:mid_point]
        test_a = data_a[mid_point:]
        predicted_a = np.linspace(train_a[-1], train_a[-1] + (train_a[-1] - train_a[0]), len(test_a))
        mse_a = np.mean((test_a - predicted_a) ** 2)
        baseline_var_a = np.var(test_a)
        accuracy_a = max(0.0, 1.0 - mse_a / (baseline_var_a + 1e-8))
        
        # Bの予測精度
        train_b = data_b[:mid_point]
        test_b = data_b[mid_point:]
        predicted_b = np.linspace(train_b[-1], train_b[-1] + (train_b[-1] - train_b[0]), len(test_b))
        mse_b = np.mean((test_b - predicted_b) ** 2)
        baseline_var_b = np.var(test_b)
        accuracy_b = max(0.0, 1.0 - mse_b / (baseline_var_b + 1e-8))
        
        return {
            'prediction_accuracy_a': accuracy_a,
            'prediction_accuracy_b': accuracy_b
        }
    
    def _calculate_interaction_quality(self, sync_results: Dict, causality_results: Dict, correlation_quality: float) -> Dict[str, float]:
        """相互作用品質計算"""
        
        # 同期品質
        sync_quality = sync_results.get('sync_strength', 0.0)
        
        # 因果品質
        causality_quality = (causality_results.get('a_to_b', 0.0) + causality_results.get('b_to_a', 0.0)) / 2
        
        # 非対称性品質
        asymmetry_quality = abs(causality_results.get('a_to_b', 0.0) - causality_results.get('b_to_a', 0.0))
        
        # 統合品質
        interaction_quality = sync_quality * 0.4
        overall_quality = (
            sync_quality * 0.3 +
            causality_quality * 0.3 +
            correlation_quality * 0.4
        )
        
        return {
            'interaction_quality': interaction_quality,
            'asymmetry_quality': asymmetry_quality,
            'synchronization_quality': sync_quality,
            'causality_quality': causality_quality,
            'overall_quality': overall_quality
        }
    
    def _fit_bayesian_pairwise_model(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        features_a: StructuralTensorProtocol,
        features_b: StructuralTensorProtocol
    ) -> Tuple[Any, Any]:
        """ベイズペアワイズモデル推定"""
        
        # 張力データ取得
        rho_a = self._get_tension_data(features_a)
        rho_b = self._get_tension_data(features_b)
        
        # データ長統一
        min_length = min(len(data_a), len(data_b), len(rho_a), len(rho_b))
        if min_length < 10:
            raise Lambda3Error("Insufficient data for Bayesian analysis")
        
        rho_a = rho_a[:min_length]
        rho_b = rho_b[:min_length]
        
        with pm.Model() as model:
            # 事前分布
            beta_a_to_b = pm.Normal('beta_a_to_b', mu=0, sigma=1)
            beta_b_to_a = pm.Normal('beta_b_to_a', mu=0, sigma=1)
            alpha_sync = pm.Normal('alpha_sync', mu=0, sigma=0.5)
            
            # 相互作用効果
            interaction_ab = beta_a_to_b * rho_a
            interaction_ba = beta_b_to_a * rho_b
            sync_effect = alpha_sync
            
            # 観測モデル（簡易版）
            mu_a = interaction_ba + sync_effect
            mu_b = interaction_ab + sync_effect
            
            sigma_a = pm.HalfNormal('sigma_a', sigma=1)
            sigma_b = pm.HalfNormal('sigma_b', sigma=1)
            
            # 尤度
            obs_a = pm.Normal('obs_a', mu=mu_a, sigma=sigma_a, observed=rho_a)
            obs_b = pm.Normal('obs_b', mu=mu_b, sigma=sigma_b, observed=rho_b)
            
            # サンプリング
            trace = pm.sample(1000, tune=500, cores=2, chains=2, return_inferencedata=True)
        
        return trace, model
    
    def _extract_bayesian_pairwise_coefficients(self, trace: Any) -> Dict[str, float]:
        """ベイズペアワイズ係数抽出"""
        
        try:
            summary = az.summary(trace)
            
            beta_a_to_b = summary.loc['beta_a_to_b', 'mean']
            beta_b_to_a = summary.loc['beta_b_to_a', 'mean']
            alpha_sync = summary.loc['alpha_sync', 'mean']
            
            coefficients = {
                'causality_a_to_b': abs(beta_a_to_b),
                'causality_b_to_a': abs(beta_b_to_a),
                'synchronization': abs(alpha_sync),
                'asymmetry': abs(beta_a_to_b - beta_b_to_a),
                'a_to_b': abs(beta_a_to_b),
                'b_to_a': abs(beta_b_to_a)
            }
            
            return coefficients
            
        except Exception as e:
            print(f"Bayesian coefficient extraction failed: {e}")
            return {
                'causality_a_to_b': 0.0,
                'causality_b_to_a': 0.0,
                'synchronization': 0.0,
                'asymmetry': 0.0,
                'a_to_b': 0.0,
                'b_to_a': 0.0
            }

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
    return analyzer.analyze_asymmetric_interaction(features_a, features_b, use_bayesian=use_bayesian)

def compare_all_pairs(
    features_dict: Dict[str, StructuralTensorProtocol],
    config: Optional[Any] = None
) -> Dict[str, PairwiseInteractionResults]:
    """
    全ペア比較分析
    
    Args:
        features_dict: 特徴量辞書
        config: 設定オブジェクト
        
    Returns:
        Dict[str, PairwiseInteractionResults]: ペア別分析結果
    """
    analyzer = PairwiseAnalyzer(config=config)
    results = {}
    
    series_names = list(features_dict.keys())
    
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i < j:  # 重複回避
                pair_key = f"{name_a}_vs_{name_b}"
                
                try:
                    result = analyzer.analyze_asymmetric_interaction(
                        features_dict[name_a],
                        features_dict[name_b]
                    )
                    results[pair_key] = result
                except Exception as e:
                    print(f"Pairwise analysis failed for {pair_key}: {e}")
                    continue
    
    return results

def analyze_network_structure(
    features_dict: Dict[str, StructuralTensorProtocol],
    config: Optional[Any] = None,
    sync_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    ネットワーク構造分析
    
    Args:
        features_dict: 特徴量辞書
        config: 設定オブジェクト
        sync_threshold: 同期閾値
        
    Returns:
        Dict[str, Any]: ネットワーク分析結果
    """
    # 全ペア分析
    pairwise_results = compare_all_pairs(features_dict, config)
    
    # ネットワーク構築
    series_names = list(features_dict.keys())
    n_series = len(series_names)
    
    # 同期行列構築
    sync_matrix = np.zeros((n_series, n_series))
    
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i != j:
                pair_key = f"{name_a}_vs_{name_b}" if i < j else f"{name_b}_vs_{name_a}"
                
                if pair_key in pairwise_results:
                    sync_strength = pairwise_results[pair_key].synchronization_strength
                    sync_matrix[i, j] = sync_strength
            else:
                sync_matrix[i, j] = 1.0
    
    # ネットワーク統計
    network_density = np.mean(sync_matrix[sync_matrix < 1.0])
    max_sync = np.max(sync_matrix[sync_matrix < 1.0]) if np.any(sync_matrix < 1.0) else 0.0
    
    # 強結合ペア
    strong_pairs = []
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i < j and sync_matrix[i, j] >= sync_threshold:
                strong_pairs.append((name_a, name_b, sync_matrix[i, j]))
    
    return {
        'synchronization_matrix': sync_matrix.tolist(),
        'series_names': series_names,
        'network_density': network_density,
        'max_synchronization': max_sync,
        'strong_pairs': strong_pairs,
        'network_size': n_series,
        'pairwise_results': pairwise_results
    }

# ==========================================================
# モジュール情報
# ==========================================================

__all__ = [
    'PairwiseInteractionResults',
    'PairwiseAnalyzer',
    'analyze_pairwise_interaction',
    'compare_all_pairs',
    'analyze_network_structure'
]

# ==========================================================
# テスト関数
# ==========================================================

def test_pairwise_analysis():
    """ペアワイズ分析のテスト"""
    print("🧪 Testing Pairwise Analysis Implementation")
    print("=" * 50)
    
    try:
        # サンプルデータ生成
        np.random.seed(42)
        sample_data_a = np.cumsum(np.random.randn(80) * 0.1)
        sample_data_b = np.cumsum(np.random.randn(80) * 0.12)
        
        # 構造テンソル特徴量作成（Protocol準拠）
        if STRUCTURAL_TENSOR_AVAILABLE:
            from ..core.structural_tensor import extract_lambda3_features
            features_a = extract_lambda3_features(sample_data_a, series_name="Test_A")
            features_b = extract_lambda3_features(sample_data_b, series_name="Test_B")
        else:
            # フォールバック：辞書形式
            features_a = {
                'data': sample_data_a,
                'series_name': 'Test_A',
                'delta_LambdaC_pos': np.random.randint(0, 2, 80).astype(np.float64),
                'delta_LambdaC_neg': np.random.randint(0, 2, 80).astype(np.float64),
                'rho_T': np.random.rand(80)
            }
            features_b = {
                'data': sample_data_b,
                'series_name': 'Test_B',
                'delta_LambdaC_pos': np.random.randint(0, 2, 80).astype(np.float64),
                'delta_LambdaC_neg': np.random.randint(0, 2, 80).astype(np.float64),
                'rho_T': np.random.rand(80)
            }
        
        # ペアワイズ分析実行
        analyzer = PairwiseAnalyzer()
        results = analyzer.analyze_asymmetric_interaction(features_a, features_b)
        
        print(f"✅ Analysis completed: {results.name_a} ↔ {results.name_b}")
        print(f"✅ Synchronization strength: {results.synchronization_strength:.3f}")
        print(f"✅ Causality A→B: {results.causality_a_to_b:.3f}")
        print(f"✅ Causality B→A: {results.causality_b_to_a:.3f}")
        print(f"✅ Asymmetry index: {results.asymmetry_index:.3f}")
        print(f"✅ Dominant direction: {results.get_dominant_direction()}")
        print(f"✅ Bidirectional coupling: {results.calculate_bidirectional_coupling():.3f}")
        
        # Protocol準拠確認
        if TYPES_AVAILABLE:
            print(f"✅ Protocol compatibility: {'OK' if hasattr(results, 'get_interaction_summary') else 'NG'}")
        
        print("🎯 Pairwise analysis test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_pairwise_analysis()
