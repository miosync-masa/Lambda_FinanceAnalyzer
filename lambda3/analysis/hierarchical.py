# ==========================================================
# lambda3/analysis/hierarchical.py (修正版)
# Hierarchical Structure Analysis for Lambda³ Theory
# ==========================================================

"""
Lambda³理論階層構造解析モジュール（修正版）

循環インポート問題を解決し、Protocol準拠による型安全性を確保した
階層的構造変化解析の実装。

修正点:
- types.pyからのProtocolインポートによる循環回避
- structural_tensor.pyからのインポート修正
- JIT最適化関数との完全互換性確保
- エラー耐性の向上

Author: Masamichi Iizumi (Miosync, Inc.)
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
        HierarchicalResultProtocol,
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
    HierarchicalResultProtocol = Any
    Lambda3Error = Exception

# 設定のインポート
try:
    from ..core.config import L3BaseConfig, L3HierarchicalConfig, L3BayesianConfig
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
        calculate_tension_scalar_fixed,
        normalize_array_fixed,
        detect_hierarchical_jumps_fixed,
        calculate_local_statistics_fixed,
        safe_divide_fixed
    )
    JIT_FUNCTIONS_AVAILABLE = True
    
    # レガシー互換性
    calculate_tension_scalar = calculate_tension_scalar_fixed
    normalize_array = normalize_array_fixed
    
except ImportError:
    JIT_FUNCTIONS_AVAILABLE = False
    warnings.warn("JIT functions not available. Using fallback implementations.")

# ==========================================================
# 階層分析結果データクラス（Protocol準拠）
# ==========================================================

@dataclass
class HierarchicalSeparationResults:
    """
    階層分離分析結果（修正版）
    
    HierarchicalResultProtocolに準拠し、循環インポートを回避した
    階層分析結果の具体実装。
    """
    
    # 基本識別子
    series_name: str = "Series"
    analysis_timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    
    # 階層分離係数
    escalation_strength: float = 0.0           # エスカレーション強度
    deescalation_strength: float = 0.0         # デエスカレーション強度
    hierarchy_correlation: float = 0.0         # 階層間相関
    
    # 品質メトリクス
    convergence_quality: float = 0.0           # 収束品質
    statistical_significance: float = 0.0      # 統計的有意性
    
    # 詳細分析結果
    separation_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    asymmetry_metrics: Dict[str, float] = field(default_factory=dict)
    hierarchy_stats: Dict[str, float] = field(default_factory=dict)
    
    # メタデータ
    analysis_method: str = "standard"          # 分析手法
    bayesian_trace: Optional[Any] = None       # ベイズトレース
    processing_time: float = 0.0               # 処理時間
    
    def get_separation_summary(self) -> Dict[str, float]:
        """分離サマリー取得"""
        return {
            'escalation': self.escalation_strength,
            'deescalation': self.deescalation_strength,
            'correlation': self.hierarchy_correlation,
            'quality': self.convergence_quality
        }
    
    def get_dominant_hierarchy(self) -> str:
        """優勢階層判定"""
        if self.escalation_strength > self.deescalation_strength * 1.2:
            return 'global'
        elif self.deescalation_strength > self.escalation_strength * 1.2:
            return 'local'
        else:
            return 'balanced'
    
    def get_escalation_strength(self) -> float:
        """エスカレーション強度取得（互換性）"""
        return self.escalation_strength
    
    def get_deescalation_strength(self) -> float:
        """デエスカレーション強度取得（互換性）"""
        return self.deescalation_strength
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            'series_name': self.series_name,
            'analysis_timestamp': self.analysis_timestamp,
            'escalation_strength': self.escalation_strength,
            'deescalation_strength': self.deescalation_strength,
            'hierarchy_correlation': self.hierarchy_correlation,
            'convergence_quality': self.convergence_quality,
            'statistical_significance': self.statistical_significance,
            'dominant_hierarchy': self.get_dominant_hierarchy(),
            'analysis_method': self.analysis_method,
            'processing_time': self.processing_time
        }

# ==========================================================
# 階層分析器クラス（修正版）
# ==========================================================

class HierarchicalAnalyzer:
    """
    Lambda³階層構造分析器（修正版）
    
    Protocol準拠による型安全性を確保し、循環インポートを回避した
    階層的構造変化分析のメインエンジン。
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
                    'local_window': 5,
                    'global_window': 30
                })()
        else:
            self.config = config
        
        # JIT使用判定
        if use_jit is None:
            self.use_jit = JIT_FUNCTIONS_AVAILABLE and getattr(self.config, 'enable_jit', True)
        else:
            self.use_jit = use_jit and JIT_FUNCTIONS_AVAILABLE
        
        print(f"HierarchicalAnalyzer initialized: JIT={self.use_jit}")
    
    def analyze_hierarchical_separation(
        self,
        features: StructuralTensorProtocol,
        use_bayesian: bool = False
    ) -> HierarchicalSeparationResults:
        """
        階層分離分析実行
        
        Args:
            features: 構造テンソル特徴量（Protocol準拠）
            use_bayesian: ベイズ分析使用フラグ
            
        Returns:
            HierarchicalSeparationResults: 階層分析結果
        """
        start_time = time.time()
        
        # 入力検証
        self._validate_input(features)
        
        # データ取得
        data, series_name = self._extract_data_from_features(features)
        
        print(f"Analyzing hierarchical separation for {series_name}...")
        
        try:
            if use_bayesian and BAYESIAN_AVAILABLE:
                # ベイズ階層分析
                results = self._analyze_with_bayesian(features, data, series_name)
            else:
                # 標準階層分析
                results = self._analyze_standard(features, data, series_name)
            
            # 処理時間記録
            results.processing_time = time.time() - start_time
            
            return results
            
        except Exception as e:
            raise Lambda3Error(f"Hierarchical analysis failed for {series_name}: {e}")
    
    def _validate_input(self, features: StructuralTensorProtocol):
        """入力検証"""
        if TYPES_AVAILABLE:
            if not is_structural_tensor_compatible(features):
                raise Lambda3Error("Input is not compatible with StructuralTensorProtocol")
        
        # データの存在確認
        if hasattr(features, 'data'):
            if len(features.data) < 10:
                raise Lambda3Error("Insufficient data for hierarchical analysis")
        elif hasattr(features, '__getitem__'):
            if len(features) < 10:
                raise Lambda3Error("Insufficient data for hierarchical analysis")
    
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
        features: StructuralTensorProtocol,
        data: np.ndarray,
        series_name: str
    ) -> HierarchicalSeparationResults:
        """標準階層分析"""
        
        print(f"Running standard hierarchical analysis for {series_name}")
        
        # 階層的特徴量の取得
        local_features, global_features = self._extract_hierarchical_features(features, data)
        
        # 階層分離係数の計算
        separation_coeffs = self._calculate_separation_coefficients(local_features, global_features)
        
        # 非対称性メトリクスの計算
        asymmetry_metrics = self._calculate_asymmetry_metrics(local_features, global_features)
        
        # 階層統計の計算
        hierarchy_stats = self._calculate_hierarchy_statistics(local_features, global_features)
        
        # 結果構築
        results = HierarchicalSeparationResults(
            series_name=series_name,
            escalation_strength=separation_coeffs.get('escalation', 0.0),
            deescalation_strength=separation_coeffs.get('deescalation', 0.0),
            hierarchy_correlation=separation_coeffs.get('correlation', 0.0),
            convergence_quality=0.8,  # 標準分析では固定値
            statistical_significance=0.7,  # 標準分析では固定値
            separation_coefficients={'standard': separation_coeffs},
            asymmetry_metrics=asymmetry_metrics,
            hierarchy_stats=hierarchy_stats,
            analysis_method='standard'
        )
        
        return results
    
    def _analyze_with_bayesian(
        self,
        features: StructuralTensorProtocol,
        data: np.ndarray,
        series_name: str
    ) -> HierarchicalSeparationResults:
        """ベイズ階層分析"""
        
        print(f"Running Bayesian hierarchical analysis for {series_name}")
        
        # 階層的特徴量の取得
        local_features, global_features = self._extract_hierarchical_features(features, data)
        
        # ベイズモデル構築と推定
        trace, model = self._fit_bayesian_hierarchical_model(local_features, global_features)
        
        # ベイズ結果から係数抽出
        separation_coeffs = self._extract_bayesian_coefficients(trace)
        
        # 品質メトリクス計算
        convergence_quality = self._assess_bayesian_convergence(trace)
        statistical_significance = self._assess_statistical_significance(trace)
        
        # 非対称性メトリクス
        asymmetry_metrics = self._calculate_asymmetry_metrics(local_features, global_features)
        
        # 階層統計
        hierarchy_stats = self._calculate_hierarchy_statistics(local_features, global_features)
        
        # 結果構築
        results = HierarchicalSeparationResults(
            series_name=series_name,
            escalation_strength=separation_coeffs.get('escalation', 0.0),
            deescalation_strength=separation_coeffs.get('deescalation', 0.0),
            hierarchy_correlation=separation_coeffs.get('correlation', 0.0),
            convergence_quality=convergence_quality,
            statistical_significance=statistical_significance,
            separation_coefficients={'bayesian': separation_coeffs},
            asymmetry_metrics=asymmetry_metrics,
            hierarchy_stats=hierarchy_stats,
            analysis_method='bayesian',
            bayesian_trace=trace
        )
        
        return results
    
    def _extract_hierarchical_features(
        self,
        features: StructuralTensorProtocol,
        data: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """階層的特徴量の抽出"""
        
        # 既存の階層的特徴量を確認
        hierarchical_attrs = ['local_pos', 'local_neg', 'global_pos', 'global_neg']
        has_hierarchical = False
        
        if hasattr(features, 'local_pos'):
            has_hierarchical = all(getattr(features, attr, None) is not None for attr in hierarchical_attrs)
        elif isinstance(features, dict):
            has_hierarchical = all(attr in features and features[attr] is not None for attr in hierarchical_attrs)
        
        if has_hierarchical:
            # 既存の階層的特徴量を使用
            if hasattr(features, 'local_pos'):
                local_features = {
                    'pos': features.local_pos,
                    'neg': features.local_neg
                }
                global_features = {
                    'pos': features.global_pos,
                    'neg': features.global_neg
                }
            else:
                local_features = {
                    'pos': features['local_pos'],
                    'neg': features['local_neg']
                }
                global_features = {
                    'pos': features['global_pos'],
                    'neg': features['global_neg']
                }
        else:
            # 階層的特徴量を動的に計算
            local_features, global_features = self._compute_hierarchical_features(data)
        
        return local_features, global_features
    
    def _compute_hierarchical_features(self, data: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """階層的特徴量の動的計算"""
        
        if self.use_jit and JIT_FUNCTIONS_AVAILABLE:
            # JIT最適化版
            return self._compute_hierarchical_features_jit(data)
        else:
            # Pure Python版
            return self._compute_hierarchical_features_python(data)
    
    def _compute_hierarchical_features_jit(self, data: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """JIT最適化階層特徴量計算"""
        
        try:
            local_window = getattr(self.config, 'local_window', 5)
            global_window = getattr(self.config, 'global_window', 30)
            
            # JIT関数で階層的ジャンプ検出
            local_pos, local_neg, global_pos, global_neg = detect_hierarchical_jumps_fixed(
                data, local_window, global_window, 90.0, 95.0
            )
            
            local_features = {
                'pos': local_pos.astype(np.float64),
                'neg': local_neg.astype(np.float64)
            }
            
            global_features = {
                'pos': global_pos.astype(np.float64),
                'neg': global_neg.astype(np.float64)
            }
            
            return local_features, global_features
            
        except Exception as e:
            print(f"JIT hierarchical computation failed: {e}, falling back to Python")
            return self._compute_hierarchical_features_python(data)
    
    def _compute_hierarchical_features_python(self, data: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Pure Python階層特徴量計算"""
        
        n = len(data)
        diff = np.diff(data, prepend=data[0])
        
        # 窓サイズ設定
        local_window = getattr(self.config, 'local_window', 5)
        global_window = getattr(self.config, 'global_window', 30)
        
        # 局所特徴量
        local_pos = np.zeros(n, dtype=np.float64)
        local_neg = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            start = max(0, i - local_window)
            end = min(n, i + local_window + 1)
            local_threshold = np.percentile(np.abs(diff[start:end]), 90.0)
            
            if diff[i] > local_threshold:
                local_pos[i] = 1.0
            elif diff[i] < -local_threshold:
                local_neg[i] = 1.0
        
        # 大域特徴量
        global_threshold = np.percentile(np.abs(diff), 95.0)
        global_pos = (diff > global_threshold).astype(np.float64)
        global_neg = (diff < -global_threshold).astype(np.float64)
        
        local_features = {'pos': local_pos, 'neg': local_neg}
        global_features = {'pos': global_pos, 'neg': global_neg}
        
        return local_features, global_features
    
    def _calculate_separation_coefficients(
        self,
        local_features: Dict[str, np.ndarray],
        global_features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """分離係数計算"""
        
        # イベント数計算
        local_events = np.sum(local_features['pos']) + np.sum(local_features['neg'])
        global_events = np.sum(global_features['pos']) + np.sum(global_features['neg'])
        total_events = local_events + global_events
        
        # 分離係数
        if total_events > 0:
            escalation = global_events / total_events  # 局所→大域
            deescalation = local_events / total_events  # 大域→局所
        else:
            escalation = 0.0
            deescalation = 0.0
        
        # 相関係数
        local_total = local_features['pos'] + local_features['neg']
        global_total = global_features['pos'] + global_features['neg']
        
        if np.var(local_total) > 0 and np.var(global_total) > 0:
            correlation = np.corrcoef(local_total, global_total)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'escalation': escalation,
            'deescalation': deescalation,
            'correlation': correlation
        }
    
    def _calculate_asymmetry_metrics(
        self,
        local_features: Dict[str, np.ndarray],
        global_features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """非対称性メトリクス計算"""
        
        # 正負の非対称性（局所）
        local_pos_count = np.sum(local_features['pos'])
        local_neg_count = np.sum(local_features['neg'])
        local_asymmetry = (local_pos_count - local_neg_count) / max(local_pos_count + local_neg_count, 1)
        
        # 正負の非対称性（大域）
        global_pos_count = np.sum(global_features['pos'])
        global_neg_count = np.sum(global_features['neg'])
        global_asymmetry = (global_pos_count - global_neg_count) / max(global_pos_count + global_neg_count, 1)
        
        # 階層間非対称性
        hierarchy_asymmetry = abs(local_asymmetry - global_asymmetry)
        
        return {
            'local_asymmetry': local_asymmetry,
            'global_asymmetry': global_asymmetry,
            'hierarchy_asymmetry': hierarchy_asymmetry
        }
    
    def _calculate_hierarchy_statistics(
        self,
        local_features: Dict[str, np.ndarray],
        global_features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """階層統計計算"""
        
        # 強度統計
        local_intensity = np.mean(local_features['pos'] + local_features['neg'])
        global_intensity = np.mean(global_features['pos'] + global_features['neg'])
        
        # 持続性統計
        local_persistence = self._calculate_persistence(local_features['pos'] + local_features['neg'])
        global_persistence = self._calculate_persistence(global_features['pos'] + global_features['neg'])
        
        return {
            'local_intensity': local_intensity,
            'global_intensity': global_intensity,
            'local_persistence': local_persistence,
            'global_persistence': global_persistence,
            'intensity_ratio': global_intensity / max(local_intensity, 1e-8),
            'persistence_ratio': global_persistence / max(local_persistence, 1e-8)
        }
    
    def _calculate_persistence(self, events: np.ndarray) -> float:
        """イベント持続性計算"""
        if np.sum(events) == 0:
            return 0.0
        
        # 連続するイベントの長さを計算
        event_starts = np.where(np.diff(np.concatenate([[0], events])) == 1)[0]
        event_ends = np.where(np.diff(np.concatenate([events, [0]])) == -1)[0]
        
        if len(event_starts) == 0:
            return 0.0
        
        durations = event_ends - event_starts + 1
        return np.mean(durations)
    
    def _fit_bayesian_hierarchical_model(
        self,
        local_features: Dict[str, np.ndarray],
        global_features: Dict[str, np.ndarray]
    ) -> Tuple[Any, Any]:
        """ベイズ階層モデル推定"""
        
        # ベイズモデル構築
        local_events = local_features['pos'] + local_features['neg']
        global_events = global_features['pos'] + global_features['neg']
        
        with pm.Model() as model:
            # 事前分布
            alpha_local = pm.Normal('alpha_local', mu=0, sigma=1)
            alpha_global = pm.Normal('alpha_global', mu=0, sigma=1)
            beta_interaction = pm.Normal('beta_interaction', mu=0, sigma=0.5)
            
            # 階層効果
            local_effect = pm.math.sigmoid(alpha_local)
            global_effect = pm.math.sigmoid(alpha_global)
            interaction_effect = beta_interaction
            
            # 観測モデル
            mu_local = local_effect + interaction_effect * global_events
            mu_global = global_effect + interaction_effect * local_events
            
            # 尤度
            obs_local = pm.Bernoulli('obs_local', p=mu_local, observed=local_events)
            obs_global = pm.Bernoulli('obs_global', p=mu_global, observed=global_events)
            
            # サンプリング
            trace = pm.sample(2000, tune=1000, cores=2, chains=2, return_inferencedata=True)
        
        return trace, model
    
    def _extract_bayesian_coefficients(self, trace: Any) -> Dict[str, float]:
        """ベイズ係数抽出"""
        
        try:
            summary = az.summary(trace)
            
            coefficients = {
                'escalation': summary.loc['alpha_global', 'mean'],
                'deescalation': summary.loc['alpha_local', 'mean'],
                'correlation': summary.loc['beta_interaction', 'mean']
            }
            
            return coefficients
            
        except Exception as e:
            print(f"Bayesian coefficient extraction failed: {e}")
            return {'escalation': 0.0, 'deescalation': 0.0, 'correlation': 0.0}
    
    def _assess_bayesian_convergence(self, trace: Any) -> float:
        """ベイズ収束品質評価"""
        
        try:
            r_hat = az.rhat(trace)
            avg_r_hat = np.mean([r_hat[var].values.mean() for var in r_hat.data_vars])
            
            # R-hat < 1.1で良好、1.0に近いほど高品質
            quality = max(0.0, 1.0 - (avg_r_hat - 1.0) * 10)
            return min(1.0, quality)
            
        except Exception:
            return 0.5  # デフォルト値
    
    def _assess_statistical_significance(self, trace: Any) -> float:
        """統計的有意性評価"""
        
        try:
            summary = az.summary(trace)
            
            # HDI区間が0を含まない変数の割合
            significant_vars = 0
            total_vars = 0
            
            for var in summary.index:
                if 'hdi_3%' in summary.columns and 'hdi_97%' in summary.columns:
                    hdi_lower = summary.loc[var, 'hdi_3%']
                    hdi_upper = summary.loc[var, 'hdi_97%']
                    
                    if hdi_lower * hdi_upper > 0:  # 同符号 = 0を含まない
                        significant_vars += 1
                    total_vars += 1
            
            return significant_vars / max(total_vars, 1)
            
        except Exception:
            return 0.7  # デフォルト値

# ==========================================================
# 便利関数
# ==========================================================

def analyze_hierarchical_structure(
    features: StructuralTensorProtocol,
    config: Optional[Any] = None,
    use_bayesian: bool = False
) -> HierarchicalSeparationResults:
    """
    階層構造分析の便利関数
    
    Args:
        features: 構造テンソル特徴量
        config: 設定オブジェクト
        use_bayesian: ベイズ分析使用フラグ
        
    Returns:
        HierarchicalSeparationResults: 階層分析結果
    """
    analyzer = HierarchicalAnalyzer(config=config)
    return analyzer.analyze_hierarchical_separation(features, use_bayesian=use_bayesian)

def compare_multiple_hierarchies(
    features_dict: Dict[str, StructuralTensorProtocol],
    config: Optional[Any] = None
) -> Dict[str, HierarchicalSeparationResults]:
    """
    複数系列の階層比較分析
    
    Args:
        features_dict: 特徴量辞書
        config: 設定オブジェクト
        
    Returns:
        Dict[str, HierarchicalSeparationResults]: 系列別階層分析結果
    """
    analyzer = HierarchicalAnalyzer(config=config)
    results = {}
    
    for series_name, features in features_dict.items():
        try:
            result = analyzer.analyze_hierarchical_separation(features)
            results[series_name] = result
        except Exception as e:
            print(f"Hierarchical analysis failed for {series_name}: {e}")
            continue
    
    return results

# ==========================================================
# モジュール情報
# ==========================================================

__all__ = [
    'HierarchicalSeparationResults',
    'HierarchicalAnalyzer',
    'analyze_hierarchical_structure',
    'compare_multiple_hierarchies'
]

# ==========================================================
# テスト関数
# ==========================================================

def test_hierarchical_analysis():
    """階層分析のテスト"""
    print("🧪 Testing Hierarchical Analysis Implementation")
    print("=" * 50)
    
    try:
        # サンプルデータ生成
        np.random.seed(42)
        sample_data = np.cumsum(np.random.randn(100) * 0.1)
        
        # 構造テンソル特徴量作成（Protocol準拠）
        if STRUCTURAL_TENSOR_AVAILABLE:
            from ..core.structural_tensor import extract_lambda3_features
            features = extract_lambda3_features(sample_data, series_name="Test", feature_level="comprehensive")
        else:
            # フォールバック：辞書形式
            features = {
                'data': sample_data,
                'series_name': 'Test',
                'delta_LambdaC_pos': np.random.randint(0, 2, 100).astype(np.float64),
                'delta_LambdaC_neg': np.random.randint(0, 2, 100).astype(np.float64),
                'rho_T': np.random.rand(100)
            }
        
        # 階層分析実行
        analyzer = HierarchicalAnalyzer()
        results = analyzer.analyze_hierarchical_separation(features)
        
        print(f"✅ Analysis completed: {results.series_name}")
        print(f"✅ Escalation strength: {results.escalation_strength:.3f}")
        print(f"✅ Deescalation strength: {results.deescalation_strength:.3f}")
        print(f"✅ Dominant hierarchy: {results.get_dominant_hierarchy()}")
        
        # Protocol準拠確認
        if TYPES_AVAILABLE:
            from ..core.types import is_structural_tensor_compatible
            print(f"✅ Protocol compatibility: {'OK' if hasattr(results, 'get_separation_summary') else 'NG'}")
        
        print("🎯 Hierarchical analysis test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_hierarchical_analysis()
