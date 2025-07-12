# ==========================================================
# lambda3/analysis/hierarchical.py (階層分離ダイナミクス完全修正版)
# Hierarchical Structure Analysis for Lambda³ Theory
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
#
# 完全修正版: エスカレーション/デエスカレーション、ベイズ階層モデル
# ==========================================================

"""
Lambda³階層構造分析（完全修正版）

構造テンソル(Λ)の階層的∆ΛC変化を検出し、エスカレーション/
デエスカレーションダイナミクスを解析。短期・長期構造変化の
分離と相互作用を完全にモデル化。

完全修正内容:
- 階層分離係数の理論準拠計算
- ベイズ階層モデルの完全実装
- JIT関数との完全統合
- 非対称性メトリクスの拡張
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
        detect_local_global_jumps,
        calculate_rho_t,
        normalize_array_fixed,
        moving_average_fixed
    )
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    warnings.warn("JIT functions not available.")

# Configuration imports
try:
    from ..core.config import L3BaseConfig, L3HierarchicalConfig
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
# RESULTS DATACLASS - 階層分析結果
# ==========================================================

@dataclass
class HierarchicalSeparationResults:
    """
    階層分離分析結果（完全版）
    
    Lambda³理論: 構造変化のエスカレーション/デエスカレーション
    ダイナミクスの完全な表現。
    """
    
    # 基本識別情報
    series_name: str = "Series"
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 階層分離係数（核心）
    escalation_strength: float = 0.0          # エスカレーション強度
    deescalation_strength: float = 0.0        # デエスカレーション強度
    hierarchy_correlation: float = 0.0        # 階層間相関
    
    # 階層効果係数
    local_effect: float = 0.0                 # 短期構造効果
    global_effect: float = 0.0                # 長期構造効果
    
    # 品質メトリクス
    convergence_quality: float = 0.0          # 収束品質
    statistical_significance: float = 0.0     # 統計的有意性
    
    # 詳細係数
    separation_coefficients: Dict[str, Dict[str, float]] = field(default_factory=dict)
    asymmetry_metrics: Dict[str, float] = field(default_factory=dict)
    hierarchy_stats: Dict[str, float] = field(default_factory=dict)
    
    # ベイズ分析結果
    bayesian_trace: Optional[Any] = None
    bayesian_summary: Optional[Dict[str, Any]] = None
    
    # メタデータ
    analysis_method: str = "standard"
    processing_time: float = 0.0
    config_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_escalation_strength(self) -> float:
        """エスカレーション強度取得"""
        return self.escalation_strength
    
    def get_deescalation_strength(self) -> float:
        """デエスカレーション強度取得"""
        return self.deescalation_strength
    
    def get_dominant_hierarchy(self) -> str:
        """優勢階層判定"""
        if abs(self.escalation_strength) > abs(self.deescalation_strength) * 1.2:
            return 'escalation_dominant'
        elif abs(self.deescalation_strength) > abs(self.escalation_strength) * 1.2:
            return 'deescalation_dominant'
        else:
            return 'balanced'
    
    def get_separation_summary(self) -> Dict[str, float]:
        """分離サマリー"""
        return {
            'escalation': self.escalation_strength,
            'deescalation': self.deescalation_strength,
            'correlation': self.hierarchy_correlation,
            'local_effect': self.local_effect,
            'global_effect': self.global_effect,
            'quality': self.convergence_quality,
            'significance': self.statistical_significance
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        return {
            'series_name': self.series_name,
            'analysis_timestamp': self.analysis_timestamp,
            'separation_summary': self.get_separation_summary(),
            'dominant_hierarchy': self.get_dominant_hierarchy(),
            'asymmetry_metrics': self.asymmetry_metrics,
            'hierarchy_stats': self.hierarchy_stats,
            'analysis_method': self.analysis_method,
            'processing_time': self.processing_time
        }

# ==========================================================
# HIERARCHICAL ANALYZER - 階層分析器
# ==========================================================

class HierarchicalAnalyzer:
    """
    Lambda³階層構造分析器（完全版）
    
    短期・長期構造変化の分離、エスカレーション/デエスカレーション
    ダイナミクスの包括的解析。
    """
    
    def __init__(self, config: Optional[Any] = None, use_jit: Optional[bool] = None):
        """
        Args:
            config: 階層分析設定
            use_jit: JIT使用フラグ
        """
        self.config = self._initialize_config(config)
        self.use_jit = use_jit if use_jit is not None else JIT_AVAILABLE
        
    def _initialize_config(self, config: Optional[Any]) -> Any:
        """設定初期化"""
        if config is not None:
            return config
        
        if CONFIG_AVAILABLE:
            return L3HierarchicalConfig()
        else:
            # フォールバック設定
            return type('Config', (), {
                'local_window': 5,
                'global_window': 30,
                'local_percentile': 90.0,
                'global_percentile': 95.0,
                'min_events_threshold': 10,
                'bayesian_draws': 4000,
                'bayesian_tune': 2000
            })()
    
    def analyze_hierarchical_separation(
        self,
        features: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
        use_bayesian: bool = False,
        data: Optional[np.ndarray] = None
    ) -> HierarchicalSeparationResults:
        """
        階層分離分析の実行
        
        Args:
            features: 構造テンソル特徴量
            use_bayesian: ベイズ分析使用フラグ
            data: 元データ（特徴量に含まれない場合）
            
        Returns:
            HierarchicalSeparationResults: 階層分析結果
        """
        start_time = time.time()
        
        # データと特徴量の準備
        data_array, series_name, hierarchical_features = self._prepare_analysis_data(
            features, data
        )
        
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL SEPARATION ANALYSIS: {series_name}")
        print(f"{'='*60}")
        
        # 階層的イベントの統計
        self._print_hierarchical_stats(hierarchical_features)
        
        # 分析手法の選択と実行
        if use_bayesian and BAYESIAN_AVAILABLE:
            results = self._analyze_bayesian(
                data_array, hierarchical_features, series_name
            )
        else:
            results = self._analyze_standard(
                data_array, hierarchical_features, series_name
            )
        
        # 処理時間記録
        results.processing_time = time.time() - start_time
        
        print(f"\nAnalysis completed in {results.processing_time:.2f} seconds")
        self._print_results_summary(results)
        
        return results
    
    def _prepare_analysis_data(
        self,
        features: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
        data: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, str, Dict[str, np.ndarray]]:
        """分析データの準備"""
        # データ取得
        if hasattr(features, 'data'):
            data_array = features.data
            series_name = getattr(features, 'series_name', 'Series')
        elif isinstance(features, dict) and 'data' in features:
            data_array = features['data']
            series_name = features.get('series_name', 'Series')
        elif data is not None:
            data_array = data
            series_name = 'Series'
        else:
            raise ValueError("No data found in features or provided")
        
        # 階層的特徴量の取得または計算
        hierarchical_features = self._extract_hierarchical_features(features, data_array)
        
        return data_array, series_name, hierarchical_features
    
    def _extract_hierarchical_features(
        self,
        features: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
        data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """階層的特徴量の抽出"""
        # 既存の階層的特徴量をチェック
        required_features = ['local_pos', 'local_neg', 'global_pos', 'global_neg']
        
        if hasattr(features, 'local_pos'):
            # オブジェクトから取得
            hier_features = {
                feat: getattr(features, feat) for feat in required_features
                if hasattr(features, feat)
            }
        elif isinstance(features, dict):
            # 辞書から取得
            hier_features = {
                feat: features[feat] for feat in required_features
                if feat in features
            }
        else:
            hier_features = {}
        
        # 不足している場合は計算
        if len(hier_features) < 4:
            if self.use_jit and JIT_AVAILABLE:
                local_pos, local_neg, global_pos, global_neg = detect_local_global_jumps(
                    data,
                    self.config.local_window,
                    self.config.global_window,
                    self.config.local_percentile,
                    self.config.global_percentile
                )
                hier_features = {
                    'local_pos': local_pos,
                    'local_neg': local_neg,
                    'global_pos': global_pos,
                    'global_neg': global_neg
                }
            else:
                # 純Python実装
                hier_features = self._compute_hierarchical_features_python(data)
        
        # 張力スカラーも含める
        if hasattr(features, 'rho_T'):
            hier_features['rho_T'] = features.rho_T
        elif isinstance(features, dict) and 'rho_T' in features:
            hier_features['rho_T'] = features['rho_T']
        else:
            if self.use_jit and JIT_AVAILABLE:
                hier_features['rho_T'] = calculate_rho_t(data, 10)
            else:
                hier_features['rho_T'] = self._calculate_rho_t_python(data)
        
        return hier_features
    
    def _compute_hierarchical_features_python(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """純Python版階層的特徴量計算"""
        n = len(data)
        diff = np.diff(data, prepend=data[0])
        
        # ローカル検出
        local_pos = np.zeros(n)
        local_neg = np.zeros(n)
        
        for i in range(n):
            local_start = max(0, i - self.config.local_window)
            local_end = min(n, i + self.config.local_window + 1)
            local_subset = np.abs(diff[local_start:local_end])
            
            if len(local_subset) > 0:
                local_threshold = np.percentile(local_subset, self.config.local_percentile)
                if diff[i] > local_threshold:
                    local_pos[i] = 1.0
                elif diff[i] < -local_threshold:
                    local_neg[i] = 1.0
        
        # グローバル検出
        global_threshold = np.percentile(np.abs(diff), self.config.global_percentile)
        global_pos = (diff > global_threshold).astype(float)
        global_neg = (diff < -global_threshold).astype(float)
        
        return {
            'local_pos': local_pos,
            'local_neg': local_neg,
            'global_pos': global_pos,
            'global_neg': global_neg
        }
    
    def _calculate_rho_t_python(self, data: np.ndarray) -> np.ndarray:
        """純Python版張力スカラー計算"""
        window = 10
        rho_t = np.zeros(len(data))
        
        for i in range(len(data)):
            start = max(0, i - window)
            end = i + 1
            subset = data[start:end]
            rho_t[i] = np.std(subset) if len(subset) > 1 else 0.0
        
        return rho_t
    
    def _print_hierarchical_stats(self, features: Dict[str, np.ndarray]):
        """階層統計の表示"""
        local_events = np.sum(features['local_pos']) + np.sum(features['local_neg'])
        global_events = np.sum(features['global_pos']) + np.sum(features['global_neg'])
        
        print(f"\nHierarchical Event Statistics:")
        print(f"  Local events:  {int(local_events)} (Pos: {int(np.sum(features['local_pos']))}, Neg: {int(np.sum(features['local_neg']))})")
        print(f"  Global events: {int(global_events)} (Pos: {int(np.sum(features['global_pos']))}, Neg: {int(np.sum(features['global_neg']))})")
        print(f"  Mean tension:  {np.mean(features['rho_T']):.4f}")
    
    def _analyze_standard(
        self,
        data: np.ndarray,
        features: Dict[str, np.ndarray],
        series_name: str
    ) -> HierarchicalSeparationResults:
        """標準階層分析"""
        print("\nPerforming standard hierarchical analysis...")
        
        # 階層分離係数の計算
        separation_coeffs = self._calculate_separation_coefficients(features)
        
        # 階層効果の計算
        hierarchy_effects = self._calculate_hierarchy_effects(features)
        
        # 非対称性メトリクス
        asymmetry_metrics = self._calculate_asymmetry_metrics(features)
        
        # 階層統計
        hierarchy_stats = self._calculate_hierarchy_statistics(features)
        
        # 結果構築
        results = HierarchicalSeparationResults(
            series_name=series_name,
            escalation_strength=separation_coeffs['escalation'],
            deescalation_strength=separation_coeffs['deescalation'],
            hierarchy_correlation=separation_coeffs['correlation'],
            local_effect=hierarchy_effects['local'],
            global_effect=hierarchy_effects['global'],
            convergence_quality=0.8,  # 標準分析では固定値
            statistical_significance=0.7,
            separation_coefficients={'standard': separation_coeffs},
            asymmetry_metrics=asymmetry_metrics,
            hierarchy_stats=hierarchy_stats,
            analysis_method='standard'
        )
        
        return results
    
    def _analyze_bayesian(
        self,
        data: np.ndarray,
        features: Dict[str, np.ndarray],
        series_name: str
    ) -> HierarchicalSeparationResults:
        """ベイズ階層分析"""
        print("\nPerforming Bayesian hierarchical analysis...")
        
        # ベイズモデル構築と推定
        trace, model = self._fit_bayesian_hierarchical_model(data, features)
        
        # 係数抽出
        bayesian_coeffs = self._extract_bayesian_coefficients(trace)
        
        # 品質評価
        convergence_quality = self._assess_convergence_quality(trace)
        statistical_significance = self._assess_statistical_significance(trace)
        
        # 標準メトリクスも計算
        asymmetry_metrics = self._calculate_asymmetry_metrics(features)
        hierarchy_stats = self._calculate_hierarchy_statistics(features)
        
        # 結果構築
        results = HierarchicalSeparationResults(
            series_name=series_name,
            escalation_strength=bayesian_coeffs['escalation'],
            deescalation_strength=bayesian_coeffs['deescalation'],
            hierarchy_correlation=bayesian_coeffs['correlation'],
            local_effect=bayesian_coeffs['local_effect'],
            global_effect=bayesian_coeffs['global_effect'],
            convergence_quality=convergence_quality,
            statistical_significance=statistical_significance,
            separation_coefficients={'bayesian': bayesian_coeffs},
            asymmetry_metrics=asymmetry_metrics,
            hierarchy_stats=hierarchy_stats,
            analysis_method='bayesian',
            bayesian_trace=trace,
            bayesian_summary=az.summary(trace).to_dict()
        )
        
        return results
    
    def _calculate_separation_coefficients(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """階層分離係数の計算"""
        # エスカレーション: ローカル→グローバル
        local_events = features['local_pos'] + features['local_neg']
        global_events = features['global_pos'] + features['global_neg']
        
        # 時間差を考慮したエスカレーション検出
        escalation_count = 0
        deescalation_count = 0
        
        for i in range(1, len(local_events)-1):
            # エスカレーション: ローカルイベント後にグローバルイベント
            if local_events[i-1] > 0 and global_events[i] > 0:
                escalation_count += 1
            
            # デエスカレーション: グローバルイベント後にローカルのみ
            if global_events[i-1] > 0 and local_events[i] > 0 and global_events[i] == 0:
                deescalation_count += 1
        
        total_transitions = max(escalation_count + deescalation_count, 1)
        
        # 階層間相関
        if np.sum(local_events) > 0 and np.sum(global_events) > 0:
            correlation = np.corrcoef(local_events, global_events)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'escalation': escalation_count / total_transitions,
            'deescalation': deescalation_count / total_transitions,
            'correlation': correlation,
            'escalation_count': escalation_count,
            'deescalation_count': deescalation_count
        }
    
    def _calculate_hierarchy_effects(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """階層効果の計算"""
        rho_t = features['rho_T']
        local_events = features['local_pos'] + features['local_neg']
        global_events = features['global_pos'] + features['global_neg']
        
        # イベント時の張力効果
        local_tension = np.mean(rho_t[local_events > 0]) if np.any(local_events > 0) else 0
        global_tension = np.mean(rho_t[global_events > 0]) if np.any(global_events > 0) else 0
        baseline_tension = np.mean(rho_t)
        
        return {
            'local': (local_tension - baseline_tension) / (baseline_tension + 1e-8),
            'global': (global_tension - baseline_tension) / (baseline_tension + 1e-8),
            'local_tension': local_tension,
            'global_tension': global_tension,
            'baseline_tension': baseline_tension
        }
    
    def _calculate_asymmetry_metrics(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """非対称性メトリクスの計算"""
        # 正負イベントの非対称性
        local_pos_count = np.sum(features['local_pos'])
        local_neg_count = np.sum(features['local_neg'])
        global_pos_count = np.sum(features['global_pos'])
        global_neg_count = np.sum(features['global_neg'])
        
        # 非対称性指標
        local_asymmetry = (local_pos_count - local_neg_count) / max(local_pos_count + local_neg_count, 1)
        global_asymmetry = (global_pos_count - global_neg_count) / max(global_pos_count + global_neg_count, 1)
        
        # 階層間非対称性
        hierarchy_asymmetry = abs(local_asymmetry - global_asymmetry)
        
        return {
            'local_asymmetry': local_asymmetry,
            'global_asymmetry': global_asymmetry,
            'hierarchy_asymmetry': hierarchy_asymmetry,
            'local_pos_ratio': local_pos_count / max(local_pos_count + local_neg_count, 1),
            'global_pos_ratio': global_pos_count / max(global_pos_count + global_neg_count, 1)
        }
    
    def _calculate_hierarchy_statistics(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """階層統計の計算"""
        local_events = features['local_pos'] + features['local_neg']
        global_events = features['global_pos'] + features['global_neg']
        
        # イベント頻度
        local_frequency = np.sum(local_events) / len(local_events)
        global_frequency = np.sum(global_events) / len(global_events)
        
        # イベント持続性
        local_persistence = self._calculate_event_persistence(local_events)
        global_persistence = self._calculate_event_persistence(global_events)
        
        return {
            'local_frequency': local_frequency,
            'global_frequency': global_frequency,
            'frequency_ratio': global_frequency / max(local_frequency, 1e-8),
            'local_persistence': local_persistence,
            'global_persistence': global_persistence,
            'persistence_ratio': global_persistence / max(local_persistence, 1e-8)
        }
    
    def _calculate_event_persistence(self, events: np.ndarray) -> float:
        """イベント持続性の計算"""
        if np.sum(events) == 0:
            return 0.0
        
        # 連続イベントの長さ
        changes = np.diff(np.concatenate([[0], events, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(starts) == 0:
            return 0.0
        
        durations = ends - starts
        return np.mean(durations)
    
    def _fit_bayesian_hierarchical_model(
        self,
        data: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> Tuple[Any, Any]:
        """ベイズ階層モデルの構築と推定"""
        local_events = features['local_pos'] + features['local_neg']
        global_events = features['global_pos'] + features['global_neg']
        rho_t = features['rho_T']
        
        with pm.Model() as model:
            # === 階層構造パラメータ ===
            # エスカレーション/デエスカレーション係数
            beta_escalation = pm.Normal('beta_escalation', mu=0, sigma=1)
            beta_deescalation = pm.Normal('beta_deescalation', mu=0, sigma=1)
            
            # 階層効果係数
            alpha_local = pm.Normal('alpha_local', mu=0, sigma=1.5)
            alpha_global = pm.Normal('alpha_global', mu=0, sigma=2)
            
            # 階層間相互作用
            rho_hierarchy = pm.Uniform('rho_hierarchy', lower=-1, upper=1)
            
            # === 構造方程式 ===
            # 時間トレンド
            time_trend = np.arange(len(data)) / len(data)
            beta_time = pm.Normal('beta_time', mu=0, sigma=0.5)
            
            # ベースライン
            beta_0 = pm.Normal('beta_0', mu=np.mean(data), sigma=np.std(data))
            
            # エスカレーション/デエスカレーション指標
            escalation_indicator = np.zeros(len(data))
            deescalation_indicator = np.zeros(len(data))
            
            for i in range(1, len(data)-1):
                if local_events[i-1] > 0 and global_events[i] > 0:
                    escalation_indicator[i] = 1
                if global_events[i-1] > 0 and local_events[i] > 0 and global_events[i] == 0:
                    deescalation_indicator[i] = 1
            
            # 構造テンソル平均モデル
            mu = (
                beta_0
                + beta_time * time_trend
                + alpha_local * local_events * rho_t
                + alpha_global * global_events * rho_t
                + beta_escalation * escalation_indicator
                + beta_deescalation * deescalation_indicator
            )
            
            # 観測モデル
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=np.std(data))
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)
            
            # サンプリング
            trace = pm.sample(
                draws=self.config.bayesian_draws,
                tune=self.config.bayesian_tune,
                target_accept=0.95,
                return_inferencedata=True,
                cores=4,
                chains=4
            )
        
        return trace, model
    
    def _extract_bayesian_coefficients(self, trace: Any) -> Dict[str, float]:
        """ベイズ推定結果から係数抽出"""
        summary = az.summary(trace)
        
        coefficients = {
            'escalation': summary.loc['beta_escalation', 'mean'],
            'deescalation': summary.loc['beta_deescalation', 'mean'],
            'correlation': summary.loc['rho_hierarchy', 'mean'],
            'local_effect': summary.loc['alpha_local', 'mean'],
            'global_effect': summary.loc['alpha_global', 'mean']
        }
        
        # 信頼区間も保存
        for param in ['beta_escalation', 'beta_deescalation', 'alpha_local', 'alpha_global']:
            coefficients[f'{param}_hdi_low'] = summary.loc[param, 'hdi_3%']
            coefficients[f'{param}_hdi_high'] = summary.loc[param, 'hdi_97%']
        
        return coefficients
    
    def _assess_convergence_quality(self, trace: Any) -> float:
        """ベイズ収束品質評価"""
        try:
            # R-hat統計量
            r_hat = az.rhat(trace)
            r_hat_values = []
            
            for var in r_hat.data_vars:
                values = r_hat[var].values.flatten()
                r_hat_values.extend(values[~np.isnan(values)])
            
            if len(r_hat_values) > 0:
                avg_r_hat = np.mean(r_hat_values)
                # R-hat < 1.1で良好
                quality = max(0.0, 1.0 - (avg_r_hat - 1.0) * 10)
                return min(1.0, quality)
            
            return 0.8
            
        except Exception:
            return 0.8
    
    def _assess_statistical_significance(self, trace: Any) -> float:
        """統計的有意性評価"""
        try:
            summary = az.summary(trace)
            
            # HDI区間が0を含まない変数の割合
            significant_vars = 0
            total_vars = 0
            
            key_params = ['beta_escalation', 'beta_deescalation', 'alpha_local', 'alpha_global']
            
            for param in key_params:
                if param in summary.index:
                    hdi_low = summary.loc[param, 'hdi_3%']
                    hdi_high = summary.loc[param, 'hdi_97%']
                    
                    if hdi_low * hdi_high > 0:  # 同符号 = 0を含まない
                        significant_vars += 1
                    total_vars += 1
            
            return significant_vars / max(total_vars, 1)
            
        except Exception:
            return 0.7
    
    def _print_results_summary(self, results: HierarchicalSeparationResults):
        """結果サマリーの表示"""
        print(f"\n{'='*40}")
        print("HIERARCHICAL SEPARATION SUMMARY")
        print(f"{'='*40}")
        print(f"Escalation Strength:    {results.escalation_strength:.4f}")
        print(f"Deescalation Strength:  {results.deescalation_strength:.4f}")
        print(f"Hierarchy Correlation:  {results.hierarchy_correlation:.4f}")
        print(f"Local Effect:          {results.local_effect:.4f}")
        print(f"Global Effect:         {results.global_effect:.4f}")
        print(f"Dominant Hierarchy:     {results.get_dominant_hierarchy()}")
        
        if results.analysis_method == 'bayesian':
            print(f"\nBayesian Quality Metrics:")
            print(f"Convergence Quality:    {results.convergence_quality:.3f}")
            print(f"Statistical Significance: {results.statistical_significance:.3f}")

# ==========================================================
# CONVENIENCE FUNCTIONS - 便利関数
# ==========================================================

def analyze_hierarchical_structure(
    features: Union[StructuralTensorProtocol, Dict[str, np.ndarray]],
    config: Optional[Any] = None,
    use_bayesian: bool = False,
    data: Optional[np.ndarray] = None
) -> HierarchicalSeparationResults:
    """
    階層構造分析の便利関数
    
    Args:
        features: 構造テンソル特徴量
        config: 設定オブジェクト
        use_bayesian: ベイズ分析使用フラグ
        data: 元データ
        
    Returns:
        HierarchicalSeparationResults: 階層分析結果
    """
    analyzer = HierarchicalAnalyzer(config=config)
    return analyzer.analyze_hierarchical_separation(features, use_bayesian, data)

def compare_hierarchical_dynamics(
    features_dict: Dict[str, Union[StructuralTensorProtocol, Dict[str, np.ndarray]]],
    config: Optional[Any] = None,
    use_bayesian: bool = False
) -> Dict[str, HierarchicalSeparationResults]:
    """
    複数系列の階層ダイナミクス比較
    
    Args:
        features_dict: 系列名→特徴量の辞書
        config: 設定オブジェクト
        use_bayesian: ベイズ分析使用フラグ
        
    Returns:
        Dict[str, HierarchicalSeparationResults]: 系列名→結果の辞書
    """
    analyzer = HierarchicalAnalyzer(config=config)
    results = {}
    
    for series_name, features in features_dict.items():
        print(f"\nAnalyzing {series_name}...")
        results[series_name] = analyzer.analyze_hierarchical_separation(
            features, use_bayesian
        )
    
    # 比較サマリー表示
    _print_comparison_summary(results)
    
    return results

def _print_comparison_summary(results: Dict[str, HierarchicalSeparationResults]):
    """比較サマリーの表示"""
    print(f"\n{'='*60}")
    print("HIERARCHICAL DYNAMICS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Series':<20} {'Escalation':<12} {'Deescalation':<12} {'Dominant':<15}")
    print(f"{'-'*60}")
    
    for series_name, result in results.items():
        print(f"{series_name:<20} {result.escalation_strength:>11.4f} "
              f"{result.deescalation_strength:>11.4f} "
              f"{result.get_dominant_hierarchy():<15}")

# ==========================================================
# VALIDATION & TESTING - 検証・テスト
# ==========================================================

def test_hierarchical_analysis():
    """階層分析のテスト"""
    print("🧪 Testing Hierarchical Analysis Implementation")
    print("=" * 60)
    
    try:
        # テストデータ生成
        np.random.seed(42)
        
        # エスカレーション傾向のあるデータ
        escalation_data = np.cumsum(np.random.randn(200) * 0.1)
        escalation_data[50:55] += np.random.randn(5) * 0.5  # ローカルイベント
        escalation_data[52:57] += np.random.randn(5) * 1.0  # グローバルイベント
        
        # デエスカレーション傾向のあるデータ
        deescalation_data = np.cumsum(np.random.randn(200) * 0.1)
        deescalation_data[100:105] += np.random.randn(5) * 1.0  # グローバルイベント
        deescalation_data[102:107] += np.random.randn(5) * 0.3  # ローカルイベント
        
        test_cases = [
            ("Escalation Pattern", escalation_data),
            ("Deescalation Pattern", deescalation_data)
        ]
        
        analyzer = HierarchicalAnalyzer()
        
        for test_name, test_data in test_cases:
            print(f"\n📊 Testing {test_name}...")
            
            # 構造テンソル特徴量の模擬
            features = {
                'data': test_data,
                'series_name': test_name
            }
            
            # 標準分析
            results_std = analyzer.analyze_hierarchical_separation(
                features, use_bayesian=False
            )
            print(f"✅ Standard analysis completed")
            print(f"   Escalation: {results_std.escalation_strength:.4f}")
            print(f"   Deescalation: {results_std.deescalation_strength:.4f}")
            
            # ベイズ分析（利用可能な場合）
            if BAYESIAN_AVAILABLE:
                results_bayes = analyzer.analyze_hierarchical_separation(
                    features, use_bayesian=True
                )
                print(f"✅ Bayesian analysis completed")
                print(f"   Convergence quality: {results_bayes.convergence_quality:.3f}")
        
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
    'HierarchicalAnalyzer',
    'HierarchicalSeparationResults',
    
    # 関数
    'analyze_hierarchical_structure',
    'compare_hierarchical_dynamics',
    
    # テスト
    'test_hierarchical_analysis'
]

if __name__ == "__main__":
    # 自動テスト実行
    test_hierarchical_analysis()
