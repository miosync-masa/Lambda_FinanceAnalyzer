# ==========================================================
# lambda3/analysis/hierarchical.py
# Hierarchical Structure Analysis for Lambda³ Theory (完全版)
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³階層的構造分析（完全版）

構造テンソルの階層的∆ΛC変化を検出し、
短期・長期構造変化の分離とその相互作用を解析。
paste.txtの完全な階層分析機能を実装。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings

# Import dependencies
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian analysis disabled.")

# Import Lambda³ components
try:
    from ..core.structural_tensor import StructuralTensorFeatures
    from ..core.config import L3Config
    from ..core.jit_functions import (
        detect_local_global_jumps,
        calculate_rho_t,
        calculate_sync_profile_jit
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    warnings.warn("Core Lambda³ components not available.")

# ==========================================================
# HIERARCHICAL ANALYSIS RESULTS - 階層分析結果
# ==========================================================

@dataclass
class HierarchicalSeparationResults:
    """階層分離分析結果"""
    
    # 基本情報
    series_name: str
    data_length: int
    
    # 階層分離係数
    escalation_coefficient: float = 0.0
    deescalation_coefficient: float = 0.0
    local_effect_coefficient: float = 0.0
    global_effect_coefficient: float = 0.0
    hierarchy_correlation: float = 0.0
    
    # 信頼区間
    escalation_hdi: Tuple[float, float] = (0.0, 0.0)
    deescalation_hdi: Tuple[float, float] = (0.0, 0.0)
    local_effect_hdi: Tuple[float, float] = (0.0, 0.0)
    global_effect_hdi: Tuple[float, float] = (0.0, 0.0)
    
    # 非対称性指標
    transition_asymmetry: float = 0.0
    escalation_dominance: float = 0.0
    deescalation_dominance: float = 0.0
    
    # 階層統計
    local_mean_intensity: float = 0.0
    global_mean_intensity: float = 0.0
    local_std_intensity: float = 0.0
    global_std_intensity: float = 0.0
    
    # 階層メトリクス
    local_dominance: float = 0.0
    global_dominance: float = 0.0
    coupling_strength: float = 0.0
    escalation_rate: float = 0.0
    
    # ベイズ推定結果（オプション）
    trace: Optional[Any] = None
    model: Optional[Any] = None
    
    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """結果サマリーを取得"""
        return {
            'series_name': self.series_name,
            'escalation_strength': abs(self.escalation_coefficient),
            'deescalation_strength': abs(self.deescalation_coefficient),
            'hierarchy_correlation': self.hierarchy_correlation,
            'transition_asymmetry': self.transition_asymmetry,
            'local_dominance': self.local_dominance,
            'global_dominance': self.global_dominance,
            'coupling_strength': self.coupling_strength
        }
    
    def get_escalation_strength(self) -> float:
        """エスカレーション強度を取得"""
        return abs(self.escalation_coefficient)
    
    def get_deescalation_strength(self) -> float:
        """デエスカレーション強度を取得"""
        return abs(self.deescalation_coefficient)

# ==========================================================
# HIERARCHICAL ANALYZER - 階層分析器
# ==========================================================

class HierarchicalAnalyzer:
    """Lambda³階層的構造分析器"""
    
    def __init__(self, config: Optional[L3Config] = None):
        """
        Args:
            config: Lambda³設定オブジェクト
        """
        self.config = config or L3Config()
        self.pymc_available = PYMC_AVAILABLE
        
    def analyze_hierarchical_separation(
        self,
        features: StructuralTensorFeatures,
        use_bayesian: bool = True
    ) -> HierarchicalSeparationResults:
        """
        階層分離ダイナミクス分析
        
        Args:
            features: 構造テンソル特徴量
            use_bayesian: ベイズ推定を使用するか
            
        Returns:
            HierarchicalSeparationResults: 階層分析結果
        """
        # 階層メトリクスを計算
        hierarchy_metrics = self._calculate_hierarchy_metrics(features)
        
        # 結果オブジェクトを初期化
        results = HierarchicalSeparationResults(
            series_name=features.series_name,
            data_length=len(features.data),
            **hierarchy_metrics
        )
        
        # 階層強度統計を計算
        if self._has_sufficient_hierarchical_data(features):
            intensity_stats = self._calculate_intensity_statistics(features)
            results.local_mean_intensity = intensity_stats['local_mean']
            results.global_mean_intensity = intensity_stats['global_mean']
            results.local_std_intensity = intensity_stats['local_std']
            results.global_std_intensity = intensity_stats['global_std']
            results.hierarchy_correlation = intensity_stats['correlation']
        
        # ベイズ推定が可能な場合
        if use_bayesian and self.pymc_available and self._has_sufficient_hierarchical_data(features):
            bayesian_results = self._fit_hierarchical_bayesian_model(features)
            if bayesian_results:
                results = self._update_results_with_bayesian(results, bayesian_results)
        
        return results
    
    def _calculate_hierarchy_metrics(
        self,
        features: StructuralTensorFeatures
    ) -> Dict[str, float]:
        """階層性メトリクスを計算"""
        metrics = {}
        
        # 階層的イベント数を取得
        if hasattr(features, 'local_pos') and features.local_pos is not None:
            total_local_pos = np.sum(features.local_pos)
            total_local_neg = np.sum(features.local_neg)
            total_global_pos = np.sum(features.global_pos)
            total_global_neg = np.sum(features.global_neg)
            
            total_local = total_local_pos + total_local_neg
            total_global = total_global_pos + total_global_neg
            total_events = total_local + total_global
            
            # 基本メトリクス
            metrics['local_dominance'] = total_local / max(total_events, 1)
            metrics['global_dominance'] = total_global / max(total_events, 1)
            
            # 純粋成分メトリクス
            if hasattr(features, 'pure_local_pos') and features.pure_local_pos is not None:
                pure_local = np.sum(features.pure_local_pos) + np.sum(features.pure_local_neg)
                pure_global = np.sum(features.pure_global_pos) + np.sum(features.pure_global_neg)
                mixed = np.sum(features.mixed_pos) + np.sum(features.mixed_neg)
                
                metrics['coupling_strength'] = mixed / max(total_events, 1)
                metrics['escalation_rate'] = mixed / max(pure_local, 1)
                
                # 非対称性
                metrics['asymmetry_local'] = (np.sum(features.pure_local_pos) - np.sum(features.pure_local_neg)) / max(pure_local, 1)
                metrics['asymmetry_global'] = (np.sum(features.pure_global_pos) - np.sum(features.pure_global_neg)) / max(pure_global, 1)
        
        return metrics
    
    def _calculate_intensity_statistics(
        self,
        features: StructuralTensorFeatures
    ) -> Dict[str, float]:
        """階層強度統計を計算"""
        stats = {
            'local_mean': 0.0,
            'global_mean': 0.0,
            'local_std': 0.0,
            'global_std': 0.0,
            'correlation': 0.0
        }
        
        # 階層強度マスク
        local_mask = (features.local_pos + features.local_neg) > 0
        global_mask = (features.global_pos + features.global_neg) > 0
        
        # ローカル強度
        if np.any(local_mask):
            local_intensity = features.rho_T[local_mask]
            stats['local_mean'] = np.mean(local_intensity)
            stats['local_std'] = np.std(local_intensity)
        
        # グローバル強度
        if np.any(global_mask):
            global_intensity = features.rho_T[global_mask]
            stats['global_mean'] = np.mean(global_intensity)
            stats['global_std'] = np.std(global_intensity)
        
        # 階層間相関
        if np.sum(local_mask) > 10 and np.sum(global_mask) > 10:
            # 共通時点での相関を計算
            common_mask = local_mask & global_mask
            if np.sum(common_mask) > 5:
                local_at_common = features.rho_T[common_mask]
                # 時間シフトした相関を近似
                stats['correlation'] = np.corrcoef(
                    local_at_common[:-1],
                    local_at_common[1:]
                )[0, 1]
        
        return stats
    
    def _has_sufficient_hierarchical_data(
        self,
        features: StructuralTensorFeatures
    ) -> bool:
        """階層分析に十分なデータがあるかチェック"""
        if not hasattr(features, 'local_pos') or features.local_pos is None:
            return False
            
        # 最小イベント数のチェック
        MIN_EVENTS = 10
        local_events = np.sum(features.local_pos) + np.sum(features.local_neg)
        global_events = np.sum(features.global_pos) + np.sum(features.global_neg)
        
        return local_events >= MIN_EVENTS and global_events >= MIN_EVENTS
    
    def _fit_hierarchical_bayesian_model(
        self,
        features: StructuralTensorFeatures
    ) -> Optional[Dict[str, Any]]:
        """階層ベイズモデルのフィッティング"""
        if not self.pymc_available:
            return None
            
        try:
            # 階層強度特徴量を準備
            hierarchical_features = self._prepare_hierarchical_features(features)
            
            # ベイズモデル構築・推定
            with pm.Model() as model:
                # 基本項
                beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
                beta_time = pm.Normal('beta_time', mu=0, sigma=1)
                
                # 構造変化項
                beta_pos = pm.Normal('beta_pos', mu=0, sigma=3)
                beta_neg = pm.Normal('beta_neg', mu=0, sigma=3)
                beta_rho = pm.Normal('beta_rho', mu=0, sigma=2)
                
                # 階層効果係数
                alpha_local = pm.Normal('alpha_local', mu=0, sigma=1.5)
                alpha_global = pm.Normal('alpha_global', mu=0, sigma=2)
                
                # 階層遷移係数
                beta_escalation = pm.Normal('beta_escalation', mu=0, sigma=1)
                beta_deescalation = pm.Normal('beta_deescalation', mu=0, sigma=1)
                
                # 平均モデル
                mu = (
                    beta_0
                    + beta_time * hierarchical_features['time_trend']
                    + beta_pos * hierarchical_features['delta_pos']
                    + beta_neg * hierarchical_features['delta_neg']
                    + beta_rho * hierarchical_features['rho_T']
                    + alpha_local * hierarchical_features['local_intensity']
                    + alpha_global * hierarchical_features['global_intensity']
                    + beta_escalation * hierarchical_features['escalation']
                    + beta_deescalation * hierarchical_features['deescalation']
                )
                
                # 観測モデル
                sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=features.data)
                
                # サンプリング
                trace = pm.sample(
                    draws=self.config.draws,
                    tune=self.config.tune,
                    target_accept=self.config.target_accept,
                    return_inferencedata=True,
                    cores=4,
                    chains=self.config.chains
                )
                
            # 結果抽出
            summary = az.summary(trace)
            
            return {
                'trace': trace,
                'model': model,
                'summary': summary,
                'coefficients': self._extract_coefficients(summary)
            }
            
        except Exception as e:
            warnings.warn(f"Bayesian fitting failed: {e}")
            return None
    
    def _prepare_hierarchical_features(
        self,
        features: StructuralTensorFeatures
    ) -> Dict[str, np.ndarray]:
        """階層ベイズ用特徴量を準備"""
        n = len(features.data)
        
        # 基本特徴量
        prepared = {
            'time_trend': features.time_trend,
            'delta_pos': features.delta_LambdaC_pos,
            'delta_neg': features.delta_LambdaC_neg,
            'rho_T': features.rho_T
        }
        
        # 階層強度
        local_mask = (features.local_pos + features.local_neg) > 0
        global_mask = (features.global_pos + features.global_neg) > 0
        
        prepared['local_intensity'] = features.rho_T * local_mask.astype(float)
        prepared['global_intensity'] = features.rho_T * global_mask.astype(float)
        
        # 階層遷移指標
        prepared['escalation'] = np.zeros(n)
        prepared['deescalation'] = np.zeros(n)
        
        for i in range(1, n):
            # エスカレーション: local(t-1) & global(t)
            if local_mask[i-1] and global_mask[i]:
                prepared['escalation'][i] = 1.0
            # デエスカレーション: global(t-1) & local(t)
            if global_mask[i-1] and local_mask[i]:
                prepared['deescalation'][i] = 1.0
        
        return prepared
    
    def _extract_coefficients(self, summary) -> Dict[str, Dict[str, float]]:
        """ベイズ推定結果から係数を抽出"""
        coefficients = {}
        
        param_names = [
            ('beta_escalation', 'escalation'),
            ('beta_deescalation', 'deescalation'),
            ('alpha_local', 'local_effect'),
            ('alpha_global', 'global_effect')
        ]
        
        for param, name in param_names:
            if param in summary.index:
                coefficients[name] = {
                    'mean': summary.loc[param, 'mean'],
                    'hdi_lower': summary.loc[param, 'hdi_3%'],
                    'hdi_upper': summary.loc[param, 'hdi_97%']
                }
        
        return coefficients
    
    def _update_results_with_bayesian(
        self,
        results: HierarchicalSeparationResults,
        bayesian_results: Dict[str, Any]
    ) -> HierarchicalSeparationResults:
        """ベイズ推定結果で結果を更新"""
        coeffs = bayesian_results['coefficients']
        
        # 係数更新
        if 'escalation' in coeffs:
            results.escalation_coefficient = coeffs['escalation']['mean']
            results.escalation_hdi = (coeffs['escalation']['hdi_lower'], coeffs['escalation']['hdi_upper'])
            
        if 'deescalation' in coeffs:
            results.deescalation_coefficient = coeffs['deescalation']['mean']
            results.deescalation_hdi = (coeffs['deescalation']['hdi_lower'], coeffs['deescalation']['hdi_upper'])
            
        if 'local_effect' in coeffs:
            results.local_effect_coefficient = coeffs['local_effect']['mean']
            results.local_effect_hdi = (coeffs['local_effect']['hdi_lower'], coeffs['local_effect']['hdi_upper'])
            
        if 'global_effect' in coeffs:
            results.global_effect_coefficient = coeffs['global_effect']['mean']
            results.global_effect_hdi = (coeffs['global_effect']['hdi_lower'], coeffs['global_effect']['hdi_upper'])
        
        # 非対称性計算
        results.transition_asymmetry = abs(results.escalation_coefficient) - abs(results.deescalation_coefficient)
        total_transition = abs(results.escalation_coefficient) + abs(results.deescalation_coefficient)
        if total_transition > 0:
            results.escalation_dominance = abs(results.escalation_coefficient) / total_transition
            results.deescalation_dominance = abs(results.deescalation_coefficient) / total_transition
        
        # トレースとモデル保存
        results.trace = bayesian_results['trace']
        results.model = bayesian_results['model']
        
        return results

# ==========================================================
# COMPLETE HIERARCHICAL ANALYSIS - 完全階層分析
# ==========================================================

def complete_hierarchical_analysis(
    data_dict: Dict[str, np.ndarray],
    config: Optional[L3Config] = None,
    series_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    完全な階層的構造変化分析（paste.txt準拠）
    
    Lambda³理論における構造テンソル(Λ)の階層的∆ΛC変化を検出し、
    進行ベクトル(ΛF)および張力スカラー(ρT)の多階層相互作用を解析する。
    
    Args:
        data_dict: 構造テンソル系列データ辞書
        config: Lambda³解析設定
        series_names: 解析対象系列名リスト
        
    Returns:
        階層的構造変化解析結果
    """
    if config is None:
        config = L3Config()
        
    if series_names is None:
        series_names = list(data_dict.keys())
    
    results = {}
    
    print("=" * 80)
    print("LAMBDA³ HIERARCHICAL STRUCTURAL ANALYSIS")
    print("=" * 80)
    print(f"構造テンソル系列数: {len(series_names)}")
    print(f"短期構造窓: {config.local_window}, 長期構造窓: {config.global_window}")
    
    # 各系列の階層的構造変化検出
    from ..core.structural_tensor import StructuralTensorExtractor
    extractor = StructuralTensorExtractor(config)
    analyzer = HierarchicalAnalyzer(config)
    
    for name in series_names:
        if name not in data_dict:
            continue
            
        print(f"\n{'─' * 60}")
        print(f"ANALYZING STRUCTURAL TENSOR: {name}")
        print(f"{'─' * 60}")
        
        data = data_dict[name]
        print(f"データ長: {len(data)}")
        
        # 階層的特徴量抽出
        features = extractor.extract_features(
            data, 
            series_name=name,
            feature_level='hierarchical'
        )
        
        # 階層分離ダイナミクス分析
        separation_results = analyzer.analyze_hierarchical_separation(features)
        
        # 結果統合
        results[name] = {
            'features': features,
            'separation_results': separation_results,
            'data': data
        }
        
        # 結果表示
        print(f"\n階層分離係数:")
        print(f"  エスカレーション: {separation_results.escalation_coefficient:.4f}")
        print(f"  デエスカレーション: {separation_results.deescalation_coefficient:.4f}")
        print(f"  短期効果強度: {separation_results.local_effect_coefficient:.4f}")
        print(f"  長期効果強度: {separation_results.global_effect_coefficient:.4f}")
        print(f"  階層間相関: {separation_results.hierarchy_correlation:.4f}")
        
        print(f"\n階層メトリクス:")
        print(f"  ローカル優勢度: {separation_results.local_dominance:.3f}")
        print(f"  グローバル優勢度: {separation_results.global_dominance:.3f}")
        print(f"  階層結合強度: {separation_results.coupling_strength:.3f}")
        print(f"  エスカレーション率: {separation_results.escalation_rate:.3f}")
    
    # ペアワイズ階層的同期分析
    if len(series_names) >= 2:
        print(f"\n{'=' * 80}")
        print("PAIRWISE HIERARCHICAL SYNCHRONIZATION ANALYSIS")
        print(f"{'=' * 80}")
        
        results['pairwise_sync'] = analyze_pairwise_hierarchical_sync(
            results[series_names[0]]['features'],
            results[series_names[1]]['features'],
            config
        )
        
        # 同期結果表示
        print(f"\n階層別同期強度:")
        for sync_type, sync_data in results['pairwise_sync'].items():
            if isinstance(sync_data, dict) and 'max_sync' in sync_data:
                print(f"  {sync_data['display_name']}: {sync_data['max_sync']:.4f} (lag={sync_data['optimal_lag']})")
    
    # 階層的因果関係分析
    if len(series_names) >= 2 and 'causality' not in kwargs.get('skip_analyses', []):
        print(f"\n{'=' * 80}")
        print("HIERARCHICAL CAUSALITY ANALYSIS")
        print(f"{'=' * 80}")
        
        results['hierarchical_causality'] = analyze_hierarchical_causality(
            results[series_names[0]]['features'],
            results[series_names[1]]['features'],
            config
        )
        
        print(f"\n階層的因果パターン検出完了")
    
    print(f"\n{'=' * 80}")
    print("LAMBDA³ HIERARCHICAL ANALYSIS COMPLETED")
    print(f"{'=' * 80}")
    
    return results

def analyze_pairwise_hierarchical_sync(
    features1: StructuralTensorFeatures,
    features2: StructuralTensorFeatures,
    config: Optional[L3Config] = None
) -> Dict[str, Any]:
    """ペアワイズ階層的同期分析"""
    if config is None:
        config = L3Config()
        
    sync_results = {}
    
    # 階層別同期率計算
    hierarchy_types = [
        ('pure_local_pos', '純粋短期正'),
        ('pure_local_neg', '純粋短期負'),
        ('pure_global_pos', '純粋長期正'),
        ('pure_global_neg', '純粋長期負'),
        ('mixed_pos', '混合正'),
        ('mixed_neg', '混合負')
    ]
    
    for attr_name, display_name in hierarchy_types:
        if hasattr(features1, attr_name) and hasattr(features2, attr_name):
            events1 = getattr(features1, attr_name)
            events2 = getattr(features2, attr_name)
            
            if events1 is not None and events2 is not None:
                # 同期プロファイル計算
                lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
                    events1.astype(np.float64),
                    events2.astype(np.float64),
                    config.lag_window_default
                )
                
                sync_results[attr_name] = {
                    'display_name': display_name,
                    'max_sync': float(max_sync),
                    'optimal_lag': int(optimal_lag),
                    'sync_profile': {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
                }
    
    return sync_results

def analyze_hierarchical_causality(
    features1: StructuralTensorFeatures,
    features2: StructuralTensorFeatures,
    config: Optional[L3Config] = None,
    lag_window: int = 5
) -> Dict[str, Any]:
    """階層的因果関係分析"""
    causality_results = {}
    
    # 階層別因果関係
    hierarchy_pairs = [
        ('local_pos', 'local_pos', 'ローカル正→ローカル正'),
        ('local_neg', 'local_neg', 'ローカル負→ローカル負'),
        ('global_pos', 'global_pos', 'グローバル正→グローバル正'),
        ('global_neg', 'global_neg', 'グローバル負→グローバル負'),
        ('local_pos', 'global_pos', 'ローカル正→グローバル正（エスカレーション）'),
        ('global_pos', 'local_pos', 'グローバル正→ローカル正（デエスカレーション）')
    ]
    
    for source_attr, target_attr, pattern_name in hierarchy_pairs:
        if (hasattr(features1, source_attr) and hasattr(features2, target_attr) and
            getattr(features1, source_attr) is not None and getattr(features2, target_attr) is not None):
            
            source_events = getattr(features1, source_attr)
            target_events = getattr(features2, target_attr)
            
            # ラグ別因果確率
            causality_by_lag = {}
            for lag in range(1, lag_window + 1):
                if lag < len(source_events):
                    cause_events = source_events[:-lag]
                    effect_events = target_events[lag:]
                    
                    joint_prob = np.mean(cause_events * effect_events)
                    cause_prob = np.mean(cause_events)
                    
                    causality_prob = joint_prob / (cause_prob + 1e-8)
                    causality_by_lag[lag] = causality_prob
            
            causality_results[pattern_name] = causality_by_lag
    
    return causality_results

# ==========================================================
# STANDALONE ANALYSIS FUNCTION - スタンドアロン分析関数
# ==========================================================

def analyze_hierarchical_structure(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    config: Optional[L3Config] = None,
    detailed: bool = True
) -> Union[HierarchicalSeparationResults, Dict[str, Any]]:
    """
    階層的構造分析の便利関数
    
    Args:
        data: 単一系列または複数系列データ
        config: 設定オブジェクト
        detailed: 詳細分析を実行するか
        
    Returns:
        単一系列の場合: HierarchicalSeparationResults
        複数系列の場合: 完全な階層分析結果
    """
    if isinstance(data, np.ndarray):
        # 単一系列
        from ..core.structural_tensor import extract_lambda3_features
        features = extract_lambda3_features(
            data,
            feature_level='hierarchical',
            config=config
        )
        analyzer = HierarchicalAnalyzer(config)
        return analyzer.analyze_hierarchical_separation(features)
    else:
        # 複数系列
        if detailed:
            return complete_hierarchical_analysis(data, config)
        else:
            # 簡易版
            results = {}
            analyzer = HierarchicalAnalyzer(config)
            from ..core.structural_tensor import extract_lambda3_features
            
            for name, series_data in data.items():
                features = extract_lambda3_features(
                    series_data,
                    series_name=name,
                    feature_level='hierarchical',
                    config=config
                )
                results[name] = analyzer.analyze_hierarchical_separation(features)
            
            return results

# ==========================================================
# MODULE EXPORTS
# ==========================================================

__all__ = [
    # クラス
    'HierarchicalAnalyzer',
    'HierarchicalSeparationResults',
    
    # 主要関数
    'complete_hierarchical_analysis',
    'analyze_hierarchical_structure',
    'analyze_pairwise_hierarchical_sync',
    'analyze_hierarchical_causality'
]
