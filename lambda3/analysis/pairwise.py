# ==========================================================
# lambda3/analysis/pairwise.py
# Pairwise Asymmetric Analysis for Lambda³ Theory (修正版)
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³ペアワイズ非対称分析（完全修正版）

構造テンソル系列間の非対称相互作用を解析。
真の相互作用は非対称的であるというLambda³理論の
核心概念を実装。
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
        calculate_sync_profile_jit,
        detect_phase_coupling
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    warnings.warn("Core Lambda³ components not available.")

# ==========================================================
# PAIRWISE INTERACTION RESULTS - ペアワイズ相互作用結果
# ==========================================================

@dataclass
class PairwiseInteractionResults:
    """ペアワイズ非対称相互作用分析結果"""
    
    # 系列名
    series_name_a: str
    series_name_b: str
    
    # 相互作用係数（A → B）
    interaction_a_to_b_pos: float = 0.0
    interaction_a_to_b_neg: float = 0.0
    interaction_a_to_b_tension: float = 0.0
    
    # 相互作用係数（B → A）
    interaction_b_to_a_pos: float = 0.0
    interaction_b_to_a_neg: float = 0.0
    interaction_b_to_a_tension: float = 0.0
    
    # 自己効果係数
    self_effect_a_pos: float = 0.0
    self_effect_a_neg: float = 0.0
    self_effect_a_tension: float = 0.0
    self_effect_b_pos: float = 0.0
    self_effect_b_neg: float = 0.0
    self_effect_b_tension: float = 0.0
    
    # 時間遅延効果
    lag_effect_a_to_b: float = 0.0
    lag_effect_b_to_a: float = 0.0
    
    # 相関
    correlation_coefficient: float = 0.0
    
    # 非対称性指標
    pos_jump_asymmetry: float = 0.0
    neg_jump_asymmetry: float = 0.0
    tension_asymmetry: float = 0.0
    total_asymmetry: float = 0.0
    
    # 同期強度
    sync_strength_a_to_b: float = 0.0
    sync_strength_b_to_a: float = 0.0
    overall_sync_strength: float = 0.0
    
    # 位相結合（オプション）
    phase_coupling_strength: float = 0.0
    phase_lag: float = 0.0
    
    # ベイズ推定結果（オプション）
    traces: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    
    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """結果サマリーを取得"""
        return {
            'series_pair': f"{self.series_name_a} ⇄ {self.series_name_b}",
            'total_asymmetry': self.total_asymmetry,
            'overall_sync_strength': self.overall_sync_strength,
            'dominant_direction': self._get_dominant_direction(),
            'phase_coupling': self.phase_coupling_strength
        }
    
    def _get_dominant_direction(self) -> str:
        """優勢な影響方向を判定"""
        strength_a_to_b = abs(self.interaction_a_to_b_pos) + abs(self.interaction_a_to_b_neg) + abs(self.interaction_a_to_b_tension)
        strength_b_to_a = abs(self.interaction_b_to_a_pos) + abs(self.interaction_b_to_a_neg) + abs(self.interaction_b_to_a_tension)
        
        if strength_a_to_b > strength_b_to_a * 1.2:
            return f"{self.series_name_a} → {self.series_name_b}"
        elif strength_b_to_a > strength_a_to_b * 1.2:
            return f"{self.series_name_b} → {self.series_name_a}"
        else:
            return "Bidirectional"

# ==========================================================
# PAIRWISE ANALYZER - ペアワイズ分析器
# ==========================================================

class PairwiseAnalyzer:
    """Lambda³ペアワイズ非対称分析器"""
    
    def __init__(self, config: Optional[L3Config] = None):
        """
        Args:
            config: Lambda³設定オブジェクト
        """
        self.config = config or L3Config()
        self.pymc_available = PYMC_AVAILABLE
        
    def analyze_asymmetric_interaction(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures,
        use_bayesian: bool = True,
        calculate_phase_coupling: bool = True
    ) -> PairwiseInteractionResults:
        """
        非対称ペアワイズ相互作用分析
        
        Args:
            features_a: 系列Aの構造テンソル特徴量
            features_b: 系列Bの構造テンソル特徴量
            use_bayesian: ベイズ推定を使用するか
            calculate_phase_coupling: 位相結合を計算するか
            
        Returns:
            PairwiseInteractionResults: 相互作用分析結果
        """
        # データ長統一
        min_length = min(len(features_a.data), len(features_b.data))
        
        # 結果オブジェクト初期化
        results = PairwiseInteractionResults(
            series_name_a=features_a.series_name,
            series_name_b=features_b.series_name
        )
        
        # ベイズ推定
        if use_bayesian and self.pymc_available:
            bayesian_results = self._fit_asymmetric_models(features_a, features_b, min_length)
            if bayesian_results:
                results = self._update_results_with_bayesian(results, bayesian_results)
        
        # 同期強度計算
        results = self._calculate_sync_strength(results)
        
        # 位相結合計算
        if calculate_phase_coupling and CORE_AVAILABLE:
            coupling, lag = detect_phase_coupling(
                features_a.data[:min_length],
                features_b.data[:min_length]
            )
            results.phase_coupling_strength = coupling
            results.phase_lag = lag
        
        return results
    
    def _fit_asymmetric_models(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures,
        min_length: int
    ) -> Optional[Dict[str, Any]]:
        """非対称ベイズモデルのフィッティング"""
        try:
            # データ準備
            data_a = features_a.data[:min_length]
            data_b = features_b.data[:min_length]
            
            # A系列に対するB系列の影響モデル
            trace_b_to_a = self._fit_single_direction_model(
                target_data=data_a,
                target_features={
                    'delta_pos': features_a.delta_LambdaC_pos[:min_length],
                    'delta_neg': features_a.delta_LambdaC_neg[:min_length],
                    'rho_T': features_a.rho_T[:min_length],
                    'time_trend': features_a.time_trend[:min_length]
                },
                source_features={
                    'delta_pos': features_b.delta_LambdaC_pos[:min_length],
                    'delta_neg': features_b.delta_LambdaC_neg[:min_length],
                    'rho_T': features_b.rho_T[:min_length]
                },
                model_name='b_to_a'
            )
            
            # B系列に対するA系列の影響モデル
            trace_a_to_b = self._fit_single_direction_model(
                target_data=data_b,
                target_features={
                    'delta_pos': features_b.delta_LambdaC_pos[:min_length],
                    'delta_neg': features_b.delta_LambdaC_neg[:min_length],
                    'rho_T': features_b.rho_T[:min_length],
                    'time_trend': features_b.time_trend[:min_length]
                },
                source_features={
                    'delta_pos': features_a.delta_LambdaC_pos[:min_length],
                    'delta_neg': features_a.delta_LambdaC_neg[:min_length],
                    'rho_T': features_a.rho_T[:min_length]
                },
                model_name='a_to_b'
            )
            
            # 双方向システムモデル
            trace_system = self._fit_bidirectional_system_model(
                data_a, data_b,
                features_a, features_b,
                min_length
            )
            
            return {
                'trace_b_to_a': trace_b_to_a,
                'trace_a_to_b': trace_a_to_b,
                'trace_system': trace_system
            }
            
        except Exception as e:
            warnings.warn(f"Bayesian fitting failed: {e}")
            return None
    
    def _fit_single_direction_model(
        self,
        target_data: np.ndarray,
        target_features: Dict[str, np.ndarray],
        source_features: Dict[str, np.ndarray],
        model_name: str
    ) -> Any:
        """単一方向モデルのフィッティング"""
        with pm.Model() as model:
            # 自己効果項
            beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
            beta_time = pm.Normal('beta_time', mu=0, sigma=1)
            beta_self_pos = pm.Normal('beta_self_pos', mu=0, sigma=3)
            beta_self_neg = pm.Normal('beta_self_neg', mu=0, sigma=3)
            beta_self_rho = pm.Normal('beta_self_rho', mu=0, sigma=2)
            
            # 相互作用項
            beta_interact_pos = pm.Normal('beta_interact_pos', mu=0, sigma=2)
            beta_interact_neg = pm.Normal('beta_interact_neg', mu=0, sigma=2)
            beta_interact_stress = pm.Normal('beta_interact_stress', mu=0, sigma=1.5)
            
            # 平均モデル
            mu = (
                beta_0
                + beta_time * target_features['time_trend']
                + beta_self_pos * target_features['delta_pos']
                + beta_self_neg * target_features['delta_neg']
                + beta_self_rho * target_features['rho_T']
                + beta_interact_pos * source_features['delta_pos']
                + beta_interact_neg * source_features['delta_neg']
                + beta_interact_stress * source_features['rho_T']
            )
            
            # 観測モデル
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=target_data)
            
            # サンプリング
            trace = pm.sample(
                draws=self.config.draws,
                tune=self.config.tune,
                target_accept=self.config.target_accept,
                return_inferencedata=True,
                cores=4,
                chains=self.config.chains
            )
            
        return trace
    
    def _fit_bidirectional_system_model(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures,
        min_length: int
    ) -> Any:
        """双方向システムモデルのフィッティング"""
        with pm.Model() as model:
            # 系列A独立項
            beta_0_a = pm.Normal('beta_0_a', mu=0, sigma=2)
            beta_time_a = pm.Normal('beta_time_a', mu=0, sigma=1)
            beta_dLC_pos_a = pm.Normal('beta_dLC_pos_a', mu=0, sigma=3)
            beta_dLC_neg_a = pm.Normal('beta_dLC_neg_a', mu=0, sigma=3)
            beta_rhoT_a = pm.Normal('beta_rhoT_a', mu=0, sigma=2)
            
            # 系列B独立項
            beta_0_b = pm.Normal('beta_0_b', mu=0, sigma=2)
            beta_time_b = pm.Normal('beta_time_b', mu=0, sigma=1)
            beta_dLC_pos_b = pm.Normal('beta_dLC_pos_b', mu=0, sigma=3)
            beta_dLC_neg_b = pm.Normal('beta_dLC_neg_b', mu=0, sigma=3)
            beta_rhoT_b = pm.Normal('beta_rhoT_b', mu=0, sigma=2)
            
            # 相互作用項
            beta_interact_ab_pos = pm.Normal('beta_interact_ab_pos', mu=0, sigma=2)
            beta_interact_ab_neg = pm.Normal('beta_interact_ab_neg', mu=0, sigma=2)
            beta_interact_ab_stress = pm.Normal('beta_interact_ab_stress', mu=0, sigma=1.5)
            beta_interact_ba_pos = pm.Normal('beta_interact_ba_pos', mu=0, sigma=2)
            beta_interact_ba_neg = pm.Normal('beta_interact_ba_neg', mu=0, sigma=2)
            beta_interact_ba_stress = pm.Normal('beta_interact_ba_stress', mu=0, sigma=1.5)
            
            # 時間遅延項
            lag_data_a = np.concatenate([[0], data_a[:-1]])
            lag_data_b = np.concatenate([[0], data_b[:-1]])
            beta_lag_ab = pm.Normal('beta_lag_ab', mu=0, sigma=1)
            beta_lag_ba = pm.Normal('beta_lag_ba', mu=0, sigma=1)
            
            # 平均モデル
            mu_a = (
                beta_0_a
                + beta_time_a * features_a.time_trend[:min_length]
                + beta_dLC_pos_a * features_a.delta_LambdaC_pos[:min_length]
                + beta_dLC_neg_a * features_a.delta_LambdaC_neg[:min_length]
                + beta_rhoT_a * features_a.rho_T[:min_length]
                + beta_interact_ba_pos * features_b.delta_LambdaC_pos[:min_length]
                + beta_interact_ba_neg * features_b.delta_LambdaC_neg[:min_length]
                + beta_interact_ba_stress * features_b.rho_T[:min_length]
                + beta_lag_ba * lag_data_b
            )
            
            mu_b = (
                beta_0_b
                + beta_time_b * features_b.time_trend[:min_length]
                + beta_dLC_pos_b * features_b.delta_LambdaC_pos[:min_length]
                + beta_dLC_neg_b * features_b.delta_LambdaC_neg[:min_length]
                + beta_rhoT_b * features_b.rho_T[:min_length]
                + beta_interact_ab_pos * features_a.delta_LambdaC_pos[:min_length]
                + beta_interact_ab_neg * features_a.delta_LambdaC_neg[:min_length]
                + beta_interact_ab_stress * features_a.rho_T[:min_length]
                + beta_lag_ab * lag_data_a
            )
            
            # 観測モデル
            sigma_a = pm.HalfNormal('sigma_a', sigma=1)
            sigma_b = pm.HalfNormal('sigma_b', sigma=1)
            rho_ab = pm.Uniform('rho_ab', lower=-1, upper=1)
            
            # 共分散行列
            cov_matrix = pm.math.stack([
                [sigma_a**2, rho_ab * sigma_a * sigma_b],
                [rho_ab * sigma_a * sigma_b, sigma_b**2]
            ])
            
            # 同時観測
            y_combined = pm.math.stack([data_a, data_b]).T
            mu_combined = pm.math.stack([mu_a, mu_b]).T
            y_obs = pm.MvNormal('y_obs', mu=mu_combined, cov=cov_matrix, observed=y_combined)
            
            # サンプリング
            trace = pm.sample(
                draws=self.config.draws,
                tune=self.config.tune,
                target_accept=self.config.target_accept,
                return_inferencedata=True,
                cores=4,
                chains=self.config.chains
            )
            
        return trace
    
    def _update_results_with_bayesian(
        self,
        results: PairwiseInteractionResults,
        bayesian_results: Dict[str, Any]
    ) -> PairwiseInteractionResults:
        """ベイズ推定結果で結果を更新"""
        # システムモデルから係数抽出
        if 'trace_system' in bayesian_results and bayesian_results['trace_system'] is not None:
            summary = az.summary(bayesian_results['trace_system'])
            
            # 自己効果
            results.self_effect_a_pos = summary.loc['beta_dLC_pos_a', 'mean']
            results.self_effect_a_neg = summary.loc['beta_dLC_neg_a', 'mean']
            results.self_effect_a_tension = summary.loc['beta_rhoT_a', 'mean']
            results.self_effect_b_pos = summary.loc['beta_dLC_pos_b', 'mean']
            results.self_effect_b_neg = summary.loc['beta_dLC_neg_b', 'mean']
            results.self_effect_b_tension = summary.loc['beta_rhoT_b', 'mean']
            
            # 相互作用
            results.interaction_a_to_b_pos = summary.loc['beta_interact_ab_pos', 'mean']
            results.interaction_a_to_b_neg = summary.loc['beta_interact_ab_neg', 'mean']
            results.interaction_a_to_b_tension = summary.loc['beta_interact_ab_stress', 'mean']
            results.interaction_b_to_a_pos = summary.loc['beta_interact_ba_pos', 'mean']
            results.interaction_b_to_a_neg = summary.loc['beta_interact_ba_neg', 'mean']
            results.interaction_b_to_a_tension = summary.loc['beta_interact_ba_stress', 'mean']
            
            # 時間遅延
            results.lag_effect_a_to_b = summary.loc['beta_lag_ab', 'mean'] if 'beta_lag_ab' in summary.index else 0
            results.lag_effect_b_to_a = summary.loc['beta_lag_ba', 'mean'] if 'beta_lag_ba' in summary.index else 0
            
            # 相関
            results.correlation_coefficient = summary.loc['rho_ab', 'mean'] if 'rho_ab' in summary.index else 0
        
        # 非対称性計算
        results.pos_jump_asymmetry = results.interaction_a_to_b_pos - results.interaction_b_to_a_pos
        results.neg_jump_asymmetry = results.interaction_a_to_b_neg - results.interaction_b_to_a_neg
        results.tension_asymmetry = results.interaction_a_to_b_tension - results.interaction_b_to_a_tension
        results.total_asymmetry = (
            abs(results.pos_jump_asymmetry) + 
            abs(results.neg_jump_asymmetry) + 
            abs(results.tension_asymmetry)
        )
        
        # トレース保存
        results.traces = bayesian_results
        
        return results
    
    def _calculate_sync_strength(
        self,
        results: PairwiseInteractionResults
    ) -> PairwiseInteractionResults:
        """同期強度を計算"""
        # A → B 同期強度
        results.sync_strength_a_to_b = (
            abs(results.interaction_a_to_b_pos) +
            abs(results.interaction_a_to_b_neg) +
            abs(results.interaction_a_to_b_tension)
        ) / 3
        
        # B → A 同期強度
        results.sync_strength_b_to_a = (
            abs(results.interaction_b_to_a_pos) +
            abs(results.interaction_b_to_a_neg) +
            abs(results.interaction_b_to_a_tension)
        ) / 3
        
        # 総合同期強度
        results.overall_sync_strength = (
            results.sync_strength_a_to_b + 
            results.sync_strength_b_to_a
        ) / 2
        
        return results

# ==========================================================
# CONVENIENCE FUNCTIONS - 便利関数
# ==========================================================

def analyze_pairwise_interaction(
    data_a: Union[np.ndarray, StructuralTensorFeatures],
    data_b: Union[np.ndarray, StructuralTensorFeatures],
    config: Optional[L3Config] = None,
    series_names: Optional[Tuple[str, str]] = None
) -> PairwiseInteractionResults:
    """
    ペアワイズ相互作用分析の便利関数
    
    Args:
        data_a: 系列Aのデータまたは特徴量
        data_b: 系列Bのデータまたは特徴量
        config: 設定オブジェクト
        series_names: 系列名のタプル
        
    Returns:
        PairwiseInteractionResults: 相互作用分析結果
    """
    # 特徴量準備
    if isinstance(data_a, np.ndarray):
        from ..core.structural_tensor import extract_lambda3_features
        features_a = extract_lambda3_features(data_a, series_name=series_names[0] if series_names else "Series_A", config=config)
    else:
        features_a = data_a
        
    if isinstance(data_b, np.ndarray):
        from ..core.structural_tensor import extract_lambda3_features
        features_b = extract_lambda3_features(data_b, series_name=series_names[1] if series_names else "Series_B", config=config)
    else:
        features_b = data_b
    
    # 分析実行
    analyzer = PairwiseAnalyzer(config)
    return analyzer.analyze_asymmetric_interaction(features_a, features_b)

# ==========================================================
# MODULE EXPORTS
# ==========================================================

__all__ = [
    # クラス
    'PairwiseAnalyzer',
    'PairwiseInteractionResults',
    
    # 便利関数
    'analyze_pairwise_interaction'
]
