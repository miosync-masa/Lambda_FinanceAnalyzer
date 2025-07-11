# ==========================================================
# lambda3/analysis/pairwise.py
# Pairwise Interaction Analysis for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論ペアワイズ相互作用解析モジュール

構造テンソル(Λ)系列間の非対称相互作用を定量化し、
∆ΛC pulsationsの相互響応パターンを解析。

核心概念:
- 非対称相互作用: A→B と B→A の方向別影響度
- 構造結合: 構造テンソル変化の相互同期
- 因果構造: 時間非依存の構造空間因果関係
- 張力伝播: ρT張力スカラーの系列間伝播
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian analysis will be disabled.")

from ..core.config import L3BaseConfig, L3PairwiseConfig, L3BayesianConfig
from ..core.structural_tensor import StructuralTensorFeatures
from ..core.jit_functions import calculate_sync_profile, calculate_sync_rate_at_lag

# ==========================================================
# PAIRWISE INTERACTION RESULTS
# ==========================================================

@dataclass
class PairwiseInteractionResults:
    """
    ペアワイズ相互作用解析結果データクラス
    
    Lambda³理論における二系列間の構造テンソル相互作用を統合管理。
    非対称係数、因果パターン、同期特性を包含。
    """
    
    series_names: Tuple[str, str]
    
    # ベイズ推定結果
    traces: Dict[str, Any] = None
    models: Dict[str, Any] = None
    
    # 相互作用係数
    interaction_coefficients: Dict[str, Dict[str, float]] = None
    
    # 非対称性メトリクス
    asymmetry_metrics: Dict[str, float] = None
    
    # 因果パターン
    causality_patterns: Dict[str, Dict[int, float]] = None
    
    # 同期特性
    synchronization_profile: Dict[str, Any] = None
    
    # 予測性能
    prediction_metrics: Dict[str, float] = None
    
    # 相互作用品質
    interaction_quality: Dict[str, float] = None
    
    def __post_init__(self):
        """初期化後処理"""
        if self.traces is None:
            self.traces = {}
        if self.models is None:
            self.models = {}
        if self.interaction_coefficients is None:
            self.interaction_coefficients = {}
        if self.asymmetry_metrics is None:
            self.asymmetry_metrics = {}
        if self.causality_patterns is None:
            self.causality_patterns = {}
        if self.synchronization_profile is None:
            self.synchronization_profile = {}
        if self.prediction_metrics is None:
            self.prediction_metrics = {}
        if self.interaction_quality is None:
            self.interaction_quality = {}
    
    def get_interaction_strength(self, direction: str) -> float:
        """相互作用強度取得"""
        if direction in self.interaction_coefficients:
            coeffs = self.interaction_coefficients[direction]
            return sum(abs(v) for v in coeffs.values() if isinstance(v, (int, float)))
        return 0.0
    
    def get_asymmetry_score(self) -> float:
        """非対称性総合スコア取得"""
        return self.asymmetry_metrics.get('total_asymmetry', 0.0)
    
    def get_dominant_direction(self) -> Tuple[str, float]:
        """優勢方向と強度を取得"""
        name_a, name_b = self.series_names
        strength_a_to_b = self.get_interaction_strength(f'{name_a}_to_{name_b}')
        strength_b_to_a = self.get_interaction_strength(f'{name_b}_to_{name_a}')
        
        if strength_a_to_b > strength_b_to_a:
            return f'{name_a}_to_{name_b}', strength_a_to_b
        else:
            return f'{name_b}_to_{name_a}', strength_b_to_a
    
    def calculate_bidirectional_coupling(self) -> float:
        """双方向結合強度計算"""
        name_a, name_b = self.series_names
        strength_a_to_b = self.get_interaction_strength(f'{name_a}_to_{name_b}')
        strength_b_to_a = self.get_interaction_strength(f'{name_b}_to_{name_a}')
        return (strength_a_to_b + strength_b_to_a) / 2

# ==========================================================
# PAIRWISE ANALYZER
# ==========================================================

class PairwiseAnalyzer:
    """
    ペアワイズ相互作用解析器
    
    Lambda³理論に基づく二系列間の構造テンソル相互作用分析。
    非対称ベイズモデリング、因果関係検出、同期解析を統合実行。
    """
    
    def __init__(self, 
                 config: Optional[Union[L3PairwiseConfig, L3BaseConfig]] = None,
                 bayesian_config: Optional[L3BayesianConfig] = None):
        """
        初期化
        
        Args:
            config: ペアワイズ解析設定
            bayesian_config: ベイズ推定設定
        """
        if config is None:
            self.config = L3PairwiseConfig()
        elif isinstance(config, L3BaseConfig) and not isinstance(config, L3PairwiseConfig):
            # L3BaseConfigからペアワイズ設定を生成
            self.config = L3PairwiseConfig()
            # 基底設定をコピー
            for field_name in L3BaseConfig.__dataclass_fields__:
                if hasattr(config, field_name):
                    setattr(self.config, field_name, getattr(config, field_name))
        else:
            self.config = config
        
        self.bayesian_config = bayesian_config or L3BayesianConfig()
        self.analysis_history = []
    
    def analyze_asymmetric_interaction(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures,
        use_bayesian: bool = True
    ) -> PairwiseInteractionResults:
        """
        非対称相互作用分析
        
        Lambda³理論: 構造テンソル系列間の非対称相互作用を定量化
        A→B と B→A の方向別影響度をベイズ推定により解析
        
        Args:
            features_a, features_b: 構造テンソル特徴量
            use_bayesian: ベイズ推定使用フラグ
            
        Returns:
            PairwiseInteractionResults: ペアワイズ相互作用結果
        """
        series_names = (features_a.series_name, features_b.series_name)
        print(f"\n{'='*60}")
        print(f"ASYMMETRIC PAIRWISE ANALYSIS: {series_names[0]} ⇄ {series_names[1]}")
        print(f"{'='*60}")
        
        # データ長の統一
        min_length = min(len(features_a.data), len(features_b.data))
        features_a = self._truncate_features(features_a, min_length)
        features_b = self._truncate_features(features_b, min_length)
        
        print(f"データ長統一: {min_length} points")
        
        # 結果オブジェクト初期化
        results = PairwiseInteractionResults(series_names=series_names)
        
        # 同期プロファイル計算
        sync_profile = self._calculate_synchronization_profile(features_a, features_b)
        results.synchronization_profile = sync_profile
        
        print(f"同期解析完了:")
        print(f"  最大同期率: {sync_profile.get('max_sync', 0):.4f}")
        print(f"  最適遅延: {sync_profile.get('optimal_lag', 0)}")
        
        # ベイズ非対称相互作用分析
        if use_bayesian and BAYESIAN_AVAILABLE:
            try:
                bayesian_results = self._bayesian_asymmetric_analysis(features_a, features_b)
                results.traces = bayesian_results['traces']
                results.models = bayesian_results['models']
                results.interaction_coefficients = bayesian_results['coefficients']
                
                print(f"ベイズ非対称分析完了:")
                for direction, coeffs in results.interaction_coefficients.items():
                    if isinstance(coeffs, dict):
                        total_strength = sum(abs(v) for v in coeffs.values() if isinstance(v, (int, float)))
                        print(f"  {direction}: {total_strength:.4f}")
                
            except Exception as e:
                print(f"ベイズ推定エラー: {e}")
                print("古典的手法にフォールバック")
                use_bayesian = False
        
        # 古典的相互作用分析（フォールバック）
        if not use_bayesian or not BAYESIAN_AVAILABLE:
            classical_results = self._classical_asymmetric_analysis(features_a, features_b)
            results.interaction_coefficients = classical_results
            
            print(f"古典的非対称分析完了:")
            for direction, coeffs in results.interaction_coefficients.items():
                if isinstance(coeffs, dict):
                    total_strength = sum(abs(v) for v in coeffs.values() if isinstance(v, (int, float)))
                    print(f"  {direction}: {total_strength:.4f}")
        
        # 非対称性メトリクス計算
        asymmetry_metrics = self._calculate_pairwise_asymmetry(results.interaction_coefficients)
        results.asymmetry_metrics = asymmetry_metrics
        
        print(f"非対称性メトリクス:")
        for metric, value in asymmetry_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 因果パターン分析
        causality_patterns = self._analyze_causality_patterns(features_a, features_b)
        results.causality_patterns = causality_patterns
        
        print(f"因果パターン分析:")
        for pattern, lags in causality_patterns.items():
            if lags:
                max_causality = max(lags.values())
                print(f"  {pattern}: 最大因果確率 {max_causality:.4f}")
        
        # 相互作用品質評価
        quality_metrics = self._evaluate_interaction_quality(results)
        results.interaction_quality = quality_metrics
        
        print(f"相互作用品質: {quality_metrics.get('overall_quality', 0):.3f}")
        
        # 解析履歴記録
        self.analysis_history.append({
            'series_pair': series_names,
            'data_length': min_length,
            'method': 'bayesian' if use_bayesian and BAYESIAN_AVAILABLE else 'classical',
            'asymmetry_score': results.get_asymmetry_score(),
            'coupling_strength': results.calculate_bidirectional_coupling(),
            'quality': quality_metrics.get('overall_quality', 0)
        })
        
        return results
    
    def compare_multiple_pairs(
        self,
        features_dict: Dict[str, StructuralTensorFeatures],
        use_bayesian: bool = True
    ) -> Dict[str, Any]:
        """
        複数ペア相互作用比較
        
        Lambda³理論: 複数の構造テンソル系列ペア間の
        相互作用パターン比較とネットワーク構造解析
        
        Args:
            features_dict: {series_name: features} 辞書
            use_bayesian: ベイズ推定使用フラグ
            
        Returns:
            Dict: 複数ペア比較結果
        """
        series_names = list(features_dict.keys())
        if len(series_names) < 2:
            raise ValueError("At least 2 series required for pairwise comparison")
        
        print(f"\n{'='*60}")
        print(f"MULTIPLE PAIRWISE COMPARISON")
        print(f"系列数: {len(series_names)}")
        print(f"{'='*60}")
        
        # 全ペア組み合わせの解析
        pairwise_results = {}
        interaction_matrix = np.zeros((len(series_names), len(series_names)))
        asymmetry_matrix = np.zeros((len(series_names), len(series_names)))
        
        for i, name_a in enumerate(series_names):
            for j, name_b in enumerate(series_names):
                if i != j:  # 自己相互作用は除外
                    pair_key = f"{name_a}_vs_{name_b}"
                    
                    try:
                        result = self.analyze_asymmetric_interaction(
                            features_dict[name_a], 
                            features_dict[name_b], 
                            use_bayesian
                        )
                        pairwise_results[pair_key] = result
                        
                        # 相互作用行列更新
                        coupling_strength = result.calculate_bidirectional_coupling()
                        interaction_matrix[i, j] = coupling_strength
                        
                        # 非対称性行列更新
                        asymmetry_score = result.get_asymmetry_score()
                        asymmetry_matrix[i, j] = asymmetry_score
                        
                    except Exception as e:
                        print(f"警告: {pair_key} の解析に失敗: {e}")
                        continue
                else:
                    interaction_matrix[i, j] = 1.0  # 自己相互作用
                    asymmetry_matrix[i, j] = 0.0   # 自己非対称性はゼロ
        
        # ネットワーク解析
        network_analysis = self._analyze_interaction_network(
            interaction_matrix, asymmetry_matrix, series_names
        )
        
        # 統計的比較
        comparative_stats = self._calculate_comparative_statistics(pairwise_results)
        
        comparison_results = {
            'series_names': series_names,
            'pairwise_results': pairwise_results,
            'interaction_matrix': interaction_matrix,
            'asymmetry_matrix': asymmetry_matrix,
            'network_analysis': network_analysis,
            'comparative_statistics': comparative_stats,
            'summary': {
                'total_pairs_analyzed': len(pairwise_results),
                'mean_interaction_strength': float(np.mean(interaction_matrix[interaction_matrix > 0])),
                'mean_asymmetry': float(np.mean(asymmetry_matrix[asymmetry_matrix > 0])),
                'max_coupling': float(np.max(interaction_matrix)),
                'strongest_pair': self._find_strongest_pair(pairwise_results)
            }
        }
        
        print(f"複数ペア比較完了:")
        print(f"  解析ペア数: {len(pairwise_results)}")
        print(f"  平均相互作用強度: {comparison_results['summary']['mean_interaction_strength']:.4f}")
        print(f"  平均非対称性: {comparison_results['summary']['mean_asymmetry']:.4f}")
        
        return comparison_results
    
    def detect_interaction_regimes(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures,
        regime_window: int = 50
    ) -> Dict[str, Any]:
        """
        相互作用レジーム検出
        
        Lambda³理論: 構造テンソル相互作用の時変パターンを解析し、
        相互作用強度の変化に基づくレジーム転換を検出
        
        Args:
            features_a, features_b: 構造テンソル特徴量
            regime_window: レジーム検出窓サイズ
            
        Returns:
            Dict: 相互作用レジーム検出結果
        """
        print(f"\n相互作用レジーム検出: {features_a.series_name} ⇄ {features_b.series_name}")
        
        # データ長統一
        min_length = min(len(features_a.data), len(features_b.data))
        
        # ローリング相互作用強度計算
        interaction_strength = []
        asymmetry_scores = []
        
        for i in range(regime_window, min_length, regime_window // 2):
            # 窓内データ抽出
            window_start = max(0, i - regime_window)
            window_end = min(min_length, i)
            
            features_a_window = self._extract_window_features(features_a, window_start, window_end)
            features_b_window = self._extract_window_features(features_b, window_start, window_end)
            
            try:
                # 窓内相互作用分析
                window_result = self.analyze_asymmetric_interaction(
                    features_a_window, features_b_window, use_bayesian=False
                )
                
                interaction_strength.append(window_result.calculate_bidirectional_coupling())
                asymmetry_scores.append(window_result.get_asymmetry_score())
                
            except Exception as e:
                print(f"警告: 窓 {i} の分析に失敗: {e}")
                interaction_strength.append(0.0)
                asymmetry_scores.append(0.0)
        
        # レジーム境界検出
        regime_boundaries = self._detect_regime_boundaries(
            np.array(interaction_strength), threshold_factor=1.5
        )
        
        # レジーム特性分析
        regime_characteristics = self._analyze_regime_characteristics(
            interaction_strength, asymmetry_scores, regime_boundaries
        )
        
        return {
            'interaction_strength_series': interaction_strength,
            'asymmetry_series': asymmetry_scores,
            'regime_boundaries': regime_boundaries,
            'regime_characteristics': regime_characteristics,
            'window_size': regime_window,
            'n_regimes': len(regime_characteristics)
        }
    
    def _truncate_features(self, features: StructuralTensorFeatures, length: int) -> StructuralTensorFeatures:
        """特徴量を指定長に切り詰め"""
        truncated = StructuralTensorFeatures(
            data=features.data[:length],
            series_name=features.series_name
        )
        
        # 各特徴量を切り詰め
        for attr_name in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend']:
            attr_value = getattr(features, attr_name)
            if attr_value is not None:
                setattr(truncated, attr_name, attr_value[:length])
        
        return truncated
    
    def _calculate_synchronization_profile(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures
    ) -> Dict[str, Any]:
        """同期プロファイル計算"""
        # 構造変化イベント系列
        events_a = features_a.delta_LambdaC_pos + features_a.delta_LambdaC_neg
        events_b = features_b.delta_LambdaC_pos + features_b.delta_LambdaC_neg
        
        # JIT関数による同期プロファイル計算
        lags, sync_values, max_sync, optimal_lag = calculate_sync_profile(
            events_a.astype(np.float64),
            events_b.astype(np.float64),
            self.config.causality_lag_window
        )
        
        # 張力スカラー同期
        rho_correlation = np.corrcoef(features_a.rho_T, features_b.rho_T)[0, 1]
        
        return {
            'lag_profile': {int(lag): float(sync) for lag, sync in zip(lags, sync_values)},
            'max_sync': float(max_sync),
            'optimal_lag': int(optimal_lag),
            'tension_correlation': float(rho_correlation),
            'sync_stability': float(np.std(sync_values))
        }
    
    def _bayesian_asymmetric_analysis(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures
    ) -> Dict[str, Any]:
        """ベイズ非対称相互作用分析"""
        if not BAYESIAN_AVAILABLE:
            raise ImportError("PyMC not available for Bayesian analysis")
        
        name_a, name_b = features_a.series_name, features_b.series_name
        print(f"  ベイズモデル構築中...")
        
        # A系列に対するB系列の影響モデル
        print(f"    {name_b} → {name_a} 影響モデル")
        trace_b_to_a, model_b_to_a = self._fit_asymmetric_model(
            target_data=features_a.data,
            target_features=features_a,
            source_features=features_b
        )
        
        # B系列に対するA系列の影響モデル
        print(f"    {name_a} → {name_b} 影響モデル")
        trace_a_to_b, model_a_to_b = self._fit_asymmetric_model(
            target_data=features_b.data,
            target_features=features_b,
            source_features=features_a
        )
        
        # 相互作用係数抽出
        coefficients = self._extract_interaction_coefficients(
            trace_b_to_a, trace_a_to_b, name_a, name_b
        )
        
        return {
            'traces': {f'{name_b}_to_{name_a}': trace_b_to_a, f'{name_a}_to_{name_b}': trace_a_to_b},
            'models': {f'{name_b}_to_{name_a}': model_b_to_a, f'{name_a}_to_{name_b}': model_a_to_b},
            'coefficients': coefficients
        }
    
    def _fit_asymmetric_model(
        self,
        target_data: np.ndarray,
        target_features: StructuralTensorFeatures,
        source_features: StructuralTensorFeatures
    ) -> Tuple[Any, Any]:
        """非対称影響モデルフィッティング"""
        with pm.Model() as model:
            # 基本項
            beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
            beta_time = pm.Normal('beta_time', mu=0, sigma=1)
            
            # 自己効果項
            beta_self_pos = pm.Normal('beta_self_pos', mu=0, sigma=3)
            beta_self_neg = pm.Normal('beta_self_neg', mu=0, sigma=3)
            beta_self_tension = pm.Normal('beta_self_tension', mu=0, sigma=2)
            
            # 相互作用項（ソース→ターゲット）
            beta_interact_pos = pm.Normal('beta_interact_pos', mu=0, sigma=2)
            beta_interact_neg = pm.Normal('beta_interact_neg', mu=0, sigma=2)
            beta_interact_tension = pm.Normal('beta_interact_tension', mu=0, sigma=1.5)
            
            # 平均モデル
            mu = (
                beta_0
                + beta_time * target_features.time_trend
                + beta_self_pos * target_features.delta_LambdaC_pos
                + beta_self_neg * target_features.delta_LambdaC_neg
                + beta_self_tension * target_features.rho_T
                + beta_interact_pos * source_features.delta_LambdaC_pos
                + beta_interact_neg * source_features.delta_LambdaC_neg
                + beta_interact_tension * source_features.rho_T
            )
            
            # 観測モデル
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=target_data)
            
            # サンプリング
            trace = pm.sample(
                draws=self.bayesian_config.draws,
                tune=self.bayesian_config.tune,
                target_accept=self.bayesian_config.target_accept,
                return_inferencedata=True,
                cores=self.bayesian_config.cores,
                chains=self.bayesian_config.chains
            )
        
        return trace, model
    
    def _extract_interaction_coefficients(
        self,
        trace_b_to_a: Any,
        trace_a_to_b: Any,
        name_a: str,
        name_b: str
    ) -> Dict[str, Dict[str, float]]:
        """相互作用係数抽出"""
        summary_b_to_a = az.summary(trace_b_to_a)
        summary_a_to_b = az.summary(trace_a_to_b)
        
        coefficients = {
            f'{name_b}_to_{name_a}': {
                'pos_jump': summary_b_to_a.loc['beta_interact_pos', 'mean'],
                'neg_jump': summary_b_to_a.loc['beta_interact_neg', 'mean'],
                'tension': summary_b_to_a.loc['beta_interact_tension', 'mean']
            },
            f'{name_a}_to_{name_b}': {
                'pos_jump': summary_a_to_b.loc['beta_interact_pos', 'mean'],
                'neg_jump': summary_a_to_b.loc['beta_interact_neg', 'mean'],
                'tension': summary_a_to_b.loc['beta_interact_tension', 'mean']
            }
        }
        
        return coefficients
    
    def _classical_asymmetric_analysis(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures
    ) -> Dict[str, Dict[str, float]]:
        """古典的非対称相互作用分析"""
        name_a, name_b = features_a.series_name, features_b.series_name
        
        # 相関ベース相互作用係数
        def calculate_cross_correlation(source_events, target_events):
            if np.sum(source_events) > 0 and np.sum(target_events) > 0:
                return np.corrcoef(source_events, target_events)[0, 1]
            return 0.0
        
        coefficients = {
            f'{name_b}_to_{name_a}': {
                'pos_jump': calculate_cross_correlation(
                    features_b.delta_LambdaC_pos, features_a.delta_LambdaC_pos
                ),
                'neg_jump': calculate_cross_correlation(
                    features_b.delta_LambdaC_neg, features_a.delta_LambdaC_neg
                ),
                'tension': calculate_cross_correlation(
                    features_b.rho_T, features_a.rho_T
                )
            },
            f'{name_a}_to_{name_b}': {
                'pos_jump': calculate_cross_correlation(
                    features_a.delta_LambdaC_pos, features_b.delta_LambdaC_pos
                ),
                'neg_jump': calculate_cross_correlation(
                    features_a.delta_LambdaC_neg, features_b.delta_LambdaC_neg
                ),
                'tension': calculate_cross_correlation(
                    features_a.rho_T, features_b.rho_T
                )
            }
        }
        
        return coefficients
    
    def _calculate_pairwise_asymmetry(self, coefficients: Dict) -> Dict[str, float]:
        """ペアワイズ非対称性計算"""
        # 方向別係数取得
        directions = list(coefficients.keys())
        if len(directions) != 2:
            return {'error': 'Invalid coefficient structure'}
        
        dir_1, dir_2 = directions
        coeffs_1 = coefficients[dir_1]
        coeffs_2 = coefficients[dir_2]
        
        # 成分別非対称性
        pos_asymmetry = coeffs_1['pos_jump'] - coeffs_2['pos_jump']
        neg_asymmetry = coeffs_1['neg_jump'] - coeffs_2['neg_jump']
        tension_asymmetry = coeffs_1['tension'] - coeffs_2['tension']
        
        # 総合非対称性
        total_asymmetry = abs(pos_asymmetry) + abs(neg_asymmetry) + abs(tension_asymmetry)
        
        return {
            'pos_jump_asymmetry': pos_asymmetry,
            'neg_jump_asymmetry': neg_asymmetry,
            'tension_asymmetry': tension_asymmetry,
            'total_asymmetry': total_asymmetry,
            'directional_bias': (pos_asymmetry + neg_asymmetry + tension_asymmetry) / 3
        }
    
    def _analyze_causality_patterns(
        self,
        features_a: StructuralTensorFeatures,
        features_b: StructuralTensorFeatures
    ) -> Dict[str, Dict[int, float]]:
        """因果パターン分析"""
        patterns = {}
        
        # 各構造変化タイプの因果パターン
        event_types = [
            ('pos', features_a.delta_LambdaC_pos, features_b.delta_LambdaC_pos),
            ('neg', features_a.delta_LambdaC_neg, features_b.delta_LambdaC_neg)
        ]
        
        for event_type, events_a, events_b in event_types:
            # A → B 因果関係
            pattern_a_to_b = {}
            for lag in range(1, self.config.causality_lag_window + 1):
                if lag < len(events_a):
                    cause_events = events_a[:-lag]
                    effect_events = events_b[lag:]
                    
                    joint_prob = np.mean(cause_events * effect_events)
                    cause_prob = np.mean(cause_events)
                    
                    causality_prob = joint_prob / max(cause_prob, 1e-8)
                    pattern_a_to_b[lag] = causality_prob
            
            patterns[f'{features_a.series_name}_{event_type}_to_{features_b.series_name}_{event_type}'] = pattern_a_to_b
            
            # B → A 因果関係
            pattern_b_to_a = {}
            for lag in range(1, self.config.causality_lag_window + 1):
                if lag < len(events_b):
                    cause_events = events_b[:-lag]
                    effect_events = events_a[lag:]
                    
                    joint_prob = np.mean(cause_events * effect_events)
                    cause_prob = np.mean(cause_events)
                    
                    causality_prob = joint_prob / max(cause_prob, 1e-8)
                    pattern_b_to_a[lag] = causality_prob
            
            patterns[f'{features_b.series_name}_{event_type}_to_{features_a.series_name}_{event_type}'] = pattern_b_to_a
        
        return patterns
    
    def _evaluate_interaction_quality(self, results: PairwiseInteractionResults) -> Dict[str, float]:
        """相互作用品質評価"""
        # 相互作用強度品質
        total_interaction = results.calculate_bidirectional_coupling()
        interaction_quality = min(1.0, total_interaction / 0.5)  # 0.5を基準とした正規化
        
        # 非対称性品質（適度な非対称性が望ましい）
        asymmetry = results.get_asymmetry_score()
        asymmetry_quality = 1 - abs(asymmetry - 0.3) / 0.7  # 0.3程度の非対称性を最適とする
        asymmetry_quality = max(0, asymmetry_quality)
        
        # 同期品質
        sync_profile = results.synchronization_profile
        max_sync = sync_profile.get('max_sync', 0)
        sync_quality = min(1.0, max_sync / 0.4)  # 0.4を基準とした正規化
        
        # 因果品質
        causality_patterns = results.causality_patterns
        max_causality = 0
        for pattern, lags in causality_patterns.items():
            if lags:
                max_causality = max(max_causality, max(lags.values()))
        causality_quality = min(1.0, max_causality / 0.3)  # 0.3を基準とした正規化
        
        # 総合品質
        overall_quality = (interaction_quality + asymmetry_quality + sync_quality + causality_quality) / 4
        
        return {
            'interaction_quality': interaction_quality,
            'asymmetry_quality': asymmetry_quality,
            'synchronization_quality': sync_quality,
            'causality_quality': causality_quality,
            'overall_quality': overall_quality
        }
    
    def _analyze_interaction_network(
        self,
        interaction_matrix: np.ndarray,
        asymmetry_matrix: np.ndarray,
        series_names: List[str]
    ) -> Dict[str, Any]:
        """相互作用ネットワーク解析"""
        n_series = len(series_names)
        
        # ネットワーク密度
        total_possible_edges = n_series * (n_series - 1)
        active_edges = np.sum(interaction_matrix > self.config.min_causality_strength) - n_series
        network_density = active_edges / total_possible_edges if total_possible_edges > 0 else 0
        
        # 中心性計算
        out_strength = np.sum(interaction_matrix, axis=1) - np.diag(interaction_matrix)
        in_strength = np.sum(interaction_matrix, axis=0) - np.diag(interaction_matrix)
        total_strength = out_strength + in_strength
        
        centrality_ranking = sorted(
            [(i, series_names[i], total_strength[i]) for i in range(n_series)],
            key=lambda x: x[2], reverse=True
        )
        
        # 非対称性統計
        asymmetry_stats = {
            'mean_asymmetry': float(np.mean(asymmetry_matrix[asymmetry_matrix > 0])) if np.any(asymmetry_matrix > 0) else 0,
            'max_asymmetry': float(np.max(asymmetry_matrix)),
            'asymmetry_std': float(np.std(asymmetry_matrix[asymmetry_matrix > 0])) if np.any(asymmetry_matrix > 0) else 0
        }
        
        return {
            'network_density': network_density,
            'centrality_ranking': [(name, strength) for _, name, strength in centrality_ranking],
            'asymmetry_statistics': asymmetry_stats,
            'strongest_connections': self._find_strongest_connections(interaction_matrix, series_names),
            'most_asymmetric_pairs': self._find_most_asymmetric_pairs(asymmetry_matrix, series_names)
        }
    
    def _calculate_comparative_statistics(self, pairwise_results: Dict) -> Dict[str, Any]:
        """比較統計計算"""
        if not pairwise_results:
            return {}
        
        # 各メトリクスの分布統計
        interaction_strengths = [r.calculate_bidirectional_coupling() for r in pairwise_results.values()]
        asymmetry_scores = [r.get_asymmetry_score() for r in pairwise_results.values()]
        quality_scores = [r.interaction_quality.get('overall_quality', 0) for r in pairwise_results.values()]
        
        return {
            'interaction_strength_stats': {
                'mean': float(np.mean(interaction_strengths)),
                'std': float(np.std(interaction_strengths)),
                'min': float(np.min(interaction_strengths)),
                'max': float(np.max(interaction_strengths))
            },
            'asymmetry_stats': {
                'mean': float(np.mean(asymmetry_scores)),
                'std': float(np.std(asymmetry_scores)),
                'min': float(np.min(asymmetry_scores)),
                'max': float(np.max(asymmetry_scores))
            },
            'quality_stats': {
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores))
            }
        }
    
    def _find_strongest_pair(self, pairwise_results: Dict) -> Tuple[str, float]:
        """最強ペア検出"""
        if not pairwise_results:
            return "None", 0.0
        
        strongest_pair = max(
            pairwise_results.items(),
            key=lambda x: x[1].calculate_bidirectional_coupling()
        )
        
        return strongest_pair[0], strongest_pair[1].calculate_bidirectional_coupling()
    
    def _find_strongest_connections(self, matrix: np.ndarray, names: List[str]) -> List[Tuple[str, str, float]]:
        """最強接続検出"""
        connections = []
        n = len(names)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    connections.append((names[i], names[j], matrix[i, j]))
        
        # 強度順にソート
        connections.sort(key=lambda x: x[2], reverse=True)
        
        return connections[:5]  # 上位5接続
    
    def _find_most_asymmetric_pairs(self, matrix: np.ndarray, names: List[str]) -> List[Tuple[str, str, float]]:
        """最非対称ペア検出"""
        asymmetric_pairs = []
        n = len(names)
        
        for i in range(n):
            for j in range(i+1, n):  # 上三角のみ
                asymmetry = abs(matrix[i, j] - matrix[j, i])
                asymmetric_pairs.append((names[i], names[j], asymmetry))
        
        # 非対称性順にソート
        asymmetric_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return asymmetric_pairs[:5]  # 上位5ペア
    
    def _extract_window_features(
        self, 
        features: StructuralTensorFeatures, 
        start: int, 
        end: int
    ) -> StructuralTensorFeatures:
        """窓内特徴量抽出"""
        window_features = StructuralTensorFeatures(
            data=features.data[start:end],
            series_name=f"{features.series_name}_window_{start}_{end}"
        )
        
        # 各特徴量を窓内で抽出
        for attr_name in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend']:
            attr_value = getattr(features, attr_name)
            if attr_value is not None:
                setattr(window_features, attr_name, attr_value[start:end])
        
        return window_features
    
    def _detect_regime_boundaries(self, signal: np.ndarray, threshold_factor: float = 1.5) -> List[int]:
        """レジーム境界検出"""
        if len(signal) < 3:
            return []
        
        # 信号の1階差分
        diff_signal = np.diff(signal)
        
        # 閾値設定
        threshold = threshold_factor * np.std(diff_signal)
        
        # 境界点検出
        boundaries = []
        for i in range(1, len(diff_signal)):
            if abs(diff_signal[i]) > threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _analyze_regime_characteristics(
        self,
        interaction_strength: List[float],
        asymmetry_scores: List[float],
        boundaries: List[int]
    ) -> List[Dict[str, Any]]:
        """レジーム特性分析"""
        if not boundaries:
            boundaries = [0, len(interaction_strength)]
        else:
            boundaries = [0] + boundaries + [len(interaction_strength)]
        
        regimes = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            if end > start:
                regime_interaction = interaction_strength[start:end]
                regime_asymmetry = asymmetry_scores[start:end]
                
                regime_char = {
                    'regime_id': i,
                    'start_idx': start,
                    'end_idx': end,
                    'duration': end - start,
                    'mean_interaction': float(np.mean(regime_interaction)),
                    'mean_asymmetry': float(np.mean(regime_asymmetry)),
                    'interaction_volatility': float(np.std(regime_interaction)),
                    'asymmetry_volatility': float(np.std(regime_asymmetry))
                }
                
                # レジーム分類
                if regime_char['mean_interaction'] > 0.3:
                    if regime_char['mean_asymmetry'] > 0.2:
                        regime_char['regime_type'] = 'high_asymmetric_interaction'
                    else:
                        regime_char['regime_type'] = 'high_symmetric_interaction'
                else:
                    if regime_char['mean_asymmetry'] > 0.2:
                        regime_char['regime_type'] = 'low_asymmetric_interaction'
                    else:
                        regime_char['regime_type'] = 'low_interaction'
                
                regimes.append(regime_char)
        
        return regimes
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """解析履歴サマリー"""
        if not self.analysis_history:
            return {"message": "No pairwise analyses performed yet"}
        
        total_analyses = len(self.analysis_history)
        bayesian_analyses = sum(1 for h in self.analysis_history if h['method'] == 'bayesian')
        
        # 統計計算
        asymmetry_scores = [h['asymmetry_score'] for h in self.analysis_history]
        coupling_strengths = [h['coupling_strength'] for h in self.analysis_history]
        qualities = [h['quality'] for h in self.analysis_history]
        
        return {
            'total_analyses': total_analyses,
            'bayesian_ratio': bayesian_analyses / total_analyses,
            'asymmetry_stats': {
                'mean': float(np.mean(asymmetry_scores)),
                'std': float(np.std(asymmetry_scores)),
                'max': float(np.max(asymmetry_scores))
            },
            'coupling_stats': {
                'mean': float(np.mean(coupling_strengths)),
                'std': float(np.std(coupling_strengths)),
                'max': float(np.max(coupling_strengths))
            },
            'quality_stats': {
                'mean': float(np.mean(qualities)),
                'std': float(np.std(qualities)),
                'max': float(np.max(qualities))
            },
            'recent_analyses': self.analysis_history[-3:]
        }

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def analyze_pairwise_interaction(
    features_a: StructuralTensorFeatures,
    features_b: StructuralTensorFeatures,
    config: Optional[L3PairwiseConfig] = None,
    use_bayesian: bool = True
) -> PairwiseInteractionResults:
    """
    ペアワイズ相互作用解析の便利関数
    
    Args:
        features_a, features_b: 構造テンソル特徴量
        config: ペアワイズ解析設定
        use_bayesian: ベイズ推定使用フラグ
        
    Returns:
        PairwiseInteractionResults: ペアワイズ相互作用結果
    """
    analyzer = PairwiseAnalyzer(config)
    return analyzer.analyze_asymmetric_interaction(features_a, features_b, use_bayesian)

def compare_all_pairs(
    features_dict: Dict[str, StructuralTensorFeatures],
    config: Optional[L3PairwiseConfig] = None,
    use_bayesian: bool = True
) -> Dict[str, Any]:
    """
    全ペア比較の便利関数
    
    Args:
        features_dict: {series_name: features} 辞書
        config: ペアワイズ解析設定
        use_bayesian: ベイズ推定使用フラグ
        
    Returns:
        Dict: 複数ペア比較結果
    """
    analyzer = PairwiseAnalyzer(config)
    return analyzer.compare_multiple_pairs(features_dict, use_bayesian)

if __name__ == "__main__":
    print("Lambda³ Pairwise Analysis Module Test")
    print("=" * 50)
    
    # テスト用構造テンソル特徴量生成は省略
    # 実際のテストは統合テストで実行
    print("Pairwise analysis module loaded successfully!")
    print("Ready for Lambda³ asymmetric interaction analysis.")
