# ==========================================================
# lambda3/analysis/hierarchical.py
# Hierarchical Structure Analysis for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論階層構造解析モジュール

構造テンソル(Λ)の階層的∆ΛC変化を解析し、
局所-大域構造変化の分離、階層遷移ダイナミクス、
およびエスカレーション・デエスカレーション現象を定量化。

核心概念:
- 階層分離: 局所構造変化と大域構造変化の分離
- 遷移ダイナミクス: 階層間の構造変化伝播
- 非対称性: 上向き・下向き遷移の非対称特性
- 結合強度: 階層間の相互作用強度
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

from ..core.config import L3BaseConfig, L3HierarchicalConfig, L3BayesianConfig
from ..core.structural_tensor import StructuralTensorFeatures
from ..core.jit_functions import calculate_tension_scalar, normalize_array

# ==========================================================
# HIERARCHICAL SEPARATION RESULTS
# ==========================================================

@dataclass
class HierarchicalSeparationResults:
    """
    階層分離解析結果データクラス
    
    Lambda³理論における階層的∆ΛC変化の分離解析結果を統合管理。
    エスカレーション・デエスカレーション係数、非対称性メトリクス、
    および階層間相互作用強度を包含。
    """
    
    series_name: str
    
    # 分離系列
    local_series: np.ndarray
    global_series: np.ndarray
    
    # ベイズ推定結果
    trace: Optional[Any] = None
    model: Optional[Any] = None
    
    # 分離係数
    separation_coefficients: Dict[str, Dict[str, float]] = None
    
    # 非対称性メトリクス
    asymmetry_metrics: Dict[str, float] = None
    
    # 階層統計
    hierarchy_stats: Dict[str, float] = None
    
    # 品質指標
    separation_quality: Dict[str, float] = None
    
    def __post_init__(self):
        """初期化後処理"""
        if self.separation_coefficients is None:
            self.separation_coefficients = {}
        if self.asymmetry_metrics is None:
            self.asymmetry_metrics = {}
        if self.hierarchy_stats is None:
            self.hierarchy_stats = {}
        if self.separation_quality is None:
            self.separation_quality = {}
    
    def get_escalation_strength(self) -> float:
        """エスカレーション強度取得"""
        return abs(self.separation_coefficients.get('escalation', {}).get('coefficient', 0.0))
    
    def get_deescalation_strength(self) -> float:
        """デエスカレーション強度取得"""
        return abs(self.separation_coefficients.get('deescalation', {}).get('coefficient', 0.0))
    
    def get_hierarchy_correlation(self) -> float:
        """階層間相関取得"""
        return self.separation_coefficients.get('hierarchy_correlation', 0.0)
    
    def calculate_transition_dominance(self) -> Dict[str, float]:
        """遷移優勢度計算"""
        escalation = self.get_escalation_strength()
        deescalation = self.get_deescalation_strength()
        total_transition = escalation + deescalation
        
        if total_transition > 1e-8:
            return {
                'escalation_dominance': escalation / total_transition,
                'deescalation_dominance': deescalation / total_transition,
                'transition_asymmetry': (escalation - deescalation) / total_transition
            }
        else:
            return {
                'escalation_dominance': 0.5,
                'deescalation_dominance': 0.5,
                'transition_asymmetry': 0.0
            }

# ==========================================================
# HIERARCHICAL ANALYZER
# ==========================================================

class HierarchicalAnalyzer:
    """
    階層構造解析器
    
    Lambda³理論に基づく階層的構造変化の包括的解析。
    局所-大域分離、階層遷移ダイナミクス、ベイズ推定による
    階層相互作用係数の定量化を実行。
    """
    
    def __init__(self, 
                 config: Optional[Union[L3HierarchicalConfig, L3BaseConfig]] = None,
                 bayesian_config: Optional[L3BayesianConfig] = None):
        """
        初期化
        
        Args:
            config: 階層解析設定
            bayesian_config: ベイズ推定設定
        """
        if config is None:
            self.config = L3HierarchicalConfig()
        elif isinstance(config, L3BaseConfig) and not isinstance(config, L3HierarchicalConfig):
            # L3BaseConfigから階層設定を生成
            self.config = L3HierarchicalConfig()
            # 基底設定をコピー
            for field_name in L3BaseConfig.__dataclass_fields__:
                if hasattr(config, field_name):
                    setattr(self.config, field_name, getattr(config, field_name))
        else:
            self.config = config
        
        self.bayesian_config = bayesian_config or L3BayesianConfig()
        self.analysis_history = []
    
    def analyze_hierarchical_separation(
        self, 
        features: StructuralTensorFeatures,
        use_bayesian: bool = True
    ) -> HierarchicalSeparationResults:
        """
        階層分離ダイナミクス解析
        
        Lambda³理論: 構造テンソルの階層的∆ΛC変化を分離し、
        局所-大域遷移の非対称ダイナミクスを定量化。
        
        Args:
            features: 構造テンソル特徴量
            use_bayesian: ベイズ推定使用フラグ
            
        Returns:
            HierarchicalSeparationResults: 階層分離解析結果
        """
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL SEPARATION ANALYSIS: {features.series_name}")
        print(f"{'='*60}")
        
        # 階層的特徴量の存在確認
        if not self._validate_hierarchical_features(features):
            raise ValueError("Hierarchical features not available in input")
        
        # 階層強度系列の構築
        local_series, global_series = self._construct_hierarchy_series(features)
        
        # 初期結果オブジェクト
        results = HierarchicalSeparationResults(
            series_name=features.series_name,
            local_series=local_series,
            global_series=global_series
        )
        
        # 階層統計計算
        hierarchy_stats = self._calculate_hierarchy_statistics(
            local_series, global_series, features
        )
        results.hierarchy_stats = hierarchy_stats
        
        print(f"階層統計計算完了:")
        print(f"  局所平均強度: {hierarchy_stats.get('local_mean', 0):.4f}")
        print(f"  大域平均強度: {hierarchy_stats.get('global_mean', 0):.4f}")
        print(f"  階層間相関: {hierarchy_stats.get('hierarchy_correlation', 0):.4f}")
        
        # ベイズ推定による階層相互作用分析
        if use_bayesian and BAYESIAN_AVAILABLE:
            try:
                bayesian_results = self._bayesian_hierarchy_analysis(
                    features.data, local_series, global_series
                )
                results.trace = bayesian_results['trace']
                results.model = bayesian_results['model']
                results.separation_coefficients = bayesian_results['coefficients']
                
                print(f"ベイズ階層分析完了:")
                for coeff_name, coeff_data in results.separation_coefficients.items():
                    if isinstance(coeff_data, dict) and 'coefficient' in coeff_data:
                        print(f"  {coeff_name}: {coeff_data['coefficient']:.4f}")
                
            except Exception as e:
                print(f"ベイズ推定エラー: {e}")
                print("非ベイズ手法にフォールバック")
                use_bayesian = False
        
        # 非ベイズ階層分析（フォールバック）
        if not use_bayesian or not BAYESIAN_AVAILABLE:
            classical_results = self._classical_hierarchy_analysis(
                local_series, global_series, features
            )
            results.separation_coefficients = classical_results
            
            print(f"古典的階層分析完了:")
            for coeff_name, coeff_value in results.separation_coefficients.items():
                if isinstance(coeff_value, (int, float)):
                    print(f"  {coeff_name}: {coeff_value:.4f}")
        
        # 非対称性メトリクス計算
        asymmetry_metrics = self._calculate_asymmetry_metrics(results.separation_coefficients)
        results.asymmetry_metrics = asymmetry_metrics
        
        print(f"非対称性メトリクス:")
        for metric_name, metric_value in asymmetry_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # 分離品質評価
        separation_quality = self._evaluate_separation_quality(results)
        results.separation_quality = separation_quality
        
        print(f"分離品質: {separation_quality.get('overall_quality', 0):.3f}")
        
        # 解析履歴記録
        self.analysis_history.append({
            'series_name': features.series_name,
            'data_length': len(features.data),
            'method': 'bayesian' if use_bayesian and BAYESIAN_AVAILABLE else 'classical',
            'separation_quality': separation_quality.get('overall_quality', 0),
            'escalation_strength': results.get_escalation_strength(),
            'deescalation_strength': results.get_deescalation_strength()
        })
        
        return results
    
    def compare_hierarchical_systems(
        self, 
        features_list: List[StructuralTensorFeatures],
        use_bayesian: bool = True
    ) -> Dict[str, Any]:
        """
        複数系列の階層システム比較
        
        Lambda³理論: 複数の構造テンソル系列における
        階層特性の比較分析とクラスタリング
        
        Args:
            features_list: 構造テンソル特徴量リスト
            use_bayesian: ベイズ推定使用フラグ
            
        Returns:
            Dict: 階層システム比較結果
        """
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL SYSTEMS COMPARISON")
        print(f"系列数: {len(features_list)}")
        print(f"{'='*60}")
        
        # 各系列の階層分析実行
        separation_results = []
        for features in features_list:
            try:
                result = self.analyze_hierarchical_separation(features, use_bayesian)
                separation_results.append(result)
            except Exception as e:
                print(f"警告: {features.series_name} の解析に失敗: {e}")
                continue
        
        if len(separation_results) == 0:
            return {'error': 'No successful hierarchical analyses'}
        
        # 比較メトリクス計算
        comparison_results = {
            'series_names': [r.series_name for r in separation_results],
            'individual_results': separation_results,
            'comparative_metrics': self._calculate_comparative_metrics(separation_results),
            'clustering_analysis': self._cluster_hierarchical_systems(separation_results),
            'ranking_analysis': self._rank_hierarchical_systems(separation_results)
        }
        
        print(f"階層システム比較完了:")
        print(f"  成功解析数: {len(separation_results)}")
        print(f"  比較メトリクス: {len(comparison_results['comparative_metrics'])}")
        
        return comparison_results
    
    def detect_hierarchical_transitions(
        self, 
        features: StructuralTensorFeatures,
        transition_window: int = 10
    ) -> Dict[str, Any]:
        """
        階層遷移イベント検出
        
        Lambda³理論: 局所↔大域構造変化の遷移パターンを
        時系列的に検出し、遷移の方向性と強度を定量化
        
        Args:
            features: 構造テンソル特徴量
            transition_window: 遷移検出窓サイズ
            
        Returns:
            Dict: 階層遷移検出結果
        """
        if not self._validate_hierarchical_features(features):
            return {'error': 'Hierarchical features not available'}
        
        # 階層イベント系列
        local_events = features.local_pos + features.local_neg
        global_events = features.global_pos + features.global_neg
        
        # エスカレーション検出（局所→大域）
        escalation_events = []
        for i in range(transition_window, len(local_events)):
            # 過去の窓で局所イベントがあり、現在大域イベントがある
            local_activity = np.sum(local_events[i-transition_window:i])
            if local_activity > 0 and global_events[i] > 0:
                escalation_events.append({
                    'position': i,
                    'local_activity': float(local_activity),
                    'intensity': float(global_events[i])
                })
        
        # デエスカレーション検出（大域→局所）
        deescalation_events = []
        for i in range(transition_window, len(global_events)):
            # 過去の窓で大域イベントがあり、現在局所イベントがある
            global_activity = np.sum(global_events[i-transition_window:i])
            if global_activity > 0 and local_events[i] > 0:
                deescalation_events.append({
                    'position': i,
                    'global_activity': float(global_activity),
                    'intensity': float(local_events[i])
                })
        
        # 遷移統計
        transition_stats = {
            'escalation_count': len(escalation_events),
            'deescalation_count': len(deescalation_events),
            'total_transitions': len(escalation_events) + len(deescalation_events),
            'escalation_rate': len(escalation_events) / max(np.sum(local_events), 1),
            'deescalation_rate': len(deescalation_events) / max(np.sum(global_events), 1),
            'transition_asymmetry': len(escalation_events) - len(deescalation_events),
            'transition_balance': abs(len(escalation_events) - len(deescalation_events)) / max(len(escalation_events) + len(deescalation_events), 1)
        }
        
        return {
            'escalation_events': escalation_events,
            'deescalation_events': deescalation_events,
            'transition_statistics': transition_stats,
            'series_name': features.series_name,
            'detection_window': transition_window
        }
    
    def _validate_hierarchical_features(self, features: StructuralTensorFeatures) -> bool:
        """階層的特徴量の妥当性確認"""
        required_features = ['local_pos', 'local_neg', 'global_pos', 'global_neg']
        return all(hasattr(features, feat) and getattr(features, feat) is not None 
                  for feat in required_features)
    
    def _construct_hierarchy_series(
        self, 
        features: StructuralTensorFeatures
    ) -> Tuple[np.ndarray, np.ndarray]:
        """階層強度系列構築"""
        # 局所イベントマスク
        local_mask = (features.local_pos + features.local_neg) > 0
        global_mask = (features.global_pos + features.global_neg) > 0
        
        # 張力スカラーを用いた階層強度系列
        local_series = features.rho_T * local_mask.astype(float)
        global_series = features.rho_T * global_mask.astype(float)
        
        return local_series, global_series
    
    def _calculate_hierarchy_statistics(
        self, 
        local_series: np.ndarray, 
        global_series: np.ndarray,
        features: StructuralTensorFeatures
    ) -> Dict[str, float]:
        """階層統計計算"""
        local_mask = local_series > 0
        global_mask = global_series > 0
        
        stats = {
            'local_events': int(np.sum(local_mask)),
            'global_events': int(np.sum(global_mask)),
            'local_mean': float(np.mean(local_series[local_mask])) if np.any(local_mask) else 0.0,
            'global_mean': float(np.mean(global_series[global_mask])) if np.any(global_mask) else 0.0,
            'local_std': float(np.std(local_series[local_mask])) if np.any(local_mask) else 0.0,
            'global_std': float(np.std(global_series[global_mask])) if np.any(global_mask) else 0.0
        }
        
        # 階層間相関計算
        if np.any(local_mask) and np.any(global_mask):
            # 共通の非ゼロ点で相関計算
            common_mask = local_mask | global_mask
            if np.sum(common_mask) > 1:
                local_common = local_series[common_mask]
                global_common = global_series[common_mask]
                
                try:
                    correlation_matrix = np.corrcoef(local_common, global_common)
                    if correlation_matrix.size > 1:
                        stats['hierarchy_correlation'] = float(correlation_matrix[0, 1])
                    else:
                        stats['hierarchy_correlation'] = 0.0
                except:
                    stats['hierarchy_correlation'] = 0.0
            else:
                stats['hierarchy_correlation'] = 0.0
        else:
            stats['hierarchy_correlation'] = 0.0
        
        return stats
    
    def _bayesian_hierarchy_analysis(
        self, 
        original_data: np.ndarray,
        local_series: np.ndarray, 
        global_series: np.ndarray
    ) -> Dict[str, Any]:
        """ベイズ階層分析"""
        if not BAYESIAN_AVAILABLE:
            raise ImportError("PyMC not available for Bayesian analysis")
        
        # 階層特徴量準備
        local_events = (local_series > 0).astype(float)
        global_events = (global_series > 0).astype(float)
        time_trend = np.arange(len(original_data), dtype=np.float64)
        
        # エスカレーション・デエスカレーション指標
        escalation_indicator = np.diff(np.concatenate([[0], global_events]))
        deescalation_indicator = np.diff(np.concatenate([[0], local_events]))
        
        print(f"  ベイズモデル構築中...")
        print(f"    局所イベント: {np.sum(local_events)}")
        print(f"    大域イベント: {np.sum(global_events)}")
        print(f"    エスカレーション遷移: {np.sum(escalation_indicator > 0)}")
        print(f"    デエスカレーション遷移: {np.sum(deescalation_indicator > 0)}")
        
        with pm.Model() as model:
            # 基本項
            beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
            beta_time = pm.Normal('beta_time', mu=0, sigma=1)
            
            # 階層効果係数
            alpha_local = pm.Normal('alpha_local', mu=0, sigma=1.5)
            alpha_global = pm.Normal('alpha_global', mu=0, sigma=2)
            
            # 階層遷移係数
            beta_escalation = pm.Normal('beta_escalation', mu=0, sigma=1)
            beta_deescalation = pm.Normal('beta_deescalation', mu=0, sigma=1)
            
            # 構造テンソル平均モデル
            mu = (
                beta_0
                + beta_time * time_trend
                + alpha_local * local_series
                + alpha_global * global_series
                + beta_escalation * escalation_indicator
                + beta_deescalation * deescalation_indicator
            )
            
            # 観測モデル
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=original_data)
            
            # サンプリング
            print(f"    MCMCサンプリング開始...")
            trace = pm.sample(
                draws=self.bayesian_config.draws,
                tune=self.bayesian_config.tune,
                target_accept=self.bayesian_config.target_accept,
                return_inferencedata=True,
                cores=self.bayesian_config.cores,
                chains=self.bayesian_config.chains
            )
        
        # 係数抽出
        summary = az.summary(trace)
        
        coefficients = {
            'escalation': {
                'coefficient': summary.loc['beta_escalation', 'mean'],
                'hdi_lower': summary.loc['beta_escalation', 'hdi_3%'],
                'hdi_upper': summary.loc['beta_escalation', 'hdi_97%']
            },
            'deescalation': {
                'coefficient': summary.loc['beta_deescalation', 'mean'],
                'hdi_lower': summary.loc['beta_deescalation', 'hdi_3%'],
                'hdi_upper': summary.loc['beta_deescalation', 'hdi_97%']
            },
            'local_effect': {
                'coefficient': summary.loc['alpha_local', 'mean'],
                'hdi_lower': summary.loc['alpha_local', 'hdi_3%'],
                'hdi_upper': summary.loc['alpha_local', 'hdi_97%']
            },
            'global_effect': {
                'coefficient': summary.loc['alpha_global', 'mean'],
                'hdi_lower': summary.loc['alpha_global', 'hdi_3%'],
                'hdi_upper': summary.loc['alpha_global', 'hdi_97%']
            }
        }
        
        return {
            'trace': trace,
            'model': model,
            'coefficients': coefficients
        }
    
    def _classical_hierarchy_analysis(
        self, 
        local_series: np.ndarray, 
        global_series: np.ndarray,
        features: StructuralTensorFeatures
    ) -> Dict[str, float]:
        """古典的階層分析（ベイズ推定フォールバック）"""
        local_events = (local_series > 0).astype(float)
        global_events = (global_series > 0).astype(float)
        
        # 相関ベース係数
        if np.sum(local_events) > 0 and np.sum(global_events) > 0:
            # エスカレーション強度（局所→大域相関）
            escalation_strength = np.corrcoef(
                np.concatenate([[0], local_events[:-1]]),  # 遅延局所
                global_events  # 現在大域
            )[0, 1]
            
            # デエスカレーション強度（大域→局所相関）
            deescalation_strength = np.corrcoef(
                np.concatenate([[0], global_events[:-1]]),  # 遅延大域
                local_events  # 現在局所
            )[0, 1]
        else:
            escalation_strength = 0.0
            deescalation_strength = 0.0
        
        # 効果強度（平均張力）
        local_effect = np.mean(local_series[local_series > 0]) if np.any(local_series > 0) else 0.0
        global_effect = np.mean(global_series[global_series > 0]) if np.any(global_series > 0) else 0.0
        
        return {
            'escalation': escalation_strength,
            'deescalation': deescalation_strength,
            'local_effect': local_effect,
            'global_effect': global_effect,
            'hierarchy_correlation': np.corrcoef(local_series, global_series)[0, 1] if len(local_series) > 1 else 0.0
        }
    
    def _calculate_asymmetry_metrics(self, coefficients: Dict) -> Dict[str, float]:
        """非対称性メトリクス計算"""
        if isinstance(coefficients, dict):
            # ベイズ結果の場合
            if 'escalation' in coefficients and isinstance(coefficients['escalation'], dict):
                escalation_coeff = coefficients['escalation']['coefficient']
                deescalation_coeff = coefficients['deescalation']['coefficient']
            else:
                # 古典的結果の場合
                escalation_coeff = coefficients.get('escalation', 0)
                deescalation_coeff = coefficients.get('deescalation', 0)
        else:
            escalation_coeff = 0
            deescalation_coeff = 0
        
        total_transition = abs(escalation_coeff) + abs(deescalation_coeff)
        
        if total_transition > 1e-8:
            asymmetry_metrics = {
                'transition_asymmetry': escalation_coeff - deescalation_coeff,
                'escalation_dominance': abs(escalation_coeff) / total_transition,
                'deescalation_dominance': abs(deescalation_coeff) / total_transition,
                'directional_bias': (escalation_coeff - deescalation_coeff) / total_transition
            }
        else:
            asymmetry_metrics = {
                'transition_asymmetry': 0.0,
                'escalation_dominance': 0.5,
                'deescalation_dominance': 0.5,
                'directional_bias': 0.0
            }
        
        return asymmetry_metrics
    
    def _evaluate_separation_quality(self, results: HierarchicalSeparationResults) -> Dict[str, float]:
        """分離品質評価"""
        # 階層間相関（低い方が良い分離）
        correlation = abs(results.get_hierarchy_correlation())
        correlation_quality = max(0, 1 - correlation)
        
        # 遷移強度（高い方が良いダイナミクス）
        escalation = results.get_escalation_strength()
        deescalation = results.get_deescalation_strength()
        transition_quality = min(1, (escalation + deescalation) / 2)
        
        # 階層バランス
        dominance_metrics = results.calculate_transition_dominance()
        balance_quality = 1 - abs(dominance_metrics['transition_asymmetry'])
        
        # 総合品質
        overall_quality = (correlation_quality + transition_quality + balance_quality) / 3
        
        return {
            'correlation_quality': correlation_quality,
            'transition_quality': transition_quality,
            'balance_quality': balance_quality,
            'overall_quality': overall_quality
        }
    
    def _calculate_comparative_metrics(self, results_list: List[HierarchicalSeparationResults]) -> Dict[str, Any]:
        """比較メトリクス計算"""
        if len(results_list) == 0:
            return {}
        
        # 各系列の主要メトリクス抽出
        escalation_strengths = [r.get_escalation_strength() for r in results_list]
        deescalation_strengths = [r.get_deescalation_strength() for r in results_list]
        correlations = [abs(r.get_hierarchy_correlation()) for r in results_list]
        qualities = [r.separation_quality.get('overall_quality', 0) for r in results_list]
        
        return {
            'escalation_stats': {
                'mean': float(np.mean(escalation_strengths)),
                'std': float(np.std(escalation_strengths)),
                'max': float(np.max(escalation_strengths)),
                'min': float(np.min(escalation_strengths))
            },
            'deescalation_stats': {
                'mean': float(np.mean(deescalation_strengths)),
                'std': float(np.std(deescalation_strengths)),
                'max': float(np.max(deescalation_strengths)),
                'min': float(np.min(deescalation_strengths))
            },
            'correlation_stats': {
                'mean': float(np.mean(correlations)),
                'std': float(np.std(correlations)),
                'max': float(np.max(correlations)),
                'min': float(np.min(correlations))
            },
            'quality_stats': {
                'mean': float(np.mean(qualities)),
                'std': float(np.std(qualities)),
                'max': float(np.max(qualities)),
                'min': float(np.min(qualities))
            }
        }
    
    def _cluster_hierarchical_systems(self, results_list: List[HierarchicalSeparationResults]) -> Dict[str, Any]:
        """階層システムクラスタリング"""
        if len(results_list) < 2:
            return {'error': 'Insufficient data for clustering'}
        
        # 特徴量行列構築
        features_matrix = []
        for result in results_list:
            feature_vector = [
                result.get_escalation_strength(),
                result.get_deescalation_strength(),
                abs(result.get_hierarchy_correlation()),
                result.separation_quality.get('overall_quality', 0)
            ]
            features_matrix.append(feature_vector)
        
        features_matrix = np.array(features_matrix)
        
        # 簡易k-meansクラスタリング（k=2）
        if len(results_list) >= 2:
            # 正規化
            normalized_features = normalize_array(features_matrix.flatten()).reshape(features_matrix.shape)
            
            # 単純な二分割（中央値ベース）
            escalation_median = np.median(features_matrix[:, 0])
            deescalation_median = np.median(features_matrix[:, 1])
            
            cluster_labels = []
            for i, result in enumerate(results_list):
                escalation = result.get_escalation_strength()
                deescalation = result.get_deescalation_strength()
                
                if escalation > escalation_median and deescalation > deescalation_median:
                    cluster_labels.append('high_activity')
                elif escalation < escalation_median and deescalation < deescalation_median:
                    cluster_labels.append('low_activity')
                elif escalation > deescalation:
                    cluster_labels.append('escalation_dominant')
                else:
                    cluster_labels.append('deescalation_dominant')
            
            return {
                'cluster_labels': cluster_labels,
                'cluster_centers': {
                    'escalation_median': escalation_median,
                    'deescalation_median': deescalation_median
                },
                'cluster_distribution': {label: cluster_labels.count(label) for label in set(cluster_labels)}
            }
        
        return {'error': 'Clustering failed'}
    
    def _rank_hierarchical_systems(self, results_list: List[HierarchicalSeparationResults]) -> Dict[str, Any]:
        """階層システムランキング"""
        if len(results_list) == 0:
            return {}
        
        # 各指標でのランキング
        rankings = {}
        
        # 分離品質ランキング
        quality_scores = [(i, r.separation_quality.get('overall_quality', 0)) for i, r in enumerate(results_list)]
        quality_ranking = sorted(quality_scores, key=lambda x: x[1], reverse=True)
        rankings['quality_ranking'] = [(results_list[i].series_name, score) for i, score in quality_ranking]
        
        # エスカレーション強度ランキング
        escalation_scores = [(i, r.get_escalation_strength()) for i, r in enumerate(results_list)]
        escalation_ranking = sorted(escalation_scores, key=lambda x: x[1], reverse=True)
        rankings['escalation_ranking'] = [(results_list[i].series_name, score) for i, score in escalation_ranking]
        
        # デエスカレーション強度ランキング
        deescalation_scores = [(i, r.get_deescalation_strength()) for i, r in enumerate(results_list)]
        deescalation_ranking = sorted(deescalation_scores, key=lambda x: x[1], reverse=True)
        rankings['deescalation_ranking'] = [(results_list[i].series_name, score) for i, score in deescalation_ranking]
        
        # 階層バランスランキング（非対称性の低さ）
        balance_scores = [(i, 1 - abs(r.asymmetry_metrics.get('transition_asymmetry', 0))) for i, r in enumerate(results_list)]
        balance_ranking = sorted(balance_scores, key=lambda x: x[1], reverse=True)
        rankings['balance_ranking'] = [(results_list[i].series_name, score) for i, score in balance_ranking]
        
        return rankings
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """解析履歴サマリー"""
        if not self.analysis_history:
            return {"message": "No hierarchical analyses performed yet"}
        
        total_analyses = len(self.analysis_history)
        bayesian_analyses = sum(1 for h in self.analysis_history if h['method'] == 'bayesian')
        
        # 平均品質
        avg_quality = np.mean([h['separation_quality'] for h in self.analysis_history])
        
        # 遷移強度統計
        escalation_strengths = [h['escalation_strength'] for h in self.analysis_history]
        deescalation_strengths = [h['deescalation_strength'] for h in self.analysis_history]
        
        return {
            'total_analyses': total_analyses,
            'bayesian_ratio': bayesian_analyses / total_analyses,
            'average_separation_quality': avg_quality,
            'escalation_strength_stats': {
                'mean': float(np.mean(escalation_strengths)),
                'std': float(np.std(escalation_strengths)),
                'max': float(np.max(escalation_strengths))
            },
            'deescalation_strength_stats': {
                'mean': float(np.mean(deescalation_strengths)),
                'std': float(np.std(deescalation_strengths)),
                'max': float(np.max(deescalation_strengths))
            },
            'recent_analyses': self.analysis_history[-3:]
        }

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def analyze_hierarchical_structure(
    features: StructuralTensorFeatures,
    config: Optional[L3HierarchicalConfig] = None,
    use_bayesian: bool = True
) -> HierarchicalSeparationResults:
    """
    階層構造解析の便利関数
    
    Args:
        features: 構造テンソル特徴量
        config: 階層解析設定
        use_bayesian: ベイズ推定使用フラグ
        
    Returns:
        HierarchicalSeparationResults: 階層分離解析結果
    """
    analyzer = HierarchicalAnalyzer(config)
    return analyzer.analyze_hierarchical_separation(features, use_bayesian)

def compare_multiple_hierarchies(
    features_list: List[StructuralTensorFeatures],
    config: Optional[L3HierarchicalConfig] = None,
    use_bayesian: bool = True
) -> Dict[str, Any]:
    """
    複数階層システム比較の便利関数
    
    Args:
        features_list: 構造テンソル特徴量リスト
        config: 階層解析設定
        use_bayesian: ベイズ推定使用フラグ
        
    Returns:
        Dict: 階層システム比較結果
    """
    analyzer = HierarchicalAnalyzer(config)
    return analyzer.compare_hierarchical_systems(features_list, use_bayesian)

if __name__ == "__main__":
    print("Lambda³ Hierarchical Analysis Module Test")
    print("=" * 50)
    
    # テスト用構造テンソル特徴量生成は省略
    # 実際のテストは統合テストで実行
    print("Hierarchical analysis module loaded successfully!")
    print("Ready for Lambda³ hierarchical structure analysis.")
