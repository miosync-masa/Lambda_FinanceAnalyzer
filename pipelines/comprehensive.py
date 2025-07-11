# ==========================================================
# lambda3/pipelines/comprehensive.py
# Comprehensive Analysis Pipeline for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論包括分析パイプライン

構造テンソル(Λ)解析の全工程を統合実行する
エンドツーエンドパイプライン。データ読み込みから
高度な可視化まで、理論的一貫性を保持した
完全自動化分析システム。

主要機能:
- マルチモーダル特徴抽出
- 階層的構造変化分析
- ペアワイズ非対称相互作用
- ベイズ推定による定量化
- 統合可視化ダッシュボード
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
from dataclasses import dataclass, field
import time
import json

# Lambda³ core modules
from ..core.config import (
    L3ComprehensiveConfig, L3BaseConfig, L3BayesianConfig,
    L3HierarchicalConfig, L3PairwiseConfig, L3VisualizationConfig
)
from ..core.structural_tensor import (
    StructuralTensorExtractor, StructuralTensorAnalyzer,
    StructuralTensorFeatures, extract_features_batch
)

# Lambda³ analysis modules
from ..analysis.hierarchical import (
    HierarchicalAnalyzer, HierarchicalSeparationResults
)
from ..analysis.pairwise import (
    PairwiseAnalyzer, PairwiseInteractionResults
)

# Lambda³ visualization modules
from ..visualization.base import (
    Lambda3BaseVisualizer, TimeSeriesVisualizer, 
    InteractionVisualizer, HierarchicalVisualizer
)

# External dependencies
try:
    import yfinance as yf
    FINANCIAL_DATA_AVAILABLE = True
except ImportError:
    FINANCIAL_DATA_AVAILABLE = False
    warnings.warn("yfinance not available. Financial data loading disabled.")

# ==========================================================
# ANALYSIS RESULTS CONTAINER
# ==========================================================

@dataclass
class Lambda3ComprehensiveResults:
    """
    Lambda³包括分析結果コンテナ
    
    全分析モジュールの結果を統合管理し、
    結果間の相互参照と品質評価を提供。
    """
    
    # メタデータ
    analysis_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    series_names: List[str] = field(default_factory=list)
    config: Optional[L3ComprehensiveConfig] = None
    
    # 原データ
    series_dict: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 特徴量
    features_dict: Dict[str, StructuralTensorFeatures] = field(default_factory=dict)
    
    # 構造解析結果
    structural_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # 階層解析結果
    hierarchical_results: Dict[str, HierarchicalSeparationResults] = field(default_factory=dict)
    
    # ペアワイズ解析結果
    pairwise_results: Dict[str, Any] = field(default_factory=dict)
    
    # レジーム・危機検出結果
    regime_results: Dict[str, Any] = field(default_factory=dict)
    crisis_results: Dict[str, Any] = field(default_factory=dict)
    
    # 同期・因果関係結果
    synchronization_results: Dict[str, Any] = field(default_factory=dict)
    causality_results: Dict[str, Any] = field(default_factory=dict)
    
    # 可視化結果
    visualization_results: Dict[str, Any] = field(default_factory=dict)
    
    # 品質・性能メトリクス
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """結果サマリー生成"""
        return {
            'analysis_info': {
                'timestamp': self.analysis_timestamp,
                'n_series': len(self.series_names),
                'series_names': self.series_names,
                'total_data_points': sum(len(data) for data in self.series_dict.values())
            },
            'feature_extraction': {
                'extracted_series': len(self.features_dict),
                'feature_types': list(self.features_dict.keys()) if self.features_dict else []
            },
            'analysis_modules': {
                'structural_analysis': bool(self.structural_analysis),
                'hierarchical_analysis': bool(self.hierarchical_results),
                'pairwise_analysis': bool(self.pairwise_results),
                'regime_analysis': bool(self.regime_results),
                'crisis_detection': bool(self.crisis_results)
            },
            'quality_metrics': self.quality_metrics,
            'performance_metrics': self.performance_metrics
        }
    
    def save_results(self, filepath: Union[str, Path]):
        """結果をファイルに保存"""
        filepath = Path(filepath)
        
        # JSONシリアライズ可能な形式に変換
        serializable_results = {
            'analysis_timestamp': self.analysis_timestamp,
            'series_names': self.series_names,
            'config': self.config.to_dict() if self.config else {},
            'quality_metrics': self.quality_metrics,
            'performance_metrics': self.performance_metrics,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"分析結果を保存しました: {filepath}")

# ==========================================================
# COMPREHENSIVE PIPELINE CLASS
# ==========================================================

class Lambda3ComprehensivePipeline:
    """
    Lambda³包括分析パイプライン
    
    構造テンソル解析の全工程を統合実行する
    メインパイプラインクラス。理論的厳密性と
    実用性のバランスを最適化。
    """
    
    def __init__(self, config: Optional[L3ComprehensiveConfig] = None):
        """
        初期化
        
        Args:
            config: 包括設定オブジェクト
        """
        self.config = config or L3ComprehensiveConfig()
        
        # 分析器の初期化
        self.structural_extractor = StructuralTensorExtractor(self.config.base)
        self.structural_analyzer = StructuralTensorAnalyzer(self.config.base)
        self.hierarchical_analyzer = HierarchicalAnalyzer(self.config.hierarchical, self.config.bayesian)
        self.pairwise_analyzer = PairwiseAnalyzer(self.config.pairwise, self.config.bayesian)
        
        # 可視化器の初期化
        self.time_series_viz = TimeSeriesVisualizer(self.config.visualization)
        self.interaction_viz = InteractionVisualizer(self.config.visualization)
        self.hierarchical_viz = HierarchicalVisualizer(self.config.visualization)
        
        # 実行履歴
        self.execution_history = []
    
    def run_comprehensive_analysis(
        self,
        data_source: Union[Dict[str, np.ndarray], str, Path],
        analysis_modes: Optional[Dict[str, bool]] = None,
        enable_visualization: bool = True
    ) -> Lambda3ComprehensiveResults:
        """
        包括分析実行
        
        Lambda³理論の全分析モジュールを統合実行し、
        構造テンソル空間の完全な解析を提供。
        
        Args:
            data_source: データソース（辞書、CSVパス、など）
            analysis_modes: 分析モード設定
            enable_visualization: 可視化有効フラグ
            
        Returns:
            Lambda3ComprehensiveResults: 包括分析結果
        """
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print("LAMBDA³ COMPREHENSIVE ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 分析モード設定
        if analysis_modes is None:
            analysis_modes = self.config.analysis_modes
        
        # 結果コンテナ初期化
        results = Lambda3ComprehensiveResults(config=self.config)
        
        try:
            # Phase 1: データ読み込みと前処理
            print(f"\n{'='*60}")
            print("PHASE 1: データ読み込みと前処理")
            print(f"{'='*60}")
            
            series_dict = self._load_and_preprocess_data(data_source)
            results.series_dict = series_dict
            results.series_names = list(series_dict.keys())
            
            print(f"データ読み込み完了: {len(series_dict)} 系列")
            for name, data in series_dict.items():
                print(f"  {name}: {len(data)} points")
            
            # Phase 2: 構造テンソル特徴抽出
            print(f"\n{'='*60}")
            print("PHASE 2: 構造テンソル特徴抽出")
            print(f"{'='*60}")
            
            features_dict = self._extract_comprehensive_features(series_dict)
            results.features_dict = features_dict
            
            # Phase 3: 構造パターン解析
            print(f"\n{'='*60}")
            print("PHASE 3: 構造パターン解析")
            print(f"{'='*60}")
            
            structural_analysis = self._analyze_structural_patterns(features_dict)
            results.structural_analysis = structural_analysis
            
            # Phase 4: 階層構造解析
            if analysis_modes.get('hierarchical_analysis', True):
                print(f"\n{'='*60}")
                print("PHASE 4: 階層構造解析")
                print(f"{'='*60}")
                
                hierarchical_results = self._analyze_hierarchical_structures(features_dict)
                results.hierarchical_results = hierarchical_results
            
            # Phase 5: ペアワイズ相互作用解析
            if analysis_modes.get('pairwise_analysis', True) and len(series_dict) >= 2:
                print(f"\n{'='*60}")
                print("PHASE 5: ペアワイズ相互作用解析")
                print(f"{'='*60}")
                
                pairwise_results = self._analyze_pairwise_interactions(features_dict)
                results.pairwise_results = pairwise_results
            
            # Phase 6: 同期・因果関係解析
            if analysis_modes.get('synchronization_analysis', True) and len(series_dict) >= 2:
                print(f"\n{'='*60}")
                print("PHASE 6: 同期・因果関係解析")
                print(f"{'='*60}")
                
                sync_results, causality_results = self._analyze_synchronization_causality(features_dict)
                results.synchronization_results = sync_results
                results.causality_results = causality_results
            
            # Phase 7: レジーム・危機検出
            if analysis_modes.get('regime_analysis', False):
                print(f"\n{'='*60}")
                print("PHASE 7: レジーム・危機検出")
                print(f"{'='*60}")
                
                regime_results, crisis_results = self._detect_regimes_and_crises(features_dict, series_dict)
                results.regime_results = regime_results
                results.crisis_results = crisis_results
            
            # Phase 8: 可視化生成
            if enable_visualization:
                print(f"\n{'='*60}")
                print("PHASE 8: 統合可視化生成")
                print(f"{'='*60}")
                
                visualization_results = self._generate_comprehensive_visualizations(results)
                results.visualization_results = visualization_results
            
            # Phase 9: 品質評価と性能測定
            print(f"\n{'='*60}")
            print("PHASE 9: 品質評価と性能測定")
            print(f"{'='*60}")
            
            quality_metrics = self._evaluate_analysis_quality(results)
            results.quality_metrics = quality_metrics
            
            # 性能メトリクス
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_metrics = {
                'total_execution_time': execution_time,
                'data_processing_rate': sum(len(data) for data in series_dict.values()) / execution_time,
                'features_per_second': len(features_dict) / execution_time,
                'memory_efficiency': self._estimate_memory_usage(results)
            }
            results.performance_metrics = performance_metrics
            
            print(f"分析完了: {execution_time:.2f}秒")
            print(f"データ処理レート: {performance_metrics['data_processing_rate']:.0f} points/sec")
            
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            results.quality_metrics['analysis_error'] = str(e)
            raise
        
        # 実行履歴記録
        self.execution_history.append({
            'timestamp': results.analysis_timestamp,
            'n_series': len(results.series_names),
            'execution_time': results.performance_metrics.get('total_execution_time', 0),
            'analysis_modes': analysis_modes,
            'success': True
        })
        
        print(f"\n{'='*80}")
        print("LAMBDA³ 包括分析完了")
        print(f"{'='*80}")
        
        return results
    
    def _load_and_preprocess_data(self, data_source: Union[Dict, str, Path]) -> Dict[str, np.ndarray]:
        """データ読み込みと前処理"""
        if isinstance(data_source, dict):
            # 辞書形式の場合はそのまま使用
            series_dict = {name: np.array(data, dtype=np.float64) 
                          for name, data in data_source.items()}
        
        elif isinstance(data_source, (str, Path)):
            # ファイルパスの場合
            file_path = Path(data_source)
            
            if file_path.suffix.lower() == '.csv':
                # CSV読み込み
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                series_dict = {col: df[col].values.astype(np.float64) 
                              for col in df.columns}
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        else:
            raise ValueError("Invalid data_source type")
        
        # データ前処理
        processed_dict = {}
        for name, data in series_dict.items():
            # 欠損値処理
            if np.isnan(data).any():
                print(f"警告: {name} に欠損値があります。前方埋めを適用します。")
                mask = np.isnan(data)
                data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            
            # 無限値処理
            if np.isinf(data).any():
                print(f"警告: {name} に無限値があります。クリッピングを適用します。")
                data = np.clip(data, -1e10, 1e10)
            
            processed_dict[name] = data
        
        return processed_dict
    
    def _extract_comprehensive_features(self, series_dict: Dict[str, np.ndarray]) -> Dict[str, StructuralTensorFeatures]:
        """包括的特徴抽出"""
        print("構造テンソル特徴抽出を実行中...")
        
        features_dict = {}
        
        for series_name, data in series_dict.items():
            print(f"  {series_name} の特徴抽出中...")
            
            try:
                # 包括的特徴抽出（階層・マルチスケール含む）
                features = self.structural_extractor.extract_comprehensive_features(
                    data, 
                    series_name,
                    include_multi_scale=True,
                    scales=[5, 10, 20, 50]
                )
                
                features_dict[series_name] = features
                
                # 特徴量統計表示
                event_counts = features.count_events_by_type()
                print(f"    イベント数: {event_counts}")
                
            except Exception as e:
                print(f"    エラー: {series_name} の特徴抽出に失敗: {e}")
                continue
        
        print(f"特徴抽出完了: {len(features_dict)} 系列")
        return features_dict
    
    def _analyze_structural_patterns(self, features_dict: Dict[str, StructuralTensorFeatures]) -> Dict[str, Any]:
        """構造パターン解析"""
        print("構造パターン解析を実行中...")
        
        structural_analysis = {}
        
        for series_name, features in features_dict.items():
            print(f"  {series_name} の構造解析中...")
            
            try:
                # 基本構造パターン解析
                pattern_analysis = self.structural_analyzer.analyze_structural_patterns(features)
                
                # 階層性メトリクス
                hierarchy_metrics = self.structural_analyzer.calculate_hierarchical_metrics(features)
                
                # 構造異常検出
                anomaly_detection = self.structural_analyzer.detect_structural_anomalies(features)
                
                structural_analysis[series_name] = {
                    'pattern_analysis': pattern_analysis,
                    'hierarchy_metrics': hierarchy_metrics,
                    'anomaly_detection': anomaly_detection
                }
                
                print(f"    構造強度: {pattern_analysis.get('structural_intensity', {}).get('total_intensity', 0):.3f}")
                print(f"    異常検出: {anomaly_detection.get('total_anomalies', 0)} 件")
                
            except Exception as e:
                print(f"    エラー: {series_name} の構造解析に失敗: {e}")
                continue
        
        return structural_analysis
    
    def _analyze_hierarchical_structures(self, features_dict: Dict[str, StructuralTensorFeatures]) -> Dict[str, HierarchicalSeparationResults]:
        """階層構造解析"""
        print("階層構造解析を実行中...")
        
        hierarchical_results = {}
        
        for series_name, features in features_dict.items():
            print(f"  {series_name} の階層解析中...")
            
            try:
                # 階層分離分析
                hierarchy_result = self.hierarchical_analyzer.analyze_hierarchical_separation(
                    features, 
                    use_bayesian=self.config.analysis_modes.get('use_bayesian', True)
                )
                
                hierarchical_results[series_name] = hierarchy_result
                
                print(f"    エスカレーション強度: {hierarchy_result.get_escalation_strength():.4f}")
                print(f"    デエスカレーション強度: {hierarchy_result.get_deescalation_strength():.4f}")
                print(f"    分離品質: {hierarchy_result.separation_quality.get('overall_quality', 0):.3f}")
                
            except Exception as e:
                print(f"    エラー: {series_name} の階層解析に失敗: {e}")
                continue
        
        return hierarchical_results
    
    def _analyze_pairwise_interactions(self, features_dict: Dict[str, StructuralTensorFeatures]) -> Dict[str, Any]:
        """ペアワイズ相互作用解析"""
        print("ペアワイズ相互作用解析を実行中...")
        
        pairwise_results = {}
        
        # 複数ペア比較
        if len(features_dict) >= 2:
            try:
                multi_pair_results = self.pairwise_analyzer.compare_multiple_pairs(
                    features_dict,
                    use_bayesian=self.config.analysis_modes.get('use_bayesian', True)
                )
                pairwise_results['multi_pair_comparison'] = multi_pair_results
                
                print(f"    ペア解析数: {multi_pair_results['summary']['total_pairs_analyzed']}")
                print(f"    平均相互作用強度: {multi_pair_results['summary']['mean_interaction_strength']:.4f}")
                print(f"    平均非対称性: {multi_pair_results['summary']['mean_asymmetry']:.4f}")
                
            except Exception as e:
                print(f"    エラー: 複数ペア解析に失敗: {e}")
        
        # 主要ペアの詳細解析
        series_names = list(features_dict.keys())
        if len(series_names) >= 2:
            primary_pair = series_names[:2]
            
            try:
                print(f"  主要ペア詳細解析: {primary_pair[0]} ⇄ {primary_pair[1]}")
                
                primary_result = self.pairwise_analyzer.analyze_asymmetric_interaction(
                    features_dict[primary_pair[0]],
                    features_dict[primary_pair[1]],
                    use_bayesian=self.config.analysis_modes.get('use_bayesian', True)
                )
                
                pairwise_results['primary_pair'] = primary_result
                
                print(f"    結合強度: {primary_result.calculate_bidirectional_coupling():.4f}")
                print(f"    非対称性: {primary_result.get_asymmetry_score():.4f}")
                print(f"    品質: {primary_result.interaction_quality.get('overall_quality', 0):.3f}")
                
            except Exception as e:
                print(f"    エラー: 主要ペア解析に失敗: {e}")
        
        return pairwise_results
    
    def _analyze_synchronization_causality(self, features_dict: Dict[str, StructuralTensorFeatures]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """同期・因果関係解析"""
        print("同期・因果関係解析を実行中...")
        
        sync_results = {}
        causality_results = {}
        
        series_names = list(features_dict.keys())
        
        if len(series_names) >= 2:
            # 同期行列計算
            print("  同期行列計算中...")
            
            # 基本的な同期行列（簡易実装）
            n_series = len(series_names)
            sync_matrix = np.eye(n_series)
            
            for i, name_a in enumerate(series_names):
                for j, name_b in enumerate(series_names):
                    if i != j:
                        # 張力スカラー相関を同期率として使用
                        corr = np.corrcoef(
                            features_dict[name_a].rho_T,
                            features_dict[name_b].rho_T
                        )[0, 1]
                        sync_matrix[i, j] = abs(corr)
            
            sync_results = {
                'sync_matrix': sync_matrix,
                'series_names': series_names,
                'mean_sync': float(np.mean(sync_matrix[sync_matrix < 1])),
                'max_sync': float(np.max(sync_matrix[sync_matrix < 1]))
            }
            
            print(f"    平均同期率: {sync_results['mean_sync']:.4f}")
            print(f"    最大同期率: {sync_results['max_sync']:.4f}")
            
            # 基本因果関係検出
            print("  因果関係検出中...")
            
            # 主要ペアの因果関係
            if len(series_names) >= 2:
                features_a = features_dict[series_names[0]]
                features_b = features_dict[series_names[1]]
                
                # 簡易因果関係（遅延相関）
                causality_patterns = {}
                
                for lag in range(1, 6):  # 1-5遅延
                    if lag < len(features_a.rho_T):
                        # A → B 因果関係
                        cause_series = features_a.rho_T[:-lag]
                        effect_series = features_b.rho_T[lag:]
                        corr_ab = np.corrcoef(cause_series, effect_series)[0, 1]
                        
                        # B → A 因果関係
                        cause_series = features_b.rho_T[:-lag]
                        effect_series = features_a.rho_T[lag:]
                        corr_ba = np.corrcoef(cause_series, effect_series)[0, 1]
                        
                        causality_patterns[f'{series_names[0]}_to_{series_names[1]}'] = {lag: abs(corr_ab)}
                        causality_patterns[f'{series_names[1]}_to_{series_names[0]}'] = {lag: abs(corr_ba)}
                
                causality_results = {
                    'basic_causality': causality_patterns,
                    'series_pair': series_names[:2]
                }
                
                # 最強因果関係
                max_causality = 0
                strongest_direction = ""
                for direction, lags in causality_patterns.items():
                    for lag, strength in lags.items():
                        if strength > max_causality:
                            max_causality = strength
                            strongest_direction = f"{direction} (lag={lag})"
                
                causality_results['strongest_causality'] = {
                    'direction': strongest_direction,
                    'strength': max_causality
                }
                
                print(f"    最強因果関係: {strongest_direction}")
                print(f"    因果強度: {max_causality:.4f}")
        
        return sync_results, causality_results
    
    def _detect_regimes_and_crises(self, features_dict: Dict[str, StructuralTensorFeatures], series_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """レジーム・危機検出"""
        print("レジーム・危機検出を実行中...")
        
        regime_results = {}
        crisis_results = {}
        
        # 簡易レジーム検出
        print("  レジーム検出中...")
        
        regime_labels = {}
        for series_name, features in features_dict.items():
            # 張力スカラーベースのレジーム分類
            rho_t = features.rho_T
            
            # 3段階レジーム（低・中・高張力）
            low_threshold = np.percentile(rho_t, 33)
            high_threshold = np.percentile(rho_t, 67)
            
            regime_series = np.zeros(len(rho_t), dtype=int)
            regime_series[rho_t > high_threshold] = 2  # 高張力レジーム
            regime_series[(rho_t > low_threshold) & (rho_t <= high_threshold)] = 1  # 中張力レジーム
            # 低張力レジームは0のまま
            
            regime_labels[series_name] = regime_series
        
        regime_results = {
            'regime_labels': regime_labels,
            'regime_definitions': {
                0: '低張力レジーム',
                1: '中張力レジーム', 
                2: '高張力レジーム'
            }
        }
        
        # 簡易危機検出
        print("  危機検出中...")
        
        crisis_indicators = {}
        for series_name, features in features_dict.items():
            # 張力スカラーと構造変化の組み合わせで危機スコア計算
            rho_t = features.rho_T
            structure_changes = features.delta_LambdaC_pos + features.delta_LambdaC_neg
            
            # 正規化
            rho_norm = (rho_t - np.mean(rho_t)) / (np.std(rho_t) + 1e-8)
            changes_norm = (structure_changes - np.mean(structure_changes)) / (np.std(structure_changes) + 1e-8)
            
            # 危機スコア = 高張力 + 高構造変化
            crisis_score = (rho_norm + changes_norm) / 2
            crisis_indicators[series_name] = crisis_score
        
        # 統合危機スコア
        if crisis_indicators:
            all_scores = np.array(list(crisis_indicators.values()))
            aggregate_crisis = np.mean(all_scores, axis=0)
            
            # 危機期間検出（閾値超過）
            crisis_threshold = 1.5  # 1.5σ以上
            crisis_periods = aggregate_crisis > crisis_threshold
            
            crisis_results = {
                'crisis_indicators': crisis_indicators,
                'aggregate_crisis': aggregate_crisis,
                'crisis_threshold': crisis_threshold,
                'crisis_periods': crisis_periods,
                'crisis_episodes': self._find_crisis_episodes(crisis_periods)
            }
            
            n_crisis_episodes = len(crisis_results['crisis_episodes'])
            print(f"    検出された危機エピソード: {n_crisis_episodes} 件")
        
        return regime_results, crisis_results
    
    def _find_crisis_episodes(self, crisis_periods: np.ndarray) -> List[Tuple[int, int]]:
        """危機エピソード検出"""
        episodes = []
        
        # 期間の開始・終了点を検出
        diff_periods = np.diff(crisis_periods.astype(int))
        starts = np.where(diff_periods == 1)[0] + 1
        ends = np.where(diff_periods == -1)[0] + 1
        
        # 最初が危機状態の場合
        if crisis_periods[0]:
            starts = np.concatenate([[0], starts])
        
        # 最後が危機状態の場合
        if crisis_periods[-1]:
            ends = np.concatenate([ends, [len(crisis_periods) - 1]])
        
        # エピソードペア生成
        for start, end in zip(starts, ends):
            episodes.append((int(start), int(end)))
        
        return episodes
    
    def _generate_comprehensive_visualizations(self, results: Lambda3ComprehensiveResults) -> Dict[str, Any]:
        """統合可視化生成"""
        print("統合可視化を生成中...")
        
        visualization_results = {}
        
        try:
            # 基本時系列可視化
            print("  基本時系列プロット生成中...")
            
            if len(results.series_names) <= 4:  # 可視化可能な系列数制限
                series_for_plot = {name: (np.arange(len(data)), data) 
                                  for name, data in results.series_dict.items()}
                
                fig_multi = self.time_series_viz.create_multi_series_plot(
                    series_for_plot, 
                    "Lambda³ 構造テンソル系列"
                )
                
                # 保存
                if self.config.visualization.save_plots:
                    saved_path = self.time_series_viz._save_figure(fig_multi, "lambda3_multi_series")
                    visualization_results['multi_series_plot'] = str(saved_path) if saved_path else None
            
            # 階層構造可視化
            if results.hierarchical_results:
                print("  階層構造プロット生成中...")
                
                for series_name, hierarchy_result in list(results.hierarchical_results.items())[:2]:  # 最大2系列
                    fig_hierarchy = self.hierarchical_viz.create_hierarchy_separation_plot(
                        np.arange(len(hierarchy_result.local_series)),
                        hierarchy_result.local_series,
                        hierarchy_result.global_series,
                        f"階層分離: {series_name}"
                    )
                    
                    if self.config.visualization.save_plots:
                        saved_path = self.hierarchical_viz._save_figure(fig_hierarchy, f"hierarchy_{series_name}")
                        visualization_results[f'hierarchy_plot_{series_name}'] = str(saved_path) if saved_path else None
            
            # 相互作用行列可視化
            if results.pairwise_results and 'multi_pair_comparison' in results.pairwise_results:
                print("  相互作用行列プロット生成中...")
                
                multi_pair = results.pairwise_results['multi_pair_comparison']
                if 'interaction_matrix' in multi_pair:
                    
                    fig_interaction = self.interaction_viz.create_interaction_matrix_plot(
                        multi_pair['interaction_matrix'],
                        multi_pair['series_names'],
                        "Lambda³ 構造テンソル相互作用行列"
                    )
                    
                    if self.config.visualization.save_plots:
                        saved_path = self.interaction_viz._save_figure(fig_interaction, "interaction_matrix")
                        visualization_results['interaction_matrix_plot'] = str(saved_path) if saved_path else None
            
            visualization_results['generation_success'] = True
            print("    可視化生成完了")
            
        except Exception as e:
            print(f"    可視化生成エラー: {e}")
            visualization_results['generation_error'] = str(e)
            visualization_results['generation_success'] = False
        
        return visualization_results
    
    def _evaluate_analysis_quality(self, results: Lambda3ComprehensiveResults) -> Dict[str, float]:
        """分析品質評価"""
        print("分析品質を評価中...")
        
        quality_metrics = {}
        
    def _evaluate_analysis_quality(self, results: Lambda3ComprehensiveResults) -> Dict[str, float]:
        """分析品質評価"""
        print("分析品質を評価中...")
        
        quality_metrics = {}
        
        # データ品質
        total_data_points = sum(len(data) for data in results.series_dict.values())
        valid_series = len(results.features_dict)
        data_quality = valid_series / max(len(results.series_dict), 1)
        quality_metrics['data_quality'] = data_quality
        
        # 特徴抽出品質
        if results.features_dict:
            total_events = 0
            for features in results.features_dict.values():
                event_counts = features.count_events_by_type()
                total_events += sum(event_counts.values())
            
            feature_density = total_events / total_data_points if total_data_points > 0 else 0
            quality_metrics['feature_density'] = feature_density
            quality_metrics['feature_extraction_success_rate'] = 1.0  # 成功した場合
        else:
            quality_metrics['feature_density'] = 0.0
            quality_metrics['feature_extraction_success_rate'] = 0.0
        
        # 階層分析品質
        if results.hierarchical_results:
            hierarchy_qualities = []
            for hierarchy_result in results.hierarchical_results.values():
                overall_quality = hierarchy_result.separation_quality.get('overall_quality', 0)
                hierarchy_qualities.append(overall_quality)
            
            quality_metrics['mean_hierarchy_quality'] = float(np.mean(hierarchy_qualities))
            quality_metrics['hierarchy_analysis_coverage'] = len(hierarchy_qualities) / valid_series
        else:
            quality_metrics['mean_hierarchy_quality'] = 0.0
            quality_metrics['hierarchy_analysis_coverage'] = 0.0
        
        # ペアワイズ分析品質
        if results.pairwise_results:
            if 'primary_pair' in results.pairwise_results:
                primary_quality = results.pairwise_results['primary_pair'].interaction_quality.get('overall_quality', 0)
                quality_metrics['pairwise_interaction_quality'] = primary_quality
            
            if 'multi_pair_comparison' in results.pairwise_results:
                multi_pair = results.pairwise_results['multi_pair_comparison']
                total_possible_pairs = len(results.series_names) * (len(results.series_names) - 1)
                actual_pairs = multi_pair['summary']['total_pairs_analyzed']
                quality_metrics['pairwise_coverage'] = actual_pairs / max(total_possible_pairs, 1)
            else:
                quality_metrics['pairwise_coverage'] = 0.0
        else:
            quality_metrics['pairwise_interaction_quality'] = 0.0
            quality_metrics['pairwise_coverage'] = 0.0
        
        # 総合品質スコア
        core_metrics = [
            quality_metrics['data_quality'],
            quality_metrics['feature_extraction_success_rate'],
            quality_metrics['mean_hierarchy_quality'],
            quality_metrics['pairwise_interaction_quality']
        ]
        
        quality_metrics['overall_analysis_quality'] = float(np.mean(core_metrics))
        
        print(f"    総合分析品質: {quality_metrics['overall_analysis_quality']:.3f}")
        print(f"    データ品質: {quality_metrics['data_quality']:.3f}")
        print(f"    特徴密度: {quality_metrics['feature_density']:.4f}")
        
        return quality_metrics
    
    def _estimate_memory_usage(self, results: Lambda3ComprehensiveResults) -> float:
        """メモリ使用量推定"""
        total_arrays = 0
        
        # 原データ
        for data in results.series_dict.values():
            total_arrays += data.nbytes
        
        # 特徴量データ
        for features in results.features_dict.values():
            for attr_name in ['data', 'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend']:
                attr_value = getattr(features, attr_name)
                if attr_value is not None and hasattr(attr_value, 'nbytes'):
                    total_arrays += attr_value.nbytes
        
        # MB単位で返す
        return total_arrays / (1024 * 1024)
    
    def run_financial_data_analysis(
        self,
        tickers: Optional[Dict[str, str]] = None,
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-31",
        enable_crisis_detection: bool = True
    ) -> Lambda3ComprehensiveResults:
        """
        金融データの包括分析実行
        
        Lambda³理論による金融市場分析の特化実行。
        自動データ取得、特化設定、金融特有の解析を提供。
        
        Args:
            tickers: {表示名: ティッカー} 辞書
            start_date, end_date: 分析期間
            enable_crisis_detection: 危機検出有効フラグ
            
        Returns:
            Lambda3ComprehensiveResults: 金融分析結果
        """
        if not FINANCIAL_DATA_AVAILABLE:
            raise ImportError("yfinance not available. Cannot perform financial data analysis.")
        
        print(f"\n{'='*80}")
        print("LAMBDA³ FINANCIAL DATA ANALYSIS")
        print(f"{'='*80}")
        
        # デフォルトティッカー
        if tickers is None:
            tickers = {
                "USD/JPY": "JPY=X",
                "OIL": "CL=F",
                "GOLD": "GC=F", 
                "Nikkei": "^N225",
                "S&P500": "^GSPC"
            }
        
        # 金融データ取得
        print(f"金融データ取得中: {start_date} - {end_date}")
        financial_data = {}
        
        for display_name, ticker in tickers.items():
            try:
                print(f"  {display_name} ({ticker}) 取得中...")
                ticker_data = yf.download(ticker, start=start_date, end=end_date)['Close']
                if len(ticker_data) > 50:  # 最小データ長チェック
                    financial_data[display_name] = ticker_data.values
                    print(f"    成功: {len(ticker_data)} データポイント")
                else:
                    print(f"    スキップ: データ不足 ({len(ticker_data)} points)")
            except Exception as e:
                print(f"    エラー: {e}")
                continue
        
        if not financial_data:
            raise ValueError("金融データの取得に失敗しました")
        
        # 金融特化設定
        financial_config = self.config
        financial_config.analysis_modes.update({
            'regime_analysis': True,
            'crisis_detection': enable_crisis_detection,
            'volatility_analysis': True
        })
        
        # 金融特化パラメータ調整
        financial_config.base.delta_percentile = 95.0  # より敏感な検出
        financial_config.hierarchical.escalation_threshold = 0.5
        
        # 包括分析実行
        results = self.run_comprehensive_analysis(
            financial_data,
            analysis_modes=financial_config.analysis_modes,
            enable_visualization=True
        )
        
        # 金融特有の追加解析
        print(f"\n{'='*60}")
        print("金融特有解析の実行")
        print(f"{'='*60}")
        
        # ボラティリティ解析
        volatility_analysis = self._analyze_financial_volatility(financial_data, results.features_dict)
        results.regime_results['volatility_analysis'] = volatility_analysis
        
        # 市場結合度分析
        if len(financial_data) >= 2:
            market_coupling = self._analyze_market_coupling(results.features_dict)
            results.synchronization_results['market_coupling'] = market_coupling
        
        print("金融分析完了")
        return results
    
    def _analyze_financial_volatility(
        self, 
        data_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, StructuralTensorFeatures]
    ) -> Dict[str, Any]:
        """金融ボラティリティ解析"""
        print("  ボラティリティ解析中...")
        
        volatility_analysis = {}
        
        for asset_name, price_data in data_dict.items():
            # リターン計算
            returns = np.diff(price_data) / price_data[:-1]
            
            # 実現ボラティリティ
            realized_vol = np.std(returns) * np.sqrt(252)  # 年率化
            
            # 張力スカラーとボラティリティの関係
            if asset_name in features_dict:
                rho_t = features_dict[asset_name].rho_T[1:]  # リターンと長さ合わせ
                
                # 張力-ボラティリティ相関
                rolling_vol = []
                window = 20  # 20日窓
                
                for i in range(window, len(returns)):
                    window_vol = np.std(returns[i-window:i]) * np.sqrt(252)
                    rolling_vol.append(window_vol)
                
                if len(rolling_vol) > 0:
                    rolling_vol = np.array(rolling_vol)
                    rho_subset = rho_t[window:]
                    
                    if len(rho_subset) == len(rolling_vol):
                        tension_vol_corr = np.corrcoef(rho_subset, rolling_vol)[0, 1]
                    else:
                        tension_vol_corr = 0.0
                else:
                    tension_vol_corr = 0.0
                
                volatility_analysis[asset_name] = {
                    'realized_volatility': float(realized_vol),
                    'mean_return': float(np.mean(returns)),
                    'sharpe_ratio': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0,
                    'tension_volatility_correlation': float(tension_vol_corr),
                    'max_drawdown': float(self._calculate_max_drawdown(price_data))
                }
            
            print(f"    {asset_name}: Vol={realized_vol:.2%}")
        
        return volatility_analysis
    
    def _calculate_max_drawdown(self, price_series: np.ndarray) -> float:
        """最大ドローダウン計算"""
        peak = price_series[0]
        max_dd = 0.0
        
        for price in price_series:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _analyze_market_coupling(self, features_dict: Dict[str, StructuralTensorFeatures]) -> Dict[str, Any]:
        """市場結合度分析"""
        print("  市場結合度解析中...")
        
        coupling_analysis = {}
        
        # 全ペアの張力相関
        series_names = list(features_dict.keys())
        n_series = len(series_names)
        
        coupling_matrix = np.eye(n_series)
        coupling_strengths = []
        
        for i, name_a in enumerate(series_names):
            for j, name_b in enumerate(series_names):
                if i != j:
                    rho_a = features_dict[name_a].rho_T
                    rho_b = features_dict[name_b].rho_T
                    
                    # 時変相関（50日窓）
                    window = min(50, len(rho_a) // 4)
                    
                    correlations = []
                    for k in range(window, len(rho_a)):
                        window_corr = np.corrcoef(
                            rho_a[k-window:k],
                            rho_b[k-window:k]
                        )[0, 1]
                        correlations.append(abs(window_corr))
                    
                    mean_coupling = np.mean(correlations) if correlations else 0.0
                    coupling_matrix[i, j] = mean_coupling
                    coupling_strengths.append(mean_coupling)
        
        # 市場統合度メトリクス
        market_integration = np.mean(coupling_strengths) if coupling_strengths else 0.0
        coupling_volatility = np.std(coupling_strengths) if len(coupling_strengths) > 1 else 0.0
        
        coupling_analysis = {
            'coupling_matrix': coupling_matrix,
            'series_names': series_names,
            'market_integration': float(market_integration),
            'coupling_volatility': float(coupling_volatility),
            'strongest_coupling': float(np.max(coupling_strengths)) if coupling_strengths else 0.0,
            'weakest_coupling': float(np.min(coupling_strengths)) if coupling_strengths else 0.0
        }
        
        print(f"    市場統合度: {market_integration:.3f}")
        print(f"    結合ボラティリティ: {coupling_volatility:.3f}")
        
        return coupling_analysis
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """パイプライン実行履歴サマリー"""
        if not self.execution_history:
            return {"message": "No pipeline executions yet"}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h['success'])
        
        execution_times = [h['execution_time'] for h in self.execution_history if h['success']]
        series_counts = [h['n_series'] for h in self.execution_history]
        
        return {
            'total_executions': total_executions,
            'success_rate': successful_executions / total_executions,
            'average_execution_time': float(np.mean(execution_times)) if execution_times else 0.0,
            'average_series_count': float(np.mean(series_counts)),
            'fastest_execution': float(np.min(execution_times)) if execution_times else 0.0,
            'slowest_execution': float(np.max(execution_times)) if execution_times else 0.0,
            'recent_executions': self.execution_history[-3:]
        }

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def run_lambda3_analysis(
    data_source: Union[Dict[str, np.ndarray], str, Path],
    config: Optional[L3ComprehensiveConfig] = None,
    analysis_type: str = "comprehensive"
) -> Lambda3ComprehensiveResults:
    """
    Lambda³分析実行の便利関数
    
    Args:
        data_source: データソース
        config: 包括設定
        analysis_type: 分析タイプ ('comprehensive', 'financial', 'rapid')
        
    Returns:
        Lambda3ComprehensiveResults: 分析結果
    """
    pipeline = Lambda3ComprehensivePipeline(config)
    
    if analysis_type == "comprehensive":
        return pipeline.run_comprehensive_analysis(data_source)
    elif analysis_type == "financial":
        if isinstance(data_source, dict):
            # 辞書データの場合は通常の包括分析
            return pipeline.run_comprehensive_analysis(data_source)
        else:
            # 金融データ特化分析
            return pipeline.run_financial_data_analysis()
    elif analysis_type == "rapid":
        # 高速分析モード
        if config is None:
            from ..core.config import create_rapid_config
            config = create_rapid_config()
        
        pipeline_rapid = Lambda3ComprehensivePipeline(config)
        return pipeline_rapid.run_comprehensive_analysis(data_source, enable_visualization=False)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

def analyze_financial_markets(
    tickers: Optional[Dict[str, str]] = None,
    period: str = "2y",
    enable_advanced_analysis: bool = True
) -> Lambda3ComprehensiveResults:
    """
    金融市場分析の便利関数
    
    Args:
        tickers: ティッカー辞書
        period: 分析期間
        enable_advanced_analysis: 高度分析有効フラグ
        
    Returns:
        Lambda3ComprehensiveResults: 金融分析結果
    """
    from ..core.config import create_financial_config
    
    config = create_financial_config()
    pipeline = Lambda3ComprehensivePipeline(config)
    
    # 期間設定
    if period == "1y":
        start_date = "2023-01-01"
        end_date = "2024-01-01"
    elif period == "2y":
        start_date = "2022-01-01"
        end_date = "2024-01-01"
    elif period == "5y":
        start_date = "2019-01-01"
        end_date = "2024-01-01"
    else:
        start_date = "2022-01-01"
        end_date = "2024-01-01"
    
    return pipeline.run_financial_data_analysis(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        enable_crisis_detection=enable_advanced_analysis
    )

def create_lambda3_report(
    results: Lambda3ComprehensiveResults,
    output_path: Union[str, Path],
    include_technical_details: bool = True
) -> Path:
    """
    Lambda³分析レポート生成
    
    Args:
        results: 分析結果
        output_path: 出力パス
        include_technical_details: 技術詳細含有フラグ
        
    Returns:
        Path: 生成されたレポートパス
    """
    output_path = Path(output_path)
    
    # レポート内容生成
    report_content = {
        'title': 'Lambda³ Theory Analysis Report',
        'timestamp': results.analysis_timestamp,
        'executive_summary': results.get_summary(),
        'analysis_results': {
            'data_overview': {
                'n_series': len(results.series_names),
                'series_names': results.series_names,
                'total_data_points': sum(len(data) for data in results.series_dict.values())
            },
            'quality_assessment': results.quality_metrics,
            'performance_metrics': results.performance_metrics
        }
    }
    
    # 技術詳細追加
    if include_technical_details:
        report_content['technical_details'] = {
            'configuration': results.config.to_dict() if results.config else {},
            'feature_extraction': {
                'n_extracted_series': len(results.features_dict),
                'feature_types': ['structural_tensor', 'hierarchical', 'multi_scale']
            },
            'analysis_modules': {
                'structural_analysis': bool(results.structural_analysis),
                'hierarchical_analysis': bool(results.hierarchical_results),
                'pairwise_analysis': bool(results.pairwise_results),
                'synchronization_analysis': bool(results.synchronization_results),
                'regime_analysis': bool(results.regime_results)
            }
        }
    
    # JSON形式で保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_content, f, indent=2, ensure_ascii=False)
    
    print(f"Lambda³分析レポートを生成しました: {output_path}")
    return output_path

# ==========================================================
# MAIN TESTING
# ==========================================================

if __name__ == "__main__":
    print("Lambda³ Comprehensive Pipeline Test")
    print("=" * 50)
    
    # テスト用合成データ生成
    np.random.seed(42)
    n_points = 500
    
    # 複数の構造変化系列
    test_data = {}
    
    for i, series_name in enumerate(['Series_A', 'Series_B', 'Series_C']):
        # ベースとなる時系列
        base_series = np.cumsum(np.random.randn(n_points) * 0.05)
        
        # 構造変化を注入
        jump_positions = np.random.choice(n_points, size=5, replace=False)
        for pos in jump_positions:
            magnitude = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)
            base_series[pos:] += magnitude
        
        # ノイズ追加
        test_data[series_name] = base_series + np.random.randn(n_points) * 0.1
    
    print(f"テストデータ生成完了: {len(test_data)} 系列、各 {n_points} ポイント")
    
    # 包括分析パイプライン実行
    try:
        pipeline = Lambda3ComprehensivePipeline()
        
        # 分析実行
        results = pipeline.run_comprehensive_analysis(
            test_data,
            analysis_modes={
                'hierarchical_analysis': True,
                'pairwise_analysis': True,
                'synchronization_analysis': True,
                'regime_analysis': True,
                'crisis_detection': True
            },
            enable_visualization=False  # テスト環境では無効化
        )
        
        # 結果サマリー表示
        summary = results.get_summary()
        print(f"\n分析結果サマリー:")
        print(f"  解析系列数: {summary['analysis_info']['n_series']}")
        print(f"  総データポイント: {summary['analysis_info']['total_data_points']}")
        print(f"  特徴抽出成功: {summary['feature_extraction']['extracted_series']}")
        print(f"  実行時間: {results.performance_metrics.get('total_execution_time', 0):.2f}秒")
        print(f"  総合品質: {results.quality_metrics.get('overall_analysis_quality', 0):.3f}")
        
        # パイプライン履歴確認
        pipeline_summary = pipeline.get_pipeline_summary()
        print(f"\nパイプライン履歴:")
        print(f"  総実行回数: {pipeline_summary['total_executions']}")
        print(f"  成功率: {pipeline_summary['success_rate']:.1%}")
        
        print("\n包括分析パイプラインテスト成功!")
        
    except Exception as e:
        print(f"テストエラー: {e}")
        raise
    
    print("\nLambda³ Comprehensive Pipeline loaded successfully!")
    print("Ready for end-to-end structural tensor analysis.")
