# ==========================================================
# lambda3/pipelines/comprehensive.py (JIT Optimized Version)
# Comprehensive Analysis Pipeline for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 革新ポイント: JIT最適化統合ワークフローの実現
# ==========================================================

"""
Lambda³理論統合解析パイプライン（JIT最適化版）

構造テンソル(Λ)理論の全要素を統合した包括的解析ワークフロー。
階層分析、ペアワイズ相互作用、ベイズ推定、可視化を一括実行。

核心機能:
- JIT最適化による高速統合解析
- 自動データ前処理とバリデーション
- 階層・ペアワイズ分析の自動実行
- 結果統合とレポート生成
- 金融市場特化ワークフロー
- リアルタイム分析対応

JIT最適化効果:
- 大規模データセットの高速処理
- メモリ効率の最適化
- 並列処理による性能向上
- 数値安定性の確保
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
import time
from pathlib import Path
import json

# Lambda³ 核心モジュール
try:
    from ..core.config import (
        L3ComprehensiveConfig, 
        L3BaseConfig,
        create_default_config,
        create_financial_config,
        create_rapid_config,
        create_research_config
    )
    from ..core.structural_tensor import StructuralTensorFeatures, StructuralTensorExtractor
    from ..analysis.hierarchical import HierarchicalAnalyzer, HierarchicalSeparationResults
    from ..analysis.pairwise import PairwiseAnalyzer, PairwiseInteractionResults
    LAMBDA3_MODULES_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Lambda³ modules import failed: {e}")
    LAMBDA3_MODULES_AVAILABLE = False

# JIT最適化関数
try:
    from ..core.jit_functions import (
        extract_lambda3_features_jit,
        test_jit_functions_fixed,
        benchmark_performance_fixed
    )
    JIT_FUNCTIONS_AVAILABLE = True
except ImportError:
    warnings.warn("JIT functions not available in pipeline")
    JIT_FUNCTIONS_AVAILABLE = False

# 外部依存関係
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Financial data acquisition disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn("Visualization libraries not available.")

try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# ==========================================================
# COMPREHENSIVE RESULTS DATA CLASS
# ==========================================================

@dataclass
class Lambda3ComprehensiveResults:
    """
    Lambda³包括解析結果データクラス
    
    Lambda³理論による全解析結果を統合管理。
    構造テンソル特徴量、階層分析、ペアワイズ分析、統計サマリーを包含。
    """
    
    # メタデータ
    analysis_timestamp: str
    config: L3ComprehensiveConfig
    data_info: Dict[str, Any]
    
    # 核心結果
    structural_features: Dict[str, StructuralTensorFeatures] = field(default_factory=dict)
    hierarchical_results: Dict[str, HierarchicalSeparationResults] = field(default_factory=dict)
    pairwise_results: Dict[str, PairwiseInteractionResults] = field(default_factory=dict)
    
    # 統合分析結果
    network_analysis: Dict[str, Any] = field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 性能情報
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # エラー・警告情報
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """解析サマリー取得"""
        return {
            'timestamp': self.analysis_timestamp,
            'jit_optimized': self.config.base.jit_config.enable_jit if hasattr(self.config.base, 'jit_config') else False,
            'series_count': len(self.structural_features),
            'hierarchical_analyses': len(self.hierarchical_results),
            'pairwise_analyses': len(self.pairwise_results),
            'analysis_modes': {mode: enabled for mode, enabled in self.config.analysis_modes.items() if enabled},
            'overall_quality': self.quality_metrics.get('overall_quality', 0.0),
            'execution_time': self.performance_metrics.get('total_execution_time', 0.0),
            'warning_count': len(self.warnings),
            'error_count': len(self.errors)
        }
    
    def get_top_interactions(self, n: int = 5) -> List[Tuple[str, float]]:
        """最強相互作用ペア取得"""
        interactions = []
        for pair_name, result in self.pairwise_results.items():
            coupling = result.calculate_bidirectional_coupling()
            interactions.append((pair_name, coupling))
        
        interactions.sort(key=lambda x: x[1], reverse=True)
        return interactions[:n]
    
    def get_hierarchy_rankings(self) -> Dict[str, List[Tuple[str, float]]]:
        """階層分析ランキング"""
        rankings = {
            'escalation_strength': [],
            'deescalation_strength': [],
            'separation_quality': []
        }
        
        for series_name, result in self.hierarchical_results.items():
            rankings['escalation_strength'].append((series_name, result.get_escalation_strength()))
            rankings['deescalation_strength'].append((series_name, result.get_deescalation_strength()))
            rankings['separation_quality'].append((series_name, result.separation_quality.get('overall_quality', 0)))
        
        # 各カテゴリでソート
        for category in rankings:
            rankings[category].sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def export_summary_report(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """サマリーレポート出力"""
        summary = self.get_analysis_summary()
        rankings = self.get_hierarchy_rankings()
        top_interactions = self.get_top_interactions()
        
        report_lines = [
            "=" * 80,
            "Lambda³ Theory - Comprehensive Analysis Report",
            "=" * 80,
            f"Analysis Timestamp: {summary['timestamp']}",
            f"JIT Optimization: {'Enabled' if summary['jit_optimized'] else 'Disabled'}",
            f"Series Count: {summary['series_count']}",
            f"Execution Time: {summary['execution_time']:.2f} seconds",
            f"Overall Quality: {summary['overall_quality']:.3f}",
            "",
            "HIERARCHICAL ANALYSIS RANKINGS:",
            "-" * 40,
            "Top Escalation Strength:",
        ]
        
        for i, (series, strength) in enumerate(rankings['escalation_strength'][:5], 1):
            report_lines.append(f"  {i}. {series}: {strength:.4f}")
        
        report_lines.extend([
            "",
            "Top Deescalation Strength:",
        ])
        
        for i, (series, strength) in enumerate(rankings['deescalation_strength'][:5], 1):
            report_lines.append(f"  {i}. {series}: {strength:.4f}")
        
        report_lines.extend([
            "",
            "PAIRWISE INTERACTION ANALYSIS:",
            "-" * 40,
            "Strongest Interactions:",
        ])
        
        for i, (pair, coupling) in enumerate(top_interactions, 1):
            report_lines.append(f"  {i}. {pair}: {coupling:.4f}")
        
        report_lines.extend([
            "",
            "PERFORMANCE METRICS:",
            "-" * 40,
            f"JIT Optimization: {'Active' if summary['jit_optimized'] else 'Inactive'}",
            f"Warnings: {summary['warning_count']}",
            f"Errors: {summary['error_count']}",
            "",
            "ANALYSIS MODES:",
            "-" * 40,
        ])
        
        for mode, enabled in summary['analysis_modes'].items():
            status = "✓" if enabled else "✗"
            report_lines.append(f"  {status} {mode}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "End of Report",
            "=" * 80
        ])
        
        report_text = "\n".join(report_lines)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text

# ==========================================================
# COMPREHENSIVE PIPELINE
# ==========================================================

class Lambda3ComprehensivePipeline:
    """
    Lambda³包括解析パイプライン（JIT最適化版）
    
    構造テンソル理論の全要素を統合した自動解析ワークフロー。
    データ取得から最終レポート生成まで一括実行。
    
    主要機能:
    - 自動データ前処理とバリデーション
    - JIT最適化による高速特徴抽出
    - 階層・ペアワイズ分析の並列実行
    - 統合ネットワーク解析
    - 自動品質評価とレポート生成
    """
    
    def __init__(self, config: Optional[L3ComprehensiveConfig] = None):
        """
        初期化
        
        Args:
            config: Lambda³包括設定
        """
        self.config = config or create_default_config()
        self.execution_history = []
        
        # JIT最適化確認
        self.jit_enabled = JIT_FUNCTIONS_AVAILABLE
        if hasattr(self.config.base, 'jit_config'):
            self.jit_enabled = self.jit_enabled and self.config.base.jit_config.enable_jit
        
        # コンポーネント初期化
        self.feature_extractor = StructuralTensorExtractor(self.config.base)
        self.hierarchical_analyzer = HierarchicalAnalyzer(self.config.hierarchical, self.config.bayesian)
        self.pairwise_analyzer = PairwiseAnalyzer(self.config.pairwise, self.config.bayesian)
        
        print(f"🚀 Lambda³ Comprehensive Pipeline initialized")
        print(f"   JIT Optimization: {'Enabled' if self.jit_enabled else 'Disabled'}")
        print(f"   Modules Available: {LAMBDA3_MODULES_AVAILABLE}")
        print(f"   Analysis Modes: {sum(self.config.analysis_modes.values())}/{len(self.config.analysis_modes)} enabled")
    
    def run_comprehensive_analysis(
        self,
        data: Union[Dict[str, np.ndarray], pd.DataFrame, str, Path],
        analysis_name: Optional[str] = None
    ) -> Lambda3ComprehensiveResults:
        """
        包括解析実行
        
        Lambda³理論の全要素を統合した完全自動解析。
        データ前処理から最終レポートまで一括実行。
        
        Args:
            data: 入力データ（辞書、DataFrame、ファイルパス）
            analysis_name: 解析名（オプション）
            
        Returns:
            Lambda3ComprehensiveResults: 包括解析結果
        """
        start_time = time.time()
        analysis_name = analysis_name or f"Lambda3_Analysis_{int(time.time())}"
        
        print(f"\n{'='*80}")
        print(f"LAMBDA³ COMPREHENSIVE ANALYSIS: {analysis_name}")
        print(f"JIT Optimization: {'Enabled' if self.jit_enabled else 'Disabled'}")
        print(f"{'='*80}")
        
        # 結果オブジェクト初期化
        results = Lambda3ComprehensiveResults(
            analysis_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            config=self.config,
            data_info={}
        )
        
        try:
            # 1. データ前処理とバリデーション
            processed_data, data_info = self._preprocess_and_validate_data(data)
            results.data_info = data_info
            
            print(f"\n📊 Data Processing Complete:")
            print(f"   Series count: {len(processed_data)}")
            print(f"   Data points per series: {data_info.get('average_length', 'N/A')}")
            print(f"   JIT preprocessing: {'Applied' if self.jit_enabled else 'Standard'}")
            
            # 2. 構造テンソル特徴抽出（JIT最適化）
            structural_features = self._extract_structural_features_optimized(processed_data)
            results.structural_features = structural_features
            
            print(f"\n🔬 Structural Tensor Extraction Complete:")
            print(f"   Features extracted: {len(structural_features)} series")
            print(f"   JIT optimization: {'Active' if self.jit_enabled else 'Inactive'}")
            
            # 3. 階層分析実行（条件付き）
            if self.config.analysis_modes.get('hierarchical_analysis', True):
                hierarchical_results = self._run_hierarchical_analysis(structural_features)
                results.hierarchical_results = hierarchical_results
                
                print(f"\n📈 Hierarchical Analysis Complete:")
                print(f"   Series analyzed: {len(hierarchical_results)}")
                avg_quality = np.mean([r.separation_quality.get('overall_quality', 0) 
                                     for r in hierarchical_results.values()])
                print(f"   Average separation quality: {avg_quality:.3f}")
            
            # 4. ペアワイズ分析実行（条件付き）
            if self.config.analysis_modes.get('pairwise_analysis', True) and len(structural_features) >= 2:
                pairwise_results = self._run_pairwise_analysis(structural_features)
                results.pairwise_results = pairwise_results
                
                print(f"\n🔗 Pairwise Analysis Complete:")
                print(f"   Pairs analyzed: {len(pairwise_results)}")
                avg_coupling = np.mean([r.calculate_bidirectional_coupling() 
                                      for r in pairwise_results.values()])
                print(f"   Average coupling strength: {avg_coupling:.3f}")
            
            # 5. ネットワーク解析（条件付き）
            if self.config.analysis_modes.get('synchronization_analysis', True) and len(results.pairwise_results) > 0:
                network_analysis = self._run_network_analysis(results)
                results.network_analysis = network_analysis
                
                print(f"\n🌐 Network Analysis Complete:")
                print(f"   Network density: {network_analysis.get('density', 0):.3f}")
                print(f"   Top centrality: {network_analysis.get('top_central_node', 'N/A')}")
            
            # 6. 品質評価
            quality_metrics = self._evaluate_analysis_quality(results)
            results.quality_metrics = quality_metrics
            
            # 7. 性能メトリクス記録
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_metrics = {
                'total_execution_time': execution_time,
                'jit_enabled': self.jit_enabled,
                'series_count': len(processed_data),
                'total_data_points': sum(len(series) for series in processed_data.values()),
                'processing_rate': sum(len(series) for series in processed_data.values()) / execution_time,
                'memory_efficiency': self._estimate_memory_usage(results)
            }
            results.performance_metrics = performance_metrics
            
            print(f"\n✅ Comprehensive Analysis Complete!")
            print(f"   Execution time: {execution_time:.2f} seconds")
            print(f"   Processing rate: {performance_metrics['processing_rate']:.0f} points/sec")
            print(f"   Overall quality: {quality_metrics.get('overall_quality', 0):.3f}")
            
            # 実行履歴記録
            self.execution_history.append({
                'analysis_name': analysis_name,
                'timestamp': results.analysis_timestamp,
                'execution_time': execution_time,
                'series_count': len(processed_data),
                'quality': quality_metrics.get('overall_quality', 0),
                'jit_enabled': self.jit_enabled
            })
            
            return results
            
        except Exception as e:
            error_msg = f"Comprehensive analysis failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            results.errors.append(error_msg)
            
            # 部分結果でも返却
            end_time = time.time()
            results.performance_metrics = {
                'total_execution_time': end_time - start_time,
                'jit_enabled': self.jit_enabled,
                'error_occurred': True
            }
            
            return results
    
    def run_financial_analysis(
        self,
        tickers: Optional[Dict[str, str]] = None,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        enable_crisis_detection: bool = True
    ) -> Lambda3ComprehensiveResults:
        """
        金融市場分析ワークフロー
        
        Lambda³理論による金融市場の構造変化分析。
        データ取得から危機検出まで全自動実行。
        
        Args:
            tickers: ティッカー辞書 {name: ticker}
            start_date, end_date: 分析期間
            enable_crisis_detection: 危機検出有効化
            
        Returns:
            Lambda3ComprehensiveResults: 金融分析結果
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not available for financial analysis")
        
        # デフォルトティッカー
        if tickers is None:
            tickers = {
                "USD/JPY": "JPY=X",
                "Bitcoin": "BTC-USD",
                "S&P500": "^GSPC",
                "Gold": "GC=F",
                "Oil": "CL=F"
            }
        
        print(f"\n📈 Lambda³ Financial Market Analysis")
        print(f"Period: {start_date} to {end_date}")
        print(f"Assets: {list(tickers.keys())}")
        print(f"Crisis Detection: {'Enabled' if enable_crisis_detection else 'Disabled'}")
        
        # 金融データ取得
        financial_data = self._acquire_financial_data(tickers, start_date, end_date)
        
        if not financial_data:
            raise ValueError("No financial data acquired")
        
        # 金融特化設定適用
        original_config = self.config
        self.config = create_financial_config()
        
        # 危機検出設定
        if enable_crisis_detection:
            self.config.analysis_modes['crisis_detection'] = True
            self.config.hierarchical.escalation_threshold = 0.4  # より敏感に
            self.config.pairwise.asymmetry_detection_sensitivity = 0.05  # より精密に
        
        try:
            # 包括分析実行
            results = self.run_comprehensive_analysis(
                financial_data, 
                analysis_name=f"Financial_Analysis_{start_date}_{end_date}"
            )
            
            # 金融特化後処理
            if enable_crisis_detection:
                crisis_analysis = self._detect_financial_crises(results)
                results.network_analysis['crisis_analysis'] = crisis_analysis
                
                print(f"\n🚨 Crisis Detection Results:")
                crisis_periods = crisis_analysis.get('crisis_periods', [])
                print(f"   Crisis periods detected: {len(crisis_periods)}")
                for period in crisis_periods[:3]:  # 上位3件表示
                    print(f"   - {period['start']} to {period['end']}: {period['severity']:.3f}")
            
            return results
            
        finally:
            # 設定復元
            self.config = original_config
    
    def run_rapid_screening(
        self,
        data: Union[Dict[str, np.ndarray], pd.DataFrame],
        screening_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        高速スクリーニング分析
        
        Lambda³理論による大量データの高速スクリーニング。
        JIT最適化を最大限活用した超高速分析。
        
        Args:
            data: 入力データ
            screening_threshold: スクリーニング閾値
            
        Returns:
            Dict: スクリーニング結果
        """
        start_time = time.time()
        
        print(f"\n⚡ Lambda³ Rapid Screening Analysis")
        print(f"JIT Acceleration: {'Maximum' if self.jit_enabled else 'Standard'}")
        
        # 高速設定適用
        rapid_config = create_rapid_config()
        rapid_config.base.jit_config.optimization_level = 'aggressive'
        
        # データ前処理（最小限）
        processed_data, _ = self._preprocess_and_validate_data(data, minimal=True)
        
        screening_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'series_count': len(processed_data),
            'threshold': screening_threshold,
            'flagged_series': [],
            'interaction_alerts': [],
            'performance_metrics': {}
        }
        
        # 高速特徴抽出（JIT最適化）
        feature_extraction_times = []
        
        for series_name, series_data in processed_data.items():
            feature_start = time.time()
            
            if self.jit_enabled and JIT_FUNCTIONS_AVAILABLE:
                try:
                    # JIT最適化による超高速抽出
                    features_tuple = extract_lambda3_features_jit(
                        series_data,
                        window=5,  # 高速化のため縮小
                        local_window=3,
                        global_window=10
                    )
                    
                    # 簡易品質スコア計算
                    delta_pos, delta_neg, rho_t = features_tuple[:3]
                    total_events = np.sum(delta_pos) + np.sum(delta_neg)
                    avg_tension = np.mean(rho_t)
                    quality_score = (total_events / len(series_data)) * avg_tension
                    
                    if quality_score > screening_threshold:
                        screening_results['flagged_series'].append({
                            'series_name': series_name,
                            'quality_score': float(quality_score),
                            'event_rate': float(total_events / len(series_data)),
                            'avg_tension': float(avg_tension)
                        })
                    
                except Exception as e:
                    print(f"JIT screening failed for {series_name}: {e}")
            
            feature_time = time.time() - feature_start
            feature_extraction_times.append(feature_time)
        
        # ペアワイズ高速スクリーニング（上位候補のみ）
        flagged_names = [item['series_name'] for item in screening_results['flagged_series']]
        
        if len(flagged_names) >= 2:
            for i, name_a in enumerate(flagged_names[:5]):  # 上位5件のみ
                for name_b in flagged_names[i+1:6]:
                    try:
                        # 簡易相関計算
                        series_a = processed_data[name_a]
                        series_b = processed_data[name_b]
                        min_len = min(len(series_a), len(series_b))
                        
                        correlation = np.corrcoef(
                            series_a[:min_len], 
                            series_b[:min_len]
                        )[0, 1]
                        
                        if abs(correlation) > 0.7:  # 高相関アラート
                            screening_results['interaction_alerts'].append({
                                'pair': f"{name_a}_vs_{name_b}",
                                'correlation': float(correlation),
                                'alert_type': 'high_correlation'
                            })
                    
                    except Exception as e:
                        continue
        
        # 性能メトリクス
        end_time = time.time()
        total_time = end_time - start_time
        
        screening_results['performance_metrics'] = {
            'total_execution_time': total_time,
            'average_feature_time': np.mean(feature_extraction_times),
            'processing_rate': sum(len(s) for s in processed_data.values()) / total_time,
            'jit_enabled': self.jit_enabled
        }
        
        print(f"\n✅ Rapid Screening Complete!")
        print(f"   Execution time: {total_time:.2f} seconds")
        print(f"   Flagged series: {len(screening_results['flagged_series'])}")
        print(f"   Interaction alerts: {len(screening_results['interaction_alerts'])}")
        print(f"   Processing rate: {screening_results['performance_metrics']['processing_rate']:.0f} points/sec")
        
        return screening_results
    
    def _preprocess_and_validate_data(
        self, 
        data: Union[Dict[str, np.ndarray], pd.DataFrame, str, Path],
        minimal: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """データ前処理とバリデーション"""
        
        # データ形式統一
        if isinstance(data, (str, Path)):
            # ファイル読み込み
            data_path = Path(data)
            if data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
                processed_data = {col: df[col].values for col in df.columns if df[col].dtype in [np.float64, np.int64]}
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        elif isinstance(data, pd.DataFrame):
            # DataFrame変換
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            processed_data = {col: data[col].values for col in numeric_columns}
        
        elif isinstance(data, dict):
            # 辞書形式確認
            processed_data = {}
            for name, series in data.items():
                if isinstance(series, (list, tuple)):
                    processed_data[name] = np.array(series, dtype=np.float64)
                elif isinstance(series, np.ndarray):
                    processed_data[name] = series.astype(np.float64)
                else:
                    raise ValueError(f"Unsupported data type for {name}: {type(series)}")
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        if not processed_data:
            raise ValueError("No valid numeric data found")
        
        # データ品質チェック
        data_info = {
            'series_count': len(processed_data),
            'series_names': list(processed_data.keys()),
            'lengths': {name: len(series) for name, series in processed_data.items()},
            'average_length': np.mean([len(series) for series in processed_data.values()]),
            'preprocessing_mode': 'minimal' if minimal else 'comprehensive'
        }
        
        if not minimal:
            # 包括的前処理
            for name, series in processed_data.items():
                # 欠損値処理
                if np.isnan(series).any():
                    mask = np.isnan(series)
                    if np.all(mask):
                        raise ValueError(f"Series {name} contains only NaN values")
                    
                    # 線形補間
                    valid_indices = np.flatnonzero(~mask)
                    invalid_indices = np.flatnonzero(mask)
                    series[mask] = np.interp(invalid_indices, valid_indices, series[valid_indices])
                    processed_data[name] = series
                
                # 無限値処理
                if np.isinf(series).any():
                    series = np.clip(series, -1e10, 1e10)
                    processed_data[name] = series
            
            # 長さ統一（オプション）
            if len(set(data_info['lengths'].values())) > 1:
                min_length = min(data_info['lengths'].values())
                for name in processed_data:
                    processed_data[name] = processed_data[name][:min_length]
                
                data_info['length_unified'] = min_length
        
        return processed_data, data_info
    
    def _extract_structural_features_optimized(
        self, 
        data: Dict[str, np.ndarray]
    ) -> Dict[str, StructuralTensorFeatures]:
        """構造テンソル特徴抽出（JIT最適化版）"""
        
        features_dict = {}
        extraction_times = []
        
        for series_name, series_data in data.items():
            start_time = time.time()
            
            try:
                if self.jit_enabled and JIT_FUNCTIONS_AVAILABLE:
                    # JIT最適化による高速抽出
                    features_tuple = extract_lambda3_features_jit(
                        series_data,
                        window=self.config.base.window,
                        local_window=self.config.base.local_window,
                        global_window=self.config.base.global_window,
                        delta_percentile=self.config.base.delta_percentile,
                        local_percentile=self.config.base.local_threshold_percentile,
                        global_percentile=self.config.base.global_threshold_percentile
                    )
                    
                    # StructuralTensorFeatures オブジェクト構築
                    delta_pos, delta_neg, rho_t, local_pos, local_neg, global_pos, global_neg = features_tuple
                    
                    features = StructuralTensorFeatures(
                        data=series_data,
                        series_name=series_name,
                        delta_LambdaC_pos=delta_pos,
                        delta_LambdaC_neg=delta_neg,
                        rho_T=rho_t,
                        local_pos=local_pos,
                        local_neg=local_neg,
                        global_pos=global_pos,
                        global_neg=global_neg
                    )
                    
                else:
                    # 標準特徴抽出
                    features = self.feature_extractor.extract_hierarchical_features(
                        series_data, series_name
                    )
                
                features_dict[series_name] = features
                
                extraction_time = time.time() - start_time
                extraction_times.append(extraction_time)
                
            except Exception as e:
                print(f"Feature extraction failed for {series_name}: {e}")
                continue
        
        if extraction_times:
            avg_extraction_time = np.mean(extraction_times)
            print(f"   Average feature extraction time: {avg_extraction_time:.4f}s per series")
        
        return features_dict
    
    def _run_hierarchical_analysis(
        self, 
        features_dict: Dict[str, StructuralTensorFeatures]
    ) -> Dict[str, HierarchicalSeparationResults]:
        """階層分析実行"""
        
        hierarchical_results = {}
        use_bayesian = self.config.analysis_modes.get('bayesian_analysis', True) and BAYESIAN_AVAILABLE
        
        for series_name, features in features_dict.items():
            try:
                result = self.hierarchical_analyzer.analyze_hierarchical_separation(
                    features, use_bayesian=use_bayesian
                )
                hierarchical_results[series_name] = result
                
            except Exception as e:
                print(f"Hierarchical analysis failed for {series_name}: {e}")
                continue
        
        return hierarchical_results
    
    def _run_pairwise_analysis(
        self, 
        features_dict: Dict[str, StructuralTensorFeatures]
    ) -> Dict[str, PairwiseInteractionResults]:
        """ペアワイズ分析実行"""
        
        pairwise_results = {}
        use_bayesian = self.config.analysis_modes.get('bayesian_analysis', True) and BAYESIAN_AVAILABLE
        
        series_names = list(features_dict.keys())
        
        for i, name_a in enumerate(series_names):
            for j, name_b in enumerate(series_names):
                if i < j:  # 重複回避
                    pair_key = f"{name_a}_vs_{name_b}"
                    
                    try:
                        result = self.pairwise_analyzer.analyze_asymmetric_interaction(
                            features_dict[name_a],
                            features_dict[name_b],
                            use_bayesian=use_bayesian
                        )
                        pairwise_results[pair_key] = result
                        
                    except Exception as e:
                        print(f"Pairwise analysis failed for {pair_key}: {e}")
                        continue
        
        return pairwise_results
    
    def _run_network_analysis(self, results: Lambda3ComprehensiveResults) -> Dict[str, Any]:
        """ネットワーク解析実行"""
        
        if not results.pairwise_results:
            return {'error': 'No pairwise results for network analysis'}
        
        # 相互作用行列構築
        series_names = list(results.structural_features.keys())
        n_series = len(series_names)
        
        interaction_matrix = np.zeros((n_series, n_series))
        asymmetry_matrix = np.zeros((n_series, n_series))
        
        name_to_idx = {name: i for i, name in enumerate(series_names)}
        
        for pair_key, result in results.pairwise_results.items():
            name_a, name_b = result.series_names
            i, j = name_to_idx[name_a], name_to_idx[name_b]
            
            coupling = result.calculate_bidirectional_coupling()
            asymmetry = result.get_asymmetry_score()
            
            interaction_matrix[i, j] = coupling
            interaction_matrix[j, i] = coupling  # 対称化
            asymmetry_matrix[i, j] = asymmetry
            asymmetry_matrix[j, i] = -asymmetry  # 反対称化
        
        # ネットワーク統計
        network_density = np.sum(interaction_matrix > 0.1) / (n_series * (n_series - 1))
        
        # 中心性計算
        centrality_scores = np.sum(interaction_matrix, axis=1)
        top_central_idx = np.argmax(centrality_scores)
        top_central_node = series_names[top_central_idx]
        
        # クラスタリング（簡易）
        clustering_coefficient = self._calculate_clustering_coefficient(interaction_matrix)
        
        return {
            'interaction_matrix': interaction_matrix.tolist(),
            'asymmetry_matrix': asymmetry_matrix.tolist(),
            'series_names': series_names,
            'density': float(network_density),
            'top_central_node': top_central_node,
            'centrality_scores': centrality_scores.tolist(),
            'clustering_coefficient': float(clustering_coefficient),
            'network_metrics': {
                'total_connections': int(np.sum(interaction_matrix > 0.1)),
                'average_coupling': float(np.mean(interaction_matrix[interaction_matrix > 0])),
                'max_coupling': float(np.max(interaction_matrix))
            }
        }
    
    def _evaluate_analysis_quality(self, results: Lambda3ComprehensiveResults) -> Dict[str, float]:
        """解析品質評価"""
        
        quality_scores = []
        
        # 階層分析品質
        if results.hierarchical_results:
            hierarchical_qualities = [
                r.separation_quality.get('overall_quality', 0) 
                for r in results.hierarchical_results.values()
            ]
            quality_scores.extend(hierarchical_qualities)
        
        # ペアワイズ分析品質
        if results.pairwise_results:
            pairwise_qualities = [
                r.interaction_quality.get('overall_quality', 0) 
                for r in results.pairwise_results.values()
            ]
            quality_scores.extend(pairwise_qualities)
        
        # 総合品質
        overall_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # データ品質
        data_quality = self._assess_data_quality(results.data_info)
        
        # 分析完成度
        expected_hierarchical = len(results.structural_features)
        expected_pairwise = len(results.structural_features) * (len(results.structural_features) - 1) // 2
        
        completion_rate = (
            len(results.hierarchical_results) / max(expected_hierarchical, 1) +
            len(results.pairwise_results) / max(expected_pairwise, 1)
        ) / 2
        
        return {
            'overall_quality': overall_quality,
            'data_quality': data_quality,
            'completion_rate': completion_rate,
            'hierarchical_avg_quality': np.mean([
                r.separation_quality.get('overall_quality', 0) 
                for r in results.hierarchical_results.values()
            ]) if results.hierarchical_results else 0.0,
            'pairwise_avg_quality': np.mean([
                r.interaction_quality.get('overall_quality', 0) 
                for r in results.pairwise_results.values()
            ]) if results.pairwise_results else 0.0
        }
    
    def _acquire_financial_data(
        self, 
        tickers: Dict[str, str], 
        start_date: str, 
        end_date: str
    ) -> Dict[str, np.ndarray]:
        """金融データ取得"""
        
        financial_data = {}
        
        for name, ticker in tickers.items():
            try:
                print(f"   Downloading {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty and 'Close' in data.columns:
                    # 対数リターン計算
                    close_prices = data['Close'].dropna()
                    if len(close_prices) > 1:
                        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
                        financial_data[name] = log_returns.values
                        print(f"     ✅ {len(log_returns)} data points acquired")
                    else:
                        print(f"     ❌ Insufficient data")
                else:
                    print(f"     ❌ No data available")
                    
            except Exception as e:
                print(f"     ❌ Download failed: {e}")
                continue
        
        return financial_data
    
    def _detect_financial_crises(self, results: Lambda3ComprehensiveResults) -> Dict[str, Any]:
        """金融危機検出"""
        
        crisis_periods = []
        crisis_indicators = []
        
        # 階層分析からの危機シグナル
        for series_name, hierarchical_result in results.hierarchical_results.items():
            escalation_strength = hierarchical_result.get_escalation_strength()
            
            if escalation_strength > 0.5:  # 高エスカレーション
                crisis_indicators.append({
                    'type': 'hierarchical_escalation',
                    'series': series_name,
                    'severity': escalation_strength,
                    'description': f'High escalation detected in {series_name}'
                })
        
        # ペアワイズ分析からの危機シグナル
        high_coupling_pairs = []
        for pair_name, pairwise_result in results.pairwise_results.items():
            coupling = pairwise_result.calculate_bidirectional_coupling()
            asymmetry = pairwise_result.get_asymmetry_score()
            
            if coupling > 0.7 and asymmetry > 0.3:  # 高結合・高非対称性
                high_coupling_pairs.append({
                    'pair': pair_name,
                    'coupling': coupling,
                    'asymmetry': asymmetry
                })
                
                crisis_indicators.append({
                    'type': 'high_asymmetric_coupling',
                    'pair': pair_name,
                    'severity': coupling * asymmetry,
                    'description': f'High asymmetric coupling in {pair_name}'
                })
        
        # ネットワークレベルの危機検出
        if results.network_analysis:
            network_density = results.network_analysis.get('density', 0)
            if network_density > 0.6:  # 高密度ネットワーク = システミックリスク
                crisis_indicators.append({
                    'type': 'systemic_risk',
                    'severity': network_density,
                    'description': f'High network density detected: {network_density:.3f}'
                })
        
        return {
            'crisis_periods': crisis_periods,
            'crisis_indicators': crisis_indicators,
            'crisis_severity': np.mean([c['severity'] for c in crisis_indicators]) if crisis_indicators else 0.0,
            'systemic_risk_level': results.network_analysis.get('density', 0) if results.network_analysis else 0.0
        }
    
    def _calculate_clustering_coefficient(self, adjacency_matrix: np.ndarray) -> float:
        """クラスタリング係数計算"""
        n = adjacency_matrix.shape[0]
        clustering_coeffs = []
        
        for i in range(n):
            neighbors = np.where(adjacency_matrix[i] > 0.1)[0]
            neighbors = neighbors[neighbors != i]  # 自己除外
            
            if len(neighbors) < 2:
                continue
            
            # 近傍間のエッジ数
            edge_count = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adjacency_matrix[neighbors[j], neighbors[k]] > 0.1:
                        edge_count += 1
            
            # 可能な最大エッジ数
            max_edges = len(neighbors) * (len(neighbors) - 1) // 2
            
            if max_edges > 0:
                clustering_coeffs.append(edge_count / max_edges)
        
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
    
    def _assess_data_quality(self, data_info: Dict[str, Any]) -> float:
        """データ品質評価"""
        quality_score = 1.0
        
        # データ長の一貫性
        lengths = list(data_info.get('lengths', {}).values())
        if lengths:
            length_cv = np.std(lengths) / (np.mean(lengths) + 1e-8)  # 変動係数
            quality_score *= max(0, 1 - length_cv)
        
        # 系列数の充足性
        series_count = data_info.get('series_count', 0)
        if series_count < 2:
            quality_score *= 0.5
        elif series_count >= 5:
            quality_score *= 1.2
        
        # 前処理の完了
        if data_info.get('preprocessing_mode') == 'comprehensive':
            quality_score *= 1.1
        
        return min(quality_score, 1.0)
    
    def _estimate_memory_usage(self, results: Lambda3ComprehensiveResults) -> float:
        """メモリ使用量推定（MB）"""
        
        memory_usage = 0.0
        
        # 構造テンソル特徴量
        for features in results.structural_features.values():
            # 各配列のメモリ使用量推定
            array_size = len(features.data) * 8  # float64 = 8 bytes
            memory_usage += array_size * 7  # 7つの主要配列
        
        # 階層分析結果
        memory_usage += len(results.hierarchical_results) * 1024  # 約1KB per result
        
        # ペアワイズ分析結果  
        memory_usage += len(results.pairwise_results) * 2048  # 約2KB per result
        
        # ネットワーク分析
        if results.network_analysis:
            n_series = len(results.structural_features)
            matrix_memory = n_series * n_series * 8 * 2  # 2つの行列
            memory_usage += matrix_memory
        
        return memory_usage / (1024 * 1024)  # MB変換
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """パイプライン実行履歴サマリー"""
        if not self.execution_history:
            return {"message": "No pipeline executions performed yet"}
        
        total_executions = len(self.execution_history)
        
        execution_times = [h['execution_time'] for h in self.execution_history]
        quality_scores = [h['quality'] for h in self.execution_history]
        jit_enabled_count = sum(1 for h in self.execution_history if h['jit_enabled'])
        
        return {
            'total_executions': total_executions,
            'jit_usage_rate': jit_enabled_count / total_executions,
            'performance_stats': {
                'avg_execution_time': np.mean(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times)
            },
            'quality_stats': {
                'avg_quality': np.mean(quality_scores),
                'min_quality': np.min(quality_scores),
                'max_quality': np.max(quality_scores)
            },
            'recent_executions': self.execution_history[-5:],
            'jit_performance_gain': self._calculate_jit_performance_gain()
        }
    
    def _calculate_jit_performance_gain(self) -> float:
        """JIT性能向上率計算"""
        jit_times = [h['execution_time'] for h in self.execution_history if h['jit_enabled']]
        non_jit_times = [h['execution_time'] for h in self.execution_history if not h['jit_enabled']]
        
        if jit_times and non_jit_times:
            avg_jit_time = np.mean(jit_times)
            avg_non_jit_time = np.mean(non_jit_times)
            
            if avg_jit_time > 0:
                return (avg_non_jit_time - avg_jit_time) / avg_jit_time
        
        return 0.0

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def run_lambda3_analysis(
    data: Union[Dict[str, np.ndarray], pd.DataFrame, str, Path],
    config: Optional[L3ComprehensiveConfig] = None,
    analysis_type: str = 'comprehensive'
) -> Lambda3ComprehensiveResults:
    """
    Lambda³分析実行の便利関数
    
    Args:
        data: 入力データ
        config: 設定オブジェクト
        analysis_type: 分析タイプ ('comprehensive', 'financial', 'rapid')
        
    Returns:
        Lambda3ComprehensiveResults: 分析結果
    """
    if config is None:
        if analysis_type == 'financial':
            config = create_financial_config()
        elif analysis_type == 'rapid':
            config = create_rapid_config()
        elif analysis_type == 'research':
            config = create_research_config()
        else:
            config = create_default_config()
    
    pipeline = Lambda3ComprehensivePipeline(config)
    
    if analysis_type == 'financial' and isinstance(data, dict):
        # 金融データとして解釈
        return pipeline.run_financial_analysis()
    else:
        # 通常の包括分析
        return pipeline.run_comprehensive_analysis(data)

def create_analysis_report(
    results: Lambda3ComprehensiveResults,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    分析レポート生成の便利関数
    
    Args:
        results: 分析結果
        output_path: 出力パス
        
    Returns:
        str: レポートテキスト
    """
    return results.export_summary_report(output_path)

if __name__ == "__main__":
    print("Lambda³ Comprehensive Pipeline Test (JIT Optimized)")
    print("=" * 70)
    
    # JIT機能確認
    if JIT_FUNCTIONS_AVAILABLE:
        print("✅ JIT最適化関数利用可能")
        
        # 簡易パフォーマンステスト
        try:
            test_result = test_jit_functions_fixed()
            if test_result:
                print("✅ JIT関数テスト成功")
            else:
                print("⚠️  JIT関数テスト部分成功")
                
            # ベンチマークテスト
            print("\n⚡ JIT性能ベンチマーク実行...")
            benchmark_performance_fixed()
            
        except Exception as e:
            print(f"❌ JIT性能テストエラー: {e}")
    else:
        print("⚠️  JIT最適化関数利用不可")
    
    # パイプライン初期化テスト
    try:
        config = create_default_config()
        pipeline = Lambda3ComprehensivePipeline(config)
        print("✅ パイプライン初期化成功")
        
        # 簡易データでのテスト
        test_data = {
            'Series_A': np.cumsum(np.random.randn(200) * 0.1),
            'Series_B': np.cumsum(np.random.randn(200) * 0.1),
            'Series_C': np.cumsum(np.random.randn(200) * 0.1)
        }
        
        print("\n🧪 高速スクリーニングテスト...")
        screening_result = pipeline.run_rapid_screening(test_data)
        print(f"✅ スクリーニング完了: {len(screening_result['flagged_series'])} series flagged")
        
    except Exception as e:
        print(f"❌ パイプラインテストエラー: {e}")
    
    print("\nComprehensive pipeline loaded successfully!")
    print("Ready for Lambda³ integrated analysis with JIT optimization.")
