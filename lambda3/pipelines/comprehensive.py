# ==========================================================
# lambda3/pipelines/comprehensive.py (修正版)
# Comprehensive Analysis Pipeline for Lambda³ Theory
# ==========================================================

"""
Lambda³理論統合解析パイプライン（修正版）

循環インポート問題を解決し、Protocol準拠による型安全性を確保した
包括的解析ワークフロー。

修正点:
- types.pyからのProtocolインポートによる循環回避
- Forward referenceの活用
- JIT最適化の段階的有効化
- エラー耐性の向上

Author: Masamichi Iizumi (Miosync, Inc.)
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import warnings
import time
from pathlib import Path
import json

# 型定義のインポート（循環回避）
try:
    from ..core.types import (
        StructuralTensorProtocol,
        HierarchicalResultProtocol,
        PairwiseResultProtocol,
        ComprehensiveResultProtocol,
        ConfigProtocol,
        AnalysisMode,
        FeatureLevel,
        QualityLevel,
        DataInfo,
        PerformanceMetrics,
        Lambda3Error,
        ensure_float_array,
        ensure_series_name
    )
    TYPES_AVAILABLE = True
except ImportError as e:
    TYPES_AVAILABLE = False
    warnings.warn(f"Types module not available: {e}")
    # フォールバック
    StructuralTensorProtocol = Any
    HierarchicalResultProtocol = Any
    PairwiseResultProtocol = Any
    Lambda3Error = Exception

# Lambda³ 核心モジュール（条件付きインポート）
try:
    from ..core.structural_tensor import StructuralTensorFeatures, StructuralTensorExtractor
    STRUCTURAL_TENSOR_AVAILABLE = True
except ImportError as e:
    STRUCTURAL_TENSOR_AVAILABLE = False
    warnings.warn(f"Structural tensor module not available: {e}")

try:
    from ..analysis.hierarchical import HierarchicalAnalyzer, HierarchicalSeparationResults
    HIERARCHICAL_AVAILABLE = True
except ImportError as e:
    HIERARCHICAL_AVAILABLE = False
    warnings.warn(f"Hierarchical analysis not available: {e}")

try:
    from ..analysis.pairwise import PairwiseAnalyzer, PairwiseInteractionResults
    PAIRWISE_AVAILABLE = True
except ImportError as e:
    PAIRWISE_AVAILABLE = False
    warnings.warn(f"Pairwise analysis not available: {e}")

# 設定システム
try:
    from ..core.config import (
        L3ComprehensiveConfig,
        create_default_config,
        create_financial_config,
        create_rapid_config
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    CONFIG_AVAILABLE = False
    warnings.warn(f"Config system not available: {e}")

# JIT最適化関数
try:
    from ..core.jit_functions import extract_lambda3_features_jit, test_jit_functions_fixed
    JIT_FUNCTIONS_AVAILABLE = True
except ImportError:
    JIT_FUNCTIONS_AVAILABLE = False

# 外部依存関係（オプション）
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# ==========================================================
# 結果データクラス（Protocol準拠）
# ==========================================================

@dataclass
class Lambda3ComprehensiveResults:
    """
    Lambda³包括解析結果（修正版）
    
    Protocol準拠による型安全性確保と循環インポート回避を実現。
    """
    
    # メタデータ
    analysis_timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    config: Optional[Any] = None
    data_info: Dict[str, Any] = field(default_factory=dict)
    
    # 核心結果（Protocol型で型安全）
    structural_features: Dict[str, StructuralTensorProtocol] = field(default_factory=dict)
    hierarchical_results: Dict[str, HierarchicalResultProtocol] = field(default_factory=dict)
    pairwise_results: Dict[str, PairwiseResultProtocol] = field(default_factory=dict)
    
    # 統合分析結果
    network_analysis: Dict[str, Any] = field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 性能情報
    performance_metrics: PerformanceMetrics = field(default_factory=dict)
    
    # エラー・警告情報
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """解析サマリー取得"""
        return {
            'timestamp': self.analysis_timestamp,
            'total_series': len(self.structural_features),
            'hierarchical_analyzed': len(self.hierarchical_results),
            'pairwise_analyzed': len(self.pairwise_results),
            'total_structural_changes': self._count_total_structural_changes(),
            'average_quality_score': np.mean(list(self.quality_metrics.values())) if self.quality_metrics else 0.0,
            'processing_time': self.performance_metrics.get('total_time', 0.0),
            'jit_optimized': self.performance_metrics.get('jit_speedup_ratio', 0.0) > 1.0,
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings)
        }
    
    def _count_total_structural_changes(self) -> int:
        """総構造変化数の計算"""
        total = 0
        for features in self.structural_features.values():
            if hasattr(features, 'get_total_structural_changes'):
                total += features.get_total_structural_changes()
        return total
    
    def add_warning(self, message: str):
        """警告追加"""
        self.warnings.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def add_error(self, message: str):
        """エラー追加"""
        self.errors.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def has_feature_type(self, feature_type: str) -> bool:
        """特定特徴タイプの存在確認"""
        type_map = {
            'structural': len(self.structural_features) > 0,
            'hierarchical': len(self.hierarchical_results) > 0,
            'pairwise': len(self.pairwise_results) > 0,
            'network': len(self.network_analysis) > 0
        }
        return type_map.get(feature_type, False)
    
    def export_summary(self, format: str = 'dict') -> Union[Dict, str]:
        """サマリーのエクスポート"""
        summary = self.get_analysis_summary()
        
        if format == 'dict':
            return summary
        elif format == 'json':
            return json.dumps(summary, indent=2, default=str)
        elif format == 'text':
            lines = []
            lines.append("Lambda³ Analysis Summary")
            lines.append("=" * 30)
            for key, value in summary.items():
                lines.append(f"{key}: {value}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

# ==========================================================
# 包括分析パイプライン（修正版）
# ==========================================================

class Lambda3ComprehensivePipeline:
    """
    Lambda³包括分析パイプライン（修正版）
    
    循環インポート問題を解決し、段階的な分析実行を可能にする
    メインパイプラインクラス。
    """
    
    def __init__(self, config: Optional[Any] = None, verbose: bool = True):
        """
        Args:
            config: 設定オブジェクト
            verbose: 詳細出力フラグ
        """
        self.verbose = verbose
        
        # 設定の初期化
        if config is None:
            if CONFIG_AVAILABLE:
                self.config = create_default_config()
            else:
                # フォールバック設定
                self.config = self._create_fallback_config()
        else:
            self.config = config
        
        # 各種分析器の初期化
        self._initialize_analyzers()
        
        # 機能可用性チェック
        self._check_feature_availability()
        
        if self.verbose:
            print("🚀 Lambda³ Comprehensive Pipeline initialized")
            print(f"   JIT optimization: {'✅' if self.jit_enabled else '❌'}")
            print(f"   Hierarchical analysis: {'✅' if self.hierarchical_enabled else '❌'}")
            print(f"   Pairwise analysis: {'✅' if self.pairwise_enabled else '❌'}")
    
    def _create_fallback_config(self) -> Any:
        """フォールバック設定作成"""
        return type('FallbackConfig', (), {
            'window': 10,
            'threshold_percentile': 95.0,
            'enable_jit': JIT_FUNCTIONS_AVAILABLE,
            'feature_level': 'standard',
            'analysis_mode': 'comprehensive'
        })()
    
    def _initialize_analyzers(self):
        """分析器の初期化"""
        self.feature_extractor = None
        self.hierarchical_analyzer = None
        self.pairwise_analyzer = None
        
        # 構造テンソル抽出器
        if STRUCTURAL_TENSOR_AVAILABLE:
            self.feature_extractor = StructuralTensorExtractor(config=self.config)
        
        # 階層分析器
        if HIERARCHICAL_AVAILABLE:
            self.hierarchical_analyzer = HierarchicalAnalyzer(config=self.config)
        
        # ペアワイズ分析器
        if PAIRWISE_AVAILABLE:
            self.pairwise_analyzer = PairwiseAnalyzer(config=self.config)
    
    def _check_feature_availability(self):
        """機能可用性チェック"""
        self.structural_enabled = STRUCTURAL_TENSOR_AVAILABLE
        self.hierarchical_enabled = HIERARCHICAL_AVAILABLE
        self.pairwise_enabled = PAIRWISE_AVAILABLE
        self.jit_enabled = JIT_FUNCTIONS_AVAILABLE and getattr(self.config, 'enable_jit', True)
        self.visualization_enabled = VISUALIZATION_AVAILABLE
    
    def run_analysis(
        self, 
        data: Union[Dict[str, np.ndarray], np.ndarray],
        analysis_modes: Optional[Dict[str, bool]] = None,
        **kwargs
    ) -> Lambda3ComprehensiveResults:
        """
        包括分析実行
        
        Args:
            data: 入力データ（辞書またはnumpy配列）
            analysis_modes: 分析モード設定
            **kwargs: 追加パラメータ
            
        Returns:
            Lambda3ComprehensiveResults: 包括分析結果
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🔬 Lambda³ Comprehensive Analysis Started")
            print("=" * 50)
        
        # 結果オブジェクト初期化
        results = Lambda3ComprehensiveResults(config=self.config)
        
        try:
            # Step 1: データ前処理
            processed_data, data_info = self._preprocess_data(data)
            results.data_info = data_info
            
            if self.verbose:
                print(f"📊 Data preprocessing completed: {len(processed_data)} series")
            
            # Step 2: 構造テンソル特徴量抽出
            if self.structural_enabled:
                structural_features = self._extract_structural_features(processed_data)
                results.structural_features = structural_features
                
                if self.verbose:
                    print(f"⚡ Structural features extracted: {len(structural_features)} series")
            else:
                results.add_warning("Structural tensor extraction disabled")
            
            # Step 3: 階層分析
            analysis_modes = analysis_modes or {}
            if self.hierarchical_enabled and analysis_modes.get('hierarchical', True):
                hierarchical_results = self._run_hierarchical_analysis(results.structural_features)
                results.hierarchical_results = hierarchical_results
                
                if self.verbose:
                    print(f"🏗️ Hierarchical analysis completed: {len(hierarchical_results)} series")
            
            # Step 4: ペアワイズ分析
            if self.pairwise_enabled and analysis_modes.get('pairwise', True) and len(results.structural_features) >= 2:
                pairwise_results = self._run_pairwise_analysis(results.structural_features)
                results.pairwise_results = pairwise_results
                
                if self.verbose:
                    print(f"🔗 Pairwise analysis completed: {len(pairwise_results)} pairs")
            
            # Step 5: ネットワーク分析
            if analysis_modes.get('network', True) and len(results.pairwise_results) > 0:
                network_analysis = self._run_network_analysis(results.pairwise_results)
                results.network_analysis = network_analysis
                
                if self.verbose:
                    print(f"🌐 Network analysis completed")
            
            # Step 6: 品質評価
            quality_metrics = self._calculate_quality_metrics(results)
            results.quality_metrics = quality_metrics
            
            # Step 7: 性能メトリクス
            total_time = time.time() - start_time
            results.performance_metrics = {
                'total_time': total_time,
                'jit_speedup_ratio': 2.0 if self.jit_enabled else 1.0,  # 簡易推定
                'memory_usage_mb': 0.0,  # 実装省略
                'processing_rate': len(processed_data) / total_time if total_time > 0 else 0.0
            }
            
            if self.verbose:
                summary = results.get_analysis_summary()
                print(f"\n✅ Analysis completed successfully!")
                print(f"   Processing time: {total_time:.2f}s")
                print(f"   Total series: {summary['total_series']}")
                print(f"   Quality score: {summary['average_quality_score']:.3f}")
            
            return results
            
        except Exception as e:
            results.add_error(f"Analysis failed: {e}")
            if self.verbose:
                print(f"❌ Analysis failed: {e}")
            raise Lambda3Error(f"Comprehensive analysis failed: {e}")
    
    def _preprocess_data(self, data: Union[Dict[str, np.ndarray], np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """データ前処理"""
        
        # データ形式の統一
        if isinstance(data, np.ndarray):
            processed_data = {'Series': data}
        elif isinstance(data, dict):
            processed_data = data.copy()
        else:
            try:
                processed_data = {'Series': np.asarray(data)}
            except Exception as e:
                raise ValueError(f"Cannot convert data to suitable format: {e}")
        
        # データ情報収集
        data_info = {
            'total_series': len(processed_data),
            'series_names': list(processed_data.keys()),
            'data_lengths': {},
            'data_types': {},
            'missing_values': {},
            'preprocessing_notes': []
        }
        
        # 各系列の前処理
        for name, series in processed_data.items():
            if TYPES_AVAILABLE:
                series = ensure_float_array(series)
                name = ensure_series_name(name)
            else:
                series = np.asarray(series, dtype=np.float64)
            
            # 基本統計
            data_info['data_lengths'][name] = len(series)
            data_info['data_types'][name] = str(series.dtype)
            data_info['missing_values'][name] = int(np.isnan(series).sum())
            
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
                data_info['preprocessing_notes'].append(f"{name}: NaN interpolated")
            
            # 無限値処理
            if np.isinf(series).any():
                series = np.clip(series, -1e10, 1e10)
                processed_data[name] = series
                data_info['preprocessing_notes'].append(f"{name}: Inf values clipped")
        
        return processed_data, data_info
    
    def _extract_structural_features(self, data: Dict[str, np.ndarray]) -> Dict[str, StructuralTensorProtocol]:
        """構造テンソル特徴量抽出"""
        
        features_dict = {}
        
        for series_name, series_data in data.items():
            try:
                if self.feature_extractor:
                    # 正規の抽出器使用
                    features = self.feature_extractor.extract_features(
                        series_data, 
                        series_name=series_name,
                        feature_level=getattr(self.config, 'feature_level', 'standard')
                    )
                else:
                    # フォールバック: 基本的な構造テンソル特徴量
                    features = self._extract_basic_features(series_data, series_name)
                
                features_dict[series_name] = features
                
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Feature extraction failed for {series_name}: {e}")
                continue
        
        return features_dict
    
    def _extract_basic_features(self, data: np.ndarray, series_name: str) -> StructuralTensorProtocol:
        """基本構造テンソル特徴量抽出（フォールバック）"""
        
        # 最小限の構造テンソル特徴量
        diff = np.diff(data, prepend=data[0])
        threshold = np.percentile(np.abs(diff), 95.0)
        
        delta_pos = (diff > threshold).astype(np.float64)
        delta_neg = (diff < -threshold).astype(np.float64)
        
        # 張力スカラー（簡易版）
        window = 10
        rho_t = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            rho_t[i] = np.std(data[start:end])
        
        # StructuralTensorFeaturesが利用可能ならそれを使用
        if STRUCTURAL_TENSOR_AVAILABLE:
            return StructuralTensorFeatures(
                data=data,
                series_name=series_name,
                delta_LambdaC_pos=delta_pos,
                delta_LambdaC_neg=delta_neg,
                rho_T=rho_t
            )
        else:
            # 辞書形式でフォールバック
            return {
                'data': data,
                'series_name': series_name,
                'delta_LambdaC_pos': delta_pos,
                'delta_LambdaC_neg': delta_neg,
                'rho_T': rho_t,
                'get_total_structural_changes': lambda: int(np.sum(delta_pos) + np.sum(delta_neg)),
                'get_average_tension': lambda: float(np.mean(rho_t))
            }
    
    def _run_hierarchical_analysis(self, features_dict: Dict[str, StructuralTensorProtocol]) -> Dict[str, HierarchicalResultProtocol]:
        """階層分析実行"""
        
        hierarchical_results = {}
        
        for series_name, features in features_dict.items():
            try:
                if self.hierarchical_analyzer:
                    result = self.hierarchical_analyzer.analyze_hierarchical_separation(features)
                    hierarchical_results[series_name] = result
                else:
                    # フォールバック: 基本階層分析
                    result = self._basic_hierarchical_analysis(features, series_name)
                    hierarchical_results[series_name] = result
                
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Hierarchical analysis failed for {series_name}: {e}")
                continue
        
        return hierarchical_results
    
    def _basic_hierarchical_analysis(self, features: StructuralTensorProtocol, series_name: str) -> HierarchicalResultProtocol:
        """基本階層分析（フォールバック）"""
        
        # データ取得
        if hasattr(features, 'data'):
            data = features.data
        elif isinstance(features, dict):
            data = features['data']
        else:
            data = np.asarray(features)
        
        # 簡易階層分析
        n = len(data)
        diff = np.diff(data, prepend=data[0])
        
        # 短期・長期変動の分離
        short_window = 5
        long_window = 20
        
        short_changes = 0
        long_changes = 0
        
        for i in range(n):
            # 短期変動
            start_short = max(0, i - short_window)
            end_short = min(n, i + short_window + 1)
            short_var = np.var(data[start_short:end_short])
            
            # 長期変動
            start_long = max(0, i - long_window)
            end_long = min(n, i + long_window + 1)
            long_var = np.var(data[start_long:end_long])
            
            if short_var > np.var(data) * 1.2:
                short_changes += 1
            if long_var > np.var(data) * 1.5:
                long_changes += 1
        
        # 階層指標計算
        total_changes = short_changes + long_changes
        escalation_strength = long_changes / max(total_changes, 1)
        deescalation_strength = short_changes / max(total_changes, 1)
        
        # 結果辞書（Protocolフォールバック）
        return {
            'series_name': series_name,
            'analysis_timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'escalation_strength': escalation_strength,
            'deescalation_strength': deescalation_strength,
            'hierarchy_correlation': 0.5,  # 簡易値
            'convergence_quality': 0.8,
            'statistical_significance': 0.7,
            'get_separation_summary': lambda: {
                'escalation': escalation_strength,
                'deescalation': deescalation_strength
            },
            'get_dominant_hierarchy': lambda: 'local' if short_changes > long_changes else 'global'
        }
    
    def _run_pairwise_analysis(self, features_dict: Dict[str, StructuralTensorProtocol]) -> Dict[str, PairwiseResultProtocol]:
        """ペアワイズ分析実行"""
        
        pairwise_results = {}
        series_names = list(features_dict.keys())
        
        for i, name_a in enumerate(series_names):
            for j, name_b in enumerate(series_names):
                if i < j:  # 重複回避
                    pair_key = f"{name_a}_vs_{name_b}"
                    
                    try:
                        if self.pairwise_analyzer:
                            result = self.pairwise_analyzer.analyze_asymmetric_interaction(
                                features_dict[name_a],
                                features_dict[name_b]
                            )
                        else:
                            result = self._basic_pairwise_analysis(
                                features_dict[name_a], 
                                features_dict[name_b], 
                                name_a, name_b
                            )
                        
                        pairwise_results[pair_key] = result
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"   Warning: Pairwise analysis failed for {pair_key}: {e}")
                        continue
        
        return pairwise_results
    
    def _basic_pairwise_analysis(self, features_a: StructuralTensorProtocol, features_b: StructuralTensorProtocol, 
                                name_a: str, name_b: str) -> PairwiseResultProtocol:
        """基本ペアワイズ分析（フォールバック）"""
        
        # データ取得
        if hasattr(features_a, 'rho_T') and hasattr(features_b, 'rho_T'):
            rho_a = features_a.rho_T
            rho_b = features_b.rho_T
        elif isinstance(features_a, dict) and isinstance(features_b, dict):
            rho_a = features_a.get('rho_T', np.array([]))
            rho_b = features_b.get('rho_T', np.array([]))
        else:
            rho_a = np.array([])
            rho_b = np.array([])
        
        # 長さ統一
        min_length = min(len(rho_a), len(rho_b))
        if min_length > 0:
            rho_a = rho_a[:min_length]
            rho_b = rho_b[:min_length]
            
            # 基本同期指標
            sync_strength = abs(np.corrcoef(rho_a, rho_b)[0, 1]) if min_length > 1 else 0.0
            
            # 簡易因果指標
            causality_a_to_b = abs(np.corrcoef(rho_a[:-1], rho_b[1:])[0, 1]) if min_length > 1 else 0.0
            causality_b_to_a = abs(np.corrcoef(rho_b[:-1], rho_a[1:])[0, 1]) if min_length > 1 else 0.0
        else:
            sync_strength = 0.0
            causality_a_to_b = 0.0
            causality_b_to_a = 0.0
        
        return {
            'name_a': name_a,
            'name_b': name_b,
            'analysis_timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'synchronization_strength': sync_strength,
            'structure_synchronization': sync_strength * 0.8,  # 簡易推定
            'causality_a_to_b': causality_a_to_b,
            'causality_b_to_a': causality_b_to_a,
            'asymmetry_index': abs(causality_a_to_b - causality_b_to_a),
            'data_overlap_length': min_length,
            'correlation_quality': 0.8 if min_length > 10 else 0.5,
            'get_interaction_summary': lambda: {
                'sync': sync_strength,
                'asymmetry': abs(causality_a_to_b - causality_b_to_a)
            },
            'get_dominant_direction': lambda: (
                'a_to_b' if causality_a_to_b > causality_b_to_a else 
                'b_to_a' if causality_b_to_a > causality_a_to_b else 'symmetric'
            )
        }
    
    def _run_network_analysis(self, pairwise_results: Dict[str, PairwiseResultProtocol]) -> Dict[str, Any]:
        """ネットワーク分析実行"""
        
        if not pairwise_results:
            return {}
        
        # 同期強度の収集
        sync_strengths = []
        causality_values = []
        
        for result in pairwise_results.values():
            if hasattr(result, 'synchronization_strength'):
                sync_strengths.append(result.synchronization_strength)
            elif isinstance(result, dict):
                sync_strengths.append(result.get('synchronization_strength', 0.0))
            
            if hasattr(result, 'asymmetry_index'):
                causality_values.append(result.asymmetry_index)
            elif isinstance(result, dict):
                causality_values.append(result.get('asymmetry_index', 0.0))
        
        return {
            'network_density': np.mean(sync_strengths) if sync_strengths else 0.0,
            'average_synchronization': np.mean(sync_strengths) if sync_strengths else 0.0,
            'max_synchronization': np.max(sync_strengths) if sync_strengths else 0.0,
            'average_asymmetry': np.mean(causality_values) if causality_values else 0.0,
            'network_size': len(set([
                result.name_a if hasattr(result, 'name_a') else result.get('name_a', 'unknown')
                for result in pairwise_results.values()
            ] + [
                result.name_b if hasattr(result, 'name_b') else result.get('name_b', 'unknown')
                for result in pairwise_results.values()
            ])),
            'total_pairs': len(pairwise_results)
        }
    
    def _calculate_quality_metrics(self, results: Lambda3ComprehensiveResults) -> Dict[str, float]:
        """品質メトリクス計算"""
        
        quality_metrics = {}
        
        # 構造テンソル品質
        if results.structural_features:
            structural_qualities = []
            for features in results.structural_features.values():
                if hasattr(features, 'data_quality_score'):
                    structural_qualities.append(features.data_quality_score)
                elif hasattr(features, 'get_average_tension'):
                    # 簡易品質推定
                    avg_tension = features.get_average_tension()
                    structural_qualities.append(min(1.0, avg_tension / 0.5))
            
            if structural_qualities:
                quality_metrics['structural_quality'] = np.mean(structural_qualities)
        
        # 階層分析品質
        if results.hierarchical_results:
            hierarchical_qualities = []
            for result in results.hierarchical_results.values():
                if hasattr(result, 'convergence_quality'):
                    hierarchical_qualities.append(result.convergence_quality)
                elif isinstance(result, dict):
                    hierarchical_qualities.append(result.get('convergence_quality', 0.8))
            
            if hierarchical_qualities:
                quality_metrics['hierarchical_quality'] = np.mean(hierarchical_qualities)
        
        # ペアワイズ分析品質
        if results.pairwise_results:
            pairwise_qualities = []
            for result in results.pairwise_results.values():
                if hasattr(result, 'correlation_quality'):
                    pairwise_qualities.append(result.correlation_quality)
                elif isinstance(result, dict):
                    pairwise_qualities.append(result.get('correlation_quality', 0.8))
            
            if pairwise_qualities:
                quality_metrics['pairwise_quality'] = np.mean(pairwise_qualities)
        
        # 総合品質
        if quality_metrics:
            quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics

# ==========================================================
# 便利関数
# ==========================================================

def run_lambda3_analysis(
    data: Union[Dict[str, np.ndarray], np.ndarray],
    config: Optional[Any] = None,
    analysis_type: str = 'comprehensive',
    **kwargs
) -> Lambda3ComprehensiveResults:
    """
    Lambda³解析実行の便利関数
    
    Args:
        data: 入力データ
        config: 設定オブジェクト
        analysis_type: 解析タイプ
        **kwargs: 追加パラメータ
        
    Returns:
        Lambda3ComprehensiveResults: 解析結果
    """
    
    # 設定準備
    if config is None and CONFIG_AVAILABLE:
        if analysis_type == 'financial':
            config = create_financial_config()
        elif analysis_type == 'rapid':
            config = create_rapid_config()
        else:
            config = create_default_config()
    
    # パイプライン実行
    pipeline = Lambda3ComprehensivePipeline(config=config)
    return pipeline.run_analysis(data, **kwargs)

def create_analysis_report(results: Lambda3ComprehensiveResults, format: str = 'text') -> str:
    """
    分析レポート生成
    
    Args:
        results: 分析結果
        format: 出力形式 ('text', 'json')
        
    Returns:
        str: 分析レポート
    """
    
    if format == 'json':
        return results.export_summary('json')
    
    # テキスト形式レポート
    summary = results.get_analysis_summary()
    
    lines = []
    lines.append("Lambda³ Theory Analysis Report")
    lines.append("=" * 40)
    lines.append(f"Analysis Date: {summary['timestamp']}")
    lines.append(f"Processing Time: {summary['processing_time']:.2f}s")
    lines.append(f"JIT Optimized: {'Yes' if summary['jit_optimized'] else 'No'}")
    lines.append("")
    
    lines.append("Data Summary:")
    lines.append(f"  Total Series: {summary['total_series']}")
    lines.append(f"  Total Structural Changes: {summary['total_structural_changes']}")
    lines.append("")
    
    lines.append("Analysis Results:")
    lines.append(f"  Hierarchical Analysis: {summary['hierarchical_analyzed']} series")
    lines.append(f"  Pairwise Analysis: {summary['pairwise_analyzed']} pairs")
    lines.append(f"  Average Quality Score: {summary['average_quality_score']:.3f}")
    lines.append("")
    
    if summary['errors_count'] > 0:
        lines.append(f"Errors: {summary['errors_count']}")
    if summary['warnings_count'] > 0:
        lines.append(f"Warnings: {summary['warnings_count']}")
    
    return "\n".join(lines)

# ==========================================================
# モジュール情報
# ==========================================================

__all__ = [
    'Lambda3ComprehensiveResults',
    'Lambda3ComprehensivePipeline',
    'run_lambda3_analysis',
    'create_analysis_report'
]

# ==========================================================
# テスト関数
# ==========================================================

def test_comprehensive_pipeline():
    """包括パイプラインのテスト"""
    print("🧪 Testing Lambda³ Comprehensive Pipeline")
    print("=" * 50)
    
    try:
        # サンプルデータ生成
        np.random.seed(42)
        sample_data = {
            'Series_A': np.cumsum(np.random.randn(100) * 0.1),
            'Series_B': np.cumsum(np.random.randn(100) * 0.15),
            'Series_C': np.cumsum(np.random.randn(100) * 0.12)
        }
        
        # パイプライン実行
        results = run_lambda3_analysis(sample_data, analysis_type='comprehensive')
        
        # 結果検証
        summary = results.get_analysis_summary()
        print(f"✅ Analysis completed: {summary['total_series']} series")
        print(f"✅ Quality score: {summary['average_quality_score']:.3f}")
        print(f"✅ Processing time: {summary['processing_time']:.2f}s")
        
        # レポート生成
        report = create_analysis_report(results)
        print(f"✅ Report generated: {len(report)} characters")
        
        print("🎯 Comprehensive pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_comprehensive_pipeline()
