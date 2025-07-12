# ==========================================================
# lambda3/core/types.py
# Lambda³ Common Type Definitions 
# ==========================================================

"""
Lambda³理論共通型定義モジュール

循環インポート問題の根本解決のため、全ての共通型を一箇所に集約。
Protocol、TypedDict、Unionを活用した型安全性確保。

構造テンソル(Λ)理論的意義:
- 型定義 = 構造テンソル空間の位相的制約
- Protocol = 構造テンソルの最小契約
- 循環回避 = ∆ΛC変化の因果順序調整

Author: Masamichi Iizumi (Miosync, Inc.)
License: MIT
"""

from typing import (
    Protocol, TypedDict, Union, Dict, List, Tuple, Optional, Any, 
    runtime_checkable, Literal, Callable, Iterator
)
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from enum import Enum

# ==========================================================
# 核心データ型定義
# ==========================================================

# NumPy配列型エイリアス
ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]
FloatArray = np.ndarray  # float64想定
IntArray = np.ndarray    # int32/int64想定
BoolArray = np.ndarray   # bool想定

# Lambda³固有型
LambdaTensorValue = float      # 構造テンソル値
DeltaLambdaC = float          # ∆ΛC構造変化値  
RhoTensor = float             # ρT張力スカラー値
ProgressionVector = float     # ΛF進行ベクトル値

# ==========================================================
# 構造テンソル特徴量Protocol（最重要）
# ==========================================================

@runtime_checkable
class StructuralTensorProtocol(Protocol):
    """
    構造テンソル特徴量の最小契約
    
    Lambda³理論の核心要素である構造テンソル特徴量の
    インターフェース定義。全ての実装がこれに準拠する必要がある。
    """
    
    # 必須プロパティ
    data: FloatArray              # 原時系列データ
    series_name: str              # 系列名
    
    # 構造テンソル成分（Optional：段階的実装可能）
    delta_LambdaC_pos: Optional[FloatArray]    # ∆ΛC⁺ 正構造変化
    delta_LambdaC_neg: Optional[FloatArray]    # ∆ΛC⁻ 負構造変化
    rho_T: Optional[FloatArray]                # ρT 張力スカラー
    time_trend: Optional[FloatArray]           # 時間トレンド
    
    # 階層的特徴（Optional：拡張機能）
    local_pos: Optional[FloatArray]            # 局所正変化
    local_neg: Optional[FloatArray]            # 局所負変化
    global_pos: Optional[FloatArray]           # 大域正変化
    global_neg: Optional[FloatArray]           # 大域負変化
    
    def get_data_length(self) -> int:
        """データ長取得"""
        ...
    
    def get_total_structural_changes(self) -> int:
        """総構造変化数取得"""
        ...
    
    def get_average_tension(self) -> float:
        """平均張力取得"""
        ...

# ==========================================================
# 階層分析結果Protocol
# ==========================================================

@runtime_checkable  
class HierarchicalResultProtocol(Protocol):
    """
    階層分析結果の最小契約
    
    局所-大域構造変化の分離分析結果に対する
    統一インターフェース。
    """
    
    # 基本識別子
    series_name: str
    analysis_timestamp: str
    
    # 階層分離係数
    escalation_strength: float           # エスカレーション強度
    deescalation_strength: float         # デエスカレーション強度  
    hierarchy_correlation: float         # 階層間相関
    
    # 品質メトリクス
    convergence_quality: float           # 収束品質
    statistical_significance: float      # 統計的有意性
    
    def get_separation_summary(self) -> Dict[str, float]:
        """分離サマリー取得"""
        ...
    
    def get_dominant_hierarchy(self) -> Literal['local', 'global', 'balanced']:
        """優勢階層判定"""
        ...

# ==========================================================
# ペアワイズ分析結果Protocol
# ==========================================================

@runtime_checkable
class PairwiseResultProtocol(Protocol):
    """
    ペアワイズ相互作用分析結果の最小契約
    
    二系列間の非対称相互作用分析に対する
    統一インターフェース。
    """
    
    # 系列識別子
    name_a: str
    name_b: str
    analysis_timestamp: str
    
    # 同期性指標
    synchronization_strength: float      # 同期強度
    structure_synchronization: float     # 構造変化同期
    
    # 因果性指標  
    causality_a_to_b: float             # A→B因果強度
    causality_b_to_a: float             # B→A因果強度
    asymmetry_index: float              # 非対称性指標
    
    # データ品質
    data_overlap_length: int            # データ重複長
    correlation_quality: float          # 相関品質
    
    def get_interaction_summary(self) -> Dict[str, float]:
        """相互作用サマリー取得"""
        ...
    
    def get_dominant_direction(self) -> Literal['a_to_b', 'b_to_a', 'symmetric']:
        """優勢方向判定"""
        ...

# ==========================================================
# 設定系Protocol
# ==========================================================

@runtime_checkable
class ConfigProtocol(Protocol):
    """設定オブジェクトの最小契約"""
    
    # 基本パラメータ
    window: int                          # 窓幅
    threshold_percentile: float          # 閾値パーセンタイル
    enable_jit: bool                     # JIT最適化有効フラグ
    
    def validate(self) -> bool:
        """設定検証"""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        ...

# ==========================================================
# 分析器Protocol（抽象基底）
# ==========================================================

@runtime_checkable  
class AnalyzerProtocol(Protocol):
    """分析器の最小契約"""
    
    config: ConfigProtocol
    use_jit: bool
    
    def analyze(self, features: StructuralTensorProtocol) -> Any:
        """分析実行"""
        ...
    
    def validate_input(self, features: StructuralTensorProtocol) -> bool:
        """入力検証"""
        ...

# ==========================================================
# 結果統合Protocol
# ==========================================================

@runtime_checkable
class ComprehensiveResultProtocol(Protocol):
    """包括分析結果の最小契約"""
    
    # メタデータ
    analysis_timestamp: str
    total_series: int
    
    # 核心結果
    structural_features: Dict[str, StructuralTensorProtocol]
    hierarchical_results: Dict[str, HierarchicalResultProtocol]
    pairwise_results: Dict[str, PairwiseResultProtocol]
    
    # 統合メトリクス
    overall_quality_score: float
    processing_time: float
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリー取得"""
        ...
    
    def export_results(self, format: Literal['json', 'csv', 'excel']) -> str:
        """結果エクスポート"""
        ...

# ==========================================================
# データフロー型定義
# ==========================================================

class AnalysisMode(Enum):
    """分析モード列挙"""
    STRUCTURAL_ONLY = "structural_only"      # 構造テンソルのみ
    HIERARCHICAL = "hierarchical"            # 階層分析
    PAIRWISE = "pairwise"                    # ペアワイズ分析
    COMPREHENSIVE = "comprehensive"          # 包括分析
    FINANCIAL = "financial"                  # 金融特化
    RAPID = "rapid"                          # 高速分析

class FeatureLevel(Enum):
    """特徴量レベル列挙"""
    BASIC = "basic"                          # 基本特徴量
    STANDARD = "standard"                    # 標準特徴量
    COMPREHENSIVE = "comprehensive"          # 包括特徴量
    RESEARCH = "research"                    # 研究レベル

class QualityLevel(Enum):
    """品質レベル列挙"""
    LOW = "low"                             # 低品質（高速）
    MEDIUM = "medium"                       # 中品質（標準）
    HIGH = "high"                           # 高品質（精密）
    RESEARCH = "research"                   # 研究品質（最高精度）

# ==========================================================
# エラー・例外型定義
# ==========================================================

class Lambda3Error(Exception):
    """Lambda³ 基底例外クラス"""
    pass

class StructuralTensorError(Lambda3Error):
    """構造テンソル関連エラー"""
    pass

class HierarchicalAnalysisError(Lambda3Error):
    """階層分析エラー"""
    pass

class PairwiseAnalysisError(Lambda3Error):
    """ペアワイズ分析エラー"""
    pass

class ConfigurationError(Lambda3Error):
    """設定エラー"""
    pass

class JITOptimizationError(Lambda3Error):
    """JIT最適化エラー"""
    pass

# ==========================================================
# TypedDict 定義（辞書型データ用）
# ==========================================================

class AnalysisConfig(TypedDict, total=False):
    """分析設定の型定義（辞書形式）"""
    analysis_mode: AnalysisMode
    feature_level: FeatureLevel
    quality_level: QualityLevel
    enable_jit: bool
    enable_bayesian: bool
    enable_visualization: bool
    output_format: Literal['json', 'pickle', 'csv']

class DataInfo(TypedDict, total=False):
    """データ情報の型定義"""
    total_series: int
    series_names: List[str]
    data_lengths: Dict[str, int]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    processing_notes: List[str]

class PerformanceMetrics(TypedDict, total=False):
    """性能メトリクスの型定義"""
    total_time: float
    feature_extraction_time: float
    hierarchical_analysis_time: float
    pairwise_analysis_time: float
    jit_speedup_ratio: float
    memory_usage_mb: float

# ==========================================================
# ファクトリー関数型定義
# ==========================================================

StructuralTensorFactory = Callable[[FloatArray, str], StructuralTensorProtocol]
HierarchicalAnalyzerFactory = Callable[[ConfigProtocol], AnalyzerProtocol]  
PairwiseAnalyzerFactory = Callable[[ConfigProtocol], AnalyzerProtocol]

# ==========================================================
# ユーティリティ型チェック関数
# ==========================================================

def is_structural_tensor_compatible(obj: Any) -> bool:
    """オブジェクトがStructuralTensorProtocolに準拠するかチェック"""
    try:
        return isinstance(obj, StructuralTensorProtocol)
    except (TypeError, AttributeError):
        return False

def is_hierarchical_result_compatible(obj: Any) -> bool:
    """オブジェクトがHierarchicalResultProtocolに準拠するかチェック"""
    try:
        return isinstance(obj, HierarchicalResultProtocol)
    except (TypeError, AttributeError):
        return False

def is_pairwise_result_compatible(obj: Any) -> bool:
    """オブジェクトがPairwiseResultProtocolに準拠するかチェック"""
    try:
        return isinstance(obj, PairwiseResultProtocol)
    except (TypeError, AttributeError):
        return False

def validate_array_like(data: Any, min_length: int = 1) -> bool:
    """配列様オブジェクトの検証"""
    try:
        arr = np.asarray(data)
        return len(arr) >= min_length and np.isfinite(arr).any()
    except (ValueError, TypeError):
        return False

def ensure_float_array(data: Any) -> FloatArray:
    """float64配列への安全変換"""
    try:
        return np.asarray(data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise Lambda3Error(f"Cannot convert to float array: {e}")

def ensure_series_name(name: Any) -> str:
    """系列名の安全確保"""
    if isinstance(name, str) and name.strip():
        return name.strip()
    elif name is None:
        return "Series"
    else:
        return str(name)

# ==========================================================
# モジュール情報
# ==========================================================

__all__ = [
    # 核心Protocol
    'StructuralTensorProtocol',
    'HierarchicalResultProtocol', 
    'PairwiseResultProtocol',
    'ComprehensiveResultProtocol',
    'ConfigProtocol',
    'AnalyzerProtocol',
    
    # データ型
    'ArrayLike', 'FloatArray', 'IntArray', 'BoolArray',
    'LambdaTensorValue', 'DeltaLambdaC', 'RhoTensor', 'ProgressionVector',
    
    # 列挙型
    'AnalysisMode', 'FeatureLevel', 'QualityLevel',
    
    # TypedDict
    'AnalysisConfig', 'DataInfo', 'PerformanceMetrics',
    
    # 例外
    'Lambda3Error', 'StructuralTensorError', 'HierarchicalAnalysisError',
    'PairwiseAnalysisError', 'ConfigurationError', 'JITOptimizationError',
    
    # ユーティリティ
    'is_structural_tensor_compatible', 'is_hierarchical_result_compatible',
    'is_pairwise_result_compatible', 'validate_array_like', 
    'ensure_float_array', 'ensure_series_name'
]

# ================================================================
# 型システム整合性チェック
# ================================================================

def _validate_type_system():
    """型システムの整合性検証（開発用）"""
    print("🔍 Lambda³ Type System Validation")
    print("=" * 40)
    
    # Protocol可用性チェック
    try:
        # ダミーオブジェクトでProtocol準拠をテスト
        class DummyStructuralTensor:
            def __init__(self):
                self.data = np.array([1.0, 2.0, 3.0])
                self.series_name = "test"
                self.delta_LambdaC_pos = None
                self.delta_LambdaC_neg = None
                self.rho_T = None
                self.time_trend = None
                self.local_pos = None
                self.local_neg = None
                self.global_pos = None
                self.global_neg = None
            
            def get_data_length(self) -> int:
                return len(self.data)
            
            def get_total_structural_changes(self) -> int:
                return 0
            
            def get_average_tension(self) -> float:
                return 0.0
        
        dummy = DummyStructuralTensor()
        is_compatible = is_structural_tensor_compatible(dummy)
        print(f"✅ StructuralTensorProtocol: {'OK' if is_compatible else 'NG'}")
        
    except Exception as e:
        print(f"❌ Protocol validation failed: {e}")
    
    # 型変換チェック
    try:
        test_data = [1, 2, 3, 4, 5]
        float_arr = ensure_float_array(test_data)
        print(f"✅ Array conversion: {float_arr.dtype}")
    except Exception as e:
        print(f"❌ Array conversion failed: {e}")
    
    print("🎯 Type system validation complete!")

if __name__ == "__main__":
    _validate_type_system()
