# ==========================================================
# lambda3/core/structural_tensor.py 
# Structural Tensor Implementation for Lambda³ Theory
# ==========================================================

"""
Lambda³理論構造テンソル実装（修正版）

循環インポート問題を解決し、Protocol準拠を確保した
構造テンソル特徴量の具体実装。

修正点:
- types.pyからのProtocolインポート
- 循環依存の完全排除  
- 型安全性の強化
- JIT最適化関数との完全互換

Author: Masamichi Iizumi (Miosync, Inc.)
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
import time
from pathlib import Path

# 型定義のインポート（循環回避）
try:
    from .types import (
        StructuralTensorProtocol,
        ConfigProtocol,
        FloatArray,
        ArrayLike,
        LambdaTensorValue,
        DeltaLambdaC,
        RhoTensor,
        ensure_float_array,
        ensure_series_name,
        validate_array_like,
        StructuralTensorError,
        FeatureLevel
    )
    TYPES_AVAILABLE = True
except ImportError as e:
    # フォールバック：最小限の型定義
    TYPES_AVAILABLE = False
    FloatArray = np.ndarray
    ArrayLike = Union[np.ndarray, List[float]]
    StructuralTensorError = Exception
    warnings.warn(f"Types module not available: {e}")

# JIT最適化関数のインポート（オプション）
try:
    from .jit_functions import (
        calculate_diff_and_threshold_fixed,
        detect_structural_jumps_fixed,
        calculate_tension_scalar_fixed,
        detect_hierarchical_jumps_fixed,
        classify_hierarchical_events_fixed,
        extract_lambda3_features_jit,
        safe_divide_fixed,
        normalize_array_fixed
    )
    JIT_FUNCTIONS_AVAILABLE = True
    
    # レガシー互換性
    calculate_diff_and_threshold = calculate_diff_and_threshold_fixed
    detect_structural_jumps = detect_structural_jumps_fixed
    calculate_tension_scalar = calculate_tension_scalar_fixed
    
except ImportError as e:
    JIT_FUNCTIONS_AVAILABLE = False
    warnings.warn(f"JIT functions not available: {e}")

# 設定のインポート（循環回避）
try:
    from .config import L3BaseConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # ダミー設定クラス
    class L3BaseConfig:
        def __init__(self):
            self.window = 10
            self.threshold_percentile = 95.0
            self.enable_jit = JIT_FUNCTIONS_AVAILABLE

# ==========================================================
# 構造テンソル特徴量クラス（Protocol準拠）
# ==========================================================

@dataclass
class StructuralTensorFeatures:
    """
    Lambda³理論構造テンソル特徴量（修正版）
    
    StructuralTensorProtocolに準拠し、循環インポートを回避した
    構造テンソル特徴量の具体実装。
    
    構造テンソル(Λ)理論:
    - data: 原時系列データ（構造テンソル基底）
    - ∆ΛC: 構造変化パルス（正負分離）
    - ρT: 張力スカラー（構造空間応力）
    - 階層成分: 局所・大域構造変化
    """
    
    # 必須プロパティ
    data: FloatArray = field(default_factory=lambda: np.array([]))
    series_name: str = "Series"
    
    # 構造テンソル成分
    delta_LambdaC_pos: Optional[FloatArray] = None
    delta_LambdaC_neg: Optional[FloatArray] = None  
    rho_T: Optional[FloatArray] = None
    time_trend: Optional[FloatArray] = None
    
    # 階層的特徴（オプション）
    local_pos: Optional[FloatArray] = None
    local_neg: Optional[FloatArray] = None
    global_pos: Optional[FloatArray] = None
    global_neg: Optional[FloatArray] = None
    
    # メタデータ
    extraction_timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    feature_level: str = "basic"
    extraction_time: float = 0.0
    
    # 品質メトリクス
    data_quality_score: float = 0.0
    feature_completeness: float = 0.0
    jit_optimized: bool = False
    
    def __post_init__(self):
        """初期化後処理"""
        # データ型の安全確保
        if TYPES_AVAILABLE:
            self.data = ensure_float_array(self.data)
            self.series_name = ensure_series_name(self.series_name)
        else:
            self.data = np.asarray(self.data, dtype=np.float64)
            if not isinstance(self.series_name, str):
                self.series_name = str(self.series_name)
        
        # 基本検証
        if len(self.data) == 0:
            warnings.warn("Empty data provided to StructuralTensorFeatures")
        
        # 時間トレンドの自動生成
        if self.time_trend is None:
            self.time_trend = np.arange(len(self.data), dtype=np.float64)
        
        # データ品質評価
        self._evaluate_data_quality()
        
        # 特徴量完全性評価
        self._evaluate_feature_completeness()
    
    # ==========================================================
    # Protocol準拠メソッド
    # ==========================================================
    
    def get_data_length(self) -> int:
        """データ長取得"""
        return len(self.data)
    
    def get_total_structural_changes(self) -> int:
        """総構造変化数取得"""
        total = 0
        if self.delta_LambdaC_pos is not None:
            total += int(np.sum(self.delta_LambdaC_pos))
        if self.delta_LambdaC_neg is not None:
            total += int(np.sum(self.delta_LambdaC_neg))
        return total
    
    def get_average_tension(self) -> float:
        """平均張力取得"""
        if self.rho_T is not None and len(self.rho_T) > 0:
            return float(np.mean(self.rho_T))
        return 0.0
    
    # ==========================================================
    # 拡張メソッド
    # ==========================================================
    
    def get_structural_summary(self) -> Dict[str, Any]:
        """構造変化サマリー取得"""
        return {
            'series_name': self.series_name,
            'data_length': self.get_data_length(),
            'total_structural_changes': self.get_total_structural_changes(),
            'average_tension': self.get_average_tension(),
            'positive_changes': int(np.sum(self.delta_LambdaC_pos)) if self.delta_LambdaC_pos is not None else 0,
            'negative_changes': int(np.sum(self.delta_LambdaC_neg)) if self.delta_LambdaC_neg is not None else 0,
            'max_tension': float(np.max(self.rho_T)) if self.rho_T is not None else 0.0,
            'data_quality_score': self.data_quality_score,
            'feature_completeness': self.feature_completeness,
            'jit_optimized': self.jit_optimized
        }
    
    def get_hierarchical_summary(self) -> Dict[str, Any]:
        """階層構造サマリー取得"""
        local_events = 0
        global_events = 0
        
        if self.local_pos is not None:
            local_events += int(np.sum(self.local_pos))
        if self.local_neg is not None:
            local_events += int(np.sum(self.local_neg))
        if self.global_pos is not None:
            global_events += int(np.sum(self.global_pos))
        if self.global_neg is not None:
            global_events += int(np.sum(self.global_neg))
        
        total_hierarchical = local_events + global_events
        
        return {
            'local_events': local_events,
            'global_events': global_events,
            'total_hierarchical_events': total_hierarchical,
            'local_dominance': local_events / max(total_hierarchical, 1),
            'global_dominance': global_events / max(total_hierarchical, 1),
            'hierarchy_ratio': global_events / max(local_events, 1) if local_events > 0 else 0.0
        }
    
    def has_hierarchical_features(self) -> bool:
        """階層的特徴量の有無確認"""
        hierarchical_attrs = ['local_pos', 'local_neg', 'global_pos', 'global_neg']
        return any(getattr(self, attr) is not None for attr in hierarchical_attrs)
    
    def has_basic_features(self) -> bool:
        """基本特徴量の有無確認"""
        basic_attrs = ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']
        return any(getattr(self, attr) is not None for attr in basic_attrs)
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """特徴量整合性検証"""
        issues = []
        data_length = len(self.data)
        
        # 配列長の整合性チェック
        for attr_name in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend']:
            attr_value = getattr(self, attr_name)
            if attr_value is not None and len(attr_value) != data_length:
                issues.append(f"{attr_name} length mismatch: {len(attr_value)} vs {data_length}")
        
        # 階層的特徴量の整合性チェック
        for attr_name in ['local_pos', 'local_neg', 'global_pos', 'global_neg']:
            attr_value = getattr(self, attr_name)
            if attr_value is not None and len(attr_value) != data_length:
                issues.append(f"Hierarchical {attr_name} length mismatch: {len(attr_value)} vs {data_length}")
        
        # 数値範囲チェック
        if self.rho_T is not None and np.any(self.rho_T < 0):
            issues.append("rho_T contains negative values")
        
        # 構造変化の妥当性
        total_changes = self.get_total_structural_changes()
        if total_changes > data_length:
            issues.append(f"Too many structural changes: {total_changes} > {data_length}")
        
        return len(issues) == 0, issues
    
    # ==========================================================
    # 内部メソッド
    # ==========================================================
    
    def _evaluate_data_quality(self):
        """データ品質評価"""
        if len(self.data) == 0:
            self.data_quality_score = 0.0
            return
        
        # 品質スコア計算
        finite_ratio = np.sum(np.isfinite(self.data)) / len(self.data)
        variance_score = 1.0 if np.var(self.data) > 1e-10 else 0.5
        length_score = min(1.0, len(self.data) / 100)  # 100点以上で満点
        
        self.data_quality_score = (finite_ratio * 0.5 + variance_score * 0.3 + length_score * 0.2)
    
    def _evaluate_feature_completeness(self):
        """特徴量完全性評価"""
        total_features = 4  # delta_pos, delta_neg, rho_T, time_trend
        available_features = 0
        
        if self.delta_LambdaC_pos is not None:
            available_features += 1
        if self.delta_LambdaC_neg is not None:
            available_features += 1
        if self.rho_T is not None:
            available_features += 1
        if self.time_trend is not None:
            available_features += 1
        
        self.feature_completeness = available_features / total_features
    
    # ==========================================================
    # 可視化サポート
    # ==========================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換（可視化・保存用）"""
        result = {
            'series_name': self.series_name,
            'data': self.data.tolist() if self.data is not None else None,
            'extraction_timestamp': self.extraction_timestamp,
            'feature_level': self.feature_level,
            'data_quality_score': self.data_quality_score,
            'feature_completeness': self.feature_completeness,
            'jit_optimized': self.jit_optimized
        }
        
        # 特徴量配列の変換
        for attr_name in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend']:
            attr_value = getattr(self, attr_name)
            result[attr_name] = attr_value.tolist() if attr_value is not None else None
        
        # 階層的特徴量の変換
        for attr_name in ['local_pos', 'local_neg', 'global_pos', 'global_neg']:
            attr_value = getattr(self, attr_name)
            result[attr_name] = attr_value.tolist() if attr_value is not None else None
        
        return result
    
    def __repr__(self) -> str:
        """文字列表現"""
        summary = self.get_structural_summary()
        return (
            f"StructuralTensorFeatures("
            f"series='{summary['series_name']}', "
            f"length={summary['data_length']}, "
            f"changes={summary['total_structural_changes']}, "
            f"quality={summary['data_quality_score']:.3f}, "
            f"completeness={summary['feature_completeness']:.3f})"
        )

# ==========================================================
# 構造テンソル抽出器クラス
# ==========================================================

class StructuralTensorExtractor:
    """
    構造テンソル特徴量抽出器（修正版）
    
    原時系列データから構造テンソル特徴量を抽出する
    メインエンジン。JIT最適化と非最適化の両方に対応。
    """
    
    def __init__(self, config: Optional[Any] = None, use_jit: Optional[bool] = None):
        """
        Args:
            config: 設定オブジェクト（L3BaseConfig想定）
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
                    'global_window': 30,
                    'delta_percentile': 95.0,
                    'local_threshold_percentile': 85.0,
                    'global_threshold_percentile': 92.5
                })()
        else:
            self.config = config
        
        # JIT使用判定
        if use_jit is None:
            self.use_jit = JIT_FUNCTIONS_AVAILABLE and getattr(self.config, 'enable_jit', True)
        else:
            self.use_jit = use_jit and JIT_FUNCTIONS_AVAILABLE
        
        print(f"StructuralTensorExtractor initialized: JIT={self.use_jit}")
    
    def extract_features(
        self, 
        data: ArrayLike, 
        series_name: str = "Series",
        feature_level: str = "standard"
    ) -> StructuralTensorFeatures:
        """
        構造テンソル特徴量抽出
        
        Args:
            data: 入力時系列データ
            series_name: 系列名
            feature_level: 特徴量レベル ('basic', 'standard', 'comprehensive')
            
        Returns:
            StructuralTensorFeatures: 抽出された特徴量
        """
        start_time = time.time()
        
        # データ前処理
        if TYPES_AVAILABLE:
            processed_data = ensure_float_array(data)
            clean_name = ensure_series_name(series_name)
        else:
            processed_data = np.asarray(data, dtype=np.float64)
            clean_name = str(series_name) if series_name else "Series"
        
        # 入力検証
        if len(processed_data) < 3:
            raise StructuralTensorError(f"Data too short: {len(processed_data)} < 3")
        
        # 特徴量抽出実行
        try:
            if self.use_jit and feature_level in ['standard', 'comprehensive']:
                features = self._extract_with_jit(processed_data, clean_name, feature_level)
            else:
                features = self._extract_basic(processed_data, clean_name, feature_level)
            
            # 実行時間記録
            features.extraction_time = time.time() - start_time
            features.feature_level = feature_level
            features.jit_optimized = self.use_jit
            
            return features
            
        except Exception as e:
            raise StructuralTensorError(f"Feature extraction failed: {e}")
    
    def _extract_with_jit(
        self, 
        data: FloatArray, 
        series_name: str, 
        feature_level: str
    ) -> StructuralTensorFeatures:
        """JIT最適化特徴量抽出"""
        
        print(f"Extracting features with JIT optimization for {series_name}")
        
        if feature_level == 'comprehensive':
            # 包括特徴量（階層的解析含む）
            features_tuple = extract_lambda3_features_jit(
                data,
                window=self.config.window,
                local_window=getattr(self.config, 'local_window', 5),
                global_window=getattr(self.config, 'global_window', 30),
                delta_percentile=getattr(self.config, 'delta_percentile', 95.0),
                local_percentile=getattr(self.config, 'local_threshold_percentile', 85.0),
                global_percentile=getattr(self.config, 'global_threshold_percentile', 92.5)
            )
            
            # 包括特徴量の展開
            delta_pos, delta_neg, rho_t, local_pos, local_neg, global_pos, global_neg = features_tuple
            
            features = StructuralTensorFeatures(
                data=data,
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
            # 標準特徴量
            diff, threshold = calculate_diff_and_threshold_fixed(data, self.config.threshold_percentile)
            delta_pos, delta_neg = detect_structural_jumps_fixed(diff, threshold)
            rho_t = calculate_tension_scalar_fixed(data, self.config.window)
            
            features = StructuralTensorFeatures(
                data=data,
                series_name=series_name,
                delta_LambdaC_pos=delta_pos.astype(np.float64),
                delta_LambdaC_neg=delta_neg.astype(np.float64),
                rho_T=rho_t
            )
        
        return features
    
    def _extract_basic(
        self, 
        data: FloatArray, 
        series_name: str, 
        feature_level: str
    ) -> StructuralTensorFeatures:
        """基本特徴量抽出（Pure Python）"""
        
        print(f"Extracting features with Pure Python for {series_name}")
        
        # 基本差分計算
        diff = np.diff(data, prepend=data[0])
        
        # 構造変化検出
        threshold = np.percentile(np.abs(diff), self.config.threshold_percentile)
        delta_pos = (diff > threshold).astype(np.float64)
        delta_neg = (diff < -threshold).astype(np.float64)
        
        # 張力スカラー計算
        rho_t = np.zeros(len(data))
        window = self.config.window
        
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            rho_t[i] = np.std(data[start:end])
        
        features = StructuralTensorFeatures(
            data=data,
            series_name=series_name,
            delta_LambdaC_pos=delta_pos,
            delta_LambdaC_neg=delta_neg,
            rho_T=rho_t
        )
        
        # 包括レベルの場合は階層解析も実行
        if feature_level == 'comprehensive':
            features = self._add_hierarchical_features(features)
        
        return features
    
    def _add_hierarchical_features(self, features: StructuralTensorFeatures) -> StructuralTensorFeatures:
        """階層的特徴量の追加（Pure Python版）"""
        
        data = features.data
        n = len(data)
        
        # 短期・長期窓の設定
        local_window = getattr(self.config, 'local_window', 5)
        global_window = getattr(self.config, 'global_window', 30)
        
        # 階層的構造変化検出（簡易版）
        local_pos = np.zeros(n, dtype=np.float64)
        local_neg = np.zeros(n, dtype=np.float64)
        global_pos = np.zeros(n, dtype=np.float64)
        global_neg = np.zeros(n, dtype=np.float64)
        
        diff = np.diff(data, prepend=data[0])
        
        # 局所構造変化
        for i in range(n):
            start = max(0, i - local_window)
            end = min(n, i + local_window + 1)
            local_threshold = np.percentile(np.abs(diff[start:end]), 90.0)
            
            if diff[i] > local_threshold:
                local_pos[i] = 1.0
            elif diff[i] < -local_threshold:
                local_neg[i] = 1.0
        
        # 大域構造変化
        global_threshold = np.percentile(np.abs(diff), 95.0)
        global_pos = (diff > global_threshold).astype(np.float64)
        global_neg = (diff < -global_threshold).astype(np.float64)
        
        # 特徴量に追加
        features.local_pos = local_pos
        features.local_neg = local_neg
        features.global_pos = global_pos
        features.global_neg = global_neg
        
        return features
    
    def extract_hierarchical_features(
        self, 
        data: ArrayLike, 
        series_name: str = "Series"
    ) -> StructuralTensorFeatures:
        """階層的特徴量専用抽出（便利メソッド）"""
        return self.extract_features(data, series_name, feature_level='comprehensive')

# ==========================================================
# 便利関数（モジュールレベル）
# ==========================================================

def extract_lambda3_features(
    data: ArrayLike,
    config: Optional[Any] = None,
    series_name: str = "Series",
    feature_level: str = "standard",
    use_jit: Optional[bool] = None
) -> StructuralTensorFeatures:
    """
    Lambda³特徴量抽出の便利関数
    
    Args:
        data: 入力時系列データ
        config: 設定オブジェクト
        series_name: 系列名
        feature_level: 特徴量レベル
        use_jit: JIT最適化使用フラグ
        
    Returns:
        StructuralTensorFeatures: 抽出された特徴量
    """
    extractor = StructuralTensorExtractor(config=config, use_jit=use_jit)
    return extractor.extract_features(data, series_name, feature_level)

def create_sample_structural_tensor(
    n_points: int = 200,
    series_name: str = "Sample",
    add_jumps: bool = True,
    random_seed: int = 42
) -> StructuralTensorFeatures:
    """
    サンプル構造テンソル特徴量生成
    
    テスト・デモ用のサンプルデータと特徴量を生成。
    """
    np.random.seed(random_seed)
    
    # 基本時系列生成
    trend = np.cumsum(np.random.randn(n_points) * 0.02)
    
    # 構造変化ジャンプ追加
    if add_jumps:
        jump_positions = np.random.choice(n_points, size=max(1, n_points // 30), replace=False)
        jumps = np.zeros(n_points)
        jumps[jump_positions] = np.random.normal(0, 1.0, len(jump_positions))
        trend += np.cumsum(jumps)
    
    # 特徴量抽出
    return extract_lambda3_features(trend, series_name=series_name, feature_level='comprehensive')

# ==========================================================
# モジュール情報
# ==========================================================

__all__ = [
    'StructuralTensorFeatures',
    'StructuralTensorExtractor', 
    'extract_lambda3_features',
    'create_sample_structural_tensor',
    'StructuralTensorError'
]

# ==========================================================
# テスト関数
# ==========================================================

def test_structural_tensor_implementation():
    """構造テンソル実装のテスト"""
    print("🧪 Testing StructuralTensorFeatures Implementation")
    print("=" * 50)
    
    # サンプルデータ生成
    sample_data = np.cumsum(np.random.randn(100) * 0.1)
    
    try:
        # 基本特徴量抽出
        features = extract_lambda3_features(sample_data, series_name="Test", feature_level="basic")
        print(f"✅ Basic extraction: {features}")
        
        # Protocol準拠確認
        if TYPES_AVAILABLE:
            from .types import is_structural_tensor_compatible
            is_compatible = is_structural_tensor_compatible(features)
            print(f"✅ Protocol compatibility: {'OK' if is_compatible else 'NG'}")
        
        # 整合性検証
        valid, issues = features.validate_consistency()
        print(f"✅ Consistency check: {'OK' if valid else 'NG'}")
        if issues:
            for issue in issues:
                print(f"   Warning: {issue}")
        
        # サマリー取得
        summary = features.get_structural_summary()
        print(f"✅ Summary: {summary}")
        
        print("🎯 StructuralTensorFeatures test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_structural_tensor_implementation()
