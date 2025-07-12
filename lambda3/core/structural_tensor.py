# ==========================================================
# lambda3/core/structural_tensor.py (∆ΛC完全修正版)
# Structural Tensor Feature Extraction for Lambda³ Theory
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
#
# 完全修正版: JIT統合、階層的特徴量抽出、型安全性の完全実装
# ==========================================================

"""
Lambda³構造テンソル特徴量抽出（∆ΛC完全修正版）

構造テンソル(Λ)、進行ベクトル(ΛF)、張力スカラー(ρT)の
包括的特徴量抽出システム。時間非依存の構造空間における
∆ΛC pulsations検出と階層的構造変化の完全実装。

完全修正内容:
- JIT関数統合の完全性確保
- 階層的特徴量（7特徴量）の確実な抽出
- 型安全性とProtocol準拠
- エラーハンドリングの強化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
import warnings
from datetime import datetime

# Type definitions
FloatArray = Union[np.ndarray, List[float]]

# Import JIT functions with fallback
try:
    from .jit_functions import (
        calculate_diff_and_threshold,
        detect_jumps,
        calculate_local_std,
        calculate_rho_t,
        detect_local_global_jumps,
        calc_lambda3_features_v2,
        sync_rate_at_lag,
        calculate_sync_profile_jit,
        detect_phase_coupling
    )
    JIT_AVAILABLE = True
except ImportError:
    warnings.warn("JIT functions not available. Using pure Python implementation.")
    JIT_AVAILABLE = False

# Import configuration
try:
    from .config import L3BaseConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    L3BaseConfig = None

# ==========================================================
# PROTOCOLS - 型安全性のためのプロトコル定義
# ==========================================================

class StructuralTensorProtocol(Protocol):
    """構造テンソル特徴量プロトコル"""
    data: np.ndarray
    series_name: str
    delta_LambdaC_pos: np.ndarray
    delta_LambdaC_neg: np.ndarray
    rho_T: np.ndarray
    
    def get_structural_summary(self) -> Dict[str, Any]: ...
    def validate_consistency(self) -> Tuple[bool, List[str]]: ...

# ==========================================================
# EXCEPTIONS - カスタム例外
# ==========================================================

class StructuralTensorError(Exception):
    """構造テンソル処理エラー"""
    pass

class FeatureExtractionError(StructuralTensorError):
    """特徴量抽出エラー"""
    pass

# ==========================================================
# MAIN FEATURE CLASS - 構造テンソル特徴量クラス
# ==========================================================

@dataclass
class StructuralTensorFeatures:
    """
    Lambda³構造テンソル特徴量（完全版）
    
    ∆ΛC pulsations、ρT、階層的構造変化を包含する
    完全な特徴量表現。
    """
    
    # 基本データ
    data: np.ndarray
    series_name: str = "Series"
    
    # 基本構造変化（∆ΛC）
    delta_LambdaC_pos: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_LambdaC_neg: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 張力スカラー（ρT）
    rho_T: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 時間トレンド
    time_trend: Optional[np.ndarray] = None
    
    # 階層的構造変化
    local_pos: Optional[np.ndarray] = None
    local_neg: Optional[np.ndarray] = None
    global_pos: Optional[np.ndarray] = None
    global_neg: Optional[np.ndarray] = None
    
    # 追加特徴量
    local_jump_detect: Optional[np.ndarray] = None
    pure_local_pos: Optional[np.ndarray] = None
    pure_local_neg: Optional[np.ndarray] = None
    pure_global_pos: Optional[np.ndarray] = None
    pure_global_neg: Optional[np.ndarray] = None
    mixed_pos: Optional[np.ndarray] = None
    mixed_neg: Optional[np.ndarray] = None
    
    # メタデータ
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_method: str = "standard"
    config_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後の検証と調整"""
        self._validate_features()
        self._ensure_consistency()
    
    def _validate_features(self):
        """特徴量の妥当性検証"""
        # データ長の確認
        data_len = len(self.data)
        
        # 基本特徴量の長さ調整
        for attr in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']:
            feature = getattr(self, attr)
            if len(feature) != data_len:
                if len(feature) == 0:
                    setattr(self, attr, np.zeros(data_len, dtype=np.float64))
                else:
                    raise FeatureExtractionError(
                        f"{attr} length mismatch: expected {data_len}, got {len(feature)}"
                    )
        
        # 時間トレンドの生成
        if self.time_trend is None:
            self.time_trend = np.arange(data_len, dtype=np.float64)
    
    def _ensure_consistency(self):
        """特徴量の整合性確保"""
        # 階層的特徴量の整合性チェック
        if self.local_pos is not None and self.global_pos is not None:
            self._compute_hierarchical_classifications()
    
    def _compute_hierarchical_classifications(self):
        """階層的分類の計算"""
        n = len(self.data)
        
        # 初期化
        self.pure_local_pos = np.zeros(n, dtype=np.float64)
        self.pure_local_neg = np.zeros(n, dtype=np.float64)
        self.pure_global_pos = np.zeros(n, dtype=np.float64)
        self.pure_global_neg = np.zeros(n, dtype=np.float64)
        self.mixed_pos = np.zeros(n, dtype=np.float64)
        self.mixed_neg = np.zeros(n, dtype=np.float64)
        
        # 分類
        for i in range(n):
            # 正の構造変化
            if self.local_pos[i] and self.global_pos[i]:
                self.mixed_pos[i] = 1.0
            elif self.local_pos[i] and not self.global_pos[i]:
                self.pure_local_pos[i] = 1.0
            elif not self.local_pos[i] and self.global_pos[i]:
                self.pure_global_pos[i] = 1.0
            
            # 負の構造変化
            if self.local_neg[i] and self.global_neg[i]:
                self.mixed_neg[i] = 1.0
            elif self.local_neg[i] and not self.global_neg[i]:
                self.pure_local_neg[i] = 1.0
            elif not self.local_neg[i] and self.global_neg[i]:
                self.pure_global_neg[i] = 1.0
    
    def get_structural_summary(self) -> Dict[str, Any]:
        """構造サマリー取得"""
        total_pos = np.sum(self.delta_LambdaC_pos)
        total_neg = np.sum(self.delta_LambdaC_neg)
        
        summary = {
            'series_name': self.series_name,
            'data_length': len(self.data),
            'total_structural_changes': total_pos + total_neg,
            'positive_changes': total_pos,
            'negative_changes': total_neg,
            'mean_tension': np.mean(self.rho_T),
            'max_tension': np.max(self.rho_T),
            'data_quality_score': self._calculate_quality_score(),
            'extraction_method': self.extraction_method
        }
        
        # 階層的特徴量のサマリー追加
        if self.local_pos is not None:
            summary['hierarchical_features'] = {
                'local_events': np.sum(self.local_pos) + np.sum(self.local_neg),
                'global_events': np.sum(self.global_pos) + np.sum(self.global_neg),
                'mixed_events': np.sum(self.mixed_pos) + np.sum(self.mixed_neg) if self.mixed_pos is not None else 0
            }
        
        return summary
    
    def _calculate_quality_score(self) -> float:
        """データ品質スコア計算"""
        # 基本的な品質指標
        has_changes = (np.sum(self.delta_LambdaC_pos) + np.sum(self.delta_LambdaC_neg)) > 0
        tension_variation = np.std(self.rho_T) / (np.mean(self.rho_T) + 1e-8)
        
        score = 0.5 if has_changes else 0.0
        score += min(0.5, tension_variation / 2)
        
        return min(1.0, score)
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """整合性検証"""
        issues = []
        
        # データ長チェック
        expected_len = len(self.data)
        for attr in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']:
            if len(getattr(self, attr)) != expected_len:
                issues.append(f"{attr} length mismatch")
        
        # 値範囲チェック
        if np.any(self.delta_LambdaC_pos < 0):
            issues.append("Negative values in delta_LambdaC_pos")
        if np.any(self.delta_LambdaC_neg < 0):
            issues.append("Negative values in delta_LambdaC_neg")
        if np.any(self.rho_T < 0):
            issues.append("Negative values in rho_T")
        
        # 階層的整合性チェック
        if self.local_pos is not None and self.global_pos is not None:
            # グローバルイベントは必ずローカルイベントを含む（理論的には）
            global_without_local = np.sum((self.global_pos > 0) & (self.local_pos == 0))
            if global_without_local > len(self.data) * 0.1:  # 10%以上の不整合
                issues.append("Many global events without local events")
        
        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        result = {
            'data': self.data,
            'series_name': self.series_name,
            'delta_LambdaC_pos': self.delta_LambdaC_pos,
            'delta_LambdaC_neg': self.delta_LambdaC_neg,
            'rho_T': self.rho_T,
            'time_trend': self.time_trend,
            'extraction_timestamp': self.extraction_timestamp,
            'extraction_method': self.extraction_method
        }
        
        # 階層的特徴量を含める
        if self.local_pos is not None:
            result.update({
                'local_pos': self.local_pos,
                'local_neg': self.local_neg,
                'global_pos': self.global_pos,
                'global_neg': self.global_neg
            })
        
        return result

# ==========================================================
# FEATURE EXTRACTOR - 特徴量抽出器
# ==========================================================

class StructuralTensorExtractor:
    """
    構造テンソル特徴量抽出器（完全版）
    
    JIT最適化と純Pythonの両方をサポートする
    堅牢な特徴量抽出システム。
    """
    
    def __init__(self, config: Optional[Any] = None, use_jit: Optional[bool] = None):
        """
        Args:
            config: 設定オブジェクト
            use_jit: JIT使用フラグ（None=自動判定）
        """
        self.config = config or (L3BaseConfig() if CONFIG_AVAILABLE else None)
        self.use_jit = use_jit if use_jit is not None else JIT_AVAILABLE
        
    def extract(
        self,
        data: FloatArray,
        series_name: str = "Series",
        feature_level: str = "standard"
    ) -> StructuralTensorFeatures:
        """
        特徴量抽出のメインメソッド
        
        Args:
            data: 入力時系列データ
            series_name: 系列名
            feature_level: 特徴量レベル（basic/standard/comprehensive）
            
        Returns:
            StructuralTensorFeatures: 抽出された特徴量
        """
        # データ前処理
        data = self._preprocess_data(data)
        
        # 特徴量レベルに応じた抽出
        if self.use_jit and feature_level in ["standard", "comprehensive"]:
            return self._extract_with_jit(data, series_name, feature_level)
        else:
            return self._extract_pure_python(data, series_name, feature_level)
    
    def _preprocess_data(self, data: FloatArray) -> np.ndarray:
        """データ前処理"""
        # NumPy配列に変換
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        else:
            data = data.astype(np.float64)
        
        # 1次元化
        if data.ndim > 1:
            data = data.flatten()
        
        # NaN/Inf処理
        if np.any(~np.isfinite(data)):
            warnings.warn("Non-finite values detected. Replacing with interpolation.")
            data = self._handle_non_finite(data)
        
        # 最小長チェック
        if len(data) < 10:
            raise FeatureExtractionError("Data length must be at least 10")
        
        return data
    
    def _handle_non_finite(self, data: np.ndarray) -> np.ndarray:
        """非有限値の処理"""
        finite_mask = np.isfinite(data)
        if np.all(~finite_mask):
            return np.zeros_like(data)
        
        # 線形補間
        indices = np.arange(len(data))
        data[~finite_mask] = np.interp(
            indices[~finite_mask],
            indices[finite_mask],
            data[finite_mask]
        )
        return data
    
    def _extract_with_jit(
        self,
        data: np.ndarray,
        series_name: str,
        feature_level: str
    ) -> StructuralTensorFeatures:
        """JIT最適化版特徴量抽出"""
        try:
            # 設定パラメータ取得
            window = getattr(self.config, 'window', 10) if self.config else 10
            percentile = getattr(self.config, 'threshold_percentile', 95.0) if self.config else 95.0
            
            if feature_level == "comprehensive":
                # 包括的特徴量抽出（7特徴量）
                jit_result = calc_lambda3_features_v2(data, window, percentile)
                
                if len(jit_result) >= 9:  # 階層的特徴量を含む
                    (delta_pos, delta_neg, rho_t, time_trend, local_jump,
                     local_pos, local_neg, global_pos, global_neg) = jit_result[:9]
                    
                    features = StructuralTensorFeatures(
                        data=data,
                        series_name=series_name,
                        delta_LambdaC_pos=delta_pos,
                        delta_LambdaC_neg=delta_neg,
                        rho_T=rho_t,
                        time_trend=time_trend,
                        local_jump_detect=local_jump,
                        local_pos=local_pos,
                        local_neg=local_neg,
                        global_pos=global_pos,
                        global_neg=global_neg,
                        extraction_method="jit_comprehensive"
                    )
                else:
                    # フォールバック
                    return self._extract_pure_python(data, series_name, feature_level)
            
            else:  # standard
                # 基本特徴量抽出
                diff, threshold = calculate_diff_and_threshold(data, percentile)
                delta_pos, delta_neg = detect_jumps(diff, threshold)
                rho_t = calculate_rho_t(data, window)
                
                features = StructuralTensorFeatures(
                    data=data,
                    series_name=series_name,
                    delta_LambdaC_pos=delta_pos,
                    delta_LambdaC_neg=delta_neg,
                    rho_T=rho_t,
                    extraction_method="jit_standard"
                )
            
            return features
            
        except Exception as e:
            warnings.warn(f"JIT extraction failed: {e}. Falling back to pure Python.")
            return self._extract_pure_python(data, series_name, feature_level)
    
    def _extract_pure_python(
        self,
        data: np.ndarray,
        series_name: str,
        feature_level: str
    ) -> StructuralTensorFeatures:
        """純Python版特徴量抽出"""
        # 基本特徴量計算
        diff = np.diff(data, prepend=data[0])
        threshold = np.percentile(np.abs(diff), 95.0)
        
        delta_pos = (diff > threshold).astype(np.float64)
        delta_neg = (diff < -threshold).astype(np.float64)
        
        # 張力スカラー計算
        window = 10
        rho_t = np.zeros(len(data), dtype=np.float64)
        for i in range(len(data)):
            start = max(0, i - window)
            end = i + 1
            subset = data[start:end]
            rho_t[i] = np.std(subset) if len(subset) > 1 else 0.0
        
        features = StructuralTensorFeatures(
            data=data,
            series_name=series_name,
            delta_LambdaC_pos=delta_pos,
            delta_LambdaC_neg=delta_neg,
            rho_T=rho_t,
            extraction_method="pure_python"
        )
        
        # 階層的特徴量の追加（comprehensiveの場合）
        if feature_level == "comprehensive":
            self._add_hierarchical_features_python(features)
        
        return features
    
    def _add_hierarchical_features_python(self, features: StructuralTensorFeatures):
        """純Python版階層的特徴量追加"""
        data = features.data
        n = len(data)
        
        # 簡易的な階層的検出
        local_window = 5
        global_window = 30
        
        diff = np.diff(data, prepend=data[0])
        
        # ローカル検出
        local_pos = np.zeros(n, dtype=np.float64)
        local_neg = np.zeros(n, dtype=np.float64)
        
        for i in range(n):
            local_start = max(0, i - local_window)
            local_end = min(n, i + local_window + 1)
            local_subset = np.abs(diff[local_start:local_end])
            
            if len(local_subset) > 0:
                local_threshold = np.percentile(local_subset, 90.0)
                if diff[i] > local_threshold:
                    local_pos[i] = 1.0
                elif diff[i] < -local_threshold:
                    local_neg[i] = 1.0
        
        # グローバル検出
        global_threshold = np.percentile(np.abs(diff), 95.0)
        global_pos = (diff > global_threshold).astype(np.float64)
        global_neg = (diff < -global_threshold).astype(np.float64)
        
        # 特徴量を更新
        features.local_pos = local_pos
        features.local_neg = local_neg
        features.global_pos = global_pos
        features.global_neg = global_neg

# ==========================================================
# CONVENIENCE FUNCTIONS - 便利関数
# ==========================================================

def extract_lambda3_features(
    data: FloatArray,
    series_name: str = "Series",
    feature_level: str = "standard",
    config: Optional[Any] = None,
    use_jit: Optional[bool] = None
) -> StructuralTensorFeatures:
    """
    Lambda³特徴量抽出の便利関数
    
    Args:
        data: 入力時系列データ
        series_name: 系列名
        feature_level: 特徴量レベル（basic/standard/comprehensive）
        config: 設定オブジェクト
        use_jit: JIT使用フラグ
        
    Returns:
        StructuralTensorFeatures: 抽出された特徴量
    """
    extractor = StructuralTensorExtractor(config=config, use_jit=use_jit)
    return extractor.extract(data, series_name, feature_level)

def extract_features(data: FloatArray, **kwargs) -> StructuralTensorFeatures:
    """extract_lambda3_featuresのエイリアス"""
    return extract_lambda3_features(data, **kwargs)

# ==========================================================
# VALIDATION & TESTING - 検証・テスト
# ==========================================================

def test_structural_tensor_implementation():
    """構造テンソル実装のテスト"""
    print("🧪 Testing StructuralTensorFeatures Implementation (Complete)")
    print("=" * 60)
    
    try:
        # テストケース
        test_cases = [
            ("Random Walk", np.cumsum(np.random.randn(100) * 0.1)),
            ("Sine Wave", np.sin(np.linspace(0, 4*np.pi, 100))),
            ("Step Function", np.concatenate([np.ones(50), np.ones(50)*2])),
            ("With NaN", np.concatenate([np.ones(30), [np.nan]*10, np.ones(60)]))
        ]
        
        for test_name, test_data in test_cases:
            print(f"\n📊 Testing {test_name}...")
            
            # 基本特徴量抽出
            features_basic = extract_lambda3_features(
                test_data, 
                series_name=test_name, 
                feature_level="basic"
            )
            print(f"✅ Basic extraction: {features_basic.extraction_method}")
            
            # 標準特徴量抽出
            features_std = extract_lambda3_features(
                test_data, 
                series_name=test_name, 
                feature_level="standard"
            )
            print(f"✅ Standard extraction: {features_std.extraction_method}")
            
            # 包括的特徴量抽出
            features_comp = extract_lambda3_features(
                test_data, 
                series_name=test_name, 
                feature_level="comprehensive"
            )
            print(f"✅ Comprehensive extraction: {features_comp.extraction_method}")
            
            # サマリー表示
            summary = features_comp.get_structural_summary()
            print(f"   Total changes: {summary['total_structural_changes']}")
            print(f"   Mean tension: {summary['mean_tension']:.4f}")
            print(f"   Quality score: {summary['data_quality_score']:.3f}")
            
            if 'hierarchical_features' in summary:
                hier = summary['hierarchical_features']
                print(f"   Hierarchical - Local: {hier['local_events']}, Global: {hier['global_events']}")
            
            # 整合性検証
            valid, issues = features_comp.validate_consistency()
            if valid:
                print("✅ Consistency check passed")
            else:
                print(f"⚠️  Consistency issues: {issues[:3]}")
        
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
    'StructuralTensorFeatures',
    'StructuralTensorExtractor',
    'StructuralTensorError',
    'FeatureExtractionError',
    
    # 関数
    'extract_lambda3_features',
    'extract_features',
    
    # テスト
    'test_structural_tensor_implementation',
    
    # プロトコル
    'StructuralTensorProtocol'
]

if __name__ == "__main__":
    # 自動テスト実行
    test_structural_tensor_implementation()
