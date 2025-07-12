# ==========================================================
# lambda3/core/structural_tensor.py
# Structural Tensor Feature Extraction for Lambda³ Theory (修正版)
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³構造テンソル特徴量抽出（完全修正版）

構造テンソル(Λ)、進行ベクトル(ΛF)、張力スカラー(ρT)の
包括的特徴量抽出システム。階層的構造変化の完全実装。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
from datetime import datetime

# Import JIT functions
try:
    from .jit_functions import (
        calculate_diff_and_threshold,
        detect_jumps,
        calculate_local_std,
        calculate_rho_t,
        detect_local_global_jumps,
        calc_lambda3_features_v2
    )
    JIT_AVAILABLE = True
except ImportError:
    warnings.warn("JIT functions not available. Using fallback implementation.")
    JIT_AVAILABLE = False

# Import configuration
try:
    from .config import L3Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    L3Config = None

# ==========================================================
# STRUCTURAL TENSOR FEATURES - 構造テンソル特徴量
# ==========================================================
@dataclass
class StructuralTensorFeatures:
    """
    Lambda³構造テンソル特徴量（型整合性強化版）
    
    ∆ΛC pulsations、ρT、階層的構造変化を包含する
    完全な特徴量表現。全配列のfloat64統一を保証。
    """
    
    # 基本データ（明示的float64）
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    series_name: str = "Series"
    
    # 基本構造変化（∆ΛC）- float64保証
    delta_LambdaC_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    delta_LambdaC_neg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # 張力スカラー（ρT）
    rho_T: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # 時間トレンド（float64統一）
    time_trend: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # 階層的構造変化（全てfloat64初期化）
    local_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    local_neg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    global_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    global_neg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # 階層的純粋成分（全てfloat64初期化）
    pure_local_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pure_local_neg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pure_global_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    pure_global_neg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    mixed_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    mixed_neg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # 追加特徴量
    local_jump_detect: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後の型変換とサイズ調整（強化版）"""
        # 基本データの型変換（入力データも強制的にfloat64へ）
        if isinstance(self.data, np.ndarray):
            self.data = self.data.astype(np.float64, copy=False)
        else:
            self.data = np.asarray(self.data, dtype=np.float64)
        
        # 全配列フィールドの型統一処理
        array_fields = [
            'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend',
            'local_pos', 'local_neg', 'global_pos', 'global_neg',
            'pure_local_pos', 'pure_local_neg', 'pure_global_pos', 'pure_global_neg',
            'mixed_pos', 'mixed_neg', 'local_jump_detect'
        ]
        
        for field_name in array_fields:
            value = getattr(self, field_name)
            
            # 配列の型変換と検証
            if value is not None:
                if isinstance(value, np.ndarray):
                    # すでにndarrayの場合、float64に変換
                    if value.dtype != np.float64:
                        setattr(self, field_name, value.astype(np.float64, copy=False))
                elif isinstance(value, (list, tuple)):
                    # リストやタプルの場合、float64配列に変換
                    setattr(self, field_name, np.array(value, dtype=np.float64))
                elif len(value) == 0:
                    # 空配列の場合も型を保証
                    setattr(self, field_name, np.array([], dtype=np.float64))
        
        # 時間トレンドの自動生成（必ずfloat64）
        if len(self.time_trend) == 0:
            self.time_trend = np.arange(len(self.data), dtype=np.float64)
        
        # データ長に基づくサイズ調整（オプション）
        self._adjust_array_sizes()
    
    def _adjust_array_sizes(self):
        """配列サイズの自動調整"""
        n = len(self.data)
        
        # 基本特徴量のサイズ調整
        for field_name in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']:
            arr = getattr(self, field_name)
            if len(arr) == 0:
                # 空の場合は適切なサイズで初期化
                setattr(self, field_name, np.zeros(n, dtype=np.float64))
            elif len(arr) != n:
                # サイズ不一致の場合は警告（自動調整は危険なので行わない）
                self.metadata.setdefault('warnings', []).append(
                    f"{field_name} size mismatch: {len(arr)} != {n}"
                )
    
    def ensure_type_consistency(self) -> 'StructuralTensorFeatures':
        """型整合性を強制的に確保（チェインメソッド用）"""
        self.__post_init__()
        return self
    
    def get_structural_summary(self) -> Dict[str, Any]:
        """構造テンソル特性の要約統計（型安全版）"""
        summary = {
            'series_name': self.series_name,
            'data_length': len(self.data),
            'data_dtype': str(self.data.dtype),
            'total_pos_jumps': float(np.sum(self.delta_LambdaC_pos)),
            'total_neg_jumps': float(np.sum(self.delta_LambdaC_neg)),
            'mean_tension': float(np.mean(self.rho_T)) if len(self.rho_T) > 0 else 0.0,
            'max_tension': float(np.max(self.rho_T)) if len(self.rho_T) > 0 else 0.0,
        }
        
        # 階層的特徴量の統計（型安全）
        if len(self.local_pos) > 0:
            summary['local_pos_events'] = float(np.sum(self.local_pos))
            summary['local_neg_events'] = float(np.sum(self.local_neg))
        if len(self.global_pos) > 0:
            summary['global_pos_events'] = float(np.sum(self.global_pos))
            summary['global_neg_events'] = float(np.sum(self.global_neg))
        
        # 階層的純粋成分の統計
        if len(self.pure_local_pos) > 0:
            summary['pure_local_events'] = float(
                np.sum(self.pure_local_pos) + np.sum(self.pure_local_neg)
            )
            summary['pure_global_events'] = float(
                np.sum(self.pure_global_pos) + np.sum(self.pure_global_neg)
            )
            summary['mixed_events'] = float(
                np.sum(self.mixed_pos) + np.sum(self.mixed_neg)
            )
            
        return summary
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """特徴量の整合性検証（強化版）"""
        errors = []
        warnings = []
        n = len(self.data)
        
        # 型チェック（全配列）
        array_fields = {
            'data': self.data,
            'delta_LambdaC_pos': self.delta_LambdaC_pos,
            'delta_LambdaC_neg': self.delta_LambdaC_neg,
            'rho_T': self.rho_T,
            'time_trend': self.time_trend
        }
        
        for name, arr in array_fields.items():
            if isinstance(arr, np.ndarray) and len(arr) > 0:
                if arr.dtype != np.float64:
                    errors.append(f"{name} dtype is {arr.dtype}, expected float64")
        
        # サイズ整合性チェック（必須フィールド）
        for name, arr in [
            ('delta_LambdaC_pos', self.delta_LambdaC_pos),
            ('delta_LambdaC_neg', self.delta_LambdaC_neg),
            ('rho_T', self.rho_T)
        ]:
            if len(arr) > 0 and len(arr) != n:
                errors.append(f"{name} size mismatch: {len(arr)} != {n}")
        
        # 値範囲チェック（∆ΛCは0または1）
        if len(self.delta_LambdaC_pos) > 0:
            unique_values = np.unique(self.delta_LambdaC_pos)
            if not np.all(np.isin(unique_values, [0.0, 1.0])):
                warnings.append("delta_LambdaC_pos contains non-binary values")
                
        if len(self.delta_LambdaC_neg) > 0:
            unique_values = np.unique(self.delta_LambdaC_neg)
            if not np.all(np.isin(unique_values, [0.0, 1.0])):
                warnings.append("delta_LambdaC_neg contains non-binary values")
        
        # NaN/Inf チェック
        for name, arr in array_fields.items():
            if isinstance(arr, np.ndarray) and len(arr) > 0:
                if np.isnan(arr).any():
                    errors.append(f"{name} contains NaN values")
                if np.isinf(arr).any():
                    errors.append(f"{name} contains Inf values")
        
        # 階層的特徴量の整合性チェック（存在する場合）
        if len(self.local_pos) > 0 and len(self.global_pos) > 0:
            # 純粋成分と混合成分の合計が元の成分と一致するか
            if len(self.pure_local_pos) > 0 and len(self.mixed_pos) > 0:
                local_total = self.pure_local_pos + self.mixed_pos
                expected_local = self.local_pos * (1 - self.global_pos)
                if not np.allclose(local_total, expected_local, atol=1e-10):
                    warnings.append("Hierarchical component consistency issue")
        
        # 警告をメタデータに保存
        if warnings:
            self.metadata['validation_warnings'] = warnings
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式への変換（型保証付き）"""
        result = {
            'series_name': self.series_name,
            'data': self.data.tolist(),
            'metadata': self.metadata
        }
        
        # 全配列フィールドを含める
        array_fields = [
            'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend',
            'local_pos', 'local_neg', 'global_pos', 'global_neg',
            'pure_local_pos', 'pure_local_neg', 'pure_global_pos', 'pure_global_neg',
            'mixed_pos', 'mixed_neg', 'local_jump_detect'
        ]
        
        for field_name in array_fields:
            arr = getattr(self, field_name)
            if isinstance(arr, np.ndarray) and len(arr) > 0:
                result[field_name] = arr.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuralTensorFeatures':
        """辞書からの復元（型保証付き）"""
        # 配列フィールドをnumpy配列に変換
        array_fields = [
            'data', 'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'time_trend',
            'local_pos', 'local_neg', 'global_pos', 'global_neg',
            'pure_local_pos', 'pure_local_neg', 'pure_global_pos', 'pure_global_neg',
            'mixed_pos', 'mixed_neg', 'local_jump_detect'
        ]
        
        kwargs = {}
        for key, value in data.items():
            if key in array_fields and isinstance(value, list):
                kwargs[key] = np.array(value, dtype=np.float64)
            else:
                kwargs[key] = value
        
        return cls(**kwargs)

# ==========================================================
# FEATURE EXTRACTION - 特徴量抽出
# ==========================================================

class StructuralTensorExtractor:
    """構造テンソル特徴量抽出器"""
    
    def __init__(self, config: Optional[L3Config] = None):
        """
        Args:
            config: Lambda³設定オブジェクト
        """
        self.config = config or (L3Config() if CONFIG_AVAILABLE else None)
        self.jit_available = JIT_AVAILABLE
        
    def extract_features(
        self,
        data: Union[np.ndarray, List[float]],
        series_name: str = "Series",
        feature_level: str = 'basic'
    ) -> StructuralTensorFeatures:
        """
        構造テンソル特徴量を抽出
        
        Args:
            data: 入力時系列データ
            series_name: 系列名
            feature_level: 特徴量レベル ('basic', 'hierarchical', 'comprehensive')
            
        Returns:
            StructuralTensorFeatures: 抽出された特徴量
        """
        # データ準備
        data = np.asarray(data, dtype=np.float64)
        
        if len(data) < 10:
            raise ValueError(f"Data too short: {len(data)} < 10")
        
        # 特徴量抽出
        if feature_level == 'basic':
            return self._extract_basic_features(data, series_name)
        elif feature_level == 'hierarchical':
            return self._extract_hierarchical_features(data, series_name)
        elif feature_level == 'comprehensive':
            return self._extract_comprehensive_features(data, series_name)
        else:
            raise ValueError(f"Unknown feature_level: {feature_level}")
    
    def _extract_basic_features(
        self,
        data: np.ndarray,
        series_name: str
    ) -> StructuralTensorFeatures:
        """基本特徴量抽出"""
        if self.jit_available and self.config:
            # JIT版
            delta_pos, delta_neg, rho_t, time_trend, local_jump = calc_lambda3_features_v2(
                data,
                window=self.config.window,
                percentile=self.config.delta_percentile
            )
        else:
            # フォールバック実装
            delta_pos, delta_neg, rho_t, time_trend = self._fallback_basic_features(data)
            local_jump = np.zeros_like(data)
        
        return StructuralTensorFeatures(
            data=data,
            series_name=series_name,
            delta_LambdaC_pos=delta_pos,
            delta_LambdaC_neg=delta_neg,
            rho_T=rho_t,
            time_trend=time_trend,
            local_jump_detect=local_jump,
            metadata={'extraction_level': 'basic'}
        )
    
   def _extract_hierarchical_features(
        self,
        data: np.ndarray,
        series_name: str
    ) -> StructuralTensorFeatures:
        """階層的特徴量抽出"""
        # 基本特徴量を先に抽出
        basic_features = self._extract_basic_features(data, series_name)
        
        if self.jit_available and self.config:
            # 階層的構造変化検出（paste.txt準拠）
            local_pos, local_neg, global_pos, global_neg = detect_local_global_jumps(
                data,
                local_window=self.config.local_window,
                global_window=self.config.global_window,
                local_percentile=self.config.local_threshold_percentile,
                global_percentile=self.config.global_threshold_percentile
            )
            
            # 階層的純粋成分の計算
            pure_local_pos, pure_local_neg, pure_global_pos, pure_global_neg, mixed_pos, mixed_neg = \
                self._calculate_hierarchical_components(
                    local_pos, local_neg, global_pos, global_neg
                )
            
            # 特徴量更新（全てfloat64保証）
            basic_features.local_pos = local_pos.astype(np.float64)
            basic_features.local_neg = local_neg.astype(np.float64)
            basic_features.global_pos = global_pos.astype(np.float64)
            basic_features.global_neg = global_neg.astype(np.float64)
            basic_features.pure_local_pos = pure_local_pos
            basic_features.pure_local_neg = pure_local_neg
            basic_features.pure_global_pos = pure_global_pos
            basic_features.pure_global_neg = pure_global_neg
            basic_features.mixed_pos = mixed_pos
            basic_features.mixed_neg = mixed_neg
            basic_features.metadata['extraction_level'] = 'hierarchical'
        
        return basic_features
    
    def _extract_comprehensive_features(
        self,
        data: np.ndarray,
        series_name: str
    ) -> StructuralTensorFeatures:
        """包括的特徴量抽出（全特徴量）"""
        # 階層的特徴量を抽出（基本特徴量も含む）
        features = self._extract_hierarchical_features(data, series_name)
        
        # 追加の高度特徴量
        features.metadata.update({
            'extraction_level': 'comprehensive',
            'extraction_timestamp': datetime.now().isoformat(),
            'config_used': self.config.to_dict() if self.config else {}
        })
        
        return features
    
    def _calculate_hierarchical_components(
        self,
        local_pos: np.ndarray,
        local_neg: np.ndarray,
        global_pos: np.ndarray,
        global_neg: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """階層的純粋成分の計算"""
        n = len(local_pos)
        
        # 初期化（float64保証）
        pure_local_pos = np.zeros(n, dtype=np.float64)
        pure_local_neg = np.zeros(n, dtype=np.float64)
        pure_global_pos = np.zeros(n, dtype=np.float64)
        pure_global_neg = np.zeros(n, dtype=np.float64)
        mixed_pos = np.zeros(n, dtype=np.float64)
        mixed_neg = np.zeros(n, dtype=np.float64)
        
        # 成分分離
        for i in range(n):
            # 正の構造変化
            if local_pos[i] and global_pos[i]:
                mixed_pos[i] = 1.0
            elif local_pos[i] and not global_pos[i]:
                pure_local_pos[i] = 1.0
            elif not local_pos[i] and global_pos[i]:
                pure_global_pos[i] = 1.0
            
            # 負の構造変化
            if local_neg[i] and global_neg[i]:
                mixed_neg[i] = 1.0
            elif local_neg[i] and not global_neg[i]:
                pure_local_neg[i] = 1.0
            elif not local_neg[i] and global_neg[i]:
                pure_global_neg[i] = 1.0
        
        return pure_local_pos, pure_local_neg, pure_global_pos, pure_global_neg, mixed_pos, mixed_neg
    
    def _fallback_basic_features(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """JIT不使用時のフォールバック実装"""
        n = len(data)
        
        # 差分計算
        diff = np.zeros(n)
        diff[1:] = np.diff(data)
        
        # 閾値
        threshold = np.percentile(np.abs(diff[1:]), 97.0)
        
        # ジャンプ検出
        delta_pos = (diff > threshold).astype(np.float64)
        delta_neg = (diff < -threshold).astype(np.float64)
        
        # 張力スカラー（簡易版）
        rho_t = np.zeros(n)
        window = 10
        for i in range(window, n):
            rho_t[i] = np.std(data[i-window:i+1])
        
        # 時間トレンド
        time_trend = np.arange(n, dtype=np.float64)
        
        return delta_pos, delta_neg, rho_t, time_trend

# ==========================================================
# CONVENIENCE FUNCTIONS - 便利関数
# ==========================================================

def extract_lambda3_features(
    data: Union[np.ndarray, List[float]],
    series_name: str = "Series",
    feature_level: str = 'basic',
    config: Optional[L3Config] = None
) -> StructuralTensorFeatures:
    """
    Lambda³特徴量抽出の便利関数
    
    Args:
        data: 入力時系列
        series_name: 系列名
        feature_level: 特徴量レベル
        config: 設定オブジェクト
        
    Returns:
        StructuralTensorFeatures: 抽出された特徴量
    """
    extractor = StructuralTensorExtractor(config)
    return extractor.extract_features(data, series_name, feature_level)

def create_sample_structural_tensor(n: int = 100, seed: int = 42) -> np.ndarray:
    """テスト用サンプル構造テンソルデータ生成"""
    np.random.seed(seed)
    
    # 基本トレンド
    trend = np.linspace(100, 120, n)
    
    # ランダムウォーク成分
    random_walk = np.cumsum(np.random.randn(n) * 0.5)
    
    # 構造変化（ジャンプ）
    jumps = np.zeros(n)
    jump_times = np.random.choice(range(10, n-10), size=5, replace=False)
    for t in jump_times:
        jumps[t] = np.random.randn() * 3
    
    # 合成
    data = trend + random_walk + np.cumsum(jumps)
    
    return data.astype(np.float64)

# ==========================================================
# MODULE EXPORTS
# ==========================================================

__all__ = [
    # クラス
    'StructuralTensorFeatures',
    'StructuralTensorExtractor',
    
    # 便利関数
    'extract_lambda3_features',
    'create_sample_structural_tensor',
    
    # 定数
    'JIT_AVAILABLE'
]
