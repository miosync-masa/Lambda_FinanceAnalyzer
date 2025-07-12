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
    Lambda³構造テンソル特徴量（完全版）
    
    ∆ΛC pulsations、ρT、階層的構造変化を包含する
    完全な特徴量表現。
    """
    
    # 基本データ
    data: np.ndarray
    series_name: str = "Series"
    
    # 基本構造変化（∆ΛC）- float64保証
    delta_LambdaC_pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    delta_LambdaC_neg: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # 張力スカラー（ρT）
    rho_T: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # 時間トレンド
    time_trend: Optional[np.ndarray] = None
    
    # 階層的構造変化（float64保証）
    local_pos: Optional[np.ndarray] = None
    local_neg: Optional[np.ndarray] = None
    global_pos: Optional[np.ndarray] = None
    global_neg: Optional[np.ndarray] = None
    
    # 階層的純粋成分
    pure_local_pos: Optional[np.ndarray] = None
    pure_local_neg: Optional[np.ndarray] = None
    pure_global_pos: Optional[np.ndarray] = None
    pure_global_neg: Optional[np.ndarray] = None
    mixed_pos: Optional[np.ndarray] = None
    mixed_neg: Optional[np.ndarray] = None
    
    # 追加特徴量
    local_jump_detect: Optional[np.ndarray] = None
    
    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後の型変換とサイズ調整"""
        # float64への統一
        self.data = np.asarray(self.data, dtype=np.float64)
        
        # 基本特徴量の型保証
        if len(self.delta_LambdaC_pos) > 0:
            self.delta_LambdaC_pos = self.delta_LambdaC_pos.astype(np.float64)
        if len(self.delta_LambdaC_neg) > 0:
            self.delta_LambdaC_neg = self.delta_LambdaC_neg.astype(np.float64)
        if len(self.rho_T) > 0:
            self.rho_T = self.rho_T.astype(np.float64)
            
        # 時間トレンドの自動生成
        if self.time_trend is None:
            self.time_trend = np.arange(len(self.data), dtype=np.float64)
    
    def get_structural_summary(self) -> Dict[str, Any]:
        """構造テンソル特性の要約統計"""
        summary = {
            'series_name': self.series_name,
            'data_length': len(self.data),
            'total_pos_jumps': np.sum(self.delta_LambdaC_pos),
            'total_neg_jumps': np.sum(self.delta_LambdaC_neg),
            'mean_tension': np.mean(self.rho_T) if len(self.rho_T) > 0 else 0.0,
            'max_tension': np.max(self.rho_T) if len(self.rho_T) > 0 else 0.0,
        }
        
        # 階層的特徴量の統計
        if self.local_pos is not None:
            summary['local_pos_events'] = np.sum(self.local_pos)
            summary['local_neg_events'] = np.sum(self.local_neg)
        if self.global_pos is not None:
            summary['global_pos_events'] = np.sum(self.global_pos)
            summary['global_neg_events'] = np.sum(self.global_neg)
            
        return summary
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """特徴量の整合性検証"""
        errors = []
        n = len(self.data)
        
        # サイズ整合性チェック
        for name, arr in [
            ('delta_LambdaC_pos', self.delta_LambdaC_pos),
            ('delta_LambdaC_neg', self.delta_LambdaC_neg),
            ('rho_T', self.rho_T)
        ]:
            if len(arr) > 0 and len(arr) != n:
                errors.append(f"{name} size mismatch: {len(arr)} != {n}")
        
        # 型チェック
        if self.data.dtype != np.float64:
            errors.append(f"data dtype is {self.data.dtype}, expected float64")
            
        # 値範囲チェック
        if len(self.delta_LambdaC_pos) > 0:
            if np.any((self.delta_LambdaC_pos < 0) | (self.delta_LambdaC_pos > 1)):
                errors.append("delta_LambdaC_pos values must be in [0, 1]")
                
        return len(errors) == 0, errors

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
            # 階層的構造変化検出
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
            
            # 特徴量更新
            basic_features.local_pos = local_pos
            basic_features.local_neg = local_neg
            basic_features.global_pos = global_pos
            basic_features.global_neg = global_neg
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
