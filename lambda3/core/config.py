# ==========================================================
# lambda3/core/config.py
# Configuration System for Lambda³ Theory (修正版)
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論設定システム（完全版）

構造テンソル分析パラメータの統一管理システム。
paste.txtとの完全互換性を確保。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

# ==========================================================
# BASE CONFIGURATION - 基本設定
# ==========================================================

@dataclass
class L3Config:
    """
    Lambda³分析パラメータ設定（完全版）
    
    構造テンソル分析の全パラメータを統一管理。
    理論準拠のデフォルト値を提供。
    """
    
    # 時系列長
    T: int = 150
    
    # 窓サイズ設定
    window: int = 10                    # 基本窓サイズ
    local_window: int = 5               # ローカル構造変化検出窓
    global_window: int = 30             # グローバル構造変化検出窓
    
    # 閾値パーセンタイル
    delta_percentile: float = 97.0      # ∆ΛC検出閾値
    local_jump_percentile: float = 95.0 # ローカルジャンプ閾値
    local_threshold_percentile: float = 85.0    # ローカル閾値（階層用）
    global_threshold_percentile: float = 92.5   # グローバル閾値（階層用）
    
    # ベイズ推定パラメータ
    draws: int = 8000                   # MCMCサンプル数
    tune: int = 8000                    # チューニングステップ数
    target_accept: float = 0.95         # 目標受容率
    chains: int = 4                     # MCMCチェーン数
    
    # 解析対象変数
    var_names: List[str] = field(default_factory=lambda: [
        'beta_time_a', 'beta_time_b', 
        'beta_interact', 'beta_rhoT_a', 'beta_rhoT_b'
    ])
    
    # 信頼区間
    hdi_prob: float = 0.94              # HDI確率
    
    # 同期分析パラメータ
    lag_window_default: int = 10        # デフォルトラグ窓
    sync_threshold_default: float = 0.3 # 同期閾値
    
    # データ検証パラメータ
    min_data_points: int = 50           # 最小データ点数
    max_missing_ratio: float = 0.2      # 最大欠損率
    
    # 数値精度設定
    dtype: type = np.float64            # デフォルト数値型
    epsilon: float = 1e-8               # 数値的安定性のための小値
    
    # 出力設定
    verbose: bool = True                # 詳細出力
    plot_results: bool = True           # 結果プロット
    save_artifacts: bool = False        # アーティファクト保存
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return {
            'T': self.T,
            'window': self.window,
            'local_window': self.local_window,
            'global_window': self.global_window,
            'delta_percentile': self.delta_percentile,
            'local_jump_percentile': self.local_jump_percentile,
            'local_threshold_percentile': self.local_threshold_percentile,
            'global_threshold_percentile': self.global_threshold_percentile,
            'draws': self.draws,
            'tune': self.tune,
            'target_accept': self.target_accept,
            'chains': self.chains,
            'var_names': self.var_names,
            'hdi_prob': self.hdi_prob,
            'lag_window_default': self.lag_window_default,
            'sync_threshold_default': self.sync_threshold_default,
            'min_data_points': self.min_data_points,
            'max_missing_ratio': self.max_missing_ratio,
            'epsilon': self.epsilon,
            'verbose': self.verbose,
            'plot_results': self.plot_results,
            'save_artifacts': self.save_artifacts
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'L3Config':
        """辞書から設定を生成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def validate(self) -> List[str]:
        """設定値の妥当性検証"""
        errors = []
        
        # 窓サイズ検証
        if self.window <= 0:
            errors.append("window must be positive")
        if self.local_window <= 0:
            errors.append("local_window must be positive")
        if self.global_window <= self.local_window:
            errors.append("global_window must be larger than local_window")
        
        # パーセンタイル検証
        for name, value in [
            ('delta_percentile', self.delta_percentile),
            ('local_jump_percentile', self.local_jump_percentile),
            ('local_threshold_percentile', self.local_threshold_percentile),
            ('global_threshold_percentile', self.global_threshold_percentile)
        ]:
            if not 0 < value < 100:
                errors.append(f"{name} must be between 0 and 100")
        
        # ベイズパラメータ検証
        if self.draws <= 0:
            errors.append("draws must be positive")
        if self.tune <= 0:
            errors.append("tune must be positive")
        if not 0 < self.target_accept <= 1:
            errors.append("target_accept must be between 0 and 1")
        if self.chains <= 0:
            errors.append("chains must be positive")
        
        # 同期パラメータ検証
        if self.lag_window_default <= 0:
            errors.append("lag_window_default must be positive")
        if not 0 <= self.sync_threshold_default <= 1:
            errors.append("sync_threshold_default must be between 0 and 1")
        
        return errors

# ==========================================================
# SPECIALIZED CONFIGURATIONS - 特化設定
# ==========================================================

@dataclass
class L3FinancialConfig(L3Config):
    """金融分析用Lambda³設定"""
    
    # 金融市場特有パラメータ
    crisis_threshold: float = 0.8       # 危機検出閾値
    regime_count: int = 4               # レジーム数（Bull/Bear/Crisis/Sideways）
    volatility_window: int = 20         # ボラティリティ計算窓
    
    # リスク管理パラメータ
    var_confidence: float = 0.95        # VaR信頼水準
    es_confidence: float = 0.975        # ES信頼水準
    
    def __post_init__(self):
        """金融分析用の調整"""
        # より保守的な閾値設定
        self.delta_percentile = 95.0
        self.sync_threshold_default = 0.4

@dataclass
class L3ResearchConfig(L3Config):
    """研究用Lambda³設定（高精度）"""
    
    def __post_init__(self):
        """研究用の高精度設定"""
        self.draws = 20000
        self.tune = 20000
        self.target_accept = 0.99
        self.chains = 8
        self.epsilon = 1e-12

@dataclass
class L3RapidConfig(L3Config):
    """高速分析用Lambda³設定"""
    
    def __post_init__(self):
        """高速分析用の軽量設定"""
        self.draws = 2000
        self.tune = 2000
        self.chains = 2
        self.verbose = False
        self.plot_results = False

# ==========================================================
# CONFIGURATION FACTORY - 設定ファクトリ
# ==========================================================

class L3ConfigFactory:
    """Lambda³設定ファクトリ"""
    
    _configs = {
        'default': L3Config,
        'financial': L3FinancialConfig,
        'research': L3ResearchConfig,
        'rapid': L3RapidConfig
    }
    
    @classmethod
    def create(cls, config_type: str = 'default', **kwargs) -> L3Config:
        """
        設定インスタンスを生成
        
        Args:
            config_type: 設定タイプ（'default', 'financial', 'research', 'rapid'）
            **kwargs: 追加パラメータ
            
        Returns:
            L3Config: 設定インスタンス
        """
        if config_type not in cls._configs:
            raise ValueError(f"Unknown config type: {config_type}")
        
        config_class = cls._configs[config_type]
        return config_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, config_class: type):
        """カスタム設定クラスを登録"""
        if not issubclass(config_class, L3Config):
            raise TypeError("config_class must be a subclass of L3Config")
        cls._configs[name] = config_class

# ==========================================================
# BACKWARDS COMPATIBILITY - 後方互換性
# ==========================================================

# 後方互換性のためのエイリアス
L3BaseConfig = L3Config
FinancialAnalysisConfig = L3FinancialConfig
create_config = L3ConfigFactory.create

# デフォルト設定インスタンス
DEFAULT_CONFIG = L3Config()
FINANCIAL_CONFIG = L3FinancialConfig()
RESEARCH_CONFIG = L3ResearchConfig()
RAPID_CONFIG = L3RapidConfig()

# ==========================================================
# MODULE EXPORTS
# ==========================================================

__all__ = [
    # メインクラス
    'L3Config',
    'L3FinancialConfig',
    'L3ResearchConfig',
    'L3RapidConfig',
    
    # ファクトリ
    'L3ConfigFactory',
    'create_config',
    
    # 後方互換性
    'L3BaseConfig',
    'FinancialAnalysisConfig',
    
    # デフォルトインスタンス
    'DEFAULT_CONFIG',
    'FINANCIAL_CONFIG',
    'RESEARCH_CONFIG',
    'RAPID_CONFIG'
]
