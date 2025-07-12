# ==========================================================
# lambda3/core/config.py (完全版)
# Lambda³ Theory Configuration System with JIT Integration
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
#
# 完全版: 全移行漏れ解決、threshold_percentile属性追加
# ==========================================================

"""
Lambda³理論統合設定システム（JIT対応完全版）

構造テンソル解析の理論的一貫性を保持しつつ、
JIT最適化との完全統合を実現する設定管理システム。

核心原理:
- 構造テンソル(Λ)の時系列解析パラメータ
- 進行ベクトル(ΛF)の方向性検出閾値  
- 張力スカラー(ρT)の変動検出感度
- ∆ΛC pulsationsの検出精度制御

完全版修正内容:
- threshold_percentile属性をL3BaseConfigに追加
- L3VisualizationConfig完全実装
- 設定管理関数群の実装
- JIT最適化設定との統合
- 循環インポート問題の解決
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# ==========================================================
# DEFAULT PARAMETERS - Lambda³理論デフォルト値
# ==========================================================

# 構造変化検出の核心パラメータ
DEFAULT_WINDOW_SIZE = 10                    # 基本構造変化検出窓
DEFAULT_LOCAL_WINDOW_SIZE = 5               # 局所構造変化窓  
DEFAULT_GLOBAL_WINDOW_SIZE = 20             # 大域構造変化窓

# 構造変化検出閾値（パーセンタイル）
DEFAULT_THRESHOLD_PERCENTILE = 95.0         # 基本構造変化閾値
DEFAULT_DELTA_PERCENTILE = 90.0             # ∆ΛC検出閾値
DEFAULT_LOCAL_JUMP_PERCENTILE = 85.0        # 局所ジャンプ検出閾値
DEFAULT_LOCAL_THRESHOLD_PERCENTILE = 80.0   # 局所構造閾値
DEFAULT_GLOBAL_THRESHOLD_PERCENTILE = 95.0  # 大域構造閾値

# ペアワイズ解析パラメータ
DEFAULT_LAG_WINDOW = 10                     # 因果関係検出遅延窓
DEFAULT_SYNC_THRESHOLD = 0.3                # 同期検出閾値

# ベイズ推定パラメータ  
DEFAULT_MCMC_DRAWS = 2000                   # 事後分布サンプル数
DEFAULT_MCMC_TUNE = 1000                    # ウォームアップサンプル数
DEFAULT_TARGET_ACCEPT = 0.9                 # MCMC受容率目標値
DEFAULT_HDI_PROB = 0.94                     # 高密度区間確率

# ==========================================================
# JIT OPTIMIZATION CONFIGURATION
# ==========================================================

@dataclass
class L3JITConfig:
    """
    Lambda³ JIT最適化設定
    
    Numba JITコンパイルとLambda³理論演算の統合設定。
    数値計算の精度と性能のトレードオフを最適化。
    """
    
    # JIT基本設定
    enable_jit: bool = True                 # JIT最適化有効化
    nopython_mode: bool = True              # nopythonモード（必須）
    fastmath: bool = True                   # 高速数学演算
    cache: bool = True                      # コンパイル結果キャッシュ
    
    # 並列計算設定  
    enable_parallel: bool = True            # 並列計算有効化
    max_workers: int = 4                    # 最大ワーカー数
    
    # 最適化レベル
    optimization_level: str = 'standard'    # conservative, standard, aggressive
    
    # 数値精度設定
    precision_mode: str = 'high'            # low, medium, high
    error_model: str = 'numpy'              # numpy, python
    
    def get_jit_options(self) -> tuple:
        """JIT設定辞書を生成"""
        base_options = {
            'nopython': self.nopython_mode,
            'fastmath': self.fastmath,
            'cache': self.cache
        }
        
        # 並列計算オプション（条件付き）
        if self.enable_parallel:
            parallel_options = {
                'parallel': True
            }
            # 基本オプションと並列オプションを分離
            return base_options, parallel_options
        else:
            return base_options, {}
    
    def get_optimization_flags(self) -> Dict[str, Any]:
        """最適化フラグ辞書を生成"""
        optimization_map = {
            'conservative': {'inline': False, 'unroll': False, 'vectorize': False},
            'standard': {'inline': True, 'unroll': False, 'vectorize': True},
            'aggressive': {'inline': True, 'unroll': True, 'vectorize': True}
        }
        
        return optimization_map.get(self.optimization_level, optimization_map['standard'])

# ==========================================================
# BASE CONFIGURATION CLASS (完全版 - threshold_percentile追加)
# ==========================================================

@dataclass
class L3BaseConfig:
    """
    Lambda³理論基底設定クラス（完全版）
    
    全ての専門設定クラスの基底となる共通パラメータを定義。
    構造テンソル解析の理論的一貫性とJIT最適化の両立を保証。
    
    完全版修正点:
    - threshold_percentile属性を追加
    - 基本構造変化検出に必要な全属性を包含
    - 品質管理パラメータの追加
    """
    
    # 基本構造変化検出パラメータ
    T: int = 150  # データ系列長（分析対象期間）
    window: int = DEFAULT_WINDOW_SIZE  # 基本構造変化検出窓
    
    # 階層的構造変化パラメータ
    local_window: int = DEFAULT_LOCAL_WINDOW_SIZE  # 局所構造変化窓
    global_window: int = DEFAULT_GLOBAL_WINDOW_SIZE  # 大域構造変化窓
    
    # 構造変化検出閾値（完全版: threshold_percentile追加）
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE  # 基本構造変化閾値
    delta_percentile: float = DEFAULT_DELTA_PERCENTILE  # 基本∆ΛC検出閾値
    local_jump_percentile: float = DEFAULT_LOCAL_JUMP_PERCENTILE  # 局所ジャンプ検出閾値
    local_threshold_percentile: float = DEFAULT_LOCAL_THRESHOLD_PERCENTILE  # 局所構造閾値
    global_threshold_percentile: float = DEFAULT_GLOBAL_THRESHOLD_PERCENTILE  # 大域構造閾値
    
    # ペアワイズ解析パラメータ
    lag_window: int = DEFAULT_LAG_WINDOW  # 因果関係検出遅延窓
    sync_threshold: float = DEFAULT_SYNC_THRESHOLD  # 同期検出閾値
    
    # 数値計算安定性パラメータ
    epsilon: float = 1e-8  # 数値ゼロ除算回避用極小値
    dtype: type = np.float64  # データ型（数値精度保証）
    
    # データ品質管理パラメータ
    min_data_points: int = 20  # 最小データ点数
    max_missing_ratio: float = 0.1  # 最大欠損データ比率
    outlier_threshold: float = 3.0  # 外れ値検出閾値（標準偏差倍数）
    
    # JIT最適化設定（統合）
    jit_config: L3JITConfig = field(default_factory=L3JITConfig)
    
    # レガシー互換性設定（削除予定）
    enable_jit: bool = True  # JITコンパイル有効化（レガシー）
    jit_parallel: bool = True  # JIT並列化有効化（レガシー）
    jit_cache: bool = True  # JITキャッシュ有効化（レガシー）
    
    def __post_init__(self):
        """設定値の妥当性検証とJIT設定統合"""
        self._validate_parameters()
        self._sync_legacy_jit_settings()
    
    def _validate_parameters(self):
        """パラメータ妥当性検証（拡張版）"""
        # 時間窓サイズ検証
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")
        if self.local_window <= 0:
            raise ValueError(f"local_window must be positive, got {self.local_window}")
        if self.global_window <= 0:
            raise ValueError(f"global_window must be positive, got {self.global_window}")
        
        # 閾値パーセンタイル検証（完全版: threshold_percentile含む）
        percentile_params = [
            ('threshold_percentile', self.threshold_percentile),
            ('delta_percentile', self.delta_percentile),
            ('local_jump_percentile', self.local_jump_percentile),
            ('local_threshold_percentile', self.local_threshold_percentile),
            ('global_threshold_percentile', self.global_threshold_percentile)
        ]
        
        for param_name, value in percentile_params:
            if not (0.0 <= value <= 100.0):
                raise ValueError(f"{param_name} must be in [0, 100], got {value}")
        
        # 時系列長検証
        if self.T <= max(self.window, self.global_window):
            raise ValueError(f"T={self.T} must be larger than window sizes")
        
        # ラグ窓検証
        if self.lag_window <= 0:
            raise ValueError(f"lag_window must be positive, got {self.lag_window}")
        
        # 同期閾値検証
        if not (0.0 <= self.sync_threshold <= 1.0):
            raise ValueError(f"sync_threshold must be in [0, 1], got {self.sync_threshold}")
        
        # データ品質パラメータ検証
        if self.min_data_points < 5:
            raise ValueError(f"min_data_points must be at least 5, got {self.min_data_points}")
        if not (0.0 <= self.max_missing_ratio <= 1.0):
            raise ValueError(f"max_missing_ratio must be in [0, 1], got {self.max_missing_ratio}")
    
    def _sync_legacy_jit_settings(self):
        """レガシーJIT設定の同期"""
        # レガシー設定から新JIT設定への同期
        if hasattr(self, 'enable_jit'):
            self.jit_config.enable_jit = self.enable_jit
        if hasattr(self, 'jit_parallel'):
            self.jit_config.enable_parallel = self.jit_parallel
        if hasattr(self, 'jit_cache'):
            self.jit_config.cache = self.jit_cache
    
    def get_structural_params(self) -> Dict[str, Any]:
        """構造テンソル解析用パラメータ辞書"""
        return {
            'window': self.window,
            'threshold_percentile': self.threshold_percentile,
            'delta_percentile': self.delta_percentile,
            'local_window': self.local_window,
            'global_window': self.global_window,
            'epsilon': self.epsilon,
            'dtype': self.dtype
        }
    
    def get_jit_params(self) -> Dict[str, Any]:
        """JIT最適化パラメータ辞書"""
        base_options, parallel_options = self.jit_config.get_jit_options()
        return {
            'base_options': base_options,
            'parallel_options': parallel_options,
            'optimization_flags': self.jit_config.get_optimization_flags()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'L3BaseConfig':
        """辞書から設定オブジェクトを生成（JIT設定含む）"""
        # JIT設定の分離処理
        jit_config_dict = config_dict.pop('jit_config', {})
        jit_config = L3JITConfig(**jit_config_dict) if jit_config_dict else L3JITConfig()
        
        # メイン設定の構築
        filtered_dict = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        config = cls(**filtered_dict)
        config.jit_config = jit_config
        
        return config

# ==========================================================
# BAYESIAN ANALYSIS CONFIGURATION (JIT対応版)
# ==========================================================

@dataclass
class L3BayesianConfig(L3BaseConfig):
    """
    Lambda³ベイズ解析設定（JIT対応版）
    
    構造テンソル相互作用のベイズ推定に特化したパラメータ。
    MCMC収束性と統計的精度のバランスを最適化。
    JIT最適化されたベイズ演算との互換性確保。
    """
    
    # MCMC サンプリングパラメータ
    draws: int = DEFAULT_MCMC_DRAWS  # 事後分布サンプル数
    tune: int = DEFAULT_MCMC_TUNE  # ウォームアップサンプル数
    chains: int = 4  # MCMCチェーン数
    cores: int = 4  # 並列処理コア数
    
    # 収束判定パラメータ
    target_accept: float = DEFAULT_TARGET_ACCEPT  # 受容率目標値
    max_treedepth: int = 10  # NUTS最大木深度
    
    # 事後分布解析パラメータ
    hdi_prob: float = DEFAULT_HDI_PROB  # 高密度区間確率
    var_names: List[str] = field(default_factory=lambda: [
        'beta_time_a', 'beta_time_b', 'beta_interact', 
        'beta_rhoT_a', 'beta_rhoT_b'
    ])  # 分析対象変数名
    
    # 事前分布パラメータ
    prior_sigma_scale: float = 2.0  # 事前分布スケール
    interaction_prior_scale: float = 3.0  # 相互作用項事前分布スケール
    
    # 収束診断設定
    check_convergence: bool = True  # 収束診断実行フラグ
    r_hat_threshold: float = 1.01  # R-hat収束判定閾値
    ess_threshold: int = 400  # 有効サンプルサイズ閾値
    
    # JIT最適化ベイズ演算設定（新規追加）
    use_jit_likelihood: bool = True  # JIT最適化尤度計算
    jit_vectorize_priors: bool = True  # 事前分布のベクトル化
    numerical_stability_mode: str = 'high'  # low, medium, high
    
    def get_sampling_kwargs(self) -> Dict[str, Any]:
        """PyMCサンプリング用パラメータ辞書を生成"""
        return {
            'draws': self.draws,
            'tune': self.tune,
            'chains': self.chains,
            'cores': self.cores,
            'target_accept': self.target_accept,
            'return_inferencedata': True
        }
    
    def get_summary_kwargs(self) -> Dict[str, Any]:
        """ArviZ summary用パラメータ辞書を生成"""
        return {
            'var_names': self.var_names,
            'hdi_prob': self.hdi_prob,
            'round_to': 4
        }
    
    def get_jit_likelihood_params(self) -> Dict[str, Any]:
        """JIT最適化尤度計算用パラメータ"""
        return {
            'use_jit': self.use_jit_likelihood,
            'vectorize_priors': self.jit_vectorize_priors,
            'stability_mode': self.numerical_stability_mode,
            'dtype': self.dtype
        }

# ==========================================================
# HIERARCHICAL ANALYSIS CONFIGURATION
# ==========================================================

@dataclass
class L3HierarchicalConfig(L3BaseConfig):
    """
    Lambda³階層解析設定
    
    構造テンソルの階層的∆ΛC変化検出に特化。
    局所-大域構造変化の分離と相互作用解析パラメータ。
    """
    
    # 階層検出感度パラメータ
    hierarchy_sensitivity: float = 0.8  # 階層検出感度
    escalation_threshold: float = 0.6  # エスカレーション検出閾値
    deescalation_threshold: float = 0.4  # デエスカレーション検出閾値
    
    # 階層分離品質パラメータ
    separation_quality_threshold: float = 0.7  # 分離品質最小閾値
    min_events_per_hierarchy: int = 10  # 階層別最小イベント数
    
    # 階層相互作用パラメータ
    cross_hierarchy_lag_window: int = 5  # 階層間相互作用遅延窓
    hierarchy_sync_threshold: float = 0.4  # 階層同期検出閾値
    
    # 階層メトリクス計算設定
    calculate_asymmetry_metrics: bool = True  # 非対称性メトリクス計算
    calculate_dominance_metrics: bool = True  # 優勢度メトリクス計算
    calculate_coupling_strength: bool = True  # 結合強度計算
    
    def get_hierarchy_detection_params(self) -> Dict[str, Any]:
        """階層検出用パラメータ辞書"""
        return {
            'local_window': self.local_window,
            'global_window': self.global_window,
            'local_percentile': self.local_threshold_percentile,
            'global_percentile': self.global_threshold_percentile,
            'sensitivity': self.hierarchy_sensitivity
        }

# ==========================================================
# PAIRWISE ANALYSIS CONFIGURATION
# ==========================================================

@dataclass 
class L3PairwiseConfig(L3BaseConfig):
    """
    Lambda³ペアワイズ解析設定
    
    系列間非対称相互作用の検出と定量化に特化。
    構造テンソル相互作用の方向性と強度解析パラメータ。
    """
    
    # 非対称性解析パラメータ
    asymmetry_detection_sensitivity: float = 0.1  # 非対称性検出感度
    interaction_significance_threshold: float = 0.05  # 相互作用有意水準
    
    # 因果関係検出パラメータ
    causality_lag_window: int = DEFAULT_LAG_WINDOW  # 因果関係検出遅延窓
    causality_confidence_level: float = 0.95  # 因果関係信頼水準
    min_causality_strength: float = 0.1  # 最小因果強度閾値
    
    # 相互作用モデリングパラメータ
    interaction_model_complexity: str = 'medium'  # low, medium, high
    include_lag_effects: bool = True  # 遅延効果包含フラグ
    include_nonlinear_terms: bool = False  # 非線形項包含フラグ
    
    # ペアワイズ品質制御
    min_overlap_ratio: float = 0.8  # 最小データ重複比率
    outlier_detection_method: str = 'iqr'  # none, iqr, zscore, isolation
    data_quality_threshold: float = 0.7  # データ品質最小閾値
    
    # 品質管理
    min_overlapping_data_ratio: float = 0.8  # 最小データ重複率
    max_missing_data_ratio: float = 0.1  # 最大欠損データ率
    
    def get_causality_params(self) -> Dict[str, Any]:
        """因果関係検出用パラメータ辞書"""
        return {
            'lag_window': self.causality_lag_window,
            'confidence_level': self.causality_confidence_level,
            'min_strength': self.min_causality_strength
        }
    
    def get_interaction_model_params(self) -> Dict[str, Any]:
        """相互作用モデル用パラメータ辞書"""
        complexity_map = {
            'low': {'max_interactions': 3, 'regularization': 'strong'},
            'medium': {'max_interactions': 6, 'regularization': 'moderate'},
            'high': {'max_interactions': 12, 'regularization': 'weak'}
        }
        
        base_params = complexity_map.get(self.interaction_model_complexity, complexity_map['medium'])
        base_params.update({
            'include_lag_effects': self.include_lag_effects,
            'include_nonlinear_terms': self.include_nonlinear_terms,
            'significance_threshold': self.interaction_significance_threshold
        })
        
        return base_params

# ==========================================================
# SYNCHRONIZATION ANALYSIS CONFIGURATION
# ==========================================================

@dataclass
class L3SynchronizationConfig(L3BaseConfig):
    """
    Lambda³同期解析設定
    
    構造テンソル変化の同期パターン検出に特化。
    ネットワーク同期と位相結合解析パラメータ。
    """
    
    # 同期検出パラメータ
    sync_threshold: float = DEFAULT_SYNC_THRESHOLD  # 同期検出閾値
    lag_window: int = DEFAULT_LAG_WINDOW  # 遅延窓サイズ
    
    # ネットワーク構築パラメータ
    network_density_threshold: float = 0.2  # ネットワーク密度閾値
    min_network_edge_weight: float = 0.1  # 最小エッジ重み
    
    # 同期品質評価パラメータ
    sync_stability_window: int = 20  # 同期安定性評価窓
    sync_persistence_threshold: float = 0.6  # 同期持続性閾値
    
    # マルチスケール同期パラメータ
    multiscale_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    cross_scale_coupling_threshold: float = 0.3  # スケール間結合閾値

# ==========================================================
# VISUALIZATION CONFIGURATION (完全版)
# ==========================================================

@dataclass
class L3VisualizationConfig:
    """
    Lambda³可視化設定（完全版）
    
    構造テンソル空間の可視化と解析結果の表示設定。
    理論的洞察を最大化する視覚表現パラメータ。
    """
    
    # 基本可視化設定
    style: str = 'lambda3'  # 可視化スタイル
    figsize_base: Tuple[float, float] = (12, 8)  # 基本図サイズ
    dpi: int = 300  # 解像度
    
    # 色彩設定
    color_scheme: str = 'structural_tensor'  # カラースキーム
    colors: Dict[str, str] = field(default_factory=lambda: {
        'pos_jump': 'blue',
        'neg_jump': 'red',
        'tension': 'green',
        'local': 'skyblue',
        'global': 'lightcoral',
        'crisis': 'darkred',
        'normal': 'lightblue'
    })
    
    # 可視化レベル設定
    visualization_level: str = 'standard'  # minimal, standard, full, debug
    show_confidence_intervals: bool = True  # 信頼区間表示
    show_annotations: bool = True  # 注釈表示
    
    # インタラクティブ設定
    enable_interactive: bool = False  # インタラクティブ機能
    save_plots: bool = False  # プロット自動保存
    output_directory: Optional[Path] = None  # 出力ディレクトリ
    
    # 3D可視化設定
    enable_3d_plots: bool = True  # 3Dプロット有効化
    camera_angles: Dict[str, float] = field(default_factory=lambda: {
        'elevation': 20, 'azimuth': 45
    })
    
    # フォント・テキスト設定
    font_family: str = 'DejaVu Sans'  # フォントファミリー
    font_size: int = 12  # 基本フォントサイズ
    title_size: int = 16  # タイトルサイズ
    
    # プロット保存設定
    save_format: str = 'png'  # 保存形式
    save_quality: int = 95  # 保存品質（JPEG用）
    transparent_background: bool = False  # 透明背景
    
    def get_matplotlib_rcparams(self) -> Dict[str, Any]:
        """matplotlib設定パラメータ"""
        return {
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 2,
            'ytick.labelsize': self.font_size - 2,
            'legend.fontsize': self.font_size - 2,
            'figure.titlesize': self.title_size + 2,
            'figure.dpi': self.dpi,
            'font.family': self.font_family
        }
    
    def get_plotly_layout(self) -> Dict[str, Any]:
        """Plotly レイアウト設定"""
        return {
            'font': {'family': self.font_family, 'size': self.font_size},
            'title': {'font': {'size': self.title_size}},
            'showlegend': True,
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        }

# ==========================================================
# COMPREHENSIVE CONFIGURATION (JIT統合完全版)
# ==========================================================

@dataclass
class L3ComprehensiveConfig:
    """
    Lambda³包括設定（JIT統合完全版）
    
    全解析モジュールの統合設定管理。
    理論的一貫性とJIT最適化の両立を保持した包括的パラメータ管理。
    """
    
    # 基底設定
    base: L3BaseConfig = field(default_factory=L3BaseConfig)
    
    # 専門解析設定
    bayesian: L3BayesianConfig = field(default_factory=L3BayesianConfig)
    hierarchical: L3HierarchicalConfig = field(default_factory=L3HierarchicalConfig)
    pairwise: L3PairwiseConfig = field(default_factory=L3PairwiseConfig)
    synchronization: L3SynchronizationConfig = field(default_factory=L3SynchronizationConfig)
    
    # 可視化設定（完全版）
    visualization: L3VisualizationConfig = field(default_factory=L3VisualizationConfig)
    
    # 解析モード設定（完全保持）
    analysis_modes: Dict[str, bool] = field(default_factory=lambda: {
        'hierarchical_analysis': True,
        'separation_dynamics': True,
        'pairwise_analysis': True,
        'asymmetric_analysis': True,
        'causality_analysis': True,
        'synchronization_analysis': True,
        'regime_analysis': True,
        'crisis_detection': True,
        'advanced_visualization': True,
        'multi_scale_analysis': False,
        'coherence_analysis': False
    })
    
    # パフォーマンス設定（JIT統合）
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'enable_parallel_processing': True,
        'max_workers': 4,
        'memory_limit_gb': 8,
        'cache_size_mb': 512,
        'jit_compilation_timeout': 300
    })
    
    def __post_init__(self):
        """包括設定の整合性確保"""
        self._sync_base_configs()
        self._validate_comprehensive_settings()
    
    def _sync_base_configs(self):
        """基底設定の同期"""
        # 基底設定を各専門設定に伝播
        base_params = self.base.get_structural_params()
        
        for config_name in ['bayesian', 'hierarchical', 'pairwise', 'synchronization']:
            config_obj = getattr(self, config_name)
            for param, value in base_params.items():
                if hasattr(config_obj, param):
                    setattr(config_obj, param, value)
    
    def _validate_comprehensive_settings(self):
        """包括設定の妥当性検証"""
        # メモリ制限確認
        if self.performance['memory_limit_gb'] < 2:
            raise ValueError("memory_limit_gb must be at least 2GB")
        
        # ワーカー数確認
        if self.performance['max_workers'] < 1:
            raise ValueError("max_workers must be at least 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return {
            'base': {
                field.name: getattr(self.base, field.name) 
                for field in self.base.__dataclass_fields__.values()
                if not field.name.startswith('_')
            },
            'bayesian': {
                field.name: getattr(self.bayesian, field.name)
                for field in self.bayesian.__dataclass_fields__.values()
                if not field.name.startswith('_')
            },
            'hierarchical': {
                field.name: getattr(self.hierarchical, field.name)
                for field in self.hierarchical.__dataclass_fields__.values()
                if not field.name.startswith('_')
            },
            'pairwise': {
                field.name: getattr(self.pairwise, field.name)
                for field in self.pairwise.__dataclass_fields__.values()
                if not field.name.startswith('_')
            },
            'synchronization': {
                field.name: getattr(self.synchronization, field.name)
                for field in self.synchronization.__dataclass_fields__.values()
                if not field.name.startswith('_')
            },
            'visualization': {
                field.name: getattr(self.visualization, field.name)
                for field in self.visualization.__dataclass_fields__.values()
                if not field.name.startswith('_')
            },
            'analysis_modes': self.analysis_modes.copy(),
            'performance': self.performance.copy()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'L3ComprehensiveConfig':
        """辞書から包括設定を生成（JIT設定含む）"""
        return cls(
            base=L3BaseConfig.from_dict(config_dict.get('base', {})),
            bayesian=L3BayesianConfig(**config_dict.get('bayesian', {})),
            hierarchical=L3HierarchicalConfig(**config_dict.get('hierarchical', {})),
            pairwise=L3PairwiseConfig(**config_dict.get('pairwise', {})),
            synchronization=L3SynchronizationConfig(**config_dict.get('synchronization', {})),
            visualization=L3VisualizationConfig(**config_dict.get('visualization', {})),
            analysis_modes=config_dict.get('analysis_modes', {}),
            performance=config_dict.get('performance', {})
        )
    
    def save_to_file(self, filepath: Union[str, Path]):
        """設定をファイルに保存"""
        import json
        
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        # numpy型の変換
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, type):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        # 型変換適用
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_types(data)
        
        converted_dict = recursive_convert(config_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'L3ComprehensiveConfig':
        """ファイルから設定を読み込み"""
        import json
        
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

# ==========================================================
# CONFIGURATION FACTORY FUNCTIONS (完全版)
# ==========================================================

def create_default_config() -> L3ComprehensiveConfig:
    """デフォルト包括設定を生成"""
    return L3ComprehensiveConfig()

def create_financial_config() -> L3ComprehensiveConfig:
    """金融市場分析特化設定を生成"""
    config = L3ComprehensiveConfig()
    
    # 金融市場向け調整
    config.base.threshold_percentile = 90.0  # やや敏感な検出
    config.base.delta_percentile = 95.0  # より敏感な構造変化検出
    config.hierarchical.escalation_threshold = 0.5  # 金融危機エスカレーション
    config.pairwise.causality_lag_window = 5  # 短期金融相互作用
    config.synchronization.sync_threshold = 0.4  # 高い同期要求
    
    # 金融特化解析モード
    config.analysis_modes.update({
        'regime_analysis': True,
        'crisis_detection': True,
        'volatility_clustering': True
    })
    
    return config

def create_rapid_config() -> L3ComprehensiveConfig:
    """高速分析特化設定を生成"""
    config = L3ComprehensiveConfig()
    
    # 高速化調整
    config.base.window = 5  # 短い検出窓
    config.bayesian.draws = 1000  # 少ないサンプル数
    config.bayesian.tune = 500
    config.bayesian.chains = 2
    
    # JIT最適化強化
    config.base.jit_config.optimization_level = 'aggressive'
    
    # 解析モード簡素化
    config.analysis_modes.update({
        'multi_scale_analysis': False,
        'coherence_analysis': False,
        'advanced_visualization': False
    })
    
    return config

def create_research_config() -> L3ComprehensiveConfig:
    """研究用高精度設定を生成"""
    config = L3ComprehensiveConfig()
    
    # 高精度調整
    config.base.window = 20  # 長い検出窓
    config.base.threshold_percentile = 98.0  # 高い閾値
    config.bayesian.draws = 4000  # 多いサンプル数
    config.bayesian.tune = 2000
    config.bayesian.chains = 4
    
    # 全解析モード有効化
    config.analysis_modes.update({
        'multi_scale_analysis': True,
        'coherence_analysis': True,
        'advanced_visualization': True
    })
    
    return config

# ==========================================================
# CONFIGURATION MANAGEMENT FUNCTIONS (完全版)
# ==========================================================

# グローバル設定インスタンス
_GLOBAL_CONFIG: Optional[L3ComprehensiveConfig] = None

def get_config() -> L3ComprehensiveConfig:
    """グローバル設定を取得"""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = create_default_config()
    return _GLOBAL_CONFIG

def set_config(config: L3ComprehensiveConfig):
    """グローバル設定を設定"""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config

def reset_config():
    """グローバル設定をリセット"""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = create_default_config()

def validate_config(config: L3ComprehensiveConfig) -> bool:
    """設定の妥当性を検証"""
    try:
        # 各設定クラスの妥当性検証
        config._validate_comprehensive_settings()
        return True
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        return False

# ==========================================================
# MODULE EXPORTS (完全版)
# ==========================================================

__all__ = [
    # 基本設定クラス
    'L3BaseConfig',
    'L3JITConfig',
    
    # 専門設定クラス
    'L3BayesianConfig', 
    'L3HierarchicalConfig',
    'L3PairwiseConfig',
    'L3SynchronizationConfig',
    'L3VisualizationConfig',
    
    # 包括設定
    'L3ComprehensiveConfig',
    
    # ファクトリー関数
    'create_default_config',
    'create_financial_config', 
    'create_rapid_config',
    'create_research_config',
    
    # 設定管理関数
    'get_config',
    'set_config',
    'reset_config',
    'validate_config',
    
    # デフォルト定数
    'DEFAULT_WINDOW_SIZE',
    'DEFAULT_THRESHOLD_PERCENTILE',
    'DEFAULT_DELTA_PERCENTILE',
    'DEFAULT_LAG_WINDOW',
    'DEFAULT_SYNC_THRESHOLD',
    'DEFAULT_MCMC_DRAWS',
    'DEFAULT_MCMC_TUNE',
    'DEFAULT_TARGET_ACCEPT',
    'DEFAULT_HDI_PROB'
]
