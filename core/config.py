# ==========================================================
# lambda3/core/config.py (JIT Compatible Version)
# Configuration Management for Lambda³ Theory
# 
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 修正点: JIT最適化関数との完全互換性確保
# ==========================================================

"""
Lambda³理論設定管理システム（JIT互換版）

構造テンソル(Λ)、進行ベクトル(ΛF)、張力スカラー(ρT)の
解析パラメータを統一管理する設定クラス群。

時間に依存しない構造空間における∆ΛC pulsationsの
検出と解析に必要な全パラメータを包含。

"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import warnings

# ==========================================================
# GLOBAL CONSTANTS - Lambda³ Theory Defaults
# ==========================================================

# 構造変化検出の基準パーセンタイル
DEFAULT_DELTA_PERCENTILE = 97.0
DEFAULT_LOCAL_JUMP_PERCENTILE = 95.0

# 時間窓パラメータ（非時間的構造空間での窓サイズ）
DEFAULT_WINDOW_SIZE = 10
DEFAULT_LOCAL_WINDOW_SIZE = 5
DEFAULT_GLOBAL_WINDOW_SIZE = 30

# 同期分析パラメータ
DEFAULT_LAG_WINDOW = 10
DEFAULT_SYNC_THRESHOLD = 0.3

# ベイズ推定パラメータ
DEFAULT_MCMC_DRAWS = 8000
DEFAULT_MCMC_TUNE = 8000
DEFAULT_TARGET_ACCEPT = 0.95
DEFAULT_HDI_PROB = 0.94

# 階層分析パラメータ
DEFAULT_LOCAL_THRESHOLD_PERCENTILE = 85.0
DEFAULT_GLOBAL_THRESHOLD_PERCENTILE = 92.5

# JIT最適化設定（新規追加）
DEFAULT_JIT_PARALLEL_THRESHOLD = 10000  # 並列化データサイズ閾値
DEFAULT_JIT_CACHE_SIZE = 100  # JITキャッシュサイズ

# ==========================================================
# JIT OPTIMIZATION CONFIGURATION (新規追加)
# ==========================================================

@dataclass
class L3JITConfig:
    """
    Lambda³ JIT最適化設定
    
    Numba JITコンパイルの最適化パラメータ管理。
    構造テンソル演算の高速化設定を統合。
    """
    
    # JIT基本設定
    enable_jit: bool = True  # JITコンパイル有効化
    nopython_mode: bool = True  # nopythonモード（必須）
    fastmath: bool = True  # 高速数学演算
    cache: bool = True  # コンパイル結果キャッシュ
    
    # 並列計算設定
    enable_parallel: bool = True  # 並列計算有効化
    parallel_threshold: int = DEFAULT_JIT_PARALLEL_THRESHOLD  # 並列化閾値
    max_threads: int = 4  # 最大スレッド数
    
    # 最適化レベル設定
    optimization_level: str = 'aggressive'  # conservative, standard, aggressive
    inline_threshold: int = 100  # インライン展開閾値
    loop_unrolling: bool = True  # ループ展開
    
    # メモリ管理設定
    memory_alignment: int = 32  # メモリアライメント（バイト）
    prefetch_distance: int = 8  # プリフェッチ距離
    
    # 型安全性設定
    strict_type_checking: bool = True  # 厳格型チェック
    type_inference_precision: str = 'high'  # low, medium, high
    
    # デバッグ設定
    enable_profiling: bool = False  # プロファイリング有効化
    debug_info: bool = False  # デバッグ情報出力
    
    def get_jit_options(self) -> Dict[str, Any]:
        """JIT関数用オプション辞書を生成"""
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
# BASE CONFIGURATION CLASS (JIT統合版)
# ==========================================================

@dataclass
class L3BaseConfig:
    """
    Lambda³理論基底設定クラス（JIT統合版）
    
    全ての専門設定クラスの基底となる共通パラメータを定義。
    構造テンソル解析の理論的一貫性とJIT最適化の両立を保証。
    """
    
    # 基本構造変化検出パラメータ
    T: int = 150  # データ系列長（分析対象期間）
    window: int = DEFAULT_WINDOW_SIZE  # 基本構造変化検出窓
    
    # 階層的構造変化パラメータ
    local_window: int = DEFAULT_LOCAL_WINDOW_SIZE  # 局所構造変化窓
    global_window: int = DEFAULT_GLOBAL_WINDOW_SIZE  # 大域構造変化窓
    
    # 構造変化検出閾値
    delta_percentile: float = DEFAULT_DELTA_PERCENTILE  # 基本∆ΛC検出閾値
    local_jump_percentile: float = DEFAULT_LOCAL_JUMP_PERCENTILE  # 局所ジャンプ検出閾値
    local_threshold_percentile: float = DEFAULT_LOCAL_THRESHOLD_PERCENTILE  # 局所構造閾値
    global_threshold_percentile: float = DEFAULT_GLOBAL_THRESHOLD_PERCENTILE  # 大域構造閾値
    
    # 数値計算安定性パラメータ
    epsilon: float = 1e-8  # 数値ゼロ除算回避用極小値
    dtype: type = np.float64  # データ型（数値精度保証）
    
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
        if self.global_window <= self.local_window:
            raise ValueError(f"global_window ({self.global_window}) must be > local_window ({self.local_window})")
        
        # パーセンタイル値検証
        percentiles = [
            ('delta_percentile', self.delta_percentile),
            ('local_jump_percentile', self.local_jump_percentile),
            ('local_threshold_percentile', self.local_threshold_percentile),
            ('global_threshold_percentile', self.global_threshold_percentile)
        ]
        
        for name, value in percentiles:
            if not (0 <= value <= 100):
                raise ValueError(f"{name} must be in [0, 100], got {value}")
        
        # データ型検証（JIT互換性）
        if self.dtype not in [np.float32, np.float64]:
            warnings.warn(f"dtype {self.dtype} may not be JIT-optimized, recommend np.float64")
        
        # 窓サイズとデータ長の整合性検証
        if self.global_window >= self.T:
            warnings.warn(f"global_window ({self.global_window}) >= T ({self.T}), may cause boundary issues")
    
    def _sync_legacy_jit_settings(self):
        """レガシーJIT設定との同期"""
        # レガシー設定をjit_configに反映
        if hasattr(self, 'enable_jit'):
            self.jit_config.enable_jit = self.enable_jit
        if hasattr(self, 'jit_parallel'):
            self.jit_config.enable_parallel = self.jit_parallel
        if hasattr(self, 'jit_cache'):
            self.jit_config.cache = self.jit_cache
    
    def get_jit_function_params(self) -> Dict[str, Any]:
        """JIT関数用パラメータ辞書を生成"""
        return {
            'window': int(self.window),
            'local_window': int(self.local_window),
            'global_window': int(self.global_window),
            'delta_percentile': float(self.delta_percentile),
            'local_percentile': float(self.local_threshold_percentile),
            'global_percentile': float(self.global_threshold_percentile),
            'dtype': self.dtype
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """設定辞書に変換（JIT設定含む）"""
        base_dict = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
        # jit_configを辞書形式で追加
        if hasattr(self.jit_config, '__dict__'):
            base_dict['jit_config'] = self.jit_config.__dict__
        return base_dict
    
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
# HIERARCHICAL ANALYSIS CONFIGURATION (完全保持)
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
# PAIRWISE ANALYSIS CONFIGURATION (完全保持)
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
    
    # ペアワイズ品質管理
    min_overlapping_data_ratio: float = 0.8  # 最小データ重複率
    max_missing_data_ratio: float = 0.1  # 最大欠損データ率
    
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
# SYNCHRONIZATION ANALYSIS CONFIGURATION (完全保持)
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
# VISUALIZATION CONFIGURATION (完全保持)
# ==========================================================

@dataclass
class L3VisualizationConfig:
    """
    Lambda³可視化設定
    
    構造テンソル空間の可視化と解析結果の表示設定。
    理論的洞察を最大化する視覚表現パラメータ。
    """
    
    # 基本可視化設定
    style: str = 'lambda3'  # 可視化スタイル
    figsize_base: tuple = (12, 8)  # 基本図サイズ
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
    
    def get_matplotlib_rcparams(self) -> Dict[str, Any]:
        """matplotlib設定パラメータ"""
        return {
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': self.dpi
        }

# ==========================================================
# COMPREHENSIVE CONFIGURATION (JIT統合版)
# ==========================================================

@dataclass
class L3ComprehensiveConfig:
    """
    Lambda³包括設定（JIT統合版）
    
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
    
    # 可視化設定
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
        'cache_intermediate_results': True,
        'jit_compilation_timeout': 300,  # JITコンパイルタイムアウト（秒）
        'auto_optimize_parameters': True  # パラメータ自動最適化
    })
    
    def __post_init__(self):
        """設定間の一貫性確保（JIT設定含む）"""
        self._ensure_consistency()
        self._validate_jit_compatibility()
    
    def _ensure_consistency(self):
        """設定間一貫性確保（拡張版）"""
        # 基底設定を各専門設定に伝播
        base_params = self.base.to_dict()
        
        # 各専門設定の基底パラメータを更新
        for config_obj in [self.bayesian, self.hierarchical, 
                          self.pairwise, self.synchronization]:
            for key, value in base_params.items():
                if hasattr(config_obj, key) and key != 'jit_config':
                    setattr(config_obj, key, value)
            
            # JIT設定の一貫性確保
            if hasattr(config_obj, 'jit_config'):
                config_obj.jit_config = self.base.jit_config
    
    def _validate_jit_compatibility(self):
        """JIT互換性の検証"""
        # データ型一貫性チェック
        if self.base.dtype != np.float64:
            warnings.warn("Non-float64 dtype may cause JIT performance issues")
        
        # 並列化設定の整合性チェック
        if (self.base.jit_config.enable_parallel and 
            self.performance['max_workers'] > self.base.jit_config.max_threads):
            warnings.warn("max_workers exceeds JIT max_threads, adjusting")
            self.performance['max_workers'] = self.base.jit_config.max_threads
    
    def get_jit_optimized_params(self) -> Dict[str, Any]:
        """JIT最適化された全パラメータを取得"""
        return {
            'base_params': self.base.get_jit_function_params(),
            'jit_options': self.base.jit_config.get_jit_options(),
            'optimization_flags': self.base.jit_config.get_optimization_flags(),
            'performance_settings': self.performance
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """完全設定辞書に変換（JIT設定含む）"""
        return {
            'base': self.base.to_dict(),
            'bayesian': self.bayesian.to_dict(),
            'hierarchical': self.hierarchical.to_dict(),
            'pairwise': self.pairwise.to_dict(),
            'synchronization': self.synchronization.to_dict(),
            'visualization': self.visualization.__dict__,
            'analysis_modes': self.analysis_modes,
            'performance': self.performance
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'L3ComprehensiveConfig':
        """辞書から包括設定を生成（JIT設定含む）"""
        return cls(
            base=L3BaseConfig.from_dict(config_dict.get('base', {})),
            bayesian=L3BayesianConfig.from_dict(config_dict.get('bayesian', {})),
            hierarchical=L3HierarchicalConfig.from_dict(config_dict.get('hierarchical', {})),
            pairwise=L3PairwiseConfig.from_dict(config_dict.get('pairwise', {})),
            synchronization=L3SynchronizationConfig.from_dict(config_dict.get('synchronization', {})),
            visualization=L3VisualizationConfig(**config_dict.get('visualization', {})),
            analysis_modes=config_dict.get('analysis_modes', {}),
            performance=config_dict.get('performance', {})
        )
    
    def save_to_file(self, filepath: Union[str, Path]):
        """設定をファイルに保存"""
        import json
        
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'L3ComprehensiveConfig':
        """ファイルから設定を読み込み"""
        import json
        
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

# ==========================================================
# CONFIGURATION FACTORY FUNCTIONS (完全保持)
# ==========================================================

def create_default_config() -> L3ComprehensiveConfig:
    """デフォルト包括設定を生成"""
    return L3ComprehensiveConfig()

def create_financial_config() -> L3ComprehensiveConfig:
    """金融市場分析特化設定を生成"""
    config = L3ComprehensiveConfig()
    
    # 金融市場向け調整
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
    config.bayesian.draws = 4000
    config.bayesian.tune = 4000
    config.bayesian.chains = 2
    
    # JIT最適化強化
    config.base.jit_config.optimization_level = 'aggressive'
    config.base.jit_config.enable_parallel = True
    
    # 解析モード簡素化
    config.analysis_modes.update({
        'advanced_visualization': False,
        'multi_scale_analysis': False,
        'coherence_analysis': False
    })
    
    return config

def create_research_config() -> L3ComprehensiveConfig:
    """研究用高精度設定を生成"""
    config = L3ComprehensiveConfig()
    
    # 高精度調整
    config.bayesian.draws = 12000
    config.bayesian.tune = 12000
    config.bayesian.target_accept = 0.98
    config.bayesian.r_hat_threshold = 1.005
    
    # JIT最適化精度重視
    config.base.jit_config.optimization_level = 'conservative'
    config.base.jit_config.strict_type_checking = True
    
    # 全解析モード有効化
    for mode in config.analysis_modes:
        config.analysis_modes[mode] = True
    
    return config

# ==========================================================
# CONFIGURATION VALIDATION (拡張版)
# ==========================================================

def validate_config(config: L3ComprehensiveConfig) -> Dict[str, List[str]]:
    """
    設定の包括的妥当性検証（JIT互換性含む）
    
    Returns:
        Dict[str, List[str]]: モジュール別検証結果
    """
    validation_results = {
        'base': [],
        'jit': [],
        'bayesian': [],
        'hierarchical': [], 
        'pairwise': [],
        'synchronization': [],
        'visualization': [],
        'overall': []
    }
    
    # 基底設定検証
    try:
        config.base._validate_parameters()
    except ValueError as e:
        validation_results['base'].append(str(e))
    
    # JIT設定検証（新規追加）
    if not config.base.jit_config.enable_jit:
        validation_results['jit'].append("JIT disabled - significant performance impact expected")
    
    if config.base.jit_config.enable_parallel and config.base.T < config.base.jit_config.parallel_threshold:
        validation_results['jit'].append(f"Data size ({config.base.T}) below parallel threshold ({config.base.jit_config.parallel_threshold})")
    
    # ベイズ設定検証
    if config.bayesian.draws < 1000:
        validation_results['bayesian'].append("draws should be >= 1000 for reliable inference")
    
    if config.bayesian.target_accept < 0.8:
        validation_results['bayesian'].append("target_accept should be >= 0.8 for good mixing")
    
    # 階層設定検証
    if config.hierarchical.min_events_per_hierarchy < 5:
        validation_results['hierarchical'].append("min_events_per_hierarchy should be >= 5")
    
    # ペアワイズ設定検証
    if config.pairwise.causality_lag_window > config.base.T // 4:
        validation_results['pairwise'].append("causality_lag_window too large relative to data length")
    
    # 同期設定検証
    if config.synchronization.lag_window > config.base.T // 3:
        validation_results['synchronization'].append("sync lag_window too large relative to data length")
    
    # 全体一貫性検証
    enabled_modes = sum(config.analysis_modes.values())
    if enabled_modes == 0:
        validation_results['overall'].append("At least one analysis mode should be enabled")
    
    return validation_results

# ==========================================================
# MAIN CONFIGURATION INTERFACE (完全保持)
# ==========================================================

# パッケージレベルのデフォルト設定
DEFAULT_L3_CONFIG = create_default_config()

def get_config(config_type: str = 'default') -> L3ComprehensiveConfig:
    """
    設定取得インターフェース
    
    Args:
        config_type: 'default', 'financial', 'rapid', 'research'
    
    Returns:
        L3ComprehensiveConfig: 指定タイプの設定
    """
    config_factory = {
        'default': create_default_config,
        'financial': create_financial_config,
        'rapid': create_rapid_config,
        'research': create_research_config
    }
    
    factory_func = config_factory.get(config_type, create_default_config)
    return factory_func()

if __name__ == "__main__":
    # 設定システムテスト（JIT互換性含む）
    print("Lambda³ Configuration System Test (JIT Compatible)")
    print("=" * 50)
    
    # デフォルト設定生成
    config = create_default_config()
    print(f"Default config created: {type(config).__name__}")
    print(f"JIT enabled: {config.base.jit_config.enable_jit}")
    print(f"JIT optimization level: {config.base.jit_config.optimization_level}")
    
    # JIT互換性検証
    jit_params = config.get_jit_optimized_params()
    print(f"JIT parameters ready: {len(jit_params)} parameter groups")
    
    # 検証実行
    validation_results = validate_config(config)
    total_issues = sum(len(errors) for errors in validation_results.values())
    print(f"Validation results: {total_issues} issues found")
    
    # 金融設定生成
    financial_config = create_financial_config()
    print(f"Financial config created with crisis detection: {financial_config.analysis_modes['crisis_detection']}")
    
    # 高速設定生成
    rapid_config = create_rapid_config()
    print(f"Rapid config created with optimization level: {rapid_config.base.jit_config.optimization_level}")
    
    print("Configuration system ready with full JIT compatibility!")
