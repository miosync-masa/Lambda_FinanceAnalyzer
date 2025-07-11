# ==========================================================
# lambda3/visualization/base.py (JIT Compatible Version)
# Advanced Visualization for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 革新ポイント: JIT最適化結果の理論的洞察最大化可視化
# ==========================================================

"""
Lambda³理論高度可視化システム（JIT互換版）

構造テンソル(Λ)解析結果の理論的洞察を最大化する視覚表現システム。
∆ΛC脈動、階層遷移ダイナミクス、非対称相互作用の美的可視化。

核心機能:
- 構造テンソル空間の3D可視化
- 階層分離ダイナミクスの動的表現
- ペアワイズ相互作用ネットワークの視覚化
- ベイズ推定結果の不確実性可視化
- JIT性能指標の統合表示
- インタラクティブダッシュボード

視覚化哲学:
- 理論的洞察の最大化
- 数学的美しさの表現
- 直感的理解の促進
- 学術的厳密性の保持
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from pathlib import Path

# 可視化ライブラリ
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualization will be disabled.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Network visualization will be limited.")

# Lambda³コンポーネント
try:
    from ..core.config import L3VisualizationConfig, L3ComprehensiveConfig
    from ..core.structural_tensor import StructuralTensorFeatures
    from ..analysis.hierarchical import HierarchicalSeparationResults
    from ..analysis.pairwise import PairwiseInteractionResults
    from ..pipelines.comprehensive import Lambda3ComprehensiveResults
    LAMBDA3_COMPONENTS_AVAILABLE = True
except ImportError:
    LAMBDA3_COMPONENTS_AVAILABLE = False
    warnings.warn("Lambda³ components not available for visualization.")

# ==========================================================
# LAMBDA³ COLOR SCHEMES
# ==========================================================

LAMBDA3_COLOR_SCHEMES = {
    'structural_tensor': {
        'primary': '#2E86AB',      # 構造テンソル青
        'secondary': '#A23B72',    # ∆ΛC変化紫
        'accent': '#F18F01',       # 張力オレンジ
        'positive': '#C73E1D',     # 正変化赤
        'negative': '#003049',     # 負変化紺
        'neutral': '#669BBC',      # 中性青灰
        'highlight': '#FFB700',    # ハイライト金
        'background': '#F8F9FA'    # 背景白
    },
    'hierarchical': {
        'local': '#36C5F0',        # 局所水色
        'global': '#E01E5A',       # 大域赤
        'escalation': '#ECB22E',   # エスカレーション黄
        'deescalation': '#2EB67D', # デエスカレーション緑
        'mixed': '#9C51B6',        # 混合紫
        'transition': '#FF6B6B',   # 遷移桃
        'stable': '#4ECDC4'        # 安定緑青
    },
    'pairwise': {
        'symmetric': '#6C5CE7',    # 対称紫
        'asymmetric': '#FD79A8',   # 非対称桃
        'coupling': '#00B894',     # 結合緑
        'causality': '#FDCB6E',    # 因果黄
        'synchrony': '#74B9FF',    # 同期青
        'discord': '#E17055',      # 不協和赤
        'network': '#81ECEC'       # ネットワーク青緑
    },
    'quality': {
        'excellent': '#00B894',    # 優秀緑
        'good': '#00CEC9',         # 良好青緑
        'fair': '#FDCB6E',         # 普通黄
        'poor': '#E17055',         # 不良赤
        'critical': '#D63031'      # 危険深赤
    }
}

def get_lambda3_colors(scheme: str = 'structural_tensor') -> Dict[str, str]:
    """Lambda³カラースキーム取得"""
    return LAMBDA3_COLOR_SCHEMES.get(scheme, LAMBDA3_COLOR_SCHEMES['structural_tensor'])

# ==========================================================
# BASE VISUALIZER CLASS
# ==========================================================

class Lambda3BaseVisualizer:
    """
    Lambda³基底可視化クラス
    
    Lambda³理論に特化した可視化の基盤クラス。
    理論的洞察を最大化する美的表現システムの基礎。
    """
    
    def __init__(self, config: Optional[L3VisualizationConfig] = None):
        """
        初期化
        
        Args:
            config: 可視化設定
        """
        self.config = config or L3VisualizationConfig()
        self.color_schemes = LAMBDA3_COLOR_SCHEMES
        
        # スタイル設定適用
        self._apply_lambda3_style()
        
        print(f"🎨 Lambda³ Visualizer initialized")
        print(f"   Matplotlib: {'Available' if MATPLOTLIB_AVAILABLE else 'Not Available'}")
        print(f"   Plotly: {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
        print(f"   NetworkX: {'Available' if NETWORKX_AVAILABLE else 'Not Available'}")
    
    def _apply_lambda3_style(self):
        """Lambda³スタイル適用"""
        if MATPLOTLIB_AVAILABLE:
            # カスタムスタイル設定
            plt.style.use('default')
            
            # Lambda³専用rcParams
            lambda3_rcparams = {
                'figure.figsize': self.config.figsize_base,
                'figure.dpi': self.config.dpi,
                'font.family': 'sans-serif',
                'font.size': 10,
                'axes.titlesize': 14,
                'axes.labelsize': 11,
                'axes.grid': True,
                'axes.grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'grid.linewidth': 0.5,
                'grid.alpha': 0.3,
                'legend.frameon': True,
                'legend.fancybox': True,
                'legend.shadow': True,
                'legend.framealpha': 0.9
            }
            
            plt.rcParams.update(lambda3_rcparams)
    
    def create_figure(self, rows: int = 1, cols: int = 1, 
                     figsize: Optional[Tuple[float, float]] = None,
                     subplot_titles: Optional[List[str]] = None) -> Tuple[Any, Any]:
        """Lambda³スタイル図表作成"""
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available for figure creation")
        
        figsize = figsize or self.config.figsize_base
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # 背景色設定
        fig.patch.set_facecolor(self.color_schemes['structural_tensor']['background'])
        
        # サブプロットタイトル設定
        if subplot_titles and hasattr(axes, '__len__'):
            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            for i, (ax, title) in enumerate(zip(axes_flat, subplot_titles)):
                ax.set_title(title, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        return fig, axes
    
    def save_figure(self, fig, filename: str, directory: Optional[Path] = None):
        """図表保存"""
        if directory is None:
            directory = self.config.output_directory or Path.cwd()
        
        filepath = directory / filename
        fig.savefig(
            filepath, 
            dpi=self.config.dpi, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        
        if self.config.save_plots:
            print(f"   📁 Figure saved: {filepath}")

# ==========================================================
# TIME SERIES VISUALIZER
# ==========================================================

class TimeSeriesVisualizer(Lambda3BaseVisualizer):
    """
    時系列可視化クラス
    
    構造テンソル時系列とその特徴量の高度可視化。
    ∆ΛC脈動、張力スカラー、階層的変化の統合表示。
    """
    
    def plot_structural_tensor_series(
        self, 
        features: StructuralTensorFeatures,
        show_events: bool = True,
        show_tension: bool = True,
        show_hierarchy: bool = True
    ) -> Tuple[Any, Any]:
        """
        構造テンソル系列可視化
        
        Lambda³理論の核心である構造テンソル系列と
        その変化パターンを包括的に可視化。
        
        Args:
            features: 構造テンソル特徴量
            show_events: ∆ΛC変化イベント表示
            show_tension: 張力スカラー表示
            show_hierarchy: 階層的変化表示
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for structural tensor visualization")
        
        # サブプロット数決定
        n_subplots = 2 + int(show_tension) + int(show_hierarchy)
        
        fig, axes = self.create_figure(
            rows=n_subplots, cols=1,
            figsize=(14, 3 * n_subplots),
            subplot_titles=[
                f'構造テンソル系列: {features.series_name}',
                '∆ΛC 構造変化パルス',
                '張力スカラー (ρT)' if show_tension else None,
                '階層的構造変化' if show_hierarchy else None
            ]
        )
        
        colors = self.color_schemes['structural_tensor']
        time_points = np.arange(len(features.data))
        
        # 1. 原系列と構造変化イベント
        ax1 = axes[0] if hasattr(axes, '__len__') else axes
        
        # 原系列プロット
        ax1.plot(time_points, features.data, 'k-', linewidth=1.5, alpha=0.8, label='構造テンソル系列')
        
        if show_events:
            # 正の構造変化
            pos_events = np.where(features.delta_LambdaC_pos > 0)[0]
            if len(pos_events) > 0:
                ax1.scatter(pos_events, features.data[pos_events], 
                           c=colors['positive'], marker='^', s=60, alpha=0.8, 
                           label='∆ΛC⁺ (正変化)', zorder=5)
            
            # 負の構造変化
            neg_events = np.where(features.delta_LambdaC_neg > 0)[0]
            if len(neg_events) > 0:
                ax1.scatter(neg_events, features.data[neg_events], 
                           c=colors['negative'], marker='v', s=60, alpha=0.8, 
                           label='∆ΛC⁻ (負変化)', zorder=5)
        
        ax1.set_ylabel('構造テンソル値', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. ∆ΛC パルス表示
        ax2 = axes[1]
        
        # パルス強度計算
        pulse_strength = features.delta_LambdaC_pos - features.delta_LambdaC_neg
        
        # 正のパルス
        pos_mask = pulse_strength > 0
        ax2.bar(time_points[pos_mask], pulse_strength[pos_mask], 
               color=colors['positive'], alpha=0.7, label='正パルス', width=0.8)
        
        # 負のパルス
        neg_mask = pulse_strength < 0
        ax2.bar(time_points[neg_mask], pulse_strength[neg_mask], 
               color=colors['negative'], alpha=0.7, label='負パルス', width=0.8)
        
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('∆ΛC パルス強度', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        subplot_idx = 2
        
        # 3. 張力スカラー（オプション）
        if show_tension:
            ax3 = axes[subplot_idx]
            
            # 張力スカラープロット
            ax3.plot(time_points, features.rho_T, color=colors['accent'], 
                    linewidth=2, label='ρT (張力スカラー)')
            ax3.fill_between(time_points, 0, features.rho_T, 
                           color=colors['accent'], alpha=0.3)
            
            # 張力レベル指標
            mean_tension = np.mean(features.rho_T)
            ax3.axhline(y=mean_tension, color=colors['accent'], 
                       linestyle='--', alpha=0.8, label=f'平均張力: {mean_tension:.3f}')
            
            ax3.set_ylabel('張力スカラー', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            subplot_idx += 1
        
        # 4. 階層的変化（オプション）
        if show_hierarchy and hasattr(features, 'local_pos') and features.local_pos is not None:
            ax4 = axes[subplot_idx]
            
            hierarchical_colors = self.color_schemes['hierarchical']
            
            # 局所イベント
            local_events = features.local_pos + features.local_neg
            local_indices = np.where(local_events > 0)[0]
            if len(local_indices) > 0:
                ax4.scatter(local_indices, [0.3] * len(local_indices), 
                           c=hierarchical_colors['local'], marker='s', s=80, 
                           alpha=0.8, label='局所構造変化')
            
            # 大域イベント
            global_events = features.global_pos + features.global_neg
            global_indices = np.where(global_events > 0)[0]
            if len(global_indices) > 0:
                ax4.scatter(global_indices, [0.7] * len(global_indices), 
                           c=hierarchical_colors['global'], marker='o', s=100, 
                           alpha=0.8, label='大域構造変化')
            
            ax4.set_ylim(0, 1)
            ax4.set_ylabel('階層レベル', fontweight='bold')
            ax4.set_yticks([0.3, 0.7])
            ax4.set_yticklabels(['局所', '大域'])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 最下段のx軸ラベル
        axes[-1].set_xlabel('時間インデックス', fontweight='bold')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            self.save_figure(fig, f'structural_tensor_{features.series_name}.png')
        
        return fig, axes
    
    def plot_multiple_series_comparison(
        self, 
        features_dict: Dict[str, StructuralTensorFeatures],
        metric: str = 'rho_T'
    ) -> Tuple[Any, Any]:
        """
        複数系列比較可視化
        
        複数の構造テンソル系列を統一的に比較表示。
        相対的パターンと特徴の識別を支援。
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for multiple series visualization")
        
        colors = self.color_schemes['structural_tensor']
        
        fig, (ax1, ax2) = self.create_figure(
            rows=2, cols=1, figsize=(16, 10),
            subplot_titles=[
                f'複数系列 {metric} 比較',
                '系列別統計サマリー'
            ]
        )
        
        # 1. 時系列比較
        series_stats = {}
        
        for i, (series_name, features) in enumerate(features_dict.items()):
            if hasattr(features, metric):
                series_data = getattr(features, metric)
                time_points = np.arange(len(series_data))
                
                # 色の循環
                color_idx = i % len(plt.cm.tab10.colors)
                color = plt.cm.tab10.colors[color_idx]
                
                # 正規化表示
                normalized_data = (series_data - np.mean(series_data)) / (np.std(series_data) + 1e-8)
                ax1.plot(time_points, normalized_data, color=color, 
                        linewidth=1.5, alpha=0.8, label=series_name)
                
                # 統計収集
                series_stats[series_name] = {
                    'mean': np.mean(series_data),
                    'std': np.std(series_data),
                    'max': np.max(series_data),
                    'events': np.sum(features.delta_LambdaC_pos) + np.sum(features.delta_LambdaC_neg)
                }
        
        ax1.set_xlabel('時間インデックス', fontweight='bold')
        ax1.set_ylabel(f'正規化 {metric}', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 統計サマリー
        if series_stats:
            stats_df = pd.DataFrame(series_stats).T
            
            # ヒートマップ表示
            sns.heatmap(stats_df, annot=True, fmt='.3f', cmap='viridis', 
                       ax=ax2, cbar_kws={'label': '統計値'})
            ax2.set_title('系列別統計サマリー', fontweight='bold')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            self.save_figure(fig, f'multiple_series_{metric}_comparison.png')
        
        return fig, (ax1, ax2)

# ==========================================================
# HIERARCHICAL VISUALIZER
# ==========================================================

class HierarchicalVisualizer(Lambda3BaseVisualizer):
    """
    階層分析可視化クラス
    
    Lambda³階層分離ダイナミクスの高度可視化。
    エスカレーション・デエスカレーション遷移の美的表現。
    """
    
    def plot_hierarchical_separation(
        self, 
        results: HierarchicalSeparationResults,
        show_coefficients: bool = True,
        show_quality_metrics: bool = True
    ) -> Tuple[Any, Any]:
        """
        階層分離解析結果可視化
        
        Lambda³理論の階層分離ダイナミクスを包括的に可視化。
        局所-大域遷移パターンの理論的洞察を提供。
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for hierarchical visualization")
        
        colors = self.color_schemes['hierarchical']
        
        fig, axes = self.create_figure(
            rows=2, cols=2, figsize=(16, 12),
            subplot_titles=[
                f'階層分離系列: {results.series_name}',
                '階層遷移係数',
                '非対称性メトリクス',
                '分離品質評価'
            ]
        )
        
        # 1. 階層分離系列表示
        ax1 = axes[0, 0]
        
        time_points = np.arange(len(results.local_series))
        
        # 局所系列
        local_mask = results.local_series > 0
        ax1.plot(time_points, results.local_series, color=colors['local'], 
                linewidth=2, alpha=0.8, label='局所構造系列')
        ax1.fill_between(time_points, 0, results.local_series, 
                        color=colors['local'], alpha=0.3)
        
        # 大域系列
        global_mask = results.global_series > 0
        ax1.plot(time_points, results.global_series, color=colors['global'], 
                linewidth=2, alpha=0.8, label='大域構造系列')
        ax1.fill_between(time_points, 0, results.global_series, 
                        color=colors['global'], alpha=0.3)
        
        ax1.set_xlabel('時間インデックス')
        ax1.set_ylabel('階層強度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 階層遷移係数
        ax2 = axes[0, 1]
        
        if show_coefficients and results.separation_coefficients:
            coeff_names = []
            coeff_values = []
            coeff_colors = []
            
            for coeff_name, coeff_data in results.separation_coefficients.items():
                if isinstance(coeff_data, dict) and 'coefficient' in coeff_data:
                    coeff_names.append(coeff_name)
                    coeff_values.append(coeff_data['coefficient'])
                    
                    # 色分け
                    if 'escalation' in coeff_name:
                        coeff_colors.append(colors['escalation'])
                    elif 'deescalation' in coeff_name:
                        coeff_colors.append(colors['deescalation'])
                    else:
                        coeff_colors.append(colors['mixed'])
            
            if coeff_names:
                bars = ax2.bar(coeff_names, coeff_values, color=coeff_colors, alpha=0.8)
                
                # 係数値表示
                for bar, value in zip(bars, coeff_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax2.set_ylabel('係数値')
                ax2.tick_params(axis='x', rotation=45)
        
        ax2.grid(True, alpha=0.3)
        
        # 3. 非対称性メトリクス
        ax3 = axes[1, 0]
        
        if results.asymmetry_metrics:
            asymmetry_data = []
            asymmetry_labels = []
            
            for metric_name, metric_value in results.asymmetry_metrics.items():
                asymmetry_labels.append(metric_name.replace('_', '\n'))
                asymmetry_data.append(metric_value)
            
            # レーダーチャート風の円形表示
            angles = np.linspace(0, 2 * np.pi, len(asymmetry_data), endpoint=False)
            asymmetry_data += asymmetry_data[:1]  # 閉じるために最初の値を追加
            angles = np.concatenate((angles, [angles[0]]))
            
            ax3.plot(angles, asymmetry_data, 'o-', linewidth=2, color=colors['mixed'])
            ax3.fill(angles, asymmetry_data, alpha=0.25, color=colors['mixed'])
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(asymmetry_labels)
            ax3.set_ylim(min(asymmetry_data) - 0.1, max(asymmetry_data) + 0.1)
        
        ax3.grid(True, alpha=0.3)
        
        # 4. 分離品質評価
        ax4 = axes[1, 1]
        
        if show_quality_metrics and results.separation_quality:
            quality_names = []
            quality_values = []
            quality_colors = []
            
            for quality_name, quality_value in results.separation_quality.items():
                quality_names.append(quality_name.replace('_', '\n'))
                quality_values.append(quality_value)
                
                # 品質による色分け
                if quality_value >= 0.8:
                    quality_colors.append(self.color_schemes['quality']['excellent'])
                elif quality_value >= 0.6:
                    quality_colors.append(self.color_schemes['quality']['good'])
                elif quality_value >= 0.4:
                    quality_colors.append(self.color_schemes['quality']['fair'])
                else:
                    quality_colors.append(self.color_schemes['quality']['poor'])
            
            bars = ax4.bar(quality_names, quality_values, color=quality_colors, alpha=0.8)
            
            # 品質レベル線
            ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='優秀')
            ax4.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='良好')
            ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='要改善')
            
            ax4.set_ylim(0, 1)
            ax4.set_ylabel('品質スコア')
            ax4.legend()
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            self.save_figure(fig, f'hierarchical_separation_{results.series_name}.png')
        
        return fig, axes
    
    def plot_escalation_deescalation_dynamics(
        self, 
        results_list: List[HierarchicalSeparationResults]
    ) -> Tuple[Any, Any]:
        """
        エスカレーション・デエスカレーション動力学可視化
        
        複数系列のエスカレーション・デエスカレーション
        パターンを統合的に解析・可視化。
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for dynamics visualization")
        
        colors = self.color_schemes['hierarchical']
        
        fig, (ax1, ax2) = self.create_figure(
            rows=2, cols=1, figsize=(14, 10),
            subplot_titles=[
                'エスカレーション vs デエスカレーション 散布図',
                '遷移優勢度ヒストグラム'
            ]
        )
        
        # データ収集
        series_names = []
        escalation_strengths = []
        deescalation_strengths = []
        transition_asymmetries = []
        
        for result in results_list:
            series_names.append(result.series_name)
            escalation_strengths.append(result.get_escalation_strength())
            deescalation_strengths.append(result.get_deescalation_strength())
            
            dominance = result.calculate_transition_dominance()
            transition_asymmetries.append(dominance['transition_asymmetry'])
        
        # 1. エスカレーション vs デエスカレーション散布図
        scatter = ax1.scatter(escalation_strengths, deescalation_strengths, 
                             c=transition_asymmetries, cmap='RdBu_r', 
                             s=100, alpha=0.7, edgecolors='black')
        
        # 対角線（バランス線）
        max_val = max(max(escalation_strengths), max(deescalation_strengths))
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='バランス線')
        
        # 系列名注釈
        for i, name in enumerate(series_names):
            ax1.annotate(name, (escalation_strengths[i], deescalation_strengths[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('エスカレーション強度', fontweight='bold')
        ax1.set_ylabel('デエスカレーション強度', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('遷移非対称性', fontweight='bold')
        
        # 2. 遷移非対称性ヒストグラム
        ax2.hist(transition_asymmetries, bins=10, color=colors['mixed'], 
                alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8, label='完全バランス')
        ax2.axvline(x=np.mean(transition_asymmetries), color='red', 
                   linestyle='--', alpha=0.8, label=f'平均: {np.mean(transition_asymmetries):.3f}')
        
        ax2.set_xlabel('遷移非対称性', fontweight='bold')
        ax2.set_ylabel('頻度', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            self.save_figure(fig, 'escalation_deescalation_dynamics.png')
        
        return fig, (ax1, ax2)

# ==========================================================
# INTERACTION VISUALIZER
# ==========================================================

class InteractionVisualizer(Lambda3BaseVisualizer):
    """
    相互作用可視化クラス
    
    Lambda³ペアワイズ相互作用とネットワーク構造の高度可視化。
    非対称性、因果関係、同期パターンの美的表現。
    """
    
    def plot_pairwise_interaction(
        self, 
        results: PairwiseInteractionResults,
        show_synchronization: bool = True,
        show_causality: bool = True
    ) -> Tuple[Any, Any]:
        """
        ペアワイズ相互作用可視化
        
        Lambda³ペアワイズ相互作用の包括的可視化。
        非対称性、同期性、因果性の統合表示。
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for interaction visualization")
        
        colors = self.color_schemes['pairwise']
        
        fig, axes = self.create_figure(
            rows=2, cols=2, figsize=(16, 12),
            subplot_titles=[
                f'相互作用係数: {results.series_names[0]} ⇄ {results.series_names[1]}',
                '同期プロファイル',
                '因果パターン分析',
                '非対称性・品質評価'
            ]
        )
        
        # 1. 相互作用係数
        ax1 = axes[0, 0]
        
        if results.interaction_coefficients:
            direction_names = []
            coefficient_types = ['pos_jump', 'neg_jump', 'tension']
            type_colors = [colors['coupling'], colors['discord'], colors['synchrony']]
            
            for direction, coeffs in results.interaction_coefficients.items():
                direction_names.append(direction.replace('_to_', ' → '))
            
            # 積み上げ棒グラフ
            x_pos = np.arange(len(direction_names))
            bottom = np.zeros(len(direction_names))
            
            for i, coeff_type in enumerate(coefficient_types):
                values = []
                for coeffs in results.interaction_coefficients.values():
                    values.append(abs(coeffs.get(coeff_type, 0)))
                
                ax1.bar(x_pos, values, bottom=bottom, label=coeff_type, 
                       color=type_colors[i], alpha=0.8)
                bottom += values
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(direction_names, rotation=45)
            ax1.set_ylabel('相互作用強度')
            ax1.legend()
        
        ax1.grid(True, alpha=0.3)
        
        # 2. 同期プロファイル
        ax2 = axes[0, 1]
        
        if show_synchronization and results.synchronization_profile:
            sync_profile = results.synchronization_profile
            if 'lag_profile' in sync_profile:
                lags = list(sync_profile['lag_profile'].keys())
                sync_values = list(sync_profile['lag_profile'].values())
                
                # 同期プロファイルプロット
                ax2.plot(lags, sync_values, 'o-', color=colors['synchrony'], 
                        linewidth=2, markersize=6)
                
                # 最適遅延ハイライト
                optimal_lag = sync_profile.get('optimal_lag', 0)
                max_sync = sync_profile.get('max_sync', 0)
                ax2.scatter([optimal_lag], [max_sync], color=colors['highlight'], 
                           s=200, marker='*', edgecolors='black', zorder=5,
                           label=f'最適遅延: {optimal_lag}')
                
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
                ax2.set_xlabel('遅延 (lag)')
                ax2.set_ylabel('同期率')
                ax2.legend()
        
        ax2.grid(True, alpha=0.3)
        
        # 3. 因果パターン分析
        ax3 = axes[1, 0]
        
        if show_causality and results.causality_patterns:
            # 因果パターンの統計表示
            pattern_strengths = []
            pattern_names = []
            
            for pattern_name, lags_dict in results.causality_patterns.items():
                if lags_dict:
                    max_causality = max(lags_dict.values())
                    pattern_strengths.append(max_causality)
                    # パターン名簡略化
                    simplified_name = pattern_name.split('_to_')[-1][:10] + '...'
                    pattern_names.append(simplified_name)
            
            if pattern_strengths:
                bars = ax3.bar(pattern_names, pattern_strengths, 
                              color=colors['causality'], alpha=0.8)
                
                # 値表示
                for bar, strength in zip(bars, pattern_strengths):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{strength:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax3.set_ylabel('最大因果確率')
                ax3.tick_params(axis='x', rotation=45)
        
        ax3.grid(True, alpha=0.3)
        
        # 4. 非対称性・品質評価
        ax4 = axes[1, 1]
        
        # 非対称性メトリクス
        if results.asymmetry_metrics:
            asymmetry_values = list(results.asymmetry_metrics.values())
            asymmetry_names = [name.replace('_', '\n') for name in results.asymmetry_metrics.keys()]
            
            # 非対称性レーダーチャート
            angles = np.linspace(0, 2 * np.pi, len(asymmetry_values), endpoint=False)
            asymmetry_values += asymmetry_values[:1]
            angles = np.concatenate((angles, [angles[0]]))
            
            ax4.plot(angles, asymmetry_values, 'o-', linewidth=2, color=colors['asymmetric'])
            ax4.fill(angles, asymmetry_values, alpha=0.25, color=colors['asymmetric'])
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(asymmetry_names, fontsize=8)
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            pair_name = f"{results.series_names[0]}_vs_{results.series_names[1]}"
            self.save_figure(fig, f'pairwise_interaction_{pair_name}.png')
        
        return fig, axes
    
    def plot_interaction_network(
        self, 
        interaction_matrix: np.ndarray,
        series_names: List[str],
        threshold: float = 0.1
    ) -> Tuple[Any, Any]:
        """
        相互作用ネットワーク可視化
        
        Lambda³相互作用ネットワークの美的可視化。
        ネットワーク構造とクラスタリングパターンの表現。
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for network visualization")
        
        if not NETWORKX_AVAILABLE:
            warnings.warn("NetworkX not available. Using simplified network visualization.")
            return self._plot_interaction_matrix(interaction_matrix, series_names)
        
        colors = self.color_schemes['pairwise']
        
        fig, (ax1, ax2) = self.create_figure(
            rows=1, cols=2, figsize=(16, 8),
            subplot_titles=[
                'Lambda³ 相互作用ネットワーク',
                '相互作用強度行列'
            ]
        )
        
        # 1. ネットワークグラフ
        G = nx.Graph()
        
        # ノード追加
        for i, name in enumerate(series_names):
            G.add_node(i, name=name)
        
        # エッジ追加（閾値以上の相互作用）
        edge_weights = []
        for i in range(len(series_names)):
            for j in range(i + 1, len(series_names)):
                weight = interaction_matrix[i, j]
                if weight > threshold:
                    G.add_edge(i, j, weight=weight)
                    edge_weights.append(weight)
        
        if G.number_of_edges() > 0:
            # レイアウト計算
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # ノード描画
            node_sizes = [1000 + 500 * np.sum(interaction_matrix[i, :]) for i in range(len(series_names))]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color=colors['network'], alpha=0.8, ax=ax1)
            
            # エッジ描画
            nx.draw_networkx_edges(G, pos, width=[5 * weight for weight in edge_weights],
                                 edge_color=colors['coupling'], alpha=0.6, ax=ax1)
            
            # ラベル描画
            labels = {i: name for i, name in enumerate(series_names)}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax1)
            
            ax1.set_title(f'ネットワーク密度: {nx.density(G):.3f}', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'ネットワーク接続なし\n(閾値以上の相互作用なし)', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        ax1.axis('off')
        
        # 2. 相互作用行列ヒートマップ
        im = ax2.imshow(interaction_matrix, cmap='viridis', aspect='auto')
        
        # 軸ラベル設定
        ax2.set_xticks(range(len(series_names)))
        ax2.set_yticks(range(len(series_names)))
        ax2.set_xticklabels(series_names, rotation=45)
        ax2.set_yticklabels(series_names)
        
        # 値表示
        for i in range(len(series_names)):
            for j in range(len(series_names)):
                text = ax2.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white" if interaction_matrix[i, j] > 0.5 else "black")
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('相互作用強度', fontweight='bold')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            self.save_figure(fig, 'interaction_network.png')
        
        return fig, (ax1, ax2)
    
    def _plot_interaction_matrix(
        self, 
        interaction_matrix: np.ndarray, 
        series_names: List[str]
    ) -> Tuple[Any, Any]:
        """相互作用行列可視化（NetworkX代替）"""
        
        fig, ax = self.create_figure(figsize=(10, 8))
        
        im = ax.imshow(interaction_matrix, cmap='viridis', aspect='auto')
        
        ax.set_xticks(range(len(series_names)))
        ax.set_yticks(range(len(series_names)))
        ax.set_xticklabels(series_names, rotation=45)
        ax.set_yticklabels(series_names)
        
        for i in range(len(series_names)):
            for j in range(len(series_names)):
                text = ax.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                              ha="center", va="center", 
                              color="white" if interaction_matrix[i, j] > 0.5 else "black")
        
        plt.colorbar(im, ax=ax, label='相互作用強度')
        ax.set_title('Lambda³ 相互作用行列', fontweight='bold')
        
        return fig, ax

# ==========================================================
# COMPREHENSIVE RESULTS VISUALIZER
# ==========================================================

class ComprehensiveResultsVisualizer(Lambda3BaseVisualizer):
    """
    包括結果可視化クラス
    
    Lambda³包括解析結果の統合ダッシュボード。
    全解析結果を統一的な視覚表現で提供。
    """
    
    def __init__(self, config: Optional[L3VisualizationConfig] = None):
        super().__init__(config)
        
        # 専門可視化器初期化
        self.timeseries_viz = TimeSeriesVisualizer(config)
        self.hierarchical_viz = HierarchicalVisualizer(config)
        self.interaction_viz = InteractionVisualizer(config)
    
    def create_comprehensive_dashboard(
        self, 
        results: 'Lambda3ComprehensiveResults'
    ) -> Tuple[Any, Any]:
        """
        包括ダッシュボード作成
        
        Lambda³包括解析結果の統合ダッシュボード。
        理論的洞察を最大化する統一視覚表現。
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for comprehensive dashboard")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # タイトル
        fig.suptitle(
            f'Lambda³ Comprehensive Analysis Dashboard\n{results.analysis_timestamp}',
            fontsize=18, fontweight='bold', y=0.95
        )
        
        # 1. 解析サマリー (左上)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_analysis_summary(ax1, results)
        
        # 2. 性能メトリクス (右上)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_performance_metrics(ax2, results)
        
        # 3. 階層分析サマリー (左中)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_hierarchical_summary(ax3, results)
        
        # 4. ペアワイズ分析サマリー (右中)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_pairwise_summary(ax4, results)
        
        # 5. ネットワーク可視化 (左下)
        ax5 = fig.add_subplot(gs[2:, :2])
        self._plot_network_summary(ax5, results)
        
        # 6. 品質評価 (右下)
        ax6 = fig.add_subplot(gs[2:, 2:])
        self._plot_quality_assessment(ax6, results)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            self.save_figure(fig, 'lambda3_comprehensive_dashboard.png')
        
        return fig, (ax1, ax2, ax3, ax4, ax5, ax6)
    
    def _plot_analysis_summary(self, ax, results):
        """解析サマリー可視化"""
        summary = results.get_analysis_summary()
        
        # 解析モード表示
        modes = summary.get('analysis_modes', {})
        mode_names = list(modes.keys())
        mode_status = ['✓' if enabled else '✗' for enabled in modes.values()]
        
        # テキスト情報表示
        summary_text = f"""
解析概要:
• 系列数: {summary.get('series_count', 0)}
• 階層分析: {summary.get('hierarchical_analyses', 0)} 系列
• ペアワイズ分析: {summary.get('pairwise_analyses', 0)} ペア
• JIT最適化: {'有効' if summary.get('jit_optimized', False) else '無効'}
• 実行時間: {summary.get('execution_time', 0):.2f} 秒
• 総合品質: {summary.get('overall_quality', 0):.3f}

解析モード:
""" + '\n'.join([f"• {name}: {status}" for name, status in zip(mode_names, mode_status)])
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('解析サマリー', fontweight='bold')
        ax.axis('off')
    
    def _plot_performance_metrics(self, ax, results):
        """性能メトリクス可視化"""
        perf = results.performance_metrics
        
        if perf:
            metrics = ['execution_time', 'processing_rate', 'memory_efficiency']
            values = []
            labels = []
            
            for metric in metrics:
                if metric in perf:
                    values.append(perf[metric])
                    labels.append(metric.replace('_', '\n'))
            
            if values:
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                bars = ax.bar(labels, values, color=colors, alpha=0.8)
                
                # 値表示
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if 'time' in bar.get_label() or height < 1000:
                        text = f'{height:.2f}'
                    else:
                        text = f'{height:.0f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           text, ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('性能メトリクス', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_hierarchical_summary(self, ax, results):
        """階層分析サマリー可視化"""
        if results.hierarchical_results:
            rankings = results.get_hierarchy_rankings()
            
            # エスカレーション強度トップ5
            top_escalation = rankings.get('escalation_strength', [])[:5]
            
            if top_escalation:
                series_names = [item[0] for item in top_escalation]
                strengths = [item[1] for item in top_escalation]
                
                colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(strengths)))
                bars = ax.barh(series_names, strengths, color=colors)
                
                ax.set_xlabel('エスカレーション強度')
                ax.set_title('階層分析: トップエスカレーション', fontweight='bold')
        else:
            ax.text(0.5, 0.5, '階層分析結果なし', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('階層分析サマリー', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_pairwise_summary(self, ax, results):
        """ペアワイズ分析サマリー可視化"""
        if results.pairwise_results:
            top_interactions = results.get_top_interactions(5)
            
            if top_interactions:
                pair_names = [item[0].replace('_vs_', '\nvs\n') for item in top_interactions]
                couplings = [item[1] for item in top_interactions]
                
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(couplings)))
                bars = ax.barh(pair_names, couplings, color=colors)
                
                ax.set_xlabel('結合強度')
                ax.set_title('ペアワイズ分析: 最強相互作用', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'ペアワイズ分析結果なし', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('ペアワイズ分析サマリー', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_network_summary(self, ax, results):
        """ネットワークサマリー可視化"""
        if results.network_analysis:
            network = results.network_analysis
            
            # ネットワーク統計表示
            stats_text = f"""
ネットワーク統計:
• 密度: {network.get('density', 0):.3f}
• 中心ノード: {network.get('top_central_node', 'N/A')}
• クラスタリング: {network.get('clustering_coefficient', 0):.3f}
• 総接続数: {network.get('network_metrics', {}).get('total_connections', 0)}
• 平均結合: {network.get('network_metrics', {}).get('average_coupling', 0):.3f}
• 最大結合: {network.get('network_metrics', {}).get('max_coupling', 0):.3f}
"""
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, 'ネットワーク分析結果なし', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title('ネットワーク分析サマリー', fontweight='bold')
        ax.axis('off')
    
    def _plot_quality_assessment(self, ax, results):
        """品質評価可視化"""
        quality = results.quality_metrics
        
        if quality:
            quality_names = []
            quality_values = []
            quality_colors = []
            
            for name, value in quality.items():
                quality_names.append(name.replace('_', '\n'))
                quality_values.append(value)
                
                # 品質による色分け
                if value >= 0.8:
                    quality_colors.append('#00B894')  # 優秀
                elif value >= 0.6:
                    quality_colors.append('#00CEC9')  # 良好
                elif value >= 0.4:
                    quality_colors.append('#FDCB6E')  # 普通
                else:
                    quality_colors.append('#E17055')  # 要改善
            
            bars = ax.bar(quality_names, quality_values, color=quality_colors, alpha=0.8)
            
            # 品質基準線
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='優秀')
            ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='良好')
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='要改善')
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('品質スコア')
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, '品質メトリクスなし', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title('解析品質評価', fontweight='bold')
        ax.grid(True, alpha=0.3)

# ==========================================================
# STYLE APPLICATION FUNCTIONS
# ==========================================================

def apply_lambda3_style(style_name: str = 'lambda3_default'):
    """
    Lambda³可視化スタイル適用
    
    Args:
        style_name: スタイル名
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib not available for style application")
        return
    
    lambda3_style = {
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'axes.grid': True,
        'axes.grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linewidth': 0.5,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True
    }
    
    plt.rcParams.update(lambda3_style)
    print(f"✅ Lambda³ style '{style_name}' applied")

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def create_lambda3_visualizer(
    config: Optional[L3VisualizationConfig] = None,
    visualizer_type: str = 'comprehensive'
) -> Lambda3BaseVisualizer:
    """
    Lambda³可視化器作成の便利関数
    
    Args:
        config: 可視化設定
        visualizer_type: 可視化器タイプ
        
    Returns:
        Lambda3BaseVisualizer: 指定タイプの可視化器
    """
    if visualizer_type == 'timeseries':
        return TimeSeriesVisualizer(config)
    elif visualizer_type == 'hierarchical':
        return HierarchicalVisualizer(config)
    elif visualizer_type == 'interaction':
        return InteractionVisualizer(config)
    elif visualizer_type == 'comprehensive':
        return ComprehensiveResultsVisualizer(config)
    else:
        return Lambda3BaseVisualizer(config)

if __name__ == "__main__":
    print("Lambda³ Visualization System Test (JIT Compatible)")
    print("=" * 70)
    
    # 可視化ライブラリ確認
    print("📊 可視化ライブラリ確認:")
    print(f"   Matplotlib: {'✅ Available' if MATPLOTLIB_AVAILABLE else '❌ Not Available'}")
    print(f"   Plotly: {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available'}")
    print(f"   NetworkX: {'✅ Available' if NETWORKX_AVAILABLE else '❌ Not Available'}")
    print(f"   Lambda³ Components: {'✅ Available' if LAMBDA3_COMPONENTS_AVAILABLE else '❌ Not Available'}")
    
    # スタイルテスト
    if MATPLOTLIB_AVAILABLE:
        print("\n🎨 Lambda³スタイルテスト...")
        try:
            apply_lambda3_style()
            print("✅ スタイル適用成功")
            
            # カラースキームテスト
            colors = get_lambda3_colors('structural_tensor')
            print(f"✅ カラースキーム取得成功: {len(colors)} colors")
            
            # 基本可視化器テスト
            visualizer = Lambda3BaseVisualizer()
            print("✅ 基本可視化器初期化成功")
            
        except Exception as e:
            print(f"❌ 可視化テストエラー: {e}")
    
    print("\nVisualization system loaded successfully!")
    print("Ready for Lambda³ theoretical insight visualization with JIT compatibility.")
