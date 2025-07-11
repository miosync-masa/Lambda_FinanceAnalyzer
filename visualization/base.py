# ==========================================================
# lambda3/visualization/base.py
# Base Visualization Classes for Lambda³ Theory
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ==========================================================

"""
Lambda³理論可視化基底クラス群

構造テンソル(Λ)空間の可視化における理論的一貫性と
美的品質を保証する基底クラスシステム。

核心原則:
- 構造空間の非時間的性質の視覚的表現
- ∆ΛC pulsations の直感的理解促進
- 階層性と非対称性の明確な可視化
- モジュラー構成による拡張性
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from pathlib import Path
import warnings

try:
    from mpl_toolkits.mplot3d import Axes3D
    THREEJS_AVAILABLE = True
except ImportError:
    THREEJS_AVAILABLE = False
    warnings.warn("3D plotting may be limited")

from ..core.config import L3VisualizationConfig

# ==========================================================
# COLOR SCHEMES AND STYLES
# ==========================================================

LAMBDA3_COLOR_SCHEMES = {
    'structural_tensor': {
        'pos_jump': '#2E86AB',      # Deep blue for positive jumps
        'neg_jump': '#A23B72',      # Deep magenta for negative jumps
        'tension': '#F18F01',       # Warm orange for tension
        'local': '#C73E1D',         # Red for local events
        'global': '#1B4332',        # Dark green for global events
        'mixed': '#7209B7',         # Purple for mixed events
        'background': '#F8F9FA',    # Light gray background
        'grid': '#DEE2E6',          # Light grid lines
        'text': '#212529'           # Dark text
    },
    'hierarchy': {
        'pure_local': '#FF6B6B',    # Bright red for pure local
        'pure_global': '#4ECDC4',   # Teal for pure global
        'mixed': '#45B7D1',         # Blue for mixed
        'escalation': '#96CEB4',    # Light green for escalation
        'deescalation': '#FFEAA7'   # Light yellow for deescalation
    },
    'interaction': {
        'strong_positive': '#00B894', # Strong green for positive interaction
        'weak_positive': '#81ECEC',   # Light cyan for weak positive
        'neutral': '#DDD',            # Gray for neutral
        'weak_negative': '#FDCB6E',   # Light orange for weak negative
        'strong_negative': '#E17055'  # Orange-red for strong negative
    },
    'asymmetry': {
        'symmetric': '#6C5CE7',       # Purple for symmetric
        'mild_asymmetric': '#A29BFE', # Light purple for mild asymmetry
        'strong_asymmetric': '#FD79A8' # Pink for strong asymmetry
    }
}

LAMBDA3_PLOT_STYLES = {
    'lambda3_default': {
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2
    },
    'lambda3_presentation': {
        'figure.figsize': (16, 10),
        'figure.dpi': 150,
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    },
    'lambda3_publication': {
        'figure.figsize': (10, 6),
        'figure.dpi': 600,
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 8,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'serif'
    }
}

# ==========================================================
# BASE VISUALIZER CLASS
# ==========================================================

class Lambda3BaseVisualizer(ABC):
    """
    Lambda³可視化基底クラス
    
    全てのLambda³可視化クラスの抽象基底クラス。
    理論的一貫性、スタイル統一、品質保証を担当。
    """
    
    def __init__(self, 
                 config: Optional[L3VisualizationConfig] = None,
                 style: str = 'lambda3_default'):
        """
        初期化
        
        Args:
            config: 可視化設定
            style: プロットスタイル
        """
        self.config = config or L3VisualizationConfig()
        self.style = style
        self.colors = LAMBDA3_COLOR_SCHEMES.get(
            self.config.color_scheme, 
            LAMBDA3_COLOR_SCHEMES['structural_tensor']
        )
        
        # プロットスタイル適用
        self._apply_style()
        
        # 可視化履歴
        self.plot_history = []
        
        # 出力ディレクトリ設定
        if self.config.save_plots and self.config.output_directory:
            self.output_dir = Path(self.config.output_directory)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
    
    def _apply_style(self):
        """プロットスタイル適用"""
        style_params = LAMBDA3_PLOT_STYLES.get(self.style, LAMBDA3_PLOT_STYLES['lambda3_default'])
        plt.rcParams.update(style_params)
        
        # 設定ファイルからの追加パラメータ
        config_params = self.config.get_matplotlib_rcparams()
        plt.rcParams.update(config_params)
    
    @abstractmethod
    def create_plot(self, *args, **kwargs) -> plt.Figure:
        """
        プロット作成の抽象メソッド
        各サブクラスで実装必須
        """
        pass
    
    def _create_figure(self, 
                      figsize: Optional[Tuple[float, float]] = None,
                      subplot_config: Optional[Dict] = None) -> Tuple[plt.Figure, Any]:
        """
        図とサブプロット作成
        
        Args:
            figsize: 図サイズ
            subplot_config: サブプロット設定
            
        Returns:
            figure, axes: 図とサブプロット
        """
        if figsize is None:
            figsize = self.config.figsize_base
        
        if subplot_config is None:
            fig, ax = plt.subplots(figsize=figsize)
            return fig, ax
        else:
            nrows = subplot_config.get('nrows', 1)
            ncols = subplot_config.get('ncols', 1)
            subplot_kw = subplot_config.get('subplot_kw', {})
            
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=subplot_kw)
            return fig, axes
    
    def _setup_axis(self, 
                   ax: plt.Axes, 
                   title: str = "",
                   xlabel: str = "",
                   ylabel: str = "",
                   show_grid: bool = True) -> plt.Axes:
        """
        軸の基本設定
        
        Args:
            ax: 軸オブジェクト
            title, xlabel, ylabel: ラベル
            show_grid: グリッド表示フラグ
            
        Returns:
            configured_ax: 設定済み軸
        """
        if title:
            ax.set_title(title, fontweight='bold', pad=20)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Lambda³特有のスタイリング
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors.get('text', '#333333'))
        ax.spines['bottom'].set_color(self.colors.get('text', '#333333'))
        
        return ax
    
    def _add_lambda3_annotations(self, 
                               ax: plt.Axes,
                               annotations: List[Dict]) -> plt.Axes:
        """
        Lambda³理論特有の注釈追加
        
        Args:
            ax: 軸オブジェクト
            annotations: 注釈リスト
            
        Returns:
            annotated_ax: 注釈付き軸
        """
        if not self.config.show_annotations:
            return ax
        
        for annotation in annotations:
            ax.annotate(
                annotation['text'],
                xy=annotation['xy'],
                xytext=annotation.get('xytext', annotation['xy']),
                arrowprops=annotation.get('arrowprops', {}),
                fontsize=annotation.get('fontsize', 9),
                ha=annotation.get('ha', 'center'),
                va=annotation.get('va', 'center'),
                bbox=annotation.get('bbox', dict(boxstyle="round,pad=0.3", 
                                                facecolor='white', alpha=0.8))
            )
        
        return ax
    
    def _plot_structural_events(self, 
                              ax: plt.Axes,
                              time_points: np.ndarray,
                              data_values: np.ndarray,
                              pos_events: np.ndarray,
                              neg_events: np.ndarray,
                              event_labels: bool = True) -> plt.Axes:
        """
        構造変化イベントプロット
        
        Lambda³理論: ∆ΛC pulsationsの可視化
        
        Args:
            ax: 軸オブジェクト
            time_points: 時間軸
            data_values: データ値
            pos_events, neg_events: 構造変化イベント
            event_labels: イベントラベル表示フラグ
            
        Returns:
            plotted_ax: プロット済み軸
        """
        # 基本時系列
        ax.plot(time_points, data_values, 'k-', alpha=0.7, linewidth=1.5, label='構造テンソル系列')
        
        # 正の構造変化
        pos_indices = np.where(pos_events > 0)[0]
        if len(pos_indices) > 0:
            ax.scatter(time_points[pos_indices], data_values[pos_indices],
                      c=self.colors['pos_jump'], marker='^', s=80, alpha=0.8,
                      label='ΔΛC⁺', zorder=5, edgecolors='white', linewidths=1)
        
        # 負の構造変化
        neg_indices = np.where(neg_events > 0)[0]
        if len(neg_indices) > 0:
            ax.scatter(time_points[neg_indices], data_values[neg_indices],
                      c=self.colors['neg_jump'], marker='v', s=80, alpha=0.8,
                      label='ΔΛC⁻', zorder=5, edgecolors='white', linewidths=1)
        
        # イベント統計表示
        if event_labels and (len(pos_indices) > 0 or len(neg_indices) > 0):
            stats_text = f'Events: ΔΛC⁺={len(pos_indices)}, ΔΛC⁻={len(neg_indices)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor='white', alpha=0.9))
        
        return ax
    
    def _plot_tension_scalar(self, 
                           ax: plt.Axes,
                           time_points: np.ndarray,
                           rho_t: np.ndarray,
                           as_twin: bool = False,
                           fill_under: bool = True) -> plt.Axes:
        """
        張力スカラー(ρT)プロット
        
        Lambda³理論: 構造テンソル空間の張力度可視化
        
        Args:
            ax: 軸オブジェクト
            time_points: 時間軸
            rho_t: 張力スカラー値
            as_twin: 双軸プロットフラグ
            fill_under: 面積塗りつぶしフラグ
            
        Returns:
            plotted_ax: プロット済み軸
        """
        if as_twin:
            ax_tension = ax.twinx()
        else:
            ax_tension = ax
        
        # 張力スカラープロット
        line = ax_tension.plot(time_points, rho_t, 
                              color=self.colors['tension'], 
                              linewidth=2, alpha=0.8, label='ρT (張力スカラー)')
        
        # 面積塗りつぶし
        if fill_under:
            ax_tension.fill_between(time_points, 0, rho_t,
                                   color=self.colors['tension'], 
                                   alpha=0.2)
        
        # 軸設定
        if as_twin:
            ax_tension.set_ylabel('張力スカラー (ρT)', 
                                 color=self.colors['tension'])
            ax_tension.tick_params(axis='y', 
                                  labelcolor=self.colors['tension'])
            ax_tension.spines['right'].set_color(self.colors['tension'])
        else:
            ax_tension.set_ylabel('張力スカラー (ρT)')
        
        # 統計情報
        if self.config.show_annotations:
            mean_tension = np.mean(rho_t)
            max_tension = np.max(rho_t)
            stats_text = f'ρT: μ={mean_tension:.3f}, max={max_tension:.3f}'
            
            ax_tension.text(0.98, 0.98, stats_text, 
                           transform=ax_tension.transAxes,
                           verticalalignment='top', 
                           horizontalalignment='right',
                           bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor=self.colors['tension'], 
                                    alpha=0.1))
        
        return ax_tension
    
    def _plot_hierarchical_events(self, 
                                 ax: plt.Axes,
                                 time_points: np.ndarray,
                                 local_events: np.ndarray,
                                 global_events: np.ndarray,
                                 mixed_events: Optional[np.ndarray] = None) -> plt.Axes:
        """
        階層的構造イベントプロット
        
        Lambda³理論: 局所-大域構造変化の階層可視化
        
        Args:
            ax: 軸オブジェクト
            time_points: 時間軸
            local_events, global_events: 階層別イベント
            mixed_events: 混合イベント（オプション）
            
        Returns:
            plotted_ax: プロット済み軸
        """
        y_positions = {'local': 0.2, 'global': 0.6, 'mixed': 0.4}
        
        # 局所イベント
        local_indices = np.where(local_events > 0)[0]
        if len(local_indices) > 0:
            ax.scatter(time_points[local_indices], 
                      [y_positions['local']] * len(local_indices),
                      c=self.colors['local'], marker='s', s=60, 
                      alpha=0.8, label='局所構造変化', zorder=4)
        
        # 大域イベント
        global_indices = np.where(global_events > 0)[0]
        if len(global_indices) > 0:
            ax.scatter(time_points[global_indices], 
                      [y_positions['global']] * len(global_indices),
                      c=self.colors['global'], marker='o', s=80, 
                      alpha=0.8, label='大域構造変化', zorder=4)
        
        # 混合イベント
        if mixed_events is not None:
            mixed_indices = np.where(mixed_events > 0)[0]
            if len(mixed_indices) > 0:
                ax.scatter(time_points[mixed_indices], 
                          [y_positions['mixed']] * len(mixed_indices),
                          c=self.colors['mixed'], marker='D', s=70, 
                          alpha=0.8, label='混合構造変化', zorder=5)
        
        # Y軸設定
        ax.set_ylim(0, 1)
        ax.set_ylabel('階層レベル')
        ax.set_yticks([y_positions['local'], y_positions['mixed'], y_positions['global']])
        ax.set_yticklabels(['局所', '混合', '大域'])
        
        return ax
    
    def _create_interaction_network_plot(self, 
                                       ax: plt.Axes,
                                       node_positions: Dict[str, Tuple[float, float]],
                                       edge_data: List[Dict],
                                       node_labels: List[str]) -> plt.Axes:
        """
        相互作用ネットワークプロット
        
        Lambda³理論: 構造テンソル系列間の相互作用ネットワーク可視化
        
        Args:
            ax: 軸オブジェクト
            node_positions: ノード位置辞書
            edge_data: エッジデータリスト
            node_labels: ノードラベル
            
        Returns:
            plotted_ax: プロット済み軸
        """
        # ノード描画
        for label, (x, y) in node_positions.items():
            circle = plt.Circle((x, y), 0.1, 
                               color=self.colors.get('interaction_a_to_b', 'lightblue'), 
                               alpha=0.8, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', 
                   fontweight='bold', fontsize=10, zorder=4)
        
        # エッジ描画
        for edge in edge_data:
            start_pos = node_positions[edge['source']]
            end_pos = node_positions['target']
            
            # 矢印の調整（ノード境界から開始・終了）
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0.2:  # ノード間距離が十分な場合のみ描画
                # 開始点と終了点の調整
                start_x = start_pos[0] + 0.1 * (dx / length)
                start_y = start_pos[1] + 0.1 * (dy / length)
                end_x = end_pos[0] - 0.1 * (dx / length)
                end_y = end_pos[1] - 0.1 * (dy / length)
                
                # エッジの重みに基づく線幅
                linewidth = max(1, edge.get('weight', 0.5) * 10)
                
                # 方向性の色分け
                color = self.colors.get('strong_positive', 'blue') if edge.get('weight', 0) > 0 else self.colors.get('strong_negative', 'red')
                
                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                           arrowprops=dict(arrowstyle='->', lw=linewidth,
                                          color=color, alpha=0.7))
                
                # 重みラベル
                if self.config.show_annotations and edge.get('weight', 0) > 0.1:
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    ax.text(mid_x, mid_y, f"{edge['weight']:.2f}",
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2",
                                    facecolor='white', alpha=0.8))
        
        # 軸設定
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return ax
    
    def _save_figure(self, 
                    fig: plt.Figure, 
                    filename: str,
                    format: str = 'png',
                    dpi: Optional[int] = None) -> Optional[Path]:
        """
        図の保存
        
        Args:
            fig: 図オブジェクト
            filename: ファイル名
            format: 保存形式
            dpi: 解像度
            
        Returns:
            saved_path: 保存パス（保存した場合）
        """
        if not self.config.save_plots or self.output_dir is None:
            return None
        
        if dpi is None:
            dpi = self.config.dpi
        
        # ファイルパス生成
        file_path = self.output_dir / f"{filename}.{format}"
        
        try:
            fig.savefig(file_path, format=format, dpi=dpi, 
                       bbox_inches='tight', facecolor='white')
            print(f"図を保存しました: {file_path}")
            return file_path
        except Exception as e:
            print(f"図の保存に失敗しました: {e}")
            return None
    
    def _add_plot_to_history(self, 
                           plot_type: str,
                           parameters: Dict[str, Any],
                           saved_path: Optional[Path] = None):
        """プロット履歴追加"""
        self.plot_history.append({
            'plot_type': plot_type,
            'timestamp': plt.datetime.datetime.now(),
            'parameters': parameters,
            'saved_path': str(saved_path) if saved_path else None,
            'style': self.style,
            'color_scheme': self.config.color_scheme
        })
    
    def get_plot_summary(self) -> Dict[str, Any]:
        """プロット履歴サマリー"""
        if not self.plot_history:
            return {"message": "No plots created yet"}
        
        total_plots = len(self.plot_history)
        plot_types = {}
        saved_plots = 0
        
        for plot in self.plot_history:
            plot_type = plot['plot_type']
            plot_types[plot_type] = plot_types.get(plot_type, 0) + 1
            if plot['saved_path']:
                saved_plots += 1
        
        return {
            'total_plots': total_plots,
            'plot_types': plot_types,
            'saved_plots': saved_plots,
            'save_ratio': saved_plots / total_plots if total_plots > 0 else 0,
            'current_style': self.style,
            'output_directory': str(self.output_dir) if self.output_dir else None
        }

# ==========================================================
# SPECIALIZED BASE CLASSES
# ==========================================================

class TimeSeriesVisualizer(Lambda3BaseVisualizer):
    """
    時系列可視化基底クラス
    
    Lambda³理論における構造テンソル時系列の
    標準的可視化パターンを提供
    """
    
    def create_basic_timeseries_plot(self, 
                                   time_points: np.ndarray,
                                   data_values: np.ndarray,
                                   title: str = "構造テンソル時系列",
                                   series_name: str = "Series") -> plt.Figure:
        """
        基本時系列プロット作成
        
        Args:
            time_points: 時間軸
            data_values: データ値
            title: プロットタイトル
            series_name: 系列名
            
        Returns:
            figure: 作成された図
        """
        fig, ax = self._create_figure()
        
        # 基本時系列プロット
        ax.plot(time_points, data_values, 'k-', linewidth=1.5, 
               alpha=0.8, label=series_name)
        
        # 軸設定
        self._setup_axis(ax, title=title, 
                        xlabel='構造空間インデックス', 
                        ylabel='構造テンソル値')
        
        ax.legend()
        
        # 履歴記録
        self._add_plot_to_history('basic_timeseries', {
            'series_name': series_name,
            'data_length': len(data_values)
        })
        
        return fig
    
    def create_multi_series_plot(self, 
                               series_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               title: str = "マルチ構造テンソル系列") -> plt.Figure:
        """
        複数系列プロット作成
        
        Args:
            series_dict: {series_name: (time_points, data_values)} 辞書
            title: プロットタイトル
            
        Returns:
            figure: 作成された図
        """
        fig, ax = self._create_figure()
        
        # 色の循環
        colors = plt.cm.Set1(np.linspace(0, 1, len(series_dict)))
        
        for i, (series_name, (time_points, data_values)) in enumerate(series_dict.items()):
            ax.plot(time_points, data_values, 
                   color=colors[i], linewidth=1.5, 
                   alpha=0.8, label=series_name)
        
        # 軸設定
        self._setup_axis(ax, title=title,
                        xlabel='構造空間インデックス',
                        ylabel='構造テンソル値')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 履歴記録
        self._add_plot_to_history('multi_series', {
            'n_series': len(series_dict),
            'series_names': list(series_dict.keys())
        })
        
        return fig

class InteractionVisualizer(Lambda3BaseVisualizer):
    """
    相互作用可視化基底クラス
    
    Lambda³理論における構造テンソル系列間の
    相互作用パターン可視化を専門化
    """
    
    def create_interaction_matrix_plot(self, 
                                     interaction_matrix: np.ndarray,
                                     series_names: List[str],
                                     title: str = "構造テンソル相互作用行列") -> plt.Figure:
        """
        相互作用行列プロット作成
        
        Args:
            interaction_matrix: 相互作用行列
            series_names: 系列名リスト
            title: プロットタイトル
            
        Returns:
            figure: 作成された図
        """
        fig, ax = self._create_figure()
        
        # ヒートマップ
        im = ax.imshow(interaction_matrix, cmap='RdBu_r', 
                      vmin=-1, vmax=1, aspect='auto')
        
        # 軸設定
        ax.set_xticks(range(len(series_names)))
        ax.set_yticks(range(len(series_names)))
        ax.set_xticklabels(series_names, rotation=45)
        ax.set_yticklabels(series_names)
        
        # 値の表示
        for i in range(len(series_names)):
            for j in range(len(series_names)):
                text_color = 'white' if abs(interaction_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                       ha="center", va="center", color=text_color, fontweight='bold')
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('相互作用強度', rotation=270, labelpad=20)
        
        self._setup_axis(ax, title=title, show_grid=False)
        
        # 履歴記録
        self._add_plot_to_history('interaction_matrix', {
            'matrix_shape': interaction_matrix.shape,
            'series_names': series_names
        })
        
        return fig

class HierarchicalVisualizer(Lambda3BaseVisualizer):
    """
    階層構造可視化基底クラス
    
    Lambda³理論における階層的構造変化の
    専門的可視化を担当
    """
    
    def create_hierarchy_separation_plot(self, 
                                       time_points: np.ndarray,
                                       local_series: np.ndarray,
                                       global_series: np.ndarray,
                                       title: str = "階層分離ダイナミクス") -> plt.Figure:
        """
        階層分離プロット作成
        
        Args:
            time_points: 時間軸
            local_series, global_series: 階層別系列
            title: プロットタイトル
            
        Returns:
            figure: 作成された図
        """
        fig, (ax1, ax2) = self._create_figure(subplot_config={'nrows': 2, 'ncols': 1})
        
        # 局所系列
        ax1.plot(time_points, local_series, 
                color=self.colors['local'], linewidth=2, 
                alpha=0.8, label='局所構造系列')
        ax1.fill_between(time_points, 0, local_series,
                        color=self.colors['local'], alpha=0.2)
        
        self._setup_axis(ax1, title=f"{title} - 局所系列",
                        ylabel='局所構造強度')
        ax1.legend()
        
        # 大域系列
        ax2.plot(time_points, global_series, 
                color=self.colors['global'], linewidth=2, 
                alpha=0.8, label='大域構造系列')
        ax2.fill_between(time_points, 0, global_series,
                        color=self.colors['global'], alpha=0.2)
        
        self._setup_axis(ax2, xlabel='構造空間インデックス',
                        ylabel='大域構造強度')
        ax2.legend()
        
        plt.tight_layout()
        
        # 履歴記録
        self._add_plot_to_history('hierarchy_separation', {
            'data_length': len(time_points),
            'local_events': int(np.sum(local_series > 0)),
            'global_events': int(np.sum(global_series > 0))
        })
        
        return fig

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def apply_lambda3_style(style_name: str = 'lambda3_default'):
    """Lambda³スタイルをグローバルに適用"""
    if style_name in LAMBDA3_PLOT_STYLES:
        plt.rcParams.update(LAMBDA3_PLOT_STYLES[style_name])
        print(f"Lambda³スタイル '{style_name}' を適用しました")
    else:
        available_styles = list(LAMBDA3_PLOT_STYLES.keys())
        print(f"未知のスタイル '{style_name}'。利用可能: {available_styles}")

def get_lambda3_colors(scheme: str = 'structural_tensor') -> Dict[str, str]:
    """Lambda³カラースキーム取得"""
    return LAMBDA3_COLOR_SCHEMES.get(scheme, LAMBDA3_COLOR_SCHEMES['structural_tensor'])

def create_lambda3_legend_elements() -> List[plt.Line2D]:
    """Lambda³標準凡例要素作成"""
    colors = get_lambda3_colors()
    
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', 
                   markerfacecolor=colors['pos_jump'], markersize=10,
                   label='ΔΛC⁺ (正の構造変化)'),
        plt.Line2D([0], [0], marker='v', color='w', 
                   markerfacecolor=colors['neg_jump'], markersize=10,
                   label='ΔΛC⁻ (負の構造変化)'),
        plt.Line2D([0], [0], color=colors['tension'], linewidth=3,
                   label='ρT (張力スカラー)')
    ]
    
    return legend_elements

def validate_plot_data(data: Union[np.ndarray, Dict, List]) -> bool:
    """プロットデータの妥当性検証"""
    try:
        if isinstance(data, np.ndarray):
            return not (np.isnan(data).any() or np.isinf(data).any())
        elif isinstance(data, dict):
            return all(validate_plot_data(v) for v in data.values() if isinstance(v, np.ndarray))
        elif isinstance(data, list):
            return all(validate_plot_data(item) for item in data if isinstance(item, np.ndarray))
        else:
            return True
    except:
        return False

# ==========================================================
# MAIN TESTING
# ==========================================================

if __name__ == "__main__":
    print("Lambda³ Base Visualization Module Test")
    print("=" * 50)
    
    # スタイル適用テスト
    apply_lambda3_style('lambda3_default')
    
    # カラースキーム取得テスト
    colors = get_lambda3_colors('structural_tensor')
    print(f"構造テンソル色: {colors['pos_jump']}, {colors['neg_jump']}")
    
    # テストデータ生成
    np.random.seed(42)
    time_points = np.arange(100)
    test_data = np.cumsum(np.random.randn(100) * 0.1)
    pos_events = np.random.random(100) > 0.95
    neg_events = np.random.random(100) > 0.95
    
    # 基底可視化クラステスト
    class TestVisualizer(TimeSeriesVisualizer):
        def create_plot(self, time_points, data_values):
            return self.create_basic_timeseries_plot(time_points, data_values)
    
    # テスト実行
    viz = TestVisualizer()
    fig = viz.create_plot(time_points, test_data)
    
    print(f"テストプロット作成完了")
    print(f"プロット履歴: {viz.get_plot_summary()}")
    
    # データ妥当性検証テスト
    valid_data = validate_plot_data(test_data)
    print(f"データ妥当性: {valid_data}")
    
    # 無効データテスト
    invalid_data = np.array([1, 2, np.nan, 4])
    invalid_check = validate_plot_data(invalid_data)
    print(f"無効データ検出: {not invalid_check}")
    
    plt.close('all')  # テスト用図を閉じる
    
    print("\nBase visualization module loaded successfully!")
    print("Ready for Lambda³ theoretical visualization framework.")
