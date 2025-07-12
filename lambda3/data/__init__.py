
# ================================================================
# lambda3/data/__init__.py
# ================================================================

"""
Lambda³ Data Management Module

構造テンソル理論のためのデータ管理システム:
- サンプルデータセット
- データ前処理ユーティリティ
- 金融データ取得インターフェース
- データ検証機能
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json

# データディレクトリのパス
DATA_DIR = Path(__file__).parent

# ================================================================
# サンプルデータ生成器
# ================================================================

class Lambda3SampleDataGenerator:
    """
    Lambda³理論検証用サンプルデータ生成器
    
    構造変化を含む時系列データを生成し、
    Lambda³理論の各機能をテストできるデータセットを提供。
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_structural_tensor_series(
        self,
        n_points: int = 500,
        base_volatility: float = 0.02,
        structural_changes: List[Dict] = None
    ) -> np.ndarray:
        """
        構造変化を含む時系列生成
        
        Args:
            n_points: データポイント数
            base_volatility: 基本ボラティリティ
            structural_changes: 構造変化仕様リスト
            
        Returns:
            np.ndarray: 構造変化を含む時系列
        """
        # 基本ランダムウォーク
        series = np.cumsum(np.random.randn(n_points) * base_volatility)
        
        # デフォルト構造変化
        if structural_changes is None:
            structural_changes = [
                {'start': 100, 'end': 150, 'type': 'jump', 'magnitude': 0.5},
                {'start': 300, 'end': 320, 'type': 'volatility_spike', 'magnitude': 5.0},
                {'start': 400, 'end': 450, 'type': 'trend', 'magnitude': 0.003}
            ]
        
        # 構造変化の注入
        for change in structural_changes:
            start, end = change['start'], change['end']
            if end <= n_points:
                if change['type'] == 'jump':
                    # 急激なジャンプ
                    series[start:end] += np.linspace(0, change['magnitude'], end - start)
                elif change['type'] == 'volatility_spike':
                    # ボラティリティスパイク
                    series[start:end] += np.random.randn(end - start) * change['magnitude'] * base_volatility
                elif change['type'] == 'trend':
                    # トレンド変化
                    series[start:end] += np.cumsum(np.ones(end - start) * change['magnitude'])
                elif change['type'] == 'regime_shift':
                    # レジームシフト
                    series[start:] += change['magnitude']
        
        return series
    
    def generate_financial_portfolio(
        self,
        assets: List[str] = None,
        n_points: int = 252,
        correlation_matrix: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        金融ポートフォリオデータ生成
        
        Args:
            assets: 資産名リスト
            n_points: データポイント数（営業日）
            correlation_matrix: 相関行列
            
        Returns:
            Dict[str, np.ndarray]: 資産別リターン系列
        """
        if assets is None:
            assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        n_assets = len(assets)
        
        # デフォルト相関行列
        if correlation_matrix is None:
            correlation_matrix = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
            np.fill_diagonal(correlation_matrix, 1.0)
            # 対称化
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        
        # 多変量正規分布からリターン生成
        mean_returns = np.random.uniform(-0.001, 0.002, n_assets)
        volatilities = np.random.uniform(0.15, 0.35, n_assets)
        
        # 共分散行列構築
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # リターン生成
        returns = np.random.multivariate_normal(
            mean_returns, covariance_matrix / 252, n_points  # 日次化
        )
        
        # 価格系列に変換（累積リターン）
        portfolio_data = {}
        for i, asset in enumerate(assets):
            portfolio_data[asset] = np.cumsum(returns[:, i])
        
        return portfolio_data
    
    def generate_crisis_scenario(
        self,
        base_series: np.ndarray,
        crisis_start: int,
        crisis_duration: int,
        crisis_severity: float = 0.3
    ) -> np.ndarray:
        """
        金融危機シナリオ生成
        
        Args:
            base_series: ベース時系列
            crisis_start: 危機開始時点
            crisis_duration: 危機継続期間
            crisis_severity: 危機の深刻度
            
        Returns:
            np.ndarray: 危機を含む時系列
        """
        series = base_series.copy()
        crisis_end = min(crisis_start + crisis_duration, len(series))
        
        # 危機期間中の特徴
        crisis_period = crisis_end - crisis_start
        
        # 急激な下落
        crash_magnitude = -crisis_severity * np.random.uniform(0.8, 1.2)
        crash_duration = max(1, crisis_period // 4)
        series[crisis_start:crisis_start + crash_duration] += crash_magnitude
        
        # 高ボラティリティ期間
        volatility_multiplier = 3.0 + crisis_severity * 2
        high_vol_returns = np.random.randn(crisis_period) * volatility_multiplier * 0.05
        series[crisis_start:crisis_end] += np.cumsum(high_vol_returns)
        
        # 部分的回復
        recovery_start = crisis_start + crisis_period // 2
        recovery_magnitude = crisis_severity * 0.6
        recovery_duration = crisis_period - (recovery_start - crisis_start)
        series[recovery_start:crisis_end] += np.linspace(0, recovery_magnitude, recovery_duration)
        
        return series

# ================================================================
# データローダー
# ================================================================

class Lambda3DataLoader:
    """
    Lambda³データローダー
    
    各種データソースからのデータ読み込みと
    Lambda³理論に適した形式への変換を提供。
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.sample_generator = Lambda3SampleDataGenerator()
    
    def load_sample_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        サンプルデータセット読み込み
        
        Args:
            dataset_name: データセット名
            
        Returns:
            Dict: データセットと메타데이터
        """
        if dataset_name == "structural_changes":
            return self._load_structural_changes_dataset()
        elif dataset_name == "financial_portfolio":
            return self._load_financial_portfolio_dataset()
        elif dataset_name == "crisis_scenarios":
            return self._load_crisis_scenarios_dataset()
        elif dataset_name == "synthetic_markets":
            return self._load_synthetic_markets_dataset()
        else:
            available_datasets = [
                "structural_changes", "financial_portfolio", 
                "crisis_scenarios", "synthetic_markets"
            ]
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available_datasets}")
    
    def _load_structural_changes_dataset(self) -> Dict[str, Any]:
        """構造変化検証用データセット"""
        datasets = {}
        
        # 基本構造変化パターン
        datasets['basic_jumps'] = self.sample_generator.generate_structural_tensor_series(
            n_points=300,
            structural_changes=[
                {'start': 50, 'end': 60, 'type': 'jump', 'magnitude': 0.3},
                {'start': 150, 'end': 160, 'type': 'jump', 'magnitude': -0.4},
                {'start': 250, 'end': 260, 'type': 'jump', 'magnitude': 0.5}
            ]
        )
        
        # ボラティリティクラスタリング
        datasets['volatility_clustering'] = self.sample_generator.generate_structural_tensor_series(
            n_points=400,
            structural_changes=[
                {'start': 80, 'end': 120, 'type': 'volatility_spike', 'magnitude': 3.0},
                {'start': 200, 'end': 280, 'type': 'volatility_spike', 'magnitude': 4.0}
            ]
        )
        
        # 複合パターン
        datasets['complex_patterns'] = self.sample_generator.generate_structural_tensor_series(
            n_points=500,
            structural_changes=[
                {'start': 50, 'end': 70, 'type': 'jump', 'magnitude': 0.2},
                {'start': 100, 'end': 150, 'type': 'trend', 'magnitude': 0.002},
                {'start': 200, 'end': 230, 'type': 'volatility_spike', 'magnitude': 2.5},
                {'start': 300, 'end': 500, 'type': 'regime_shift', 'magnitude': 0.1}
            ]
        )
        
        metadata = {
            'description': 'Structural changes validation dataset for Lambda³ theory',
            'series_count': len(datasets),
            'theory_focus': 'ΔΛC pulsation detection and structural tensor analysis',
            'use_cases': ['Feature extraction validation', 'JIT optimization testing', 'Hierarchical analysis']
        }
        
        return {'data': datasets, 'metadata': metadata}
    
    def _load_financial_portfolio_dataset(self) -> Dict[str, Any]:
        """金融ポートフォリオデータセット"""
        # 大型株ポートフォリオ
        large_cap = self.sample_generator.generate_financial_portfolio(
            assets=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            n_points=252
        )
        
        # セクター別ポートフォリオ
        sectors = self.sample_generator.generate_financial_portfolio(
            assets=['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer'],
            n_points=252,
            correlation_matrix=np.array([
                [1.0, 0.6, 0.4, 0.3, 0.5],
                [0.6, 1.0, 0.5, 0.2, 0.4],
                [0.4, 0.5, 1.0, 0.3, 0.6],
                [0.3, 0.2, 0.3, 1.0, 0.4],
                [0.5, 0.4, 0.6, 0.4, 1.0]
            ])
        )
        
        # 通貨ペア
        currencies = self.sample_generator.generate_financial_portfolio(
            assets=['USD_JPY', 'EUR_USD', 'GBP_USD', 'USD_CHF'],
            n_points=252
        )
        
        datasets = {
            'large_cap_stocks': large_cap,
            'sector_indices': sectors,
            'currency_pairs': currencies
        }
        
        metadata = {
            'description': 'Financial portfolio dataset for Lambda³ market analysis',
            'time_horizon': '1 year (252 trading days)',
            'theory_focus': 'Pairwise interactions and network analysis',
            'use_cases': ['Asset correlation analysis', 'Portfolio optimization', 'Risk assessment']
        }
        
        return {'data': datasets, 'metadata': metadata}
    
    def _load_crisis_scenarios_dataset(self) -> Dict[str, Any]:
        """金融危機シナリオデータセット"""
        base_series = self.sample_generator.generate_structural_tensor_series(n_points=500)
        
        scenarios = {}
        
        # 軽度の危機
        scenarios['mild_crisis'] = self.sample_generator.generate_crisis_scenario(
            base_series, crisis_start=150, crisis_duration=50, crisis_severity=0.2
        )
        
        # 中程度の危機
        scenarios['moderate_crisis'] = self.sample_generator.generate_crisis_scenario(
            base_series, crisis_start=200, crisis_duration=80, crisis_severity=0.4
        )
        
        # 深刻な危機
        scenarios['severe_crisis'] = self.sample_generator.generate_crisis_scenario(
            base_series, crisis_start=100, crisis_duration=120, crisis_severity=0.7
        )
        
        metadata = {
            'description': 'Financial crisis scenarios for Lambda³ crisis detection validation',
            'scenario_count': len(scenarios),
            'theory_focus': 'Crisis detection and escalation dynamics',
            'use_cases': ['Crisis detection testing', 'Early warning systems', 'Risk management']
        }
        
        return {'data': scenarios, 'metadata': metadata}
    
    def _load_synthetic_markets_dataset(self) -> Dict[str, Any]:
        """合成市場データセット"""
        # マルチアセット市場シミュレーション
        n_assets = 10
        n_points = 500
        
        asset_names = [f'Asset_{i:02d}' for i in range(n_assets)]
        
        # 相関構造のある市場
        correlation_blocks = np.block([
            [np.ones((3, 3)) * 0.7, np.ones((3, 7)) * 0.2],
            [np.ones((7, 3)) * 0.2, np.ones((7, 7)) * 0.5]
        ])
        np.fill_diagonal(correlation_blocks, 1.0)
        
        synthetic_market = self.sample_generator.generate_financial_portfolio(
            assets=asset_names,
            n_points=n_points,
            correlation_matrix=correlation_blocks
        )
        
        # 各資産に異なる構造変化を注入
        for i, (asset, series) in enumerate(synthetic_market.items()):
            if i < 3:  # 第1グループ: 早期構造変化
                change_point = 100 + i * 10
                synthetic_market[asset] = self.sample_generator.generate_structural_tensor_series(
                    n_points=n_points,
                    structural_changes=[
                        {'start': change_point, 'end': change_point + 30, 'type': 'jump', 'magnitude': 0.3}
                    ]
                )[:n_points]
            elif i < 7:  # 第2グループ: 中期ボラティリティ変化
                change_point = 250 + (i - 3) * 15
                synthetic_market[asset] = self.sample_generator.generate_structural_tensor_series(
                    n_points=n_points,
                    structural_changes=[
                        {'start': change_point, 'end': change_point + 50, 'type': 'volatility_spike', 'magnitude': 2.0}
                    ]
                )[:n_points]
        
        metadata = {
            'description': 'Synthetic multi-asset market for comprehensive Lambda³ testing',
            'asset_count': n_assets,
            'correlation_structure': 'Block correlation with groups',
            'theory_focus': 'Network analysis and multi-asset interactions',
            'use_cases': ['Network centrality analysis', 'Systemic risk assessment', 'Multi-asset modeling']
        }
        
        return {'data': {'synthetic_market': synthetic_market}, 'metadata': metadata}

# ================================================================
# データ検証ユーティリティ
# ================================================================

def validate_lambda3_data(data: Union[np.ndarray, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """
    Lambda³分析用データの検証
    
    Args:
        data: 検証対象データ
        
    Returns:
        Dict: 検証結果
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'recommendations': [],
        'statistics': {}
    }
    
    if isinstance(data, np.ndarray):
        data_dict = {'series': data}
    elif isinstance(data, dict):
        data_dict = data
    else:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Data must be numpy array or dict of arrays")
        return validation_results
    
    for series_name, series in data_dict.items():
        series = np.asarray(series)
        
        # 基本検証
        if len(series) < 50:
            validation_results['issues'].append(f"{series_name}: Too short (< 50 points)")
            validation_results['recommendations'].append(f"Increase {series_name} length for reliable analysis")
        
        # 欠損値検証
        if np.isnan(series).any():
            nan_count = np.sum(np.isnan(series))
            validation_results['issues'].append(f"{series_name}: Contains {nan_count} NaN values")
            validation_results['recommendations'].append(f"Handle missing values in {series_name}")
        
        # 無限値検証
        if np.isinf(series).any():
            inf_count = np.sum(np.isinf(series))
            validation_results['issues'].append(f"{series_name}: Contains {inf_count} infinite values")
            validation_results['recommendations'].append(f"Handle infinite values in {series_name}")
        
        # 統計検証
        if np.std(series) < 1e-10:
            validation_results['issues'].append(f"{series_name}: Nearly constant (std < 1e-10)")
            validation_results['recommendations'].append(f"Check {series_name} for variation")
        
        # 統計サマリー
        validation_results['statistics'][series_name] = {
            'length': len(series),
            'mean': float(np.mean(series)),
            'std': float(np.std(series)),
            'min': float(np.min(series)),
            'max': float(np.max(series)),
            'nan_count': int(np.sum(np.isnan(series))),
            'inf_count': int(np.sum(np.isinf(series)))
        }
    
    # 全体妥当性判定
    if validation_results['issues']:
        validation_results['is_valid'] = False
    
    return validation_results

# ================================================================
# パッケージ初期化
# ================================================================

# 利用可能なデータセット
AVAILABLE_DATASETS = [
    "structural_changes",
    "financial_portfolio", 
    "crisis_scenarios",
    "synthetic_markets"
]

# データローダーインスタンス
data_loader = Lambda3DataLoader()

# 便利関数
def load_sample_data(dataset_name: str) -> Dict[str, Any]:
    """サンプルデータ読み込みの便利関数"""
    return data_loader.load_sample_data(dataset_name)

def generate_test_data(
    data_type: str = "structural_changes",
    n_points: int = 300,
    **kwargs
) -> np.ndarray:
    """テストデータ生成の便利関数"""
    generator = Lambda3SampleDataGenerator()
    
    if data_type == "structural_changes":
        return generator.generate_structural_tensor_series(n_points, **kwargs)
    elif data_type == "financial_portfolio":
        portfolio = generator.generate_financial_portfolio(n_points=n_points, **kwargs)
        return next(iter(portfolio.values()))  # 最初の系列を返す
    elif data_type == "crisis_scenario":
        base = generator.generate_structural_tensor_series(n_points)
        return generator.generate_crisis_scenario(base, n_points//3, n_points//4, **kwargs)
    else:
        return generator.generate_structural_tensor_series(n_points, **kwargs)

# エクスポート
__all__ = [
    'Lambda3SampleDataGenerator',
    'Lambda3DataLoader', 
    'validate_lambda3_data',
    'load_sample_data',
    'generate_test_data',
    'AVAILABLE_DATASETS',
    'data_loader'
]
