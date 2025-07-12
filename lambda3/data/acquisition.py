# ==========================================================
# lambda3/data/acquisition.py
# Advanced Real Data Acquisition System for Lambda³ Theory
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# 
# 革新ポイント: 多様なデータソースの統合取得・前処理システム
# ==========================================================

"""
Lambda³理論リアルデータ取得システム

多様な金融・経済データソースからの自動取得・前処理・検証システム。
構造テンソル分析に最適化されたデータパイプライン。

対応データソース:
- Yahoo Finance (株価、為替、商品、指数)
- FRED (経済統計)
- Quandl/NASDAQ Data Link
- Alpha Vantage
- Custom CSV/Excel files
- Real-time streaming data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import time
from datetime import datetime, timedelta
import json
import asyncio
import aiohttp

# データ取得ライブラリ
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available")

try:
    import pandas_datareader as pdr
    from pandas_datareader import fred
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False
    warnings.warn("pandas_datareader not available")

try:
    import quandl
    QUANDL_AVAILABLE = True
except ImportError:
    QUANDL_AVAILABLE = False
    warnings.warn("quandl not available")

try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.foreignexchange import ForeignExchange
    ALPHAVANTAGE_AVAILABLE = True
except ImportError:
    ALPHAVANTAGE_AVAILABLE = False
    warnings.warn("alpha_vantage not available")

# Lambda³ components
try:
    from ..utils.helpers import safe_divide, robust_normalize
    from ..core.config import L3BaseConfig
    LAMBDA3_AVAILABLE = True
except ImportError:
    LAMBDA3_AVAILABLE = False

# ==========================================================
# DATA SOURCE CONFIGURATION
# ==========================================================

@dataclass
class DataSourceConfig:
    """データソース設定"""
    source_name: str
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    retry_attempts: int = 3
    timeout_seconds: int = 30
    cache_duration_hours: int = 24
    enable_async: bool = True

@dataclass
class DataAcquisitionConfig:
    """データ取得設定"""
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None  # None = today
    frequency: str = "daily"  # daily, weekly, monthly
    adjust_prices: bool = True
    fill_missing_method: str = "forward"  # forward, backward, interpolate
    outlier_detection: bool = True
    outlier_threshold: float = 5.0  # sigma
    min_data_points: int = 50
    output_directory: Optional[Path] = None
    save_raw_data: bool = True
    save_processed_data: bool = True
    data_validation: bool = True

# ==========================================================
# UNIFIED DATA ACQUISITION SYSTEM
# ==========================================================

class Lambda3DataAcquisition:
    """
    Lambda³統合データ取得システム
    
    多様なデータソースからの自動取得・前処理・検証を提供。
    構造テンソル分析に最適化されたデータパイプライン。
    """
    
    def __init__(self, config: Optional[DataAcquisitionConfig] = None):
        self.config = config or DataAcquisitionConfig()
        self.data_sources = {}
        self.cache = {}
        self.acquisition_log = []
        
        # データソース初期化
        self._initialize_data_sources()
        
        print(f"🔄 Lambda³ Data Acquisition System initialized")
        print(f"   Available sources: {list(self.data_sources.keys())}")
    
    def _initialize_data_sources(self):
        """データソース初期化"""
        if YFINANCE_AVAILABLE:
            self.data_sources['yahoo'] = DataSourceConfig(
                source_name='yahoo',
                rate_limit_per_minute=2000,  # Yahoo Finance は比較的寛容
            )
        
        if DATAREADER_AVAILABLE:
            self.data_sources['fred'] = DataSourceConfig(
                source_name='fred',
                rate_limit_per_minute=120,
            )
        
        if QUANDL_AVAILABLE:
            self.data_sources['quandl'] = DataSourceConfig(
                source_name='quandl',
                api_key=None,  # 設定で指定
                rate_limit_per_minute=50,
            )
        
        if ALPHAVANTAGE_AVAILABLE:
            self.data_sources['alphavantage'] = DataSourceConfig(
                source_name='alphavantage',
                api_key=None,  # 設定で指定
                rate_limit_per_minute=5,  # 無料版制限
            )
    
    def fetch_financial_markets_comprehensive(
        self,
        asset_categories: Optional[Dict[str, Dict[str, str]]] = None,
        regions: Optional[List[str]] = None,
        custom_tickers: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        包括的金融市場データ取得
        
        Args:
            asset_categories: 資産カテゴリ別ティッカー
            regions: 地域リスト
            custom_tickers: カスタムティッカー
            
        Returns:
            Dict: カテゴリ別データフレーム
        """
        if asset_categories is None:
            asset_categories = self._get_default_asset_categories()
        
        if regions is None:
            regions = ['US', 'Europe', 'Asia']
        
        print(f"\n🌍 Comprehensive Financial Markets Data Acquisition")
        print(f"Asset Categories: {list(asset_categories.keys())}")
        print(f"Regions: {regions}")
        print(f"Period: {self.config.start_date} to {self.config.end_date or 'today'}")
        
        all_market_data = {}
        
        # 各資産カテゴリの取得
        for category, tickers in asset_categories.items():
            print(f"\n📊 Fetching {category} data...")
            
            try:
                category_data = self._fetch_category_data(category, tickers)
                if not category_data.empty:
                    all_market_data[category] = category_data
                    print(f"   ✅ {category}: {category_data.shape[0]} days, {category_data.shape[1]} assets")
                else:
                    print(f"   ❌ {category}: No data retrieved")
            
            except Exception as e:
                print(f"   ❌ {category}: Error - {e}")
                continue
        
        # カスタムティッカー取得
        if custom_tickers:
            print(f"\n📋 Fetching custom tickers...")
            try:
                custom_data = self._fetch_yahoo_batch(custom_tickers)
                if not custom_data.empty:
                    all_market_data['custom'] = custom_data
                    print(f"   ✅ Custom: {custom_data.shape[0]} days, {custom_data.shape[1]} assets")
            except Exception as e:
                print(f"   ❌ Custom: Error - {e}")
        
        # データ統合と前処理
        processed_data = self._post_process_market_data(all_market_data)
        
        # 保存
        if self.config.output_directory:
            self._save_market_data(processed_data)
        
        return processed_data
    
    def fetch_economic_indicators(
        self,
        indicators: Optional[Dict[str, str]] = None,
        regions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        経済指標データ取得
        
        Args:
            indicators: 指標名とコードの辞書
            regions: 対象地域
            
        Returns:
            pd.DataFrame: 経済指標データ
        """
        if not DATAREADER_AVAILABLE:
            raise ImportError("pandas_datareader required for economic indicators")
        
        if indicators is None:
            indicators = self._get_default_economic_indicators()
        
        print(f"\n📈 Economic Indicators Data Acquisition")
        print(f"Indicators: {list(indicators.keys())}")
        
        economic_data = pd.DataFrame()
        
        for indicator_name, fred_code in indicators.items():
            try:
                print(f"   Fetching {indicator_name} ({fred_code})...")
                
                data = fred.FredReader(
                    fred_code,
                    start=self.config.start_date,
                    end=self.config.end_date
                ).read()
                
                if not data.empty:
                    economic_data[indicator_name] = data.iloc[:, 0]
                    print(f"   ✅ {indicator_name}: {len(data)} points")
                else:
                    print(f"   ❌ {indicator_name}: No data")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"   ❌ {indicator_name}: Error - {e}")
                continue
        
        # 前処理
        if not economic_data.empty:
            economic_data = self._preprocess_economic_data(economic_data)
        
        return economic_data
    
    def fetch_cryptocurrency_markets(
        self,
        cryptocurrencies: Optional[Dict[str, str]] = None,
        vs_currencies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        暗号通貨市場データ取得
        
        Args:
            cryptocurrencies: 暗号通貨ティッカー
            vs_currencies: 対基軸通貨
            
        Returns:
            pd.DataFrame: 暗号通貨データ
        """
        if cryptocurrencies is None:
            cryptocurrencies = {
                'Bitcoin': 'BTC-USD',
                'Ethereum': 'ETH-USD',
                'Solana': 'SOL-USD',
                'Cardano': 'ADA-USD',
                'Polygon': 'MATIC-USD',
                'Chainlink': 'LINK-USD'
            }
        
        print(f"\n₿ Cryptocurrency Markets Data Acquisition")
        print(f"Cryptocurrencies: {list(cryptocurrencies.keys())}")
        
        crypto_data = self._fetch_yahoo_batch(cryptocurrencies)
        
        if not crypto_data.empty:
            # 暗号通貨特有の前処理
            crypto_data = self._preprocess_crypto_data(crypto_data)
        
        return crypto_data
    
    def fetch_commodities_complex(
        self,
        commodity_categories: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        商品市場包括データ取得
        
        Args:
            commodity_categories: 商品カテゴリ別ティッカー
            
        Returns:
            Dict: カテゴリ別商品データ
        """
        if commodity_categories is None:
            commodity_categories = {
                'Energy': {
                    'WTI_Oil': 'CL=F',
                    'Brent_Oil': 'BZ=F',
                    'Natural_Gas': 'NG=F',
                    'Gasoline': 'RB=F'
                },
                'Precious_Metals': {
                    'Gold': 'GC=F',
                    'Silver': 'SI=F',
                    'Platinum': 'PL=F',
                    'Palladium': 'PA=F'
                },
                'Industrial_Metals': {
                    'Copper': 'HG=F',
                    'Aluminum': 'ALI=F',
                    'Zinc': 'ZI=F'
                },
                'Agriculture': {
                    'Corn': 'C=F',
                    'Wheat': 'W=F',
                    'Soybeans': 'S=F',
                    'Sugar': 'SB=F',
                    'Coffee': 'KC=F'
                }
            }
        
        print(f"\n🏭 Commodities Complex Data Acquisition")
        
        commodities_data = {}
        
        for category, tickers in commodity_categories.items():
            print(f"\n   📦 Fetching {category}...")
            
            try:
                category_data = self._fetch_yahoo_batch(tickers)
                if not category_data.empty:
                    commodities_data[category] = category_data
                    print(f"   ✅ {category}: {category_data.shape[1]} commodities")
                else:
                    print(f"   ❌ {category}: No data")
            
            except Exception as e:
                print(f"   ❌ {category}: Error - {e}")
                continue
        
        return commodities_data
    
    def fetch_fx_majors_and_crosses(
        self,
        base_currencies: Optional[List[str]] = None,
        quote_currencies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        主要通貨ペア・クロス通貨データ取得
        
        Args:
            base_currencies: 基軸通貨リスト
            quote_currencies: 対象通貨リスト
            
        Returns:
            pd.DataFrame: 為替データ
        """
        if base_currencies is None:
            base_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD']
        
        if quote_currencies is None:
            quote_currencies = ['USD', 'EUR', 'JPY', 'GBP']
        
        print(f"\n💱 FX Majors and Crosses Data Acquisition")
        
        fx_pairs = {}
        
        for base in base_currencies:
            for quote in quote_currencies:
                if base != quote:
                    pair_name = f"{base}/{quote}"
                    
                    # Yahoo Finance format
                    if quote == 'USD':
                        ticker = f"{base}=X"
                    elif base == 'USD':
                        ticker = f"{quote}=X"
                        pair_name = f"USD/{quote}"
                    else:
                        ticker = f"{base}{quote}=X"
                    
                    fx_pairs[pair_name] = ticker
        
        print(f"   Currency pairs: {len(fx_pairs)}")
        
        fx_data = self._fetch_yahoo_batch(fx_pairs)
        
        if not fx_data.empty:
            # FX特有の前処理
            fx_data = self._preprocess_fx_data(fx_data)
        
        return fx_data
    
    def fetch_custom_data_from_file(
        self,
        filepath: Union[str, Path],
        date_column: str = 'Date',
        value_columns: Optional[List[str]] = None,
        sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        カスタムファイルからデータ取得
        
        Args:
            filepath: ファイルパス
            date_column: 日付列名
            value_columns: 値列名リスト
            sheet_name: Excelシート名
            
        Returns:
            pd.DataFrame: ファイルデータ
        """
        filepath = Path(filepath)
        
        print(f"\n📄 Custom File Data Loading: {filepath.name}")
        
        try:
            if filepath.suffix.lower() == '.csv':
                data = pd.read_csv(filepath, parse_dates=[date_column], index_col=date_column)
            elif filepath.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(filepath, sheet_name=sheet_name, 
                                   parse_dates=[date_column], index_col=date_column)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            print(f"   ✅ Loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # 列選択
            if value_columns:
                available_columns = [col for col in value_columns if col in data.columns]
                data = data[available_columns]
                print(f"   📊 Selected columns: {available_columns}")
            
            # 前処理
            data = self._preprocess_custom_data(data)
            
            return data
            
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            return pd.DataFrame()
    
    async def fetch_realtime_data_stream(
        self,
        tickers: List[str],
        duration_minutes: int = 60,
        update_interval_seconds: int = 60
    ) -> Dict[str, List[Dict]]:
        """
        リアルタイムデータストリーミング取得
        
        Args:
            tickers: ティッカーリスト
            duration_minutes: 取得期間（分）
            update_interval_seconds: 更新間隔（秒）
            
        Returns:
            Dict: リアルタイムデータ
        """
        print(f"\n🔴 Real-time Data Stream: {len(tickers)} tickers for {duration_minutes} minutes")
        
        realtime_data = {ticker: [] for ticker in tickers}
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            timestamp = datetime.now()
            
            for ticker in tickers:
                try:
                    # Yahoo Finance からリアルタイム価格取得
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.history(period='1d', interval='1m').tail(1)
                    
                    if not info.empty:
                        current_price = float(info['Close'].iloc[0])
                        volume = float(info['Volume'].iloc[0])
                        
                        realtime_data[ticker].append({
                            'timestamp': timestamp,
                            'price': current_price,
                            'volume': volume
                        })
                
                except Exception as e:
                    print(f"   ⚠️ Error fetching {ticker}: {e}")
                    continue
            
            print(f"   📊 Update: {timestamp.strftime('%H:%M:%S')} - {len(tickers)} tickers")
            await asyncio.sleep(update_interval_seconds)
        
        return realtime_data
    
    def _get_default_asset_categories(self) -> Dict[str, Dict[str, str]]:
        """デフォルト資産カテゴリ"""
        return {
            'US_Equities': {
                'S&P500': '^GSPC',
                'NASDAQ': '^IXIC',
                'Dow_Jones': '^DJI',
                'Russell_2000': '^RUT',
                'VIX': '^VIX'
            },
            'International_Equities': {
                'Nikkei225': '^N225',
                'FTSE100': '^FTSE',
                'DAX': '^GDAXI',
                'CAC40': '^FCHI',
                'Hang_Seng': '^HSI'
            },
            'Bonds': {
                'US_10Y': '^TNX',
                'US_30Y': '^TYX',
                'US_2Y': '^IRX',
                'German_10Y': '^TNX'  # Note: Limited availability
            },
            'Commodities': {
                'Gold': 'GC=F',
                'Silver': 'SI=F',
                'Oil_WTI': 'CL=F',
                'Copper': 'HG=F'
            },
            'Currencies': {
                'USD_JPY': 'JPY=X',
                'EUR_USD': 'EURUSD=X',
                'GBP_USD': 'GBPUSD=X',
                'USD_CHF': 'CHF=X'
            }
        }
    
    def _get_default_economic_indicators(self) -> Dict[str, str]:
        """デフォルト経済指標"""
        return {
            'GDP_Growth': 'GDP',
            'Unemployment_Rate': 'UNRATE',
            'Inflation_CPI': 'CPIAUCSL',
            'Fed_Funds_Rate': 'FEDFUNDS',
            'Consumer_Confidence': 'UMCSENT',
            'Industrial_Production': 'INDPRO',
            'Housing_Starts': 'HOUST',
            'Retail_Sales': 'RSAFS'
        }
    
    def _fetch_category_data(self, category: str, tickers: Dict[str, str]) -> pd.DataFrame:
        """カテゴリ別データ取得"""
        return self._fetch_yahoo_batch(tickers)
    
    def _fetch_yahoo_batch(self, tickers: Dict[str, str]) -> pd.DataFrame:
        """Yahoo Finance バッチ取得"""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required for Yahoo Finance data")
        
        ticker_symbols = list(tickers.values())
        ticker_names = list(tickers.keys())
        
        try:
            # バッチダウンロード
            data = yf.download(
                ticker_symbols,
                start=self.config.start_date,
                end=self.config.end_date,
                group_by='ticker'
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # データ整形
            if len(ticker_symbols) == 1:
                # 単一ティッカーの場合
                close_data = pd.DataFrame({ticker_names[0]: data['Close']})
            else:
                # 複数ティッカーの場合
                close_data = pd.DataFrame()
                for symbol, name in zip(ticker_symbols, ticker_names):
                    if symbol in data.columns.levels[0]:
                        close_data[name] = data[symbol]['Close']
            
            # 欠損値処理
            close_data = close_data.dropna()
            
            return close_data
        
        except Exception as e:
            print(f"   Error in Yahoo Finance batch fetch: {e}")
            return pd.DataFrame()
    
    def _preprocess_economic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """経済指標データ前処理"""
        # 月次・四半期データの日次変換
        data = data.resample('D').ffill()
        
        # 単位統一・スケーリング
        for col in data.columns:
            if 'Rate' in col or 'Inflation' in col:
                # パーセンテージデータの正規化
                data[col] = data[col] / 100
        
        return data
    
    def _preprocess_crypto_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """暗号通貨データ前処理"""
        # 対数変換（価格の極端な変動対応）
        for col in data.columns:
            data[f'{col}_log'] = np.log(data[col])
        
        # ボラティリティ計算
        for col in data.columns:
            if '_log' not in col:
                data[f'{col}_volatility'] = data[col].pct_change().rolling(window=7).std()
        
        return data
    
    def _preprocess_fx_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """FX データ前処理"""
        # 為替レート正規化
        for col in data.columns:
            # 日次リターン計算
            data[f'{col}_return'] = data[col].pct_change()
            
            # 移動平均
            data[f'{col}_ma20'] = data[col].rolling(window=20).mean()
            data[f'{col}_ma50'] = data[col].rolling(window=50).mean()
        
        return data
    
    def _preprocess_custom_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """カスタムデータ前処理"""
        # 基本クリーニング
        data = data.select_dtypes(include=[np.number])
        data = data.fillna(method=self.config.fill_missing_method)
        
        # 外れ値検出・処理
        if self.config.outlier_detection:
            for col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                outlier_mask = (
                    (data[col] < (Q1 - 1.5 * IQR)) | 
                    (data[col] > (Q3 + 1.5 * IQR))
                )
                
                if outlier_mask.sum() > 0:
                    print(f"   ⚠️ Outliers detected in {col}: {outlier_mask.sum()} points")
                    # 外れ値を前後の値で補間
                    data.loc[outlier_mask, col] = np.nan
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _post_process_market_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """市場データ後処理"""
        processed_data = {}
        
        for category, data in market_data.items():
            if data.empty:
                continue
            
            # 基本統計計算
            print(f"\n   📊 Processing {category}:")
            print(f"      Shape: {data.shape}")
            print(f"      Date range: {data.index[0]} to {data.index[-1]}")
            
            # リターン計算
            returns_data = data.pct_change().dropna()
            returns_data.columns = [f"{col}_return" for col in returns_data.columns]
            
            # 統合データフレーム
            combined_data = pd.concat([data, returns_data], axis=1)
            
            processed_data[category] = combined_data
        
        return processed_data
    
    def _save_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """市場データ保存"""
        if not self.config.output_directory:
            return
        
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for category, data in market_data.items():
            filename = f"lambda3_{category}_{timestamp}.csv"
            filepath = output_dir / filename
            
            data.to_csv(filepath)
            print(f"   💾 Saved {category}: {filepath}")
    
    def get_acquisition_summary(self) -> Dict[str, Any]:
        """取得サマリー"""
        return {
            'config': self.config.__dict__,
            'available_sources': list(self.data_sources.keys()),
            'acquisition_history': self.acquisition_log,
            'cache_size': len(self.cache)
        }

# ==========================================================
# CONVENIENCE FUNCTIONS
# ==========================================================

def quick_financial_data_fetch(
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None,
    asset_types: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    クイック金融データ取得
    
    Args:
        start_date: 開始日
        end_date: 終了日
        asset_types: 資産タイプリスト
        
    Returns:
        Dict: 資産タイプ別データ
    """
    if asset_types is None:
        asset_types = ['US_Equities', 'Currencies', 'Commodities']
    
    config = DataAcquisitionConfig(
        start_date=start_date,
        end_date=end_date
    )
    
    acquisition = Lambda3DataAcquisition(config)
    
    # 資産カテゴリをフィルタリング
    all_categories = acquisition._get_default_asset_categories()
    selected_categories = {k: v for k, v in all_categories.items() if k in asset_types}
    
    return acquisition.fetch_financial_markets_comprehensive(selected_categories)

def setup_data_acquisition_config(
    output_directory: str,
    frequency: str = "daily",
    outlier_detection: bool = True
) -> DataAcquisitionConfig:
    """
    データ取得設定セットアップ
    
    Args:
        output_directory: 出力ディレクトリ
        frequency: データ頻度
        outlier_detection: 外れ値検出有効化
        
    Returns:
        DataAcquisitionConfig: 設定オブジェクト
    """
    return DataAcquisitionConfig(
        output_directory=Path(output_directory),
        frequency=frequency,
        outlier_detection=outlier_detection,
        save_raw_data=True,
        save_processed_data=True
    )

# ==========================================================
# EXAMPLE USAGE
# ==========================================================

if __name__ == "__main__":
    print("Lambda³ Real Data Acquisition System Test")
    print("=" * 60)
    
    # データ取得テスト
    config = DataAcquisitionConfig(
        start_date="2023-01-01",
        end_date="2024-01-01",
        output_directory=Path("./data_output")
    )
    
    acquisition = Lambda3DataAcquisition(config)
    
    # 包括的市場データ取得テスト
    print("\n1. Testing comprehensive market data acquisition...")
    market_data = acquisition.fetch_financial_markets_comprehensive()
    
    for category, data in market_data.items():
        print(f"   {category}: {data.shape}")
    
    # 暗号通貨データ取得テスト
    if YFINANCE_AVAILABLE:
        print("\n2. Testing cryptocurrency data acquisition...")
        crypto_data = acquisition.fetch_cryptocurrency_markets()
        print(f"   Cryptocurrency data: {crypto_data.shape}")
    
    # 商品データ取得テスト
    print("\n3. Testing commodities data acquisition...")
    commodities_data = acquisition.fetch_commodities_complex()
    
    for category, data in commodities_data.items():
        print(f"   {category}: {data.shape}")
    
    print("\n✅ Data acquisition system test completed!")
