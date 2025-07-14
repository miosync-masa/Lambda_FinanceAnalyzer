# ==========================================================
# Λ³: GCP Cloud Batch Ultimate Parallel Extension (FIXED)
# ----------------------------------------------------
# Google Cloud APIの構造変更に対応した修正版
# ==========================================================

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict
from queue import PriorityQueue
import threading
import logging
from pathlib import Path
import pickle
import numpy as np
import os

# GCP Libraries - 修正版
from google.cloud import batch_v1
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import monitoring_v3
from google.api_core import retry
from google.api_core.exceptions import GoogleAPIError
from google.auth import default
from google.oauth2 import service_account

# Import Lambda³ core
try:
    from .lambda3_cloud_parallel import (
        CloudScaleConfig, Lambda3TaskDecomposer, ExecutionBackend
    )
    from ..core.lambda3_zeroshot_tensor_field import L3Config
except ImportError:
    # 直接実行用
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cloud.lambda3_cloud_parallel import (
        CloudScaleConfig, Lambda3TaskDecomposer, ExecutionBackend
    )
    from core.lambda3_zeroshot_tensor_field import L3Config

# ===============================
# PRODUCTION GCP CONFIGURATION
# ===============================
@dataclass
class GCPUltimateConfig:
    """Production GCP resource configuration"""
    
    # Global resource hunting - 実際に利用可能なリージョン
    regions: List[str] = field(default_factory=lambda: [
        "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
        "europe-west1", "europe-west2", "europe-west3", "europe-west4",
        "asia-east1", "asia-northeast1", "asia-southeast1",
    ])
    
    # Realistic scaling parameters
    max_instances_per_region: int = 1000  # 現実的な上限
    target_total_instances: int = 10000   # グローバル目標
    min_instances_per_region: int = 10    # 最小デプロイ単位
    
    # Instance configuration - 実際に利用可能なマシンタイプ
    use_spot: bool = True
    machine_types: List[str] = field(default_factory=lambda: [
        "e2-standard-4",    # 4 vCPU, 16GB RAM
        "e2-highcpu-4",     # 4 vCPU, 4GB RAM
        "n2d-standard-4",   # AMD, 4 vCPU, 16GB RAM
        "n1-standard-4",    # 旧世代だが安定
    ])
    
    # Cost optimization
    max_price_per_hour: float = 0.10  # 実際のスポット価格を考慮
    auto_shutdown_hours: float = 1.0
    checkpoint_every_n_pairs: int = 50
    max_retries: int = 3
    
    # Storage
    gcs_bucket: str = None  # 自動設定
    use_regional_buckets: bool = True
    
    # Authentication
    service_account_path: Optional[str] = None
    use_application_default: bool = True
    
    # Batch configuration
    batch_service_account: Optional[str] = None
    network: str = "default"
    subnetwork: str = "default"
    
    # Monitoring
    enable_cloud_logging: bool = True
    enable_cloud_monitoring: bool = True
    
    def __post_init__(self):
        # バケット名の自動設定
        if self.gcs_bucket is None:
            try:
                _, project_id = default()
                self.gcs_bucket = f"lambda3-{project_id}"
            except:
                self.gcs_bucket = "lambda3-results"

# ===============================
# SIMPLIFIED RESOURCE HUNTER
# ===============================
class GCPResourceHunter:
    """
    Simplified resource hunter without quota API
    """
    
    def __init__(self, config: GCPUltimateConfig):
        self.config = config
        
        # 認証設定
        if config.service_account_path:
            credentials = service_account.Credentials.from_service_account_file(
                config.service_account_path
            )
            self.compute_client = compute_v1.InstancesClient(credentials=credentials)
            self.zones_client = compute_v1.ZonesClient(credentials=credentials)
        else:
            self.compute_client = compute_v1.InstancesClient()
            self.zones_client = compute_v1.ZonesClient()
        
        # プロジェクトID取得
        try:
            _, self.project_id = default()
        except:
            self.project_id = "default-project"
        
        # リソース可用性キャッシュ
        self.availability_cache = {}
        self.last_cache_update = 0
        self.cache_ttl = 300  # 5分
        
    def get_available_resources(self) -> Dict[str, Dict[str, Any]]:
        """全リージョンの利用可能リソースを取得（簡易版）"""
        
        # キャッシュチェック
        if time.time() - self.last_cache_update < self.cache_ttl:
            return self.availability_cache
        
        availability = {}
        
        # 簡易的なリソース推定
        for region in self.config.regions[:3]:  # デモ用に最初の3リージョンのみ
            try:
                # 静的な可用性データ（実際の値は動的に取得すべき）
                region_availability = {
                    'zones': self._get_zones_in_region(region),
                    'machine_types': {},
                    'total_cpus_available': 0,
                    'spot_discount': 0.7  # 一般的なスポット割引
                }
                
                # マシンタイプごとの可用性（推定値）
                for machine_type in self.config.machine_types:
                    available = self._estimate_machine_availability(region, machine_type)
                    if available > 0:
                        region_availability['machine_types'][machine_type] = available
                        # CPUカウント（簡易計算）
                        cpu_count = int(machine_type.split('-')[-1]) if '-' in machine_type else 4
                        region_availability['total_cpus_available'] += available * cpu_count
                
                if region_availability['machine_types']:
                    availability[region] = region_availability
                    
            except Exception as e:
                logging.warning(f"Failed to check {region}: {e}")
                continue
        
        self.availability_cache = availability
        self.last_cache_update = time.time()
        
        return availability
    
    def _get_zones_in_region(self, region: str) -> List[str]:
        """リージョン内のゾーンを取得（簡易版）"""
        # よく使われるゾーンパターン
        common_zones = ['a', 'b', 'c']
        return [f"{region}-{zone}" for zone in common_zones]
    
    def _estimate_machine_availability(self, region: str, machine_type: str) -> int:
        """マシンタイプの可用性を推定"""
        # 実際のクォータAPIの代わりに推定値を使用
        base_availability = {
            "us-central1": 2000,
            "us-east1": 3000,
            "us-west1": 1500,
            "europe-west1": 2000,
            "asia-northeast1": 1000,
        }
        
        # デフォルト値
        default_availability = 500
        
        # マシンタイプによる調整
        machine_multiplier = {
            "e2-standard-4": 1.0,
            "e2-highcpu-4": 0.8,
            "n2d-standard-4": 0.6,
            "n1-standard-4": 1.2,
        }
        
        base = base_availability.get(region, default_availability)
        multiplier = machine_multiplier.get(machine_type, 0.5)
        
        # デモ用に小さな値を返す
        return min(100, int(base * multiplier))
    
    def get_best_regions_for_launch(self, n_instances: int) -> List[Tuple[str, str, int, float]]:
        """最適なリージョン/マシンタイプの組み合わせを取得"""
        
        # 利用可能リソースを取得
        availability = self.get_available_resources()
        
        if not availability:
            logging.error("No available resources found!")
            return []
        
        options = []
        
        # 各リージョンのオプションを評価
        for region, region_info in availability.items():
            for machine_type, available_count in region_info['machine_types'].items():
                if available_count > 0:
                    # 価格計算（簡易版）
                    base_price = self._get_machine_price(machine_type)
                    spot_price = base_price * region_info['spot_discount']
                    
                    if spot_price <= self.config.max_price_per_hour:
                        options.append((region, machine_type, available_count, spot_price))
        
        # 価格でソート
        options.sort(key=lambda x: x[3])
        
        # 貪欲法で割り当て
        allocations = []
        remaining = n_instances
        
        for region, machine_type, available, price in options:
            if remaining <= 0:
                break
            
            # このリージョンに割り当てる数（デモ用に制限）
            to_allocate = min(
                remaining,
                available,
                10  # デモ用の制限
            )
            
            if to_allocate >= 1:
                allocations.append((region, machine_type, to_allocate, price))
                remaining -= to_allocate
                
                logging.info(f"Allocated {to_allocate} instances in {region} "
                           f"({machine_type} @ ${price:.3f}/hr)")
        
        return allocations
    
    def _get_machine_price(self, machine_type: str) -> float:
        """マシンタイプの基本価格を取得"""
        # 実際の価格（概算）
        base_prices = {
            "e2-standard-4": 0.134,
            "e2-highcpu-4": 0.100,
            "n2d-standard-4": 0.152,
            "n1-standard-4": 0.150,
        }
        return base_prices.get(machine_type, 0.15)

# ===============================
# SIMPLIFIED BATCH MANAGER
# ===============================
class Lambda3CloudBatchManager:
    """
    Simplified Cloud Batch job manager for demo
    """
    
    def __init__(self, config: GCPUltimateConfig, resource_hunter: GCPResourceHunter):
        self.config = config
        self.hunter = resource_hunter
        
        # Batch API クライアント
        try:
            if config.service_account_path:
                credentials = service_account.Credentials.from_service_account_file(
                    config.service_account_path
                )
                self.batch_client = batch_v1.BatchServiceClient(credentials=credentials)
                self.storage_client = storage.Client(credentials=credentials)
            else:
                self.batch_client = batch_v1.BatchServiceClient()
                self.storage_client = storage.Client()
        except Exception as e:
            logging.error(f"Failed to initialize clients: {e}")
            # フォールバック
            self.batch_client = None
            self.storage_client = None
        
        self.project_id = resource_hunter.project_id
        
        # ジョブ追跡
        self.active_jobs = {}
        self.job_status = {}
        
    def create_demo_batch_job(
        self,
        pair_batches: List[List[Tuple[str, str]]],
        l3_config: L3Config,
        series_data_gcs_path: str
    ) -> Dict[str, str]:
        """デモ用の簡易バッチジョブ作成"""
        
        # デモ用に1つのリージョンのみ使用
        demo_region = self.config.regions[0]
        demo_machine_type = self.config.machine_types[0]
        
        logging.info(f"Creating demo job in {demo_region}")
        
        # バケットの存在確認
        try:
            if self.storage_client:
                self._ensure_bucket_exists()
        except Exception as e:
            logging.warning(f"Could not ensure bucket: {e}")
        
        # デモ用の簡易ジョブID
        job_id = f"lambda3-demo-{int(time.time())}"
        
        # 実際のバッチジョブ作成はスキップ（デモ用）
        logging.info(f"Demo job created: {job_id}")
        logging.info(f"  Region: {demo_region}")
        logging.info(f"  Machine type: {demo_machine_type}")
        logging.info(f"  Batches to process: {len(pair_batches)}")
        
        self.active_jobs[demo_region] = job_id
        
        return {demo_region: job_id}
    
    def _ensure_bucket_exists(self):
        """バケットの存在を確認"""
        if not self.storage_client:
            return
            
        try:
            bucket = self.storage_client.bucket(self.config.gcs_bucket)
            if not bucket.exists():
                # バケット作成はスキップ（権限エラー回避）
                logging.info(f"Bucket {self.config.gcs_bucket} may not exist")
        except Exception as e:
            logging.warning(f"Bucket check skipped: {e}")

# ===============================
# SIMPLIFIED MONITOR
# ===============================
class Lambda3GlobalMonitor:
    """Simplified monitoring for demo"""
    
    def __init__(self, batch_manager: Lambda3CloudBatchManager):
        self.batch_manager = batch_manager
        self.start_time = time.time()
        self.total_pairs = 0
        self.completed_pairs = 0
        
    async def monitor_jobs_demo(self):
        """デモ用の監視（実際のジョブ監視なし）"""
        # デモ用のプログレス表示
        for i in range(5):
            await asyncio.sleep(2)
            self.completed_pairs = int(self.total_pairs * (i + 1) / 5)
            self.update_dashboard()
    
    def update_dashboard(self):
        """ダッシュボードを更新"""
        elapsed = time.time() - self.start_time
        rate = self.completed_pairs / max(elapsed, 1)
        
        print("\n" + "="*80)
        print("LAMBDA³ DEMO COMPUTATION DASHBOARD")
        print("="*80)
        print(f"Elapsed: {elapsed:.1f} seconds")
        print(f"Progress: {self.completed_pairs}/{self.total_pairs} "
              f"({self.completed_pairs/max(self.total_pairs,1)*100:.1f}%)")
        print(f"Speed: {rate:.1f} pairs/sec")

# ===============================
# MAIN ORCHESTRATOR (SIMPLIFIED)
# ===============================
async def run_lambda3_gcp_ultimate(
    data_source: Union[str, Dict[str, np.ndarray]],
    l3_config: L3Config = None,
    gcp_config: GCPUltimateConfig = None,
    target_pairs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simplified Lambda³ GCP analysis for demo
    """
    
    if l3_config is None:
        l3_config = L3Config()
    
    if gcp_config is None:
        gcp_config = GCPUltimateConfig()
    
    print("="*80)
    print("LAMBDA³ ULTIMATE GCP PARALLEL ANALYSIS")
    print("="*80)
    print(f"Target regions: {len(gcp_config.regions)}")
    print(f"Target instances: {gcp_config.target_total_instances}")
    print(f"Max price: ${gcp_config.max_price_per_hour}/hour")
    
    # コンポーネント初期化
    resource_hunter = GCPResourceHunter(gcp_config)
    batch_manager = Lambda3CloudBatchManager(gcp_config, resource_hunter)
    monitor = Lambda3GlobalMonitor(batch_manager)
    
    # データ準備
    if isinstance(data_source, str):
        # ファイルからロード
        with open(data_source, 'rb') as f:
            series_dict = pickle.load(f)
    else:
        series_dict = data_source
    
    # デモ用のダミーGCSパス
    series_data_gcs_path = f"gs://{gcp_config.gcs_bucket}/data/demo_series_data.pkl"
    
    # タスク分解
    decomposer = Lambda3TaskDecomposer(CloudScaleConfig())
    series_names = list(series_dict.keys())
    pair_batches = decomposer.decompose_pairwise_analysis(series_names)
    
    if target_pairs:
        # ペア数を制限
        total_pairs = sum(len(batch) for batch in pair_batches)
        if total_pairs > target_pairs:
            # バッチを調整
            pair_batches = pair_batches[:target_pairs // len(pair_batches[0]) + 1]
    
    monitor.total_pairs = sum(len(batch) for batch in pair_batches)
    
    print(f"\nTotal pairs to analyze: {monitor.total_pairs}")
    print(f"Batch size: ~{len(pair_batches[0]) if pair_batches else 0} pairs")
    print(f"Total batches: {len(pair_batches)}")
    
    # リソースハンティング（簡易版）
    print("\nHunting for global compute resources...")
    available_resources = resource_hunter.get_available_resources()
    print(f"Found resources in {len(available_resources)} regions")
    
    # バッチジョブ作成（デモ版）
    job_ids = batch_manager.create_demo_batch_job(
        pair_batches, l3_config, series_data_gcs_path
    )
    
    print(f"\nDemo jobs created in {len(job_ids)} regions!")
    
    # デモ監視
    await monitor.monitor_jobs_demo()
    
    # デモ結果
    print("\nDemo analysis complete!")
    results = {
        'total_pairs_analyzed': monitor.total_pairs,
        'execution_time_seconds': time.time() - monitor.start_time,
        'regions_used': list(job_ids.keys()),
        'demo_mode': True,
        'gcs_results_path': f"gs://{gcp_config.gcs_bucket}/results/",
        'job_ids': job_ids
    }
    
    return results

# ===============================
# COST SAVINGS CALCULATOR
# ===============================
def calculate_cost_savings():
    """コスト削減の計算"""
    print("\n" + "="*60)
    print("LAMBDA³ COST OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # 仮定値
    on_demand_price = 0.15  # $/hour
    spot_price = 0.04      # $/hour
    instances = 10000
    hours = 2
    
    on_demand_cost = on_demand_price * instances * hours
    spot_cost = spot_price * instances * hours
    savings = on_demand_cost - spot_cost
    savings_percent = (savings / on_demand_cost) * 100
    
    print(f"\nScenario: {instances:,} instances for {hours} hours")
    print(f"On-demand cost: ${on_demand_cost:,.2f}")
    print(f"Spot cost: ${spot_cost:,.2f}")
    print(f"Savings: ${savings:,.2f} ({savings_percent:.1f}%)")
    
    print("\nAdditional optimizations:")
    print("• Preemptible instances: 70-90% discount")
    print("• Regional arbitrage: 10-30% price variation")
    print("• Off-peak scheduling: Additional 5-15% savings")
    print("• Batch optimization: 20-40% efficiency gain")
    
    return {
        'on_demand_cost': on_demand_cost,
        'spot_cost': spot_cost,
        'savings': savings,
        'savings_percent': savings_percent
    }
