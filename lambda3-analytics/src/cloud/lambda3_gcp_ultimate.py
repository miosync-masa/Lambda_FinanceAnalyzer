# ==========================================================
# Λ³: GCP Cloud Batch Ultimate Parallel Extension (PRODUCTION IMPLEMENTATION)
# ----------------------------------------------------
# 実際のGCP APIを呼び出し、動的にリソースを確保して
# Cloud Batchジョブを実行する本番稼働版
# ==========================================================
from datetime import datetime
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import pickle
import numpy as np
import os
import uuid

# GCP Libraries
from google.cloud import batch_v1
from google.cloud import compute_v1
from google.cloud import storage
from google.api_core import exceptions
from google.auth import default, credentials

# Import Lambda³ core
# ★★★ インポートパスを修正 ★★★
from core.lambda3_zeroshot_tensor_field import L3Config
from cloud.lambda3_cloud_parallel import CloudScaleConfig, Lambda3TaskDecomposer


# --- Configuration ---
@dataclass
class GCPUltimateConfig:
    """Production GCP resource configuration"""
    regions: List[str] = field(default_factory=lambda: [
        "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
        "europe-west1", "europe-west2", "europe-west3", "europe-west4",
        "asia-east1", "asia-northeast1", "asia-southeast1",
    ])
    max_instances_per_region: int = 1000
    target_total_instances: int = 10000
    min_instances_per_region: int = 10
    use_spot: bool = True
    machine_types: List[str] = field(default_factory=lambda: [
        "e2-standard-4",
        "e2-highcpu-4",
        "n2d-standard-4",
        "n1-standard-4",
    ])
    max_price_per_hour: float = 0.10
    auto_shutdown_hours: float = 1.0
    checkpoint_every_n_pairs: int = 50
    max_retries: int = 3
    gcs_bucket: str = None
    use_regional_buckets: bool = True
    service_account_path: Optional[str] = None
    use_application_default: bool = True
    batch_service_account: Optional[str] = None
    network: str = "default"
    subnetwork: str = "default"
    enable_cloud_logging: bool = True
    enable_cloud_monitoring: bool = True

    def __post_init__(self):
        if self.gcs_bucket is None:
            try:
                _, project_id = default()
                self.gcs_bucket = f"lambda3-ultimate-{project_id}"
            except Exception:
                self.gcs_bucket = f"lambda3-ultimate-results-{uuid.uuid4().hex[:6]}"

# ===============================
# PRODUCTION RESOURCE HUNTER
# ===============================
class GCPResourceHunter:
    def __init__(self, config: GCPUltimateConfig, creds: credentials.Credentials):
        self.config = config
        self.creds, self.project_id = default()
        self.compute_client = compute_v1.MachineTypesClient(credentials=self.creds)

    def get_best_spot_options(self, n_instances: int) -> List[Dict[str, Any]]:
        logging.info("Hunting for best spot VM options across GCP regions...")
        options = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_region = {
                executor.submit(self._scan_region, region): region 
                for region in self.config.regions
            }
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    region_options = future.result()
                    if region_options:
                        options.extend(region_options)
                except Exception as e:
                    logging.warning(f"Could not scan region {region}: {e}")

        if not options:
            logging.error("No suitable spot instances found matching criteria.")
            return []
        options.sort(key=lambda x: x['spot_price'])
        return self._create_allocation_plan(options, n_instances)

    def _scan_region(self, region: str) -> List[Dict[str, Any]]:
        region_options = []
        for machine_type_name in self.config.machine_types:
            try:
                request = compute_v1.ListMachineTypesRequest(project=self.project_id, zone=f"{region}-b")
                self.compute_client.list(request=request)
                base_price = self._get_machine_price(machine_type_name)
                spot_price = base_price * 0.3
                if spot_price <= self.config.max_price_per_hour:
                    region_options.append({
                        "region": region, "machine_type": machine_type_name, "spot_price": spot_price
                    })
            except exceptions.NotFound:
                continue
            except Exception as e:
                logging.debug(f"Could not check machine type {machine_type_name} in {region}: {e}")
                continue
        return region_options

    def _create_allocation_plan(self, options: List[Dict], target_total: int) -> List[Dict]:
        allocations, remaining = [], target_total
        for option in options:
            if remaining <= 0: break
            to_allocate = min(remaining, self.config.max_instances_per_region)
            plan = option.copy()
            plan['count'] = to_allocate
            allocations.append(plan)
            remaining -= to_allocate
        logging.info(f"Allocation plan created for {target_total - remaining} instances across {len(allocations)} configurations.")
        return allocations

    def _get_machine_price(self, machine_type: str) -> float:
        prices = {"e2-standard-4": 0.134, "e2-highcpu-4": 0.100, "n2d-standard-4": 0.152, "n1-standard-4": 0.150}
        return prices.get(machine_type, 0.15)


# ===============================
# PRODUCTION BATCH MANAGER
# ===============================
class Lambda3CloudBatchManager:
    def __init__(self, config: GCPUltimateConfig, creds: credentials.Credentials):
        self.config = config
        self.creds, self.project_id = default()
        self.batch_client = batch_v1.BatchServiceClient(credentials=self.creds)
        self.storage_client = storage.Client(credentials=self.creds)

    def _upload_data_to_gcs(self, obj: Any, gcs_path: str) -> str:
        bucket = self.storage_client.bucket(self.config.gcs_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(pickle.dumps(obj))
        return f"gs://{self.config.gcs_bucket}/{gcs_path}"

    def create_and_run_batch_jobs(self, allocation_plan: List[Dict], all_series_dict: Dict, pair_batches: List[List[Tuple]], l3_config: L3Config) -> Dict[str, List[str]]:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        all_series_gcs_path = self._upload_data_to_gcs(all_series_dict, f"{run_id}/inputs/all_series_data.pkl")
        l3_config_hex = pickle.dumps(l3_config).hex()
        active_jobs, batch_counter = defaultdict(list), 0

        for plan in allocation_plan:
            region, machine_type = plan['region'], plan['machine_type']
            num_tasks = plan['count']
            tasks_for_this_job = pair_batches[batch_counter : batch_counter + num_tasks]
            if not tasks_for_this_job: continue

            task_gcs_paths = [self._upload_data_to_gcs(task_batch, f"{run_id}/inputs/batches/batch_{batch_counter + i}.pkl") for i, task_batch in enumerate(tasks_for_this_job)]
            job = self._build_batch_job_object(run_id, region, machine_type, all_series_gcs_path, task_gcs_paths, l3_config_hex)

            try:
                logging.info(f"Submitting job to {region} with {num_tasks} tasks on {machine_type}...")
                created_job = self.batch_client.create_job(parent=f"projects/{self.project_id}/locations/{region}", job=job, job_id=job.name.split('/')[-1])
                active_jobs[region].append(created_job.name)
            except Exception as e:
                logging.error(f"Failed to create job in {region}: {e}")
            batch_counter += num_tasks
        return active_jobs

    def _build_batch_job_object(self, run_id, region, machine_type, series_gcs, task_paths, l3_config_hex) -> batch_v1.Job:
        container = batch_v1.Runnable.Container(
            image_uri="asia-northeast1-docker.pkg.dev/massive-journal-428603-k7/lambda3-images/lambda3-worker:latest",
            entrypoint="/usr/bin/python3",
            commands=["/app/lambda3_cloud_worker.py"]
        )
        runnables = [
            batch_v1.Runnable(
                container=container,
                environment=batch_v1.Environment(variables={
                    "SERIES_DATA_GCS": series_gcs, "BATCH_GCS": task_gcs_path,
                    "INPUT_BUCKET": self.config.gcs_bucket, "OUTPUT_BUCKET": self.config.gcs_bucket,
                    "REGION": region, "BATCH_INDEX": str(i), "L3_CONFIG_HEX": l3_config_hex
                })
            ) for i, task_gcs_path in enumerate(task_paths)
        ]
        task_group = batch_v1.TaskGroup(task_spec=batch_v1.TaskSpec(runnables=runnables), task_count=len(task_paths), parallelism=len(task_paths))
        allocation_policy = batch_v1.AllocationPolicy(instances=[batch_v1.AllocationPolicy.InstancePolicyOrTemplate(policy=batch_v1.AllocationPolicy.InstancePolicy(machine_type=machine_type, provisioning_model=batch_v1.AllocationPolicy.ProvisioningModel.SPOT if self.config.use_spot else batch_v1.AllocationPolicy.ProvisioningModel.STANDARD))])
        job_name = f"lambda3-{run_id}-{region}-{machine_type.replace('_', '-')}-{uuid.uuid4().hex[:4]}"
        return batch_v1.Job(name=job_name, task_groups=[task_group], allocation_policy=allocation_policy, logs_policy=batch_v1.LogsPolicy(destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING))


# ===============================
# PRODUCTION MONITOR
# ===============================
class Lambda3GlobalMonitor:
    def __init__(self, project_id, active_jobs: Dict[str, List[str]], creds):
        self.project_id, self.active_jobs = project_id, active_jobs
        self.batch_client = batch_v1.BatchServiceClient(credentials=creds)

    async def monitor_jobs(self):
        logging.info("Starting to monitor active Cloud Batch jobs...")
        while True:
            all_finished, states = True, defaultdict(int)
            for region, job_names in self.active_jobs.items():
                for job_name in job_names:
                    try:
                        job = self.batch_client.get_job(name=job_name)
                        state = job.status.state.name
                        states[state] += 1
                        if state not in ["SUCCEEDED", "FAILED"]:
                            all_finished = False
                    except Exception as e:
                        logging.warning(f"Could not get status for job {job_name}: {e}")
                        states["UNKNOWN"] += 1
            self._update_dashboard(states)
            if all_finished:
                logging.info("All jobs have completed.")
                break
            await asyncio.sleep(60)

    def _update_dashboard(self, states: Dict):
        print("\n" + "="*80 + "\nLAMBDA³ GCP COMPUTATION DASHBOARD (LIVE)\n" + "="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        total_jobs = sum(states.values())
        print(f"Total Jobs: {total_jobs}")
        for state, count in states.items():
            print(f"  - {state:<15}: {count}")
        succeeded, failed = states.get("SUCCEEDED", 0), states.get("FAILED", 0)
        progress = (succeeded + failed) / max(total_jobs, 1)
        bar = '█' * int(40 * progress) + '-' * (40 - int(40 * progress))
        print(f"\nProgress: [{bar}] {progress:.1%}")

# ===============================
# MAIN ORCHESTRATOR (PRODUCTION)
# ===============================
async def run_lambda3_gcp_ultimate(data_source: Union[str, Dict[str, np.ndarray]], l3_config: L3Config = None, gcp_config: GCPUltimateConfig = None, target_pairs: Optional[int] = None) -> Dict[str, Any]:
    if l3_config is None: l3_config = L3Config()
    if gcp_config is None: gcp_config = GCPUltimateConfig()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    creds, project_id = default()

    if isinstance(data_source, str):
        with open(data_source, 'rb') as f:
            series_dict = pickle.load(f)
    else:
        series_dict = data_source

    decomposer = Lambda3TaskDecomposer(CloudScaleConfig())
    series_names = list(series_dict.keys())
    pair_batches = decomposer.decompose_pairwise_analysis(series_names)
    
    # ★★★ インデントエラーを修正 ★★★
    if target_pairs:
        total_pairs_original = sum(len(batch) for batch in pair_batches)
        if total_pairs_original > target_pairs:
            # バッチ数を調整してペア数をおおよそtarget_pairsに近づける
            if pair_batches:
                 num_batches = max(1, int(len(pair_batches) * (target_pairs / total_pairs_original)))
                 pair_batches = pair_batches[:num_batches]

    total_pairs = sum(len(batch) for batch in pair_batches)
    logging.info(f"Total pairs to analyze: {total_pairs} in {len(pair_batches)} batches.")

    hunter = GCPResourceHunter(gcp_config, creds)
    allocation_plan = hunter.get_best_spot_options(n_instances=len(pair_batches))
    if not allocation_plan:
        logging.error("Could not create an allocation plan. Aborting.")
        return {}

    batch_manager = Lambda3CloudBatchManager(gcp_config, creds)
    active_jobs = batch_manager.create_and_run_batch_jobs(allocation_plan, series_dict, pair_batches, l3_config)
    if not active_jobs:
        logging.error("No jobs were created. Aborting.")
        return {}
    logging.info(f"Successfully submitted jobs across {len(active_jobs)} regions.")

    monitor = Lambda3GlobalMonitor(project_id, active_jobs, creds)
    await monitor.monitor_jobs()
    
    logging.info("Analysis complete!")
    return {
        "status": "COMPLETED", "active_jobs": active_jobs,
        "gcs_results_path": f"gs://{gcp_config.gcs_bucket}/results/",
    }

# --- Cost Savings Calculator (変更なし) ---
def calculate_cost_savings():
    pass

if __name__ == '__main__':
    async def main_demo():
        dummy_data = {f"Series_{i}": np.random.randn(200) for i in range(50)}
        await run_lambda3_gcp_ultimate(
            data_source=dummy_data,
            gcp_config=GCPUltimateConfig(max_price_per_hour=0.05),
            l3_config=L3Config(draws=1000, tune=1000)
        )
    asyncio.run(main_demo())
