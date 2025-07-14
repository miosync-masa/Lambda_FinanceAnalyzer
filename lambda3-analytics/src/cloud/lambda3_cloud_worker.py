#!/usr/bin/env python3
# ==========================================================
# Lambda³ Cloud Worker - Individual Task Executor (PRODUCTION IMPLEMENTATION)
# ==========================================================
# Runs on each Cloud Batch instance
# Handles checkpointing, fault tolerance, and result upload

import argparse
import pickle
import time
import signal
import sys
from pathlib import Path
import numpy as np
import arviz as az # 結果抽出のためにarvizをインポート
from google.cloud import storage
import logging

# Lambda³ imports - Cloud Batch環境用に修正
try:
    # パッケージとして実行される場合
    from core.lambda3_zeroshot_tensor_field import (
        calc_lambda3_features,
        fit_l3_pairwise_bayesian_system,
        L3Config
    )
    # 既存の係数抽出関数をインポート
    from core.lambda3_regime_aware_extension import _extract_regime_interaction_coefficients as extract_interaction_coefficients

except ImportError:
    # 直接実行される場合（Cloud Batch環境）
    # Cloud Batchのワーカーでは、GCSからコードをダウンロードして実行するため、
    # PYTHONPATHが通っていない可能性があります。
    # そのため、実行時にパスを追加する処理が有効です。
    sys.path.insert(0, '/workspace') # コンテナ内のワークスペースを指す
    from core.lambda3_zeroshot_tensor_field import (
        calc_lambda3_features,
        fit_l3_pairwise_bayesian_system,
        L3Config
    )
    from core.lambda3_regime_aware_extension import _extract_regime_interaction_coefficients as extract_interaction_coefficients

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreemptionHandler:
    """Handle preemption signals gracefully"""
    def __init__(self, worker):
        self.worker = worker
        self.preempted = False
        signal.signal(signal.SIGTERM, self._handle_preemption)
        
    def _handle_preemption(self, signum, frame):
        logger.warning("Preemption signal received! Saving progress...")
        self.preempted = True
        self.worker.save_checkpoint()
        sys.exit(0)

class Lambda3CloudWorker:
    """
    Worker that processes a batch of Lambda³ pair analyses
    """
    
    def __init__(self, args):
        self.args = args
        self.storage_client = storage.Client()
        self.completed_pairs = []
        self.current_pair_index = 0
        
        # L3Configを引数からデシリアライズ
        self.l3_config = pickle.loads(bytes.fromhex(self.args.l3_config_hex))
        
        # Load data
        self._load_data()
        
        # Setup preemption handler
        self.preemption_handler = PreemptionHandler(self)
        
    def _load_data(self):
        """Load series data and batch assignment"""
        logger.info("Loading series data...")
        series_blob = self.storage_client.bucket(self.args.input_bucket).blob(self.args.series_data_gcs)
        self.series_dict = pickle.loads(series_blob.download_as_bytes())
        
        logger.info("Loading batch assignment...")
        batch_blob = self.storage_client.bucket(self.args.input_bucket).blob(self.args.batch_gcs)
        self.pair_batch = pickle.loads(batch_blob.download_as_bytes())
            
        logger.info(f"Loaded {len(self.series_dict)} series")
        logger.info(f"Assigned {len(self.pair_batch)} pairs to process")
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        checkpoint_path = f"checkpoints/{self.args.region}/batch_{self.args.batch_index}_checkpoint.pkl"
        blob = self.storage_client.bucket(self.args.output_bucket).blob(checkpoint_path)
        
        if blob.exists():
            logger.info("Found checkpoint, resuming...")
            try:
                checkpoint_data = pickle.loads(blob.download_as_bytes())
                self.completed_pairs = checkpoint_data.get('completed_pairs', [])
                self.current_pair_index = checkpoint_data.get('current_index', 0)
                logger.info(f"Resuming from pair {self.current_pair_index}/{len(self.pair_batch)}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

    def save_checkpoint(self):
        """Save current progress"""
        if self.preemption_handler.preempted and not self.completed_pairs:
             logger.warning("No pairs completed before preemption. Not saving empty checkpoint.")
             return

        checkpoint_data = {
            'completed_pairs': self.completed_pairs,
            'current_index': self.current_pair_index,
            'timestamp': time.time()
        }
        
        checkpoint_path = f"checkpoints/{self.args.region}/batch_{self.args.batch_index}_checkpoint.pkl"
        blob = self.storage_client.bucket(self.args.output_bucket).blob(checkpoint_path)
        blob.upload_from_string(pickle.dumps(checkpoint_data))
        
        logger.info(f"Checkpoint saved at pair {self.current_pair_index}")

    def process_pair(self, name_a: str, name_b: str) -> dict:
        """
        Process a single pair using Lambda³ analysis and extract real results.
        """
        logger.info(f"Processing pair: {name_a} vs {name_b}")
        
        try:
            # Stage 1: Feature Extraction
            features_dict = {
                name_a: calc_lambda3_features(self.series_dict[name_a], self.l3_config),
                name_b: calc_lambda3_features(self.series_dict[name_b], self.l3_config)
            }
            
            # Stage 2: Fit Bayesian model
            trace, model = fit_l3_pairwise_bayesian_system(
                {name_a: self.series_dict[name_a], name_b: self.series_dict[name_b]},
                features_dict,
                self.l3_config,
                series_pair=(name_a, name_b)
            )
            
            # Stage 3: Extract meaningful interaction coefficients
            # `lambda3_regime_aware_extension.py`から流用した関数で結果を抽出
            interaction_coeffs = extract_interaction_coefficients(
                trace, [name_a, name_b]
            )
            
            # Return a structured result
            result = {
                'pair': f"{name_a}_vs_{name_b}",
                'completed': True,
                'timestamp': time.time(),
                'worker': f"{self.args.region}_batch_{self.args.batch_index}",
                'coefficients': interaction_coeffs # 抽出した係数をすべて格納
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {name_a} vs {name_b}: {e}", exc_info=True)
            return {
                'pair': f"{name_a}_vs_{name_b}",
                'completed': False,
                'error': str(e),
                'timestamp': time.time(),
                'worker': f"{self.args.region}_batch_{self.args.batch_index}"
            }

    def upload_result(self, result: dict):
        """Upload individual result to GCS"""
        result_path = f"results/{self.args.region}/batch_{self.args.batch_index}/{result['pair']}.pkl"
        blob = self.storage_client.bucket(self.args.output_bucket).blob(result_path)
        blob.upload_from_string(pickle.dumps(result))
    
    def run(self):
        """Main processing loop"""
        logger.info(f"Starting processing from index {self.current_pair_index}")
        
        for i in range(self.current_pair_index, len(self.pair_batch)):
            self.current_pair_index = i
            
            # Check for preemption before starting a new task
            if self.preemption_handler.preempted:
                logger.warning("Preemption detected, stopping loop.")
                break

            # Process pair
            name_a, name_b = self.pair_batch[self.current_pair_index]
            result = self.process_pair(name_a, name_b)
            
            # Upload result immediately
            self.upload_result(result)
            
            # Track completion
            if result['completed']:
                self.completed_pairs.append(result['pair'])

            logger.info(f"Progress: {self.current_pair_index + 1}/{len(self.pair_batch)} pairs")

            # Checkpoint periodically
            if (self.current_pair_index + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint and mark as complete
        self.save_checkpoint()
        if not self.preemption_handler.preempted:
            self._mark_batch_complete()
            logger.info("Batch processing complete!")
        else:
            logger.warning("Batch processing preempted.")
    
    def _mark_batch_complete(self):
        """Mark this batch as complete in GCS"""
        complete_path = f"completed/{self.args.region}/batch_{self.args.batch_index}_complete.txt"
        blob = self.storage_client.bucket(self.args.output_bucket).blob(complete_path)
        completion_info = (
            f"Completed at {time.time()}\n"
            f"Total pairs processed: {len(self.completed_pairs)}\n"
            f"Worker: {self.args.region}_batch_{self.args.batch_index}"
        )
        blob.upload_from_string(completion_info)

def main():
    parser = argparse.ArgumentParser(description='Lambda³ Cloud Worker')
    # GCS上のパスを受け取るように変更
    parser.add_argument('--series-data-gcs', required=True, help='GCS path to series data pickle')
    parser.add_argument('--batch-gcs', required=True, help='GCS path to batch assignment pickle')
    parser.add_argument('--input-bucket', required=True, help='GCS bucket for input data')
    parser.add_argument('--output-bucket', required=True, help='GCS bucket for results')
    parser.add_argument('--region', required=True, help='GCP region')
    parser.add_argument('--batch-index', type=int, required=True)
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    # L3Configを文字列として受け取る
    parser.add_argument('--l3-config-hex', required=True, help='Hex-encoded pickled L3Config object')
    
    args = parser.parse_args()
    
    worker = Lambda3CloudWorker(args)
    worker.run()

if __name__ == "__main__":
    main()
