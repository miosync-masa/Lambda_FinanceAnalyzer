# ==========================================================
# Lambda³ Global Result Aggregator
# ==========================================================
# Efficiently collects and merges results from thousands of workers

import asyncio
import pickle
from typing import Dict, List, Set, Any
from collections import defaultdict
import numpy as np
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd

class Lambda3ResultAggregator:
    """
    Aggregates Lambda³ results from distributed GCP computation
    """
    
    def __init__(self, bucket_name: str, regions: List[str]):
        self.bucket_name = bucket_name
        self.regions = regions
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Result tracking
        self.completed_pairs = set()
        self.failed_pairs = []
        self.interaction_matrix = defaultdict(dict)
        self.aggregate_statistics = {}
        
    async def aggregate_all_results(self) -> Dict[str, Any]:
        """
        Aggregate all results from all regions in parallel
        """
        print("="*80)
        print("LAMBDA³ GLOBAL RESULT AGGREGATION")
        print("="*80)
        
        start_time = time.time()
        
        # Check completion status for all regions
        completion_status = await self._check_completion_status()
        
        print(f"\nCompletion Status:")
        for region, status in completion_status.items():
            print(f"  {region}: {status['completed']}/{status['total']} batches")
        
        # Parallel result collection
        with ThreadPoolExecutor(max_workers=20) as executor:
            region_results = []
            
            for region in self.regions:
                future = executor.submit(self._collect_region_results, region)
                region_results.append((region, future))
            
            # Collect results
            for region, future in region_results:
                try:
                    results = future.result(timeout=300)
                    self._merge_region_results(results)
                    print(f"✓ Collected {len(results)} results from {region}")
                except Exception as e:
                    print(f"✗ Error collecting from {region}: {e}")
        
        # Build final matrices and statistics
        final_results = self._build_final_results()
        
        elapsed = time.time() - start_time
        print(f"\nAggregation completed in {elapsed:.2f} seconds")
        print(f"Total pairs processed: {len(self.completed_pairs)}")
        print(f"Failed pairs: {len(self.failed_pairs)}")
        
        return final_results
    
    async def _check_completion_status(self) -> Dict[str, Dict[str, int]]:
        """Check completion status for all regions"""
        status = {}
        
        for region in self.regions:
            # Count completed batches
            prefix = f"completed/{region}/"
            completed_blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            # Count total batches
            batch_prefix = f"results/{region}/"
            batch_dirs = set()
            for blob in self.bucket.list_blobs(prefix=batch_prefix):
                parts = blob.name.split('/')
                if len(parts) >= 3:
                    batch_dirs.add(parts[2])
            
            status[region] = {
                'completed': len(completed_blobs),
                'total': len(batch_dirs)
            }
        
        return status
    
    def _collect_region_results(self, region: str) -> List[Dict[str, Any]]:
        """Collect all results from a specific region"""
        results = []
        prefix = f"results/{region}/"
        
        # List all result blobs
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        # Download in batches
        batch_size = 100
        for i in range(0, len(blobs), batch_size):
            batch_blobs = blobs[i:i+batch_size]
            
            for blob in batch_blobs:
                if blob.name.endswith('.pkl'):
                    try:
                        result = pickle.loads(blob.download_as_bytes())
                        results.append(result)
                    except Exception as e:
                        print(f"Error loading {blob.name}: {e}")
        
        return results
    
    # _merge_region_results 関数の内部
    def _merge_region_results(self, results: List[Dict[str, Any]]):
        """Merge results from a region into global structures"""
        for result in results:
            pair_key = result.get('pair', '')
            
            if result.get('completed', False):
                self.completed_pairs.add(pair_key)
                
                # Extract series names
                if '_vs_' in pair_key:
                    name_a, name_b = pair_key.split('_vs_')
                    
                    # Workerが出力するネストされた構造から値を取得する
                    coeffs = result.get('coefficients', {})
                    regime_specific = coeffs.get('regime_specific', {})
                    strength_a_to_b = regime_specific.get('strength_a_to_b', 0)
                    strength_b_to_a = regime_specific.get('strength_b_to_a', 0)
                    
                    # Store interaction strength (非対称な値を考慮)
                    if name_a in self.interaction_matrix and name_b in self.interaction_matrix[name_a]:
                        pass
                    else:
                        self.interaction_matrix[name_a][name_b] = strength_a_to_b
                        self.interaction_matrix[name_b][name_a] = strength_b_to_a
                    
            else:
                self.failed_pairs.append({
                    'pair': pair_key,
                    'error': result.get('error', 'Unknown error'),
                    'worker': result.get('worker', 'Unknown')
                })
    
    def _build_final_results(self) -> Dict[str, Any]:
        """Build final result structures"""
        
        # Convert interaction matrix to numpy array
        series_names = sorted(set(
            list(self.interaction_matrix.keys()) + 
            [b for a_dict in self.interaction_matrix.values() for b in a_dict.keys()]
        ))
        
        n = len(series_names)
        matrix = np.zeros((n, n))
        
        for i, name_a in enumerate(series_names):
            for j, name_b in enumerate(series_names):
                if name_a in self.interaction_matrix and name_b in self.interaction_matrix[name_a]:
                    matrix[i, j] = self.interaction_matrix[name_a][name_b]
        
        # Calculate aggregate statistics
        self.aggregate_statistics = {
            'mean_interaction': np.mean(matrix[matrix > 0]),
            'max_interaction': np.max(matrix),
            'interaction_density': np.sum(matrix > 0.1) / (n * n),
            'network_modularity': self._calculate_modularity(matrix),
            'top_interactions': self._get_top_interactions(matrix, series_names, top_k=10)
        }
        
        return {
            'series_names': series_names,
            'interaction_matrix': matrix,
            'completed_pairs': list(self.completed_pairs),
            'failed_pairs': self.failed_pairs,
            'aggregate_statistics': self.aggregate_statistics,
            'metadata': {
                'total_pairs_analyzed': len(self.completed_pairs),
                'failure_rate': len(self.failed_pairs) / max(len(self.completed_pairs) + len(self.failed_pairs), 1),
                'regions_used': self.regions
            }
        }
    
    def _calculate_modularity(self, matrix: np.ndarray) -> float:
        """Calculate network modularity (simplified)"""
        # Simplified modularity calculation
        if matrix.size == 0:
            return 0.0
        
        # Threshold to create adjacency matrix
        adj = (matrix > np.percentile(matrix[matrix > 0], 75)).astype(int)
        
        # Simple modularity: ratio of within-cluster to total edges
        # (Would use proper community detection in practice)
        return np.trace(adj) / (np.sum(adj) + 1e-8)
    
    def _get_top_interactions(
        self, 
        matrix: np.ndarray, 
        series_names: List[str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top k strongest interactions"""
        interactions = []
        
        for i in range(len(series_names)):
            for j in range(i+1, len(series_names)):
                if matrix[i, j] > 0 or matrix[j, i] > 0:
                    interactions.append({
                        'pair': f"{series_names[i]}_vs_{series_names[j]}",
                        'strength': max(matrix[i, j], matrix[j, i]),
                        'asymmetry': abs(matrix[i, j] - matrix[j, i])
                    })
        
        # Sort by strength
        interactions.sort(key=lambda x: x['strength'], reverse=True)
        
        return interactions[:top_k]
    
    async def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        report = []
        report.append("="*80)
        report.append("LAMBDA³ GLOBAL ANALYSIS SUMMARY")
        report.append("="*80)
        
        meta = results['metadata']
        stats = results['aggregate_statistics']
        
        report.append(f"\nAnalysis Overview:")
        report.append(f"  Total pairs analyzed: {meta['total_pairs_analyzed']:,}")
        report.append(f"  Failure rate: {meta['failure_rate']*100:.2f}%")
        report.append(f"  Regions used: {len(meta['regions_used'])}")
        
        report.append(f"\nStructural Tensor Statistics:")
        report.append(f"  Mean interaction strength: {stats['mean_interaction']:.4f}")
        report.append(f"  Maximum interaction: {stats['max_interaction']:.4f}")
        report.append(f"  Interaction density: {stats['interaction_density']*100:.1f}%")
        report.append(f"  Network modularity: {stats['network_modularity']:.4f}")
        
        report.append(f"\nTop 10 Structural Interactions:")
        for i, interaction in enumerate(stats['top_interactions']):
            report.append(f"  {i+1}. {interaction['pair']}: "
                         f"strength={interaction['strength']:.4f}, "
                         f"asymmetry={interaction['asymmetry']:.4f}")
        
        # Lambda³ theoretical interpretation
        report.append(f"\n" + "-"*80)
        report.append("LAMBDA³ THEORETICAL INSIGHTS:")
        report.append("-"*80)
        
        # Structural phase detection
        high_interaction_threshold = np.percentile(
            results['interaction_matrix'][results['interaction_matrix'] > 0], 90
        )
        crisis_pairs = np.sum(results['interaction_matrix'] > high_interaction_threshold)
        
        report.append(f"• Detected {crisis_pairs} high-tension structural couplings")
        report.append(f"• Network shows {stats['network_modularity']:.1%} modularity")
        
        if stats['interaction_density'] > 0.3:
            report.append("• HIGH COHERENCE: System-wide structural synchronization detected")
        elif stats['interaction_density'] < 0.1:
            report.append("• LOW COHERENCE: Isolated structural dynamics dominate")
        else:
            report.append("• MODERATE COHERENCE: Mixed local/global structural patterns")
        
        return "\n".join(report)
    
    def save_consolidated_results(self, results: Dict[str, Any], output_path: str):
        """Save consolidated results locally and to GCS"""
        
        # Save locally
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Upload to GCS
        blob = self.bucket.blob(f"final_results/lambda3_consolidated_{int(time.time())}.pkl")
        blob.upload_from_filename(output_path)
        
        print(f"\nResults saved to:")
        print(f"  Local: {output_path}")
        print(f"  GCS: gs://{self.bucket_name}/{blob.name}")

# ===============================
# STREAMING AGGREGATOR
# ===============================
class StreamingResultAggregator(Lambda3ResultAggregator):
    """
    Real-time streaming aggregation as results arrive
    """
    
    def __init__(self, bucket_name: str, regions: List[str]):
        super().__init__(bucket_name, regions)
        self.result_queue = asyncio.Queue()
        self.processed_blobs = set()
        
    async def start_streaming_aggregation(self):
        """Start streaming aggregation process"""
        
        # Start watcher tasks for each region
        watcher_tasks = []
        for region in self.regions:
            task = asyncio.create_task(self._watch_region(region))
            watcher_tasks.append(task)
        
        # Start aggregator task
        aggregator_task = asyncio.create_task(self._aggregate_stream())
        
        # Wait for all tasks
        await asyncio.gather(*watcher_tasks, aggregator_task)
    
    async def _watch_region(self, region: str):
        """Watch for new results in a region"""
        prefix = f"results/{region}/"
        
        while True:
            try:
                # List new blobs
                for blob in self.bucket.list_blobs(prefix=prefix):
                    if blob.name not in self.processed_blobs and blob.name.endswith('.pkl'):
                        # Queue for processing
                        await self.result_queue.put((region, blob))
                        self.processed_blobs.add(blob.name)
                
                # Check every 10 seconds
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"Error watching {region}: {e}")
                await asyncio.sleep(30)
    
    async def _aggregate_stream(self):
        """Process results as they arrive"""
        
        while True:
            try:
                # Get next result
                region, blob = await self.result_queue.get()
                
                # Process result
                result = pickle.loads(blob.download_as_bytes())
                self._merge_region_results([result])
                
                # Periodic summary
                if len(self.completed_pairs) % 1000 == 0:
                    print(f"\rProcessed {len(self.completed_pairs)} pairs...", end='')
                
            except Exception as e:
                print(f"Error processing stream: {e}")

# ===============================
# MAIN EXECUTION
# ===============================
async def aggregate_lambda3_results(
    bucket_name: str,
    regions: List[str],
    output_path: str = "lambda3_global_results.pkl",
    streaming: bool = False
):
    """
    Main function to aggregate Lambda³ results
    """
    
    if streaming:
        print("Starting streaming aggregation...")
        aggregator = StreamingResultAggregator(bucket_name, regions)
        await aggregator.start_streaming_aggregation()
    else:
        print("Starting batch aggregation...")
        aggregator = Lambda3ResultAggregator(bucket_name, regions)
        
        # Aggregate all results
        results = await aggregator.aggregate_all_results()
        
        # Generate report
        report = await aggregator.generate_summary_report(results)
        print("\n" + report)
        
        # Save results
        aggregator.save_consolidated_results(results, output_path)
        
        return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python lambda3_result_aggregator.py <bucket_name> [output_path]")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "lambda3_results.pkl"
    
    # Default regions (would be dynamic in practice)
    regions = [
        "us-central1", "us-east1", "us-west1",
        "europe-west1", "asia-northeast1"
    ]
    
    # Run aggregation
    asyncio.run(aggregate_lambda3_results(bucket_name, regions, output_path))
