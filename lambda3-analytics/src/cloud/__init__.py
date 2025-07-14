"""
Lambda³ Cloud Module

Cloud-scale parallel execution of Lambda³ analytics.
"""

from .lambda3_cloud_parallel import (
    # Configuration
    CloudScaleConfig,
    ExecutionBackend,
    
    # Core classes
    Lambda3TaskDecomposer,
    Lambda3ParallelExecutor,
    LocalMultiprocessExecutor,
    
    # Main functions
    run_lambda3_cloud_scale,
    benchmark_backends,
    
    # Monitoring
    ProgressMonitor,
    CloudStorageHandler
)

# Optional GCP imports
try:
    from .lambda3_gcp_ultimate import (
        # Configuration
        GCPUltimateConfig,
        
        # Core classes
        GCPResourceHunter,
        Lambda3CloudBatchManager,
        UltimatePreemptionHandler,
        Lambda3GlobalMonitor,
        
        # Main functions
        run_lambda3_gcp_ultimate,
        calculate_cost_savings
    )
    
    from .lambda3_cloud_worker import (
        Lambda3CloudWorker,
        PreemptionHandler
    )
    
    from .lambda3_result_aggregator import (
        Lambda3ResultAggregator,
        StreamingResultAggregator,
        aggregate_lambda3_results
    )
    
    _HAS_GCP = True
    
except ImportError:
    _HAS_GCP = False
    print("GCP modules not available. Install with: pip install lambda3-analytics[gcp]")

# Export list
__all__ = [
    # From lambda3_cloud_parallel
    'CloudScaleConfig',
    'ExecutionBackend',
    'Lambda3TaskDecomposer',
    'Lambda3ParallelExecutor',
    'run_lambda3_cloud_scale',
    'benchmark_backends',
    'ProgressMonitor',
    'CloudStorageHandler'
]

# Add GCP exports if available
if _HAS_GCP:
    __all__.extend([
        # From lambda3_gcp_ultimate
        'GCPUltimateConfig',
        'GCPResourceHunter',
        'Lambda3CloudBatchManager',
        'UltimatePreemptionHandler',
        'Lambda3GlobalMonitor',
        'run_lambda3_gcp_ultimate',
        'calculate_cost_savings',
        
        # From lambda3_cloud_worker
        'Lambda3CloudWorker',
        'PreemptionHandler',
        
        # From lambda3_result_aggregator
        'Lambda3ResultAggregator',
        'StreamingResultAggregator',
        'aggregate_lambda3_results'
    ])

# Optional imports for other cloud providers
try:
    import dask
    from .lambda3_cloud_parallel import DaskDistributedExecutor
    __all__.append('DaskDistributedExecutor')
    _HAS_DASK = True
except ImportError:
    _HAS_DASK = False

try:
    import ray
    from .lambda3_cloud_parallel import RayClusterExecutor
    __all__.append('RayClusterExecutor')
    _HAS_RAY = True
except ImportError:
    _HAS_RAY = False

def get_available_backends():
    """Get list of available execution backends"""
    backends = ['local_mp']  # Always available
    
    if _HAS_DASK:
        backends.append('dask')
    if _HAS_RAY:
        backends.append('ray')
    if _HAS_GCP:
        backends.extend(['gcp_batch', 'gcp_ultimate'])
    
    return backends

def print_backend_info():
    """Print information about available backends"""
    print("Lambda³ Cloud Execution Backends:")
    print("-" * 40)
    
    backends = {
        'local_mp': ('Local Multiprocessing', True),
        'dask': ('Dask Distributed', _HAS_DASK),
        'ray': ('Ray Cluster', _HAS_RAY),
        'gcp_batch': ('Google Cloud Batch', _HAS_GCP),
        'gcp_ultimate': ('GCP Ultimate (10k+ instances)', _HAS_GCP)
    }
    
    for backend, (name, available) in backends.items():
        status = "✓" if available else "✗"
        print(f"  {status} {backend:<15} - {name}")
    
    print("\nTo enable more backends:")
    if not _HAS_DASK:
        print("  pip install dask[distributed]")
    if not _HAS_RAY:
        print("  pip install ray")
    if not _HAS_GCP:
        print("  pip install lambda3-analytics[gcp]")
