"""
Lambda³ Analytics Framework

Universal Structural Tensor Field Analytics - Beyond time, beyond causality.
"""

__version__ = "1.0.0"
__author__ = "Masamichi Iizumi"

# Make key modules available at package level
from .core.lambda3_zeroshot_tensor_field import (
    L3Config,
    run_lambda3_analysis,
    calc_lambda3_features,
    fit_l3_pairwise_bayesian_system,
    Lambda3BayesianLogger,
    Lambda3FinancialRegimeDetector,
    plot_lambda3_summary,
    quick_lambda3_plot
)

from .core.lambda3_regime_aware_extension import (
    HierarchicalRegimeConfig,
    run_lambda3_regime_aware_analysis,
    HierarchicalRegimeDetector,
    RegimeAwareBayesianAnalysis
)

from .cloud.lambda3_cloud_parallel import (
    CloudScaleConfig,
    ExecutionBackend,
    run_lambda3_cloud_scale
)

# GCP specific imports (optional)
try:
    from .cloud.lambda3_gcp_ultimate import (
        GCPUltimateConfig,
        run_lambda3_gcp_ultimate,
        calculate_cost_savings
    )
    _HAS_GCP = True
except ImportError:
    _HAS_GCP = False

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core functionality
    "L3Config",
    "run_lambda3_analysis",
    "calc_lambda3_features",
    "fit_l3_pairwise_bayesian_system",
    "Lambda3BayesianLogger",
    "Lambda3FinancialRegimeDetector",
    "plot_lambda3_summary",
    "quick_lambda3_plot",
    
    # Regime-aware extension
    "HierarchicalRegimeConfig",
    "run_lambda3_regime_aware_analysis",
    "HierarchicalRegimeDetector",
    "RegimeAwareBayesianAnalysis",
    
    # Cloud parallel
    "CloudScaleConfig",
    "ExecutionBackend",
    "run_lambda3_cloud_scale",
]

# Add GCP exports if available
if _HAS_GCP:
    __all__.extend([
        "GCPUltimateConfig",
        "run_lambda3_gcp_ultimate",
        "calculate_cost_savings",
    ])

def check_dependencies():
    """Check if all required dependencies are installed"""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'pymc': 'pymc',
        'arviz': 'arviz',
        'numba': 'numba',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'networkx': 'networkx',
        'yfinance': 'yfinance'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Warning: Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

# Check dependencies on import
_DEPS_OK = check_dependencies()
