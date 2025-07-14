"""
Lambda³ Core Module

Core theoretical implementation of Lambda³ structural tensor analytics.
"""

from .lambda3_zeroshot_tensor_field import (
    # Configuration
    L3Config,
    
    # Core functions
    calc_lambda3_features,
    fit_l3_pairwise_bayesian_system,
    fit_l3_bayesian_regression_asymmetric,
    fit_hierarchical_bayesian,
    
    # Feature extraction
    calculate_diff_and_threshold,
    detect_jumps,
    calculate_local_std,
    calculate_rho_t,
    detect_local_global_jumps,
    
    # Analysis functions
    run_lambda3_analysis,
    analyze_all_pairwise_interactions,
    complete_hierarchical_analysis,
    analyze_hierarchical_synchronization,
    
    # Regime detection
    Lambda3RegimeDetector,
    Lambda3FinancialRegimeDetector,
    
    # Interaction analysis
    extract_interaction_coefficients,
    predict_with_interactions,
    
    # Causality analysis
    detect_basic_structural_causality,
    analyze_comprehensive_causality,
    
    # Synchronization
    calculate_sync_profile,
    sync_matrix,
    build_sync_network,
    
    # Crisis detection
    detect_structural_crisis,
    
    # Bayesian utilities
    Lambda3BayesianLogger,
    
    # Visualization
    Lambda3Visualizer,
    plot_lambda3_summary,
    quick_lambda3_plot,
    
    # Data utilities
    fetch_financial_data,
    load_csv_data,
    lambda3_analyze_financial_data,
    lambda3_batch_analysis,
    lambda3_streaming_analysis
)

from .lambda3_regime_aware_extension import (
    # Configuration
    HierarchicalRegimeConfig,
    
    # Core classes
    HierarchicalRegimeDetector,
    RegimeAwareBayesianAnalysis,
    
    # Main functions
    run_lambda3_regime_aware_analysis,
    
    # Utilities
    calculate_returns,
    create_regime_aware_config,
    validate_regime_features,
    merge_regime_results_with_base,
    
    # Visualization
    plot_regime_timeline,
    plot_regime_interaction_heatmap
)

__all__ = [
    # From lambda3_zeroshot_tensor_field
    'L3Config',
    'calc_lambda3_features',
    'fit_l3_pairwise_bayesian_system',
    'run_lambda3_analysis',
    'Lambda3RegimeDetector',
    'Lambda3FinancialRegimeDetector',
    'Lambda3BayesianLogger',
    'Lambda3Visualizer',
    'plot_lambda3_summary',
    'quick_lambda3_plot',
    'fetch_financial_data',
    'load_csv_data',
    
    # From lambda3_regime_aware_extension
    'HierarchicalRegimeConfig',
    'HierarchicalRegimeDetector',
    'RegimeAwareBayesianAnalysis',
    'run_lambda3_regime_aware_analysis',
    'plot_regime_timeline',
    'plot_regime_interaction_heatmap'
]
