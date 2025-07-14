#!/usr/bin/env python3
"""
Lambda³ Local Demo Script

Demonstrates Lambda³ analysis on local machine without cloud dependencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.lambda3_zeroshot_tensor_field import (
    L3Config,
    run_lambda3_analysis,
    Lambda3BayesianLogger,
    plot_lambda3_summary,
    fetch_financial_data
)
from src.utils.data_loader import (
    generate_synthetic_data,
    generate_structural_jumps,
    generate_regime_switching_data,
    check_data_quality,
    save_results
)

def demo_synthetic_data():
    """Demo with synthetic data"""
    print("\n" + "="*80)
    print("LAMBDA³ DEMO: SYNTHETIC DATA ANALYSIS")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_series = 5
    n_points = 500
    
    print(f"\nGenerating {n_series} synthetic time series with {n_points} points each...")
    
    # Create different types of series
    synthetic_data = {}
    
    # Series 1: Random walk with structural jumps
    data1 = np.cumsum(np.random.randn(n_points) * 0.1)
    data1 = generate_structural_jumps(data1, n_jumps=3)
    synthetic_data['RandomWalk_Jumps'] = data1
    
    # Series 2: Oscillatory with trend
    t = np.linspace(0, 10, n_points)
    data2 = np.sin(2 * np.pi * t) + 0.1 * t + 0.2 * np.random.randn(n_points)
    synthetic_data['Oscillatory_Trend'] = data2
    
    # Series 3: Regime switching
    data3, regimes = generate_regime_switching_data(n_points)
    synthetic_data['Regime_Switching'] = data3
    
    # Series 4 & 5: Correlated series
    base = np.cumsum(np.random.randn(n_points) * 0.1)
    data4 = base + 0.3 * np.random.randn(n_points)
    data5 = 0.7 * base + 0.5 * np.random.randn(n_points)
    synthetic_data['Correlated_A'] = data4
    synthetic_data['Correlated_B'] = data5
    
    # Check data quality
    print("\nData Quality Check:")
    check_data_quality(synthetic_data)
    
    # Configure Lambda³ analysis
    config = L3Config(
        draws=2000,  # Reduced for demo
        tune=2000,
        hierarchical=True,
        hdi_prob=0.94
    )
    
    print(f"\nRunning Lambda³ analysis...")
    print(f"  Configuration: {config.draws} draws, {config.tune} tune")
    
    start_time = time.time()
    
    # Run analysis
    results = run_lambda3_analysis(
        data_source=synthetic_data,
        config=config,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.2f} seconds")
    
    # Save results
    output_dir = Path("results") / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_results(results, output_dir / "synthetic_results.pkl")
    
    # Visualize if possible
    try:
        plot_lambda3_summary(results, save_path=output_dir / "synthetic_summary.png")
        print(f"\nVisualization saved to: {output_dir / 'synthetic_summary.png'}")
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    return results

def demo_financial_data():
    """Demo with real financial data"""
    print("\n" + "="*80)
    print("LAMBDA³ DEMO: FINANCIAL MARKET ANALYSIS")
    print("="*80)
    
    # Try to fetch financial data
    print("\nFetching financial data...")
    
    try:
        # Define tickers
        tickers = {
            "USD/JPY": "JPY=X",
            "Gold": "GC=F",
            "Oil": "CL=F", 
            "S&P 500": "^GSPC",
            "Bitcoin": "BTC-USD"
        }
        
        # Fetch recent data
        data_df = fetch_financial_data(
            start_date="2023-01-01",
            end_date="2024-12-31",
            tickers=tickers,
            csv_filename="data/financial_demo.csv"
        )
        
        if data_df is None:
            print("Failed to fetch financial data, using synthetic data instead")
            return demo_synthetic_data()
        
        # Convert to dict
        financial_data = {col: data_df[col].values for col in data_df.columns}
        
        # Check data quality
        print("\nData Quality Check:")
        check_data_quality(financial_data)
        
    except Exception as e:
        print(f"Error fetching financial data: {e}")
        print("Falling back to synthetic data...")
        return demo_synthetic_data()
    
    # Configure Lambda³ analysis for financial data
    config = L3Config(
        draws=4000,
        tune=4000,
        hierarchical=True,
        regime_specific=True
    )
    
    print(f"\nRunning Lambda³ financial analysis...")
    
    start_time = time.time()
    
    # Run analysis with regime detection
    from src.core.lambda3_regime_aware_extension import (
        run_lambda3_regime_aware_analysis,
        HierarchicalRegimeConfig
    )
    
    regime_config = HierarchicalRegimeConfig(
        n_global_regimes=3,
        global_regime_names=['Bull', 'Neutral', 'Bear']
    )
    
    results = run_lambda3_regime_aware_analysis(
        data_source=financial_data,
        base_config=config,
        hierarchical_config=regime_config,
        max_pairs=10,  # Limit for demo
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.2f} seconds")
    
    # Save results
    output_dir = Path("results") / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_results(results, output_dir / "financial_results.pkl")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if 'regime_analysis' in results:
        regime_stats = results['regime_analysis']['regime_statistics']
        print("\nMarket Regime Distribution:")
        for regime_id, stats in regime_stats.items():
            print(f"  {stats['regime_name']}: {stats['frequency']*100:.1f}%")
    
    return results

def demo_interactive():
    """Interactive demo with user choices"""
    print("\n" + "="*80)
    print("LAMBDA³ INTERACTIVE DEMO")
    print("="*80)
    
    print("\nSelect data source:")
    print("1. Synthetic data (fast)")
    print("2. Financial data (requires internet)")
    print("3. Load from CSV file")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        return demo_synthetic_data()
    elif choice == "2":
        return demo_financial_data()
    elif choice == "3":
        csv_path = input("Enter CSV file path: ").strip()
        if Path(csv_path).exists():
            from src.utils.data_loader import load_csv_data
            data = load_csv_data(csv_path)
            
            config = L3Config()
            results = run_lambda3_analysis(data, config)
            return results
        else:
            print(f"File not found: {csv_path}")
            return None
    else:
        print("Invalid choice")
        return None

def main():
    """Main demo execution"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         LAMBDA³ ANALYTICS FRAMEWORK - LOCAL DEMO          ║")
    print("║                                                           ║")
    print("║    Universal Structural Tensor Field Analytics            ║")
    print("║    Beyond time, beyond causality                          ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    
    print("\nDemo Options:")
    print("1. Quick synthetic data demo")
    print("2. Financial markets demo")
    print("3. Interactive mode")
    
    mode = input("\nSelect mode (1-3) [default: 1]: ").strip() or "1"
    
    if mode == "1":
        results = demo_synthetic_data()
    elif mode == "2":
        results = demo_financial_data()
    elif mode == "3":
        results = demo_interactive()
    else:
        print("Invalid mode")
        return
    
    if results:
        print("\n" + "="*80)
        print("DEMO COMPLETE")
        print("="*80)
        print("\nResults saved to: results/demo/")
        print("\nNext steps:")
        print("  • Explore results with Jupyter notebooks")
        print("  • Scale up with cloud parallel processing")
        print("  • Apply to your own data")
        
        # Print Lambda³ insights
        print("\n" + "-"*80)
        print("LAMBDA³ THEORETICAL INSIGHTS")
        print("-"*80)
        print("• Structural tensor Λ reveals hidden system dynamics")
        print("• ΔΛC pulsations mark critical state transitions")
        print("• Tension scalar ρT quantifies structural stress")
        print("• Time-independent analysis transcends causality")

if __name__ == "__main__":
    main()
