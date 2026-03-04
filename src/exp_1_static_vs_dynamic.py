"""
Experiment 1: Static vs Dynamic Tracking Error Portfolio Simulation

Constructs and compares three portfolios:
1. Monthly-rebalanced 70/30 benchmark
2. Static 2% TE portfolio
3. Dynamic regime-switching TE portfolio (0.5%/2%/5%)

Validates ~53 bps CAGR advantage of dynamic over static strategy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    calculate_cagr, calculate_volatility, calculate_sharpe,
    calculate_max_drawdown, block_bootstrap, jobson_korkie_test,
    compute_regime_from_vix, compute_spread_volatility,
    compute_active_weight, compute_portfolio_weights,
    compute_portfolio_returns, compute_realized_te,
    create_monthly_rebalance_weights, performance_summary,
    calculate_drawdown_series
)

sns.set_style("whitegrid")


def construct_benchmark(spx_returns: pd.Series, agg_returns: pd.Series,
                       equity_weight: float = 0.70) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Construct monthly-rebalanced benchmark portfolio.
    
    Parameters
    ----------
    spx_returns : pd.Series
        SPX daily returns
    agg_returns : pd.Series
        AGG daily returns
    equity_weight : float
        Target equity allocation
        
    Returns
    -------
    tuple
        (benchmark_returns, equity_weights, bond_weights)
    """
    # Create monthly rebalancing weights
    equity_weights, bond_weights = create_monthly_rebalance_weights(
        spx_returns, equity_weight
    )
    
    # Compute returns
    benchmark_returns = compute_portfolio_returns(
        equity_weights, bond_weights, spx_returns, agg_returns
    )
    
    return benchmark_returns, equity_weights, bond_weights


def construct_static_te_portfolio(spx_returns: pd.Series, agg_returns: pd.Series,
                                  te_target: float = 0.02) -> Tuple[pd.Series, pd.Series]:
    """
    Construct static TE portfolio with constant tracking error target.
    
    Parameters
    ----------
    spx_returns : pd.Series
        SPX daily returns
    agg_returns : pd.Series
        AGG daily returns
    te_target : float
        Static TE target (e.g., 0.02 for 2%)
        
    Returns
    -------
    tuple
        (portfolio_returns, realized_te)
    """
    # Compute spread volatility
    spread_vol = compute_spread_volatility(spx_returns, agg_returns)
    
    # Constant TE target
    te_target_series = pd.Series(te_target, index=spx_returns.index)
    
    # Compute active weight
    theta = compute_active_weight(te_target_series, spread_vol)
    
    # Compute portfolio weights
    equity_weight, bond_weight = compute_portfolio_weights(theta, base_equity_weight=0.70)
    
    # Compute portfolio returns
    portfolio_returns = compute_portfolio_returns(
        equity_weight, bond_weight, spx_returns, agg_returns
    )
    
    # Compute benchmark for active returns
    benchmark_returns, _, _ = construct_benchmark(spx_returns, agg_returns)
    
    # Active returns
    active_returns = portfolio_returns - benchmark_returns
    
    # Realized TE
    realized_te = compute_realized_te(active_returns)
    
    return portfolio_returns, realized_te


def construct_dynamic_te_portfolio(spx_returns: pd.Series, agg_returns: pd.Series,
                                   vix: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Construct dynamic TE portfolio with regime-based tracking error targets.
    
    Parameters
    ----------
    spx_returns : pd.Series
        SPX daily returns
    agg_returns : pd.Series
        AGG daily returns
    vix : pd.Series
        VIX daily close
        
    Returns
    -------
    tuple
        (portfolio_returns, realized_te, regime_te_target)
    """
    # Compute regime-based TE target
    regime_te_target = compute_regime_from_vix(vix, window=21, 
                                               low_threshold=13, 
                                               high_threshold=22)
    
    # Compute spread volatility
    spread_vol = compute_spread_volatility(spx_returns, agg_returns)
    
    # Compute active weight
    theta = compute_active_weight(regime_te_target, spread_vol)
    
    # Compute portfolio weights
    equity_weight, bond_weight = compute_portfolio_weights(theta, base_equity_weight=0.70)
    
    # Compute portfolio returns
    portfolio_returns = compute_portfolio_returns(
        equity_weight, bond_weight, spx_returns, agg_returns
    )
    
    # Compute benchmark for active returns
    benchmark_returns, _, _ = construct_benchmark(spx_returns, agg_returns)
    
    # Active returns
    active_returns = portfolio_returns - benchmark_returns
    
    # Realized TE
    realized_te = compute_realized_te(active_returns)
    
    return portfolio_returns, realized_te, regime_te_target


def run_experiment_1(spx_returns: pd.Series, agg_returns: pd.Series, 
                    vix: pd.Series) -> Dict:
    """
    Run Experiment 1: Static vs Dynamic TE Portfolio Simulation.
    
    Parameters
    ----------
    spx_returns : pd.Series
        SPX daily returns
    agg_returns : pd.Series
        AGG daily returns
    vix : pd.Series
        VIX daily close
        
    Returns
    -------
    dict
        Results dictionary with metrics and series
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Static vs Dynamic Tracking Error Portfolio Simulation")
    print("="*80)
    
    # Align data to common index (drop NaNs after rolling calculations)
    common_idx = spx_returns.index
    
    # 1. Construct Benchmark
    print("\n1. Constructing 70/30 monthly-rebalanced benchmark...")
    benchmark_returns, _, _ = construct_benchmark(spx_returns, agg_returns)
    
    # 2. Construct Static TE Portfolio
    print("2. Constructing static 2% TE portfolio...")
    static_returns, static_te = construct_static_te_portfolio(spx_returns, agg_returns, te_target=0.02)
    
    # 3. Construct Dynamic TE Portfolio
    print("3. Constructing dynamic TE portfolio (0.5%/2%/5%)...")
    dynamic_returns, dynamic_te, regime_target = construct_dynamic_te_portfolio(
        spx_returns, agg_returns, vix
    )
    
    # Drop initial NaN period (burn-in for rolling calculations)
    valid_idx = ~benchmark_returns.isna() & ~static_returns.isna() & ~dynamic_returns.isna()
    
    benchmark_returns_clean = benchmark_returns[valid_idx]
    static_returns_clean = static_returns[valid_idx]
    dynamic_returns_clean = dynamic_returns[valid_idx]
    static_te_clean = static_te[valid_idx]
    dynamic_te_clean = dynamic_te[valid_idx]
    vix_clean = vix[valid_idx]
    
    print(f"\nSimulation period: {benchmark_returns_clean.index[0]} to {benchmark_returns_clean.index[-1]}")
    print(f"Total trading days: {len(benchmark_returns_clean)}")
    
    # 4. Performance Metrics
    print("\n4. Computing performance metrics...")
    
    bench_perf = performance_summary(benchmark_returns_clean, "Benchmark 70/30")
    static_perf = performance_summary(static_returns_clean, "Static 2% TE")
    dynamic_perf = performance_summary(dynamic_returns_clean, "Dynamic TE (0.5/2/5%)")
    
    # Add TE metrics for active portfolios
    static_active = static_returns_clean - benchmark_returns_clean
    dynamic_active = dynamic_returns_clean - benchmark_returns_clean
    
    static_te_mean = static_te_clean.mean() * 100
    static_te_std = static_te_clean.std() * 100
    dynamic_te_mean = dynamic_te_clean.mean() * 100
    dynamic_te_std = dynamic_te_clean.std() * 100
    
    static_perf['mean_te'] = static_te_mean
    static_perf['sigma_te'] = static_te_std
    dynamic_perf['mean_te'] = dynamic_te_mean
    dynamic_perf['sigma_te'] = dynamic_te_std
    
    # 5. Statistical Tests
    print("\n5. Running statistical tests...")
    
    # Block bootstrap for Sharpe uncertainty
    print("   - Block bootstrap (10,000 iterations)...")
    bench_boot = block_bootstrap(benchmark_returns_clean, n_iterations=10000, block_size=63, seed=42)
    static_boot = block_bootstrap(static_returns_clean, n_iterations=10000, block_size=63, seed=43)
    dynamic_boot = block_bootstrap(dynamic_returns_clean, n_iterations=10000, block_size=63, seed=44)
    
    # Jobson-Korkie tests
    print("   - Jobson-Korkie Sharpe equality tests...")
    jk_dyn_static = jobson_korkie_test(dynamic_returns_clean, static_returns_clean)
    jk_dyn_bench = jobson_korkie_test(dynamic_returns_clean, benchmark_returns_clean)
    jk_static_bench = jobson_korkie_test(static_returns_clean, benchmark_returns_clean)
    
    # 6. TE Cyclicality
    print("\n6. Computing TE cyclicality...")
    
    # Correlation of realized TE with VIX
    te_vix_corr_static = static_te_clean.corr(vix_clean)
    te_vix_corr_dynamic = dynamic_te_clean.corr(vix_clean)
    
    # VIX SMA correlation
    vix_sma21 = vix_clean.rolling(21).mean()
    te_vix_sma_corr_static = static_te_clean.corr(vix_sma21)
    te_vix_sma_corr_dynamic = dynamic_te_clean.corr(vix_sma21)
    
    # 7. Print Results
    print("\n" + "="*80)
    print("RESULTS - EXHIBIT 3: Performance Metrics")
    print("="*80)
    
    results_table = pd.DataFrame({
        'Benchmark': [
            bench_perf['cagr'] * 100,
            bench_perf['volatility'] * 100,
            bench_perf['sharpe'],
            bench_perf['max_drawdown'] * 100,
            bench_perf['cagr_maxdd'],
            np.nan,
            np.nan
        ],
        'Static 2% TE': [
            static_perf['cagr'] * 100,
            static_perf['volatility'] * 100,
            static_perf['sharpe'],
            static_perf['max_drawdown'] * 100,
            static_perf['cagr_maxdd'],
            static_perf['mean_te'],
            static_perf['sigma_te']
        ],
        'Dynamic TE': [
            dynamic_perf['cagr'] * 100,
            dynamic_perf['volatility'] * 100,
            dynamic_perf['sharpe'],
            dynamic_perf['max_drawdown'] * 100,
            dynamic_perf['cagr_maxdd'],
            dynamic_perf['mean_te'],
            dynamic_perf['sigma_te']
        ]
    }, index=['CAGR (%)', 'Volatility (%)', 'Sharpe', 'Max DD (%)', 
              'CAGR/MaxDD', 'Mean TE (%)', 'σ(TE) (%)'])
    
    print(results_table.round(3))
    
    print("\n" + "="*80)
    print("Dynamic vs Static Advantage:")
    print("="*80)
    cagr_advantage_bps = (dynamic_perf['cagr'] - static_perf['cagr']) * 10000
    print(f"CAGR Advantage: {cagr_advantage_bps:.1f} bps")
    print(f"σ(TE) Ratio (Dynamic/Static): {dynamic_perf['sigma_te'] / static_perf['sigma_te']:.2f}x")
    
    print("\n" + "="*80)
    print("Sharpe Ratio Bootstrap 95% Confidence Intervals:")
    print("="*80)
    print(f"Benchmark:   [{np.percentile(bench_boot, 2.5):.3f}, {np.percentile(bench_boot, 97.5):.3f}] (width: {np.percentile(bench_boot, 97.5) - np.percentile(bench_boot, 2.5):.3f})")
    print(f"Static:      [{np.percentile(static_boot, 2.5):.3f}, {np.percentile(static_boot, 97.5):.3f}] (width: {np.percentile(static_boot, 97.5) - np.percentile(static_boot, 2.5):.3f})")
    print(f"Dynamic:     [{np.percentile(dynamic_boot, 2.5):.3f}, {np.percentile(dynamic_boot, 97.5):.3f}] (width: {np.percentile(dynamic_boot, 97.5) - np.percentile(dynamic_boot, 2.5):.3f})")
    
    print("\n" + "="*80)
    print("Jobson-Korkie Sharpe Equality Tests:")
    print("="*80)
    print(f"Dynamic vs Static:     t-stat = {jk_dyn_static[0]:.3f}, p-value = {jk_dyn_static[1]:.3f}")
    print(f"Dynamic vs Benchmark:  t-stat = {jk_dyn_bench[0]:.3f}, p-value = {jk_dyn_bench[1]:.3f}")
    print(f"Static vs Benchmark:   t-stat = {jk_static_bench[0]:.3f}, p-value = {jk_static_bench[1]:.3f}")
    
    print("\n" + "="*80)
    print("TE Cyclicality (Correlation with VIX):")
    print("="*80)
    print(f"Static TE vs VIX:       {te_vix_corr_static:.3f}")
    print(f"Dynamic TE vs VIX:      {te_vix_corr_dynamic:.3f}")
    print(f"Static TE vs VIX_SMA21: {te_vix_sma_corr_static:.3f}")
    print(f"Dynamic TE vs VIX_SMA21:{te_vix_sma_corr_dynamic:.3f}")
    
    # Return results
    results = {
        'benchmark_returns': benchmark_returns_clean,
        'static_returns': static_returns_clean,
        'dynamic_returns': dynamic_returns_clean,
        'static_te': static_te_clean,
        'dynamic_te': dynamic_te_clean,
        'vix': vix_clean,
        'regime_target': regime_target[valid_idx],
        'metrics': {
            'benchmark': bench_perf,
            'static': static_perf,
            'dynamic': dynamic_perf
        },
        'bootstrap': {
            'benchmark': bench_boot,
            'static': static_boot,
            'dynamic': dynamic_boot
        },
        'tests': {
            'jk_dyn_static': jk_dyn_static,
            'jk_dyn_bench': jk_dyn_bench,
            'jk_static_bench': jk_static_bench
        },
        'cyclicality': {
            'te_vix_corr_static': te_vix_corr_static,
            'te_vix_corr_dynamic': te_vix_corr_dynamic,
            'te_vix_sma_corr_static': te_vix_sma_corr_static,
            'te_vix_sma_corr_dynamic': te_vix_sma_corr_dynamic
        },
        'summary_table': results_table
    }
    
    return results


def plot_experiment_1_results(results: Dict, save_dir: str = "results"):
    """
    Create visualizations for Experiment 1.
    
    Parameters
    ----------
    results : dict
        Results from run_experiment_1
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Cumulative Returns
    fig, ax = plt.subplots(figsize=(14, 7))
    
    (1 + results['benchmark_returns']).cumprod().plot(ax=ax, label='Benchmark 70/30', linewidth=2)
    (1 + results['static_returns']).cumprod().plot(ax=ax, label='Static 2% TE', linewidth=2)
    (1 + results['dynamic_returns']).cumprod().plot(ax=ax, label='Dynamic TE (0.5/2/5%)', linewidth=2)
    
    ax.set_title('Cumulative Returns: Benchmark vs Static vs Dynamic TE', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp1_cumulative_returns.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Realized TE Time Series (Exhibit 4)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top panel: Realized TE
    (results['static_te'] * 100).plot(ax=ax1, label='Static TE', linewidth=1.5, alpha=0.8)
    (results['dynamic_te'] * 100).plot(ax=ax1, label='Dynamic TE', linewidth=1.5, alpha=0.8)
    
    ax1.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='2% Target')
    ax1.axhline(y=0.5, color='lightgray', linestyle='--', alpha=0.5, label='0.5% Target')
    ax1.axhline(y=5, color='lightgray', linestyle='--', alpha=0.5, label='5% Target')
    
    ax1.set_title('Exhibit 4: Realized Tracking Error Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Realized TE (%)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: VIX overlay
    results['vix'].plot(ax=ax2, color='orange', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=13, color='green', linestyle='--', alpha=0.5, label='Low Threshold (13)')
    ax2.axhline(y=22, color='red', linestyle='--', alpha=0.5, label='High Threshold (22)')
    ax2.set_ylabel('VIX Level', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp1_exhibit4_realized_te.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Bootstrap Sharpe Distributions
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.hist(results['bootstrap']['benchmark'], bins=50, alpha=0.5, label='Benchmark', density=True)
    ax.hist(results['bootstrap']['static'], bins=50, alpha=0.5, label='Static', density=True)
    ax.hist(results['bootstrap']['dynamic'], bins=50, alpha=0.5, label='Dynamic', density=True)
    
    ax.axvline(results['metrics']['benchmark']['sharpe'], color='C0', linestyle='--', linewidth=2)
    ax.axvline(results['metrics']['static']['sharpe'], color='C1', linestyle='--', linewidth=2)
    ax.axvline(results['metrics']['dynamic']['sharpe'], color='C2', linestyle='--', linewidth=2)
    
    ax.set_title('Bootstrap Distribution of Sharpe Ratios (10,000 iterations)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp1_bootstrap_sharpe.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")


if __name__ == "__main__":
    from data_loader import load_all_experiment_data, prepare_returns_data
    
    # Load data
    data = load_all_experiment_data()
    returns = prepare_returns_data(data)
    
    # Run experiment
    results = run_experiment_1(returns['spx'], returns['agg'], data['vix'])
    
    # Create plots
    plot_experiment_1_results(results)
    
    print("\nExperiment 1 completed successfully!")
