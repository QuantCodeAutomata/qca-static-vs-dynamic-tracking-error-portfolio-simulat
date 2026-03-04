"""
Experiment 2: TE Governance Constraint Spectrum Convergence Analysis

Tests 11 TE constraint levels from 0.5% to 5.0% to demonstrate:
- Sharpe ratios are statistically indistinguishable across constraints
- σ(TE) varies dramatically (approximately 12×)
- Portfolio returns plateau above ~3.5% TE cap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    calculate_sharpe, block_bootstrap, jobson_korkie_test,
    compute_regime_from_vix, compute_spread_volatility,
    compute_active_weight, compute_portfolio_weights,
    compute_portfolio_returns, compute_realized_te,
    performance_summary
)
from exp_1_static_vs_dynamic import construct_benchmark

sns.set_style("whitegrid")


def construct_constrained_te_portfolio(spx_returns: pd.Series, agg_returns: pd.Series,
                                       vix: pd.Series, te_cap: float) -> Tuple[pd.Series, pd.Series]:
    """
    Construct dynamic TE portfolio with constraint cap.
    
    Parameters
    ----------
    spx_returns : pd.Series
        SPX daily returns
    agg_returns : pd.Series
        AGG daily returns
    vix : pd.Series
        VIX daily close
    te_cap : float
        TE constraint cap (e.g., 0.02 for 2%)
        
    Returns
    -------
    tuple
        (portfolio_returns, realized_te)
    """
    # Compute regime-based TE target
    regime_te_target = compute_regime_from_vix(vix, window=21)
    
    # Apply cap
    te_target_capped = regime_te_target.clip(upper=te_cap)
    
    # Compute spread volatility
    spread_vol = compute_spread_volatility(spx_returns, agg_returns)
    
    # Compute active weight
    theta = compute_active_weight(te_target_capped, spread_vol)
    
    # Compute portfolio weights
    equity_weight, bond_weight = compute_portfolio_weights(theta)
    
    # Compute portfolio returns
    portfolio_returns = compute_portfolio_returns(
        equity_weight, bond_weight, spx_returns, agg_returns
    )
    
    # Compute benchmark for active returns
    benchmark_returns, _, _ = construct_benchmark(spx_returns, agg_returns)
    
    # Active returns and realized TE
    active_returns = portfolio_returns - benchmark_returns
    realized_te = compute_realized_te(active_returns)
    
    return portfolio_returns, realized_te


def run_experiment_2(spx_returns: pd.Series, agg_returns: pd.Series, 
                    vix: pd.Series) -> Dict:
    """
    Run Experiment 2: TE Governance Constraint Spectrum.
    
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
        Results dictionary
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: TE Governance Constraint Spectrum Convergence Analysis")
    print("="*80)
    
    # Define constraint levels
    te_caps = np.arange(0.005, 0.055, 0.005)  # 0.5% to 5.0% in 0.5% increments
    
    results_list = []
    portfolio_returns_dict = {}
    realized_te_dict = {}
    
    print(f"\nRunning simulations for {len(te_caps)} constraint levels...")
    
    for i, te_cap in enumerate(te_caps):
        print(f"  {i+1}/{len(te_caps)}: TE cap = {te_cap*100:.1f}%...")
        
        port_returns, realized_te = construct_constrained_te_portfolio(
            spx_returns, agg_returns, vix, te_cap
        )
        
        # Drop NaN
        valid_idx = ~port_returns.isna()
        port_returns_clean = port_returns[valid_idx]
        realized_te_clean = realized_te[valid_idx]
        
        # Compute metrics
        perf = performance_summary(port_returns_clean, f"TE Cap {te_cap*100:.1f}%")
        
        # Add TE metrics
        perf['te_cap'] = te_cap * 100
        perf['mean_te'] = realized_te_clean.mean() * 100
        perf['sigma_te'] = realized_te_clean.std() * 100
        
        results_list.append(perf)
        portfolio_returns_dict[te_cap] = port_returns_clean
        realized_te_dict[te_cap] = realized_te_clean
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print("RESULTS - EXHIBIT 7: Convergence Analysis")
    print("="*80)
    
    summary_table = results_df[['te_cap', 'cagr', 'volatility', 'sharpe', 
                                 'max_drawdown', 'mean_te', 'sigma_te']].copy()
    summary_table['cagr'] = summary_table['cagr'] * 100
    summary_table['volatility'] = summary_table['volatility'] * 100
    summary_table['max_drawdown'] = summary_table['max_drawdown'] * 100
    
    print(summary_table.round(3))
    
    # Bootstrap Sharpe CIs
    print("\n" + "="*80)
    print("Block Bootstrap Sharpe 95% CIs:")
    print("="*80)
    
    bootstrap_results = []
    for te_cap in te_caps:
        boot_sharpes = block_bootstrap(portfolio_returns_dict[te_cap], 
                                       n_iterations=10000, 
                                       block_size=63,
                                       seed=int(te_cap*10000))
        ci_low = np.percentile(boot_sharpes, 2.5)
        ci_high = np.percentile(boot_sharpes, 97.5)
        
        bootstrap_results.append({
            'te_cap': te_cap * 100,
            'sharpe': calculate_sharpe(portfolio_returns_dict[te_cap]),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'ci_width': ci_high - ci_low
        })
        
        print(f"TE Cap {te_cap*100:4.1f}%: Sharpe = {bootstrap_results[-1]['sharpe']:.3f}, "
              f"CI = [{ci_low:.3f}, {ci_high:.3f}], Width = {bootstrap_results[-1]['ci_width']:.3f}")
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    
    # Sharpe equality tests (pairwise adjacent)
    print("\n" + "="*80)
    print("Jobson-Korkie Sharpe Equality Tests (Adjacent Pairs):")
    print("="*80)
    
    for i in range(len(te_caps) - 1):
        returns1 = portfolio_returns_dict[te_caps[i]]
        returns2 = portfolio_returns_dict[te_caps[i+1]]
        t_stat, p_val = jobson_korkie_test(returns1, returns2)
        print(f"{te_caps[i]*100:.1f}% vs {te_caps[i+1]*100:.1f}%: t-stat = {t_stat:.3f}, p-value = {p_val:.3f}")
    
    # Test extremes
    t_stat, p_val = jobson_korkie_test(portfolio_returns_dict[te_caps[0]], 
                                       portfolio_returns_dict[te_caps[-1]])
    print(f"\nExtreme: {te_caps[0]*100:.1f}% vs {te_caps[-1]*100:.1f}%: "
          f"t-stat = {t_stat:.3f}, p-value = {p_val:.3f}")
    
    # Returns plateau analysis
    print("\n" + "="*80)
    print("Returns Plateau Analysis:")
    print("="*80)
    
    for i in range(len(te_caps) - 1):
        cagr_diff = (results_df.iloc[i+1]['cagr'] - results_df.iloc[i]['cagr']) * 10000  # bps
        print(f"{te_caps[i]*100:.1f}% → {te_caps[i+1]*100:.1f}%: +{cagr_diff:.1f} bps")
    
    results = {
        'summary_df': results_df,
        'bootstrap_df': bootstrap_df,
        'portfolio_returns': portfolio_returns_dict,
        'realized_te': realized_te_dict,
        'te_caps': te_caps
    }
    
    return results


def plot_experiment_2_results(results: Dict, save_dir: str = "results"):
    """
    Create visualizations for Experiment 2.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Sharpe vs TE Cap
    fig, ax = plt.subplots(figsize=(12, 7))
    
    te_caps_pct = results['summary_df']['te_cap']
    sharpes = results['summary_df']['sharpe']
    
    ax.plot(te_caps_pct, sharpes, marker='o', linewidth=2, markersize=8, color='steelblue')
    
    # Add bootstrap CIs
    ax.fill_between(te_caps_pct, 
                    results['bootstrap_df']['ci_low'],
                    results['bootstrap_df']['ci_high'],
                    alpha=0.2, color='steelblue')
    
    ax.set_title('Exhibit 7A: Sharpe Ratio Convergence Across TE Constraints', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('TE Constraint Level (%)', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp2_sharpe_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: σ(TE) vs TE Cap
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sigma_te = results['summary_df']['sigma_te']
    
    ax.plot(te_caps_pct, sigma_te, marker='s', linewidth=2, markersize=8, color='coral')
    
    ax.set_title('Exhibit 7B: σ(TE) Variation Across TE Constraints', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('TE Constraint Level (%)', fontsize=12)
    ax.set_ylabel('σ(TE) (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp2_sigma_te_variation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Realized TE Fan-Out
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot subset of constraint levels for clarity
    selected_caps = [0.005, 0.015, 0.025, 0.035, 0.050]
    
    for te_cap in selected_caps:
        te_series = results['realized_te'][te_cap] * 100
        ax.plot(te_series.index, te_series, label=f'{te_cap*100:.1f}% cap', alpha=0.7, linewidth=1.5)
    
    ax.set_title('Exhibit 7C: Realized TE Fan-Out During Stress Episodes', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Realized TE (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp2_te_fanout.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")


if __name__ == "__main__":
    from data_loader import load_all_experiment_data, prepare_returns_data
    
    data = load_all_experiment_data()
    returns = prepare_returns_data(data)
    
    results = run_experiment_2(returns['spx'], returns['agg'], data['vix'])
    plot_experiment_2_results(results)
    
    print("\nExperiment 2 completed successfully!")
