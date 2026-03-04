"""
Experiment 3: Omega Premium - VIX Quintile Forward Return Analysis

Sorts daily observations into VIX quintiles and computes forward returns
at multiple horizons to validate that opportunity cost spikes during stress.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import newey_west_tstat

sns.set_style("whitegrid")


def compute_forward_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """
    Compute forward cumulative returns over specified horizon.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    horizon : int
        Forward horizon in trading days
        
    Returns
    -------
    pd.Series
        Forward cumulative returns (annualized)
    """
    fwd_returns = pd.Series(index=returns.index, dtype=float)
    
    for i in range(len(returns) - horizon):
        cum_ret = (1 + returns.iloc[i+1:i+1+horizon]).prod() - 1
        # Annualize
        ann_ret = (1 + cum_ret) ** (252 / horizon) - 1
        fwd_returns.iloc[i] = ann_ret
    
    return fwd_returns


def run_experiment_3(spx_returns: pd.Series, vix: pd.Series, 
                    start_date: str = "1990-01-01") -> Dict:
    """
    Run Experiment 3: VIX Quintile Forward Return Analysis.
    
    Parameters
    ----------
    spx_returns : pd.Series
        SPX daily returns (from 1990)
    vix : pd.Series
        VIX daily close (from 1990)
    start_date : str
        Analysis start date
        
    Returns
    -------
    dict
        Results dictionary
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Omega Premium - VIX Quintile Forward Return Analysis")
    print("="*80)
    
    # Align data
    common_idx = spx_returns.index.intersection(vix.index)
    spx_returns = spx_returns.loc[common_idx]
    vix = vix.loc[common_idx]
    
    # Filter to start date
    spx_returns = spx_returns[spx_returns.index >= start_date]
    vix = vix[vix.index >= start_date]
    
    print(f"\nAnalysis period: {spx_returns.index[0]} to {spx_returns.index[-1]}")
    print(f"Total observations: {len(spx_returns)}")
    
    # Compute VIX quintiles
    vix_quintiles = pd.qcut(vix, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    quintile_bounds = pd.qcut(vix, q=5, retbins=True)[1]
    
    print("\nVIX Quintile Boundaries:")
    for i, (low, high) in enumerate(zip(quintile_bounds[:-1], quintile_bounds[1:])):
        print(f"  Q{i+1}: {low:.2f} to {high:.2f}")
    
    # Define horizons
    horizons = {
        '1M': 21,
        '3M': 63,
        '6M': 126,
        '1Y': 252
    }
    
    results_dict = {}
    
    for horizon_name, horizon_days in horizons.items():
        print(f"\nComputing {horizon_name} forward returns (h={horizon_days} days)...")
        
        # Compute forward returns
        fwd_returns = compute_forward_returns(spx_returns, horizon_days)
        
        # Drop observations with incomplete forward windows
        valid_fwd = fwd_returns.dropna()
        valid_vix_quintiles = vix_quintiles.loc[valid_fwd.index]
        
        print(f"  Valid observations: {len(valid_fwd)}")
        
        # Average forward returns by quintile
        quintile_means = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_returns = valid_fwd[valid_vix_quintiles == q]
            quintile_means[q] = q_returns.mean()
        
        # Q5 - Q1 spread with Newey-West t-stat
        q5_returns = valid_fwd[valid_vix_quintiles == 'Q5']
        q1_returns = valid_fwd[valid_vix_quintiles == 'Q1']
        
        spread = q5_returns.mean() - q1_returns.mean()
        
        # Newey-West for the spread (using difference series)
        diff_series = pd.Series(0.0, index=valid_fwd.index)
        diff_series[valid_vix_quintiles == 'Q5'] = valid_fwd[valid_vix_quintiles == 'Q5']
        diff_series[valid_vix_quintiles == 'Q1'] = -valid_fwd[valid_vix_quintiles == 'Q1']
        
        _, nw_se, nw_tstat = newey_west_tstat(diff_series[diff_series != 0], lags=horizon_days)
        
        results_dict[horizon_name] = {
            'quintile_means': quintile_means,
            'spread': spread,
            'nw_tstat': nw_tstat,
            'n_obs': len(valid_fwd)
        }
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS - EXHIBIT 5: Forward Returns by VIX Quintile")
    print("="*80)
    
    # Create table
    quintile_table = pd.DataFrame({
        horizon: [results_dict[horizon]['quintile_means'][q] * 100 
                 for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
        for horizon in ['1M', '3M', '6M', '1Y']
    }, index=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    print(quintile_table.round(2))
    
    print("\n" + "="*80)
    print("Q5 - Q1 Spread (High VIX minus Low VIX):")
    print("="*80)
    
    for horizon in ['1M', '3M', '6M', '1Y']:
        spread_pp = results_dict[horizon]['spread'] * 100
        tstat = results_dict[horizon]['nw_tstat']
        print(f"{horizon:3s}: {spread_pp:+6.2f} pp  (NW t-stat = {tstat:6.2f})")
    
    results = {
        'quintile_table': quintile_table,
        'results_dict': results_dict,
        'quintile_bounds': quintile_bounds,
        'vix_quintiles': vix_quintiles
    }
    
    return results


def plot_experiment_3_results(results: Dict, save_dir: str = "results"):
    """
    Create visualizations for Experiment 3.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot: Forward Returns by Quintile (grouped bars)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    quintile_table = results['quintile_table']
    
    x = np.arange(len(quintile_table.index))
    width = 0.2
    
    for i, horizon in enumerate(['1M', '3M', '6M', '1Y']):
        ax.bar(x + i * width, quintile_table[horizon], width, label=horizon, alpha=0.8)
    
    ax.set_title('Exhibit 5: Annualized Forward Returns by VIX Quintile', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('VIX Quintile', fontsize=12)
    ax.set_ylabel('Annualized Forward Return (%)', fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(quintile_table.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp3_exhibit5_forward_returns.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot: Q5-Q1 Spread with significance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = ['1M', '3M', '6M', '1Y']
    spreads = [results['results_dict'][h]['spread'] * 100 for h in horizons]
    tstats = [results['results_dict'][h]['nw_tstat'] for h in horizons]
    
    colors = ['green' if abs(t) > 1.96 else 'orange' for t in tstats]
    
    ax.bar(horizons, spreads, color=colors, alpha=0.7)
    
    for i, (h, s, t) in enumerate(zip(horizons, spreads, tstats)):
        ax.text(i, s + 1 if s > 0 else s - 1, f't={t:.2f}', 
                ha='center', fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_title('Q5 - Q1 Spread: Omega Premium by Horizon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Spread (pp)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp3_omega_premium_spread.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")


if __name__ == "__main__":
    from data_loader import load_all_experiment_data, prepare_returns_data
    
    # Need extended history back to 1990
    data = load_all_experiment_data(start_date="1990-01-01")
    returns = prepare_returns_data(data)
    
    results = run_experiment_3(returns['spx'], data['vix'])
    plot_experiment_3_results(results)
    
    print("\nExperiment 3 completed successfully!")
