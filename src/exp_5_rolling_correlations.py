"""
Experiment 5: Rolling Correlation Exhibits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sns.set_style("whitegrid")


def run_experiment_5(sector_returns: Dict[str, pd.Series], spx_returns: pd.Series,
                    agg_returns: pd.Series, tlt_returns: pd.Series, vix: pd.Series) -> Dict:
    """Run Experiment 5: Rolling Correlations."""
    print("\n" + "="*80)
    print("EXPERIMENT 5: Rolling Correlation Exhibits")
    print("="*80)
    
    # Part A: Sector correlations
    print("\nPart A: Computing average pairwise sector correlations (63-day window)...")
    
    sectors = ['xlb', 'xle', 'xlf', 'xli', 'xlk', 'xlp', 'xlu', 'xlv', 'xly']
    sector_df = pd.DataFrame({s: sector_returns[s] for s in sectors if s in sector_returns})
    
    # Rolling pairwise correlations
    window = 63
    avg_corr = []
    dates = []
    
    for i in range(window, len(sector_df)):
        window_data = sector_df.iloc[i-window:i]
        corr_matrix = window_data.corr()
        
        # Extract upper triangle (exclude diagonal)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        avg = upper_tri.stack().mean()
        
        avg_corr.append(avg)
        dates.append(sector_df.index[i])
    
    avg_sector_corr = pd.Series(avg_corr, index=dates)
    
    print(f"  Average sector correlation: mean = {avg_sector_corr.mean():.3f}")
    
    # Part B: Stock-bond correlation
    print("\nPart B: Computing stock-bond rolling correlations (126-day window)...")
    
    window_sb = 126
    
    # SPX vs AGG
    rolling_corr_agg = spx_returns.rolling(window_sb).corr(agg_returns)
    
    # SPX vs TLT
    rolling_corr_tlt = spx_returns.rolling(window_sb).corr(tlt_returns)
    
    print(f"  SPX-AGG correlation: mean = {rolling_corr_agg.mean():.3f}")
    print(f"  SPX-TLT correlation: mean = {rolling_corr_tlt.mean():.3f}")
    
    return {
        'avg_sector_corr': avg_sector_corr,
        'spx_agg_corr': rolling_corr_agg,
        'spx_tlt_corr': rolling_corr_tlt,
        'vix': vix
    }


def plot_experiment_5_results(results: Dict, save_dir: str = "results"):
    """Create visualizations for Experiment 5."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Sector correlations with VIX overlay
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    results['avg_sector_corr'].plot(ax=ax1, linewidth=1.5, color='steelblue')
    ax1.set_title('Exhibit 1: Average Sector Pairwise Correlation', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Correlation', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    results['vix'].plot(ax=ax2, linewidth=1.5, color='orange', alpha=0.7)
    ax2.set_ylabel('VIX Level', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp5_exhibit1_sector_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Stock-bond correlations
    fig, ax = plt.subplots(figsize=(14, 7))
    
    results['spx_agg_corr'].plot(ax=ax, label='SPX-AGG', linewidth=1.5)
    results['spx_tlt_corr'].plot(ax=ax, label='SPX-TLT', linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Exhibit 2: Stock-Bond Rolling Correlation (126-day)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp5_exhibit2_stock_bond_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")


if __name__ == "__main__":
    from data_loader import load_all_experiment_data, prepare_returns_data
    
    data = load_all_experiment_data(start_date="2000-01-01")
    returns = prepare_returns_data(data)
    
    sector_returns = {k: v for k, v in returns.items() if k.startswith('xl')}
    
    results = run_experiment_5(sector_returns, returns['spx'], returns['agg'], 
                              returns['tlt'], data['vix'])
    plot_experiment_5_results(results)
    
    print("\nExperiment 5 completed!")
