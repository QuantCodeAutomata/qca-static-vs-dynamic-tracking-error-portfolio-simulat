"""
Experiment 4: Regret Analysis - Cost of De-Risking at Crisis Troughs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import calculate_drawdown_series, find_drawdown_troughs
from exp_1_static_vs_dynamic import construct_benchmark

sns.set_style("whitegrid")


def compute_forward_portfolio_return(spx_ret: pd.Series, agg_ret: pd.Series,
                                     start_date: pd.Timestamp, horizon: int,
                                     equity_weight: float = 0.70) -> float:
    """Compute forward return for fixed allocation portfolio."""
    # Get returns from start_date+1 for horizon days
    try:
        start_loc = spx_ret.index.get_loc(start_date)
        end_loc = min(start_loc + horizon, len(spx_ret) - 1)
        
        fwd_spx = spx_ret.iloc[start_loc+1:end_loc+1]
        fwd_agg = agg_ret.iloc[start_loc+1:end_loc+1]
        
        # Simulate portfolio with monthly rebalancing
        port_ret = 0
        for i in range(len(fwd_spx)):
            port_ret += equity_weight * fwd_spx.iloc[i] + (1-equity_weight) * fwd_agg.iloc[i]
        
        cum_ret = (1 + fwd_spx * equity_weight + fwd_agg * (1-equity_weight)).prod() - 1
        return cum_ret
    except:
        return np.nan


def run_experiment_4(spx_returns: pd.Series, agg_returns: pd.Series, 
                    vix: pd.Series) -> Dict:
    """Run Experiment 4: Regret Analysis."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Regret Analysis - Cost of De-Risking at Crisis Troughs")
    print("="*80)
    
    # Construct benchmark to find troughs
    benchmark_returns, _, _ = construct_benchmark(spx_returns, agg_returns)
    troughs = find_drawdown_troughs(benchmark_returns.dropna(), threshold=-0.15)
    
    print(f"\nIdentified {len(troughs)} major drawdown troughs:")
    for trough_date, dd_val in troughs:
        vix_at_trough = vix.loc[trough_date] if trough_date in vix.index else np.nan
        print(f"  {trough_date.date()}: DD = {dd_val:.1%}, VIX = {vix_at_trough:.1f}")
    
    # Horizons
    horizons = {'3M': 63, '6M': 126, '12M': 252}
    
    results_list = []
    
    for trough_date, dd_val in troughs:
        vix_val = vix.loc[trough_date] if trough_date in vix.index else np.nan
        
        row = {
            'date': trough_date,
            'drawdown': dd_val,
            'vix': vix_val
        }
        
        for h_name, h_days in horizons.items():
            # Stay at 70/30
            stay_ret = compute_forward_portfolio_return(spx_returns, agg_returns, 
                                                        trough_date, h_days, 0.70)
            # De-risk to 30/70
            derisk_ret = compute_forward_portfolio_return(spx_returns, agg_returns, 
                                                          trough_date, h_days, 0.30)
            
            regret = stay_ret - derisk_ret
            
            row[f'stay_{h_name}'] = stay_ret * 100
            row[f'derisk_{h_name}'] = derisk_ret * 100
            row[f'regret_{h_name}'] = regret * 100
        
        results_list.append(row)
    
    results_df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print("RESULTS - EXHIBIT 6: Regret from De-Risking at Troughs")
    print("="*80)
    print(results_df.round(2))
    
    return {'results_df': results_df, 'troughs': troughs}


def plot_experiment_4_results(results: Dict, save_dir: str = "results"):
    """Create visualizations for Experiment 4."""
    os.makedirs(save_dir, exist_ok=True)
    
    df = results['results_df']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df))
    width = 0.25
    
    for i, horizon in enumerate(['3M', '6M', '12M']):
        ax.bar(x + i*width, df[f'regret_{horizon}'], width, 
               label=f'{horizon} Regret', alpha=0.8)
    
    ax.set_title('Exhibit 6: Regret from De-Risking at Crisis Troughs', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Crisis Episode', fontsize=12)
    ax.set_ylabel('Regret (Stay 70/30 - De-risk 30/70) %', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.strftime('%Y-%m') for d in df['date']], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp4_regret_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    from data_loader import load_all_experiment_data, prepare_returns_data
    
    data = load_all_experiment_data()
    returns = prepare_returns_data(data)
    
    results = run_experiment_4(returns['spx'], returns['agg'], data['vix'])
    plot_experiment_4_results(results)
    
    print("\nExperiment 4 completed!")
