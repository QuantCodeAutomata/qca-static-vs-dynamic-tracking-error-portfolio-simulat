"""
Quick demonstration version of experiments with smaller data.
"""

import sys
import os
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

warnings.filterwarnings('ignore')

from data_loader import load_all_experiment_data, prepare_returns_data
from exp_1_static_vs_dynamic import run_experiment_1, plot_experiment_1_results

print("\n" + "="*80)
print("QUICK DEMONSTRATION: Running Experiment 1")
print("="*80)

# Load data with shorter period
data = load_all_experiment_data(start_date="2010-01-01", end_date="2023-12-31")
returns = prepare_returns_data(data)

# Run Experiment 1
results = run_experiment_1(returns['spx'], returns['agg'], data['vix'])

# Create plots
plot_experiment_1_results(results)

print("\n" + "="*80)
print("DEMONSTRATION COMPLETED!")
print("="*80)
print("Check results/ directory for outputs")
