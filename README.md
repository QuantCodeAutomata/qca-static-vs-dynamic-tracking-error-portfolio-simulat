# Static vs Dynamic Tracking Error Portfolio Simulation

This repository implements a comprehensive quantitative finance research study on static versus dynamic tracking error (TE) governance in portfolio management.

## Overview

The project validates that dynamic TE strategies (varying TE with VIX regime) produce superior risk-adjusted returns compared to static TE strategies, while demonstrating that traditional Sharpe ratios fail to distinguish between governance approaches.

## Experiments

### Experiment 1: Static vs Dynamic TE Portfolio Simulation
- Constructs three portfolios: 70/30 benchmark, static 2% TE, dynamic 0.5/2/5% TE
- Time period: September 2004 - February 2026
- Validates ~53 bps CAGR advantage of dynamic over static strategy

### Experiment 2: TE Governance Constraint Spectrum Convergence
- Tests 11 TE constraint levels from 0.5% to 5.0%
- Demonstrates Sharpe ratios are indistinguishable across constraints
- Shows σ(TE) varies 12× across the spectrum

### Experiment 3: Omega Premium - VIX Quintile Forward Returns
- Sorts observations into VIX quintiles from 1990-2026
- Computes forward returns at 1M/3M/6M/1Y horizons
- Validates highest returns occur after peak-stress episodes

### Experiment 4: Regret Analysis at Crisis Troughs
- Identifies major drawdown troughs (GFC, COVID, 2022)
- Compares staying 70/30 vs de-risking to 30/70
- Quantifies opportunity cost of de-risking at worst moments

### Experiment 5: Rolling Correlation Exhibits
- Sector ETF pairwise correlations (63-day window)
- Stock-bond correlation regime shifts (126-day window)
- Validates VIX-driven correlation structure

### Experiment 6: Robustness Tests
- VIX smoothing window sensitivity (1d/5d/21d/63d)
- Markov-switching regime model comparison
- Validates 21-day window optimality

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Set your MASSIVE_TOKEN environment variable:
```bash
export MASSIVE_TOKEN="your_api_key_here"
```

Run all experiments:
```bash
python src/run_experiments.py
```

Run tests:
```bash
pytest tests/ -v
```

## Data Requirements

- S&P 500 Total Return Index (SPXT)
- AGG (iShares Core US Aggregate Bond ETF)
- TLT (iShares 20+ Year Treasury ETF)
- VIX Index
- 9 Sector SPDR ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)

All data retrieved via the Massive API.

## Results

Results are saved to the `results/` directory:
- `RESULTS.md`: Comprehensive metrics and findings
- `*.png`: Visualization charts
- `*.csv`: Detailed numerical results

## Structure

```
.
├── src/
│   ├── data_loader.py          # Data retrieval via Massive API
│   ├── utils.py                # Common utilities and metrics
│   ├── exp_1_static_vs_dynamic.py
│   ├── exp_2_constraint_spectrum.py
│   ├── exp_3_omega_premium.py
│   ├── exp_4_regret_analysis.py
│   ├── exp_5_rolling_correlations.py
│   ├── exp_6_robustness_tests.py
│   └── run_experiments.py      # Main execution script
├── tests/
│   └── test_*.py               # Test suite
├── results/
│   └── RESULTS.md              # Generated results
└── requirements.txt
```

## Methodology

All implementations strictly follow the paper methodology:
- No look-ahead bias (all signals lagged by 1 day)
- Exact 63-day rolling windows for volatility estimation
- Monthly benchmark rebalancing at month-end
- Daily portfolio rebalancing for TE strategies
- Block bootstrap (63-day blocks) for Sharpe uncertainty
- Newey-West HAC standard errors for overlapping returns

## License

MIT License

## Authors

QCA Agent - Quantitative Code Automata
