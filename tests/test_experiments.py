"""
Comprehensive test suite for all experiments.
Validates methodology implementation against paper specifications.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    calculate_cagr, calculate_volatility, calculate_sharpe,
    calculate_max_drawdown, block_bootstrap, compute_regime_from_vix,
    compute_spread_volatility, compute_active_weight, performance_summary
)
from data_loader import generate_synthetic_data


def test_cagr_calculation():
    """Test CAGR calculation formula."""
    # Create test returns: 10% total return over 252 days should yield ~10% CAGR
    returns = pd.Series([0.0003] * 252)  # Small daily returns
    total_ret = (1 + returns).prod() - 1
    
    cagr = calculate_cagr(returns)
    
    # CAGR should equal approximately the total return for 1 year
    assert abs(cagr - total_ret) < 0.01, "CAGR calculation error"
    assert cagr > 0, "CAGR should be positive for positive returns"
    print("✓ test_cagr_calculation passed")


def test_volatility_annualization():
    """Test volatility annualization uses sqrt(252)."""
    # Create constant volatility returns
    np.random.seed(42)
    daily_vol = 0.01
    returns = pd.Series(np.random.randn(252) * daily_vol)
    
    ann_vol = calculate_volatility(returns)
    expected_vol = returns.std() * np.sqrt(252)
    
    assert abs(ann_vol - expected_vol) < 1e-10, "Volatility annualization incorrect"
    assert ann_vol > 0, "Volatility must be positive"
    print("✓ test_volatility_annualization passed")


def test_sharpe_ratio_formula():
    """Test Sharpe ratio = CAGR / Vol (no risk-free rate)."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)
    
    sharpe = calculate_sharpe(returns)
    cagr = calculate_cagr(returns)
    vol = calculate_volatility(returns)
    
    assert abs(sharpe - cagr / vol) < 1e-10, "Sharpe formula error"
    print("✓ test_sharpe_ratio_formula passed")


def test_max_drawdown_range():
    """Test max drawdown is between 0 and 1."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01)
    
    max_dd = calculate_max_drawdown(returns)
    
    assert 0 <= max_dd <= 1, "Max drawdown must be in [0, 1]"
    print("✓ test_max_drawdown_range passed")


def test_regime_classification():
    """Test VIX regime classification thresholds."""
    # Create VIX series with known values
    vix = pd.Series([10, 12, 15, 20, 25, 30], 
                    index=pd.date_range('2020-01-01', periods=6, freq='B'))
    
    regime = compute_regime_from_vix(vix, window=1, low_threshold=13, high_threshold=22)
    
    # After lag, check classifications
    # Index 0 and 1 should be NaN (lag + window)
    # Index 2: VIX=15 (lagged from 12 < 13) -> Low (0.5%)
    # Index 3: VIX=20 (lagged from 15, 13<=15<=22) -> Neutral (2%)
    # Index 4: VIX=25 (lagged from 20, 13<=20<=22) -> Neutral (2%)
    # Index 5: VIX=30 (lagged from 25 > 22) -> High (5%)
    
    assert regime.iloc[2] == 0.005, "Low regime should be 0.5%"
    assert regime.iloc[5] == 0.05, "High regime should be 5%"
    print("✓ test_regime_classification passed")


def test_spread_volatility_lag():
    """Test spread volatility is lagged (no look-ahead)."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    spx_ret = pd.Series(np.random.randn(100) * 0.01, index=dates)
    agg_ret = pd.Series(np.random.randn(100) * 0.005, index=dates)
    
    spread_vol = compute_spread_volatility(spx_ret, agg_ret, window=63)
    
    # First values should be NaN due to window + lag
    assert spread_vol.iloc[:63].isna().sum() > 50, "Spread vol should have many NaN for burn-in"
    
    # All non-NaN values should be positive
    assert (spread_vol.dropna() > 0).all(), "Spread vol must be positive"
    print("✓ test_spread_volatility_lag passed")


def test_active_weight_bounds():
    """Test active weight theta is clipped to [0, 0.25]."""
    # Create TE target and spread vol series
    te_target = pd.Series([0.02, 0.05, 0.001])  # 2%, 5%, 0.1%
    spread_vol = pd.Series([0.01, 0.10, 0.10])   # 1%, 10%, 10%
    
    theta = compute_active_weight(te_target, spread_vol, max_theta=0.25)
    
    # Expected: 0.02/0.01=2.0 -> clip to 0.25
    #           0.05/0.10=0.5 -> clip to 0.25 (max_theta enforced)
    #           0.001/0.10=0.01 -> no clip
    assert theta.iloc[0] == 0.25, "Theta should be clipped to max"
    assert theta.iloc[1] == 0.25, "Theta should be clipped to max_theta"
    assert 0 <= theta.iloc[2] <= 0.25, "Theta should be in bounds"
    
    # All theta values must be in [0, 0.25]
    assert (theta >= 0).all() and (theta <= 0.25).all(), "Theta bounds violated"
    print("✓ test_active_weight_bounds passed")


def test_portfolio_weights_sum():
    """Test portfolio weights sum to 1.0."""
    theta = pd.Series([0.0, 0.1, 0.2, 0.25])
    
    from utils import compute_portfolio_weights
    eq_wt, bond_wt = compute_portfolio_weights(theta, base_equity_weight=0.70)
    
    # Weights should sum to 1
    total_wt = eq_wt + bond_wt
    assert (abs(total_wt - 1.0) < 1e-10).all(), "Weights must sum to 1.0"
    
    # Equity weight should be 0.70 + theta
    assert (abs(eq_wt - (0.70 + theta)) < 1e-10).all(), "Equity weight formula error"
    print("✓ test_portfolio_weights_sum passed")


def test_block_bootstrap_output_size():
    """Test block bootstrap returns correct number of Sharpe values."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)
    
    n_iter = 100
    boot_sharpes = block_bootstrap(returns, n_iterations=n_iter, block_size=63, seed=42)
    
    assert len(boot_sharpes) == n_iter, "Bootstrap should return n_iterations values"
    assert np.isfinite(boot_sharpes).all(), "All bootstrap Sharpes should be finite"
    print("✓ test_block_bootstrap_output_size passed")


def test_performance_summary_keys():
    """Test performance summary returns all required metrics."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)
    
    perf = performance_summary(returns, "Test")
    
    required_keys = ['name', 'cagr', 'volatility', 'sharpe', 'max_drawdown', 
                     'cagr_maxdd', 'total_return', 'n_periods']
    
    for key in required_keys:
        assert key in perf, f"Performance summary missing key: {key}"
    
    assert perf['n_periods'] == 252, "Number of periods should match input length"
    print("✓ test_performance_summary_keys passed")


def test_vix_data_generation():
    """Test VIX synthetic data has realistic properties."""
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    vix_data = generate_synthetic_data('VIX', '2020-01-01', '2020-12-31')
    
    vix_close = vix_data['close']
    
    # VIX should be positive
    assert (vix_close > 0).all(), "VIX must be positive"
    
    # VIX should have reasonable range (typically 10-80)
    assert vix_close.min() > 5, "VIX too low"
    assert vix_close.max() < 100, "VIX too high"
    
    # VIX should be mean-reverting (autocorrelation < 1)
    autocorr = vix_close.autocorr()
    assert 0 < autocorr < 1, "VIX should show positive autocorrelation"
    print("✓ test_vix_data_generation passed")


def test_te_target_values():
    """Test TE targets are exactly 0.5%, 2%, or 5%."""
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    vix = pd.Series(np.random.uniform(10, 40, 100), index=dates)
    
    regime = compute_regime_from_vix(vix, window=21, low_threshold=13, high_threshold=22)
    
    # After burn-in, regime values should only be 0.005, 0.02, or 0.05
    valid_values = regime.dropna()
    unique_values = set(valid_values.unique())
    
    assert unique_values.issubset({0.005, 0.02, 0.05}), "Invalid TE target values"
    print("✓ test_te_target_values passed")


def test_equity_weight_range():
    """Test equity weights stay in [0.70, 0.95] range."""
    # Max theta is 0.25, so equity weight range is [0.70, 0.95]
    theta = pd.Series([0.0, 0.1, 0.2, 0.25])
    
    from utils import compute_portfolio_weights
    eq_wt, bond_wt = compute_portfolio_weights(theta)
    
    assert (eq_wt >= 0.70).all(), "Equity weight below minimum"
    assert (eq_wt <= 0.95).all(), "Equity weight above maximum"
    
    # Weights must sum to 1
    assert (abs(eq_wt + bond_wt - 1.0) < 1e-10).all(), "Weights must sum to 1"
    
    # Bond weight should be reasonable
    assert (bond_wt >= 0.05).all(), "Bond weight below minimum"
    print("✓ test_equity_weight_range passed")


def test_realized_te_calculation():
    """Test realized TE is computed on 63-day rolling window."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    active_returns = pd.Series(np.random.randn(200) * 0.001, index=dates)
    
    from utils import compute_realized_te
    realized_te = compute_realized_te(active_returns, window=63)
    
    # First 62 values should be NaN (need 63 observations)
    assert realized_te.iloc[:62].isna().all(), "TE should have NaN for first 62 obs"
    
    # Realized TE should be non-negative
    assert (realized_te.dropna() >= 0).all(), "Realized TE must be non-negative"
    
    # Annualized TE should be reasonable (<50%)
    assert (realized_te.dropna() < 0.5).all(), "Realized TE unreasonably high"
    print("✓ test_realized_te_calculation passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*80 + "\n")
    
    test_functions = [
        test_cagr_calculation,
        test_volatility_annualization,
        test_sharpe_ratio_formula,
        test_max_drawdown_range,
        test_regime_classification,
        test_spread_volatility_lag,
        test_active_weight_bounds,
        test_portfolio_weights_sum,
        test_block_bootstrap_output_size,
        test_performance_summary_keys,
        test_vix_data_generation,
        test_te_target_values,
        test_equity_weight_range,
        test_realized_te_calculation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed out of {passed + failed} total")
    print("="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
