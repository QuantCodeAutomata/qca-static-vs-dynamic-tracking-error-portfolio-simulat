"""
Utility functions for portfolio analysis and statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def calculate_cagr(returns: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        CAGR as decimal (e.g., 0.089 for 8.9%)
    """
    terminal_wealth = (1 + returns).prod()
    n_days = len(returns)
    cagr = terminal_wealth ** (252 / n_days) - 1
    return cagr


def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        Annualized volatility
    """
    return returns.std() * np.sqrt(252)


def calculate_sharpe(returns: pd.Series) -> float:
    """
    Calculate Sharpe ratio (CAGR / Vol, no risk-free rate).
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        Sharpe ratio
    """
    cagr = calculate_cagr(returns)
    vol = calculate_volatility(returns)
    return cagr / vol if vol > 0 else 0.0


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        Maximum drawdown as positive decimal
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    pd.Series
        Drawdown series
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def find_drawdown_troughs(returns: pd.Series, threshold: float = -0.15) -> List[Tuple[pd.Timestamp, float]]:
    """
    Find major drawdown trough dates.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    threshold : float
        Minimum drawdown to consider (e.g., -0.15 for -15%)
        
    Returns
    -------
    list
        List of (date, drawdown) tuples for major troughs
    """
    drawdown = calculate_drawdown_series(returns)
    
    # Find local minima in major drawdown episodes
    troughs = []
    
    # Define crisis windows
    crisis_windows = [
        ('2008-09-01', '2009-03-31'),  # GFC
        ('2020-02-01', '2020-04-30'),  # COVID
        ('2022-01-01', '2022-10-31'),  # 2022
    ]
    
    for start, end in crisis_windows:
        try:
            window_dd = drawdown.loc[start:end]
            if len(window_dd) > 0:
                min_idx = window_dd.idxmin()
                min_val = window_dd.min()
                if min_val < threshold:
                    troughs.append((min_idx, min_val))
        except:
            pass
    
    return troughs


def block_bootstrap(returns: pd.Series, n_iterations: int = 10000, 
                    block_size: int = 63, seed: int = 42) -> np.ndarray:
    """
    Perform block bootstrap on return series.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    n_iterations : int
        Number of bootstrap iterations
    block_size : int
        Block size in trading days
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Array of Sharpe ratios from bootstrap samples
    """
    np.random.seed(seed)
    returns_array = returns.values
    n = len(returns_array)
    n_blocks = n // block_size
    
    sharpe_boots = []
    
    for _ in range(n_iterations):
        # Sample blocks with replacement
        boot_returns = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            boot_returns.extend(returns_array[start_idx:start_idx + block_size])
        
        boot_returns = np.array(boot_returns[:n])  # Trim to original length
        boot_series = pd.Series(boot_returns)
        
        sharpe_boots.append(calculate_sharpe(boot_series))
    
    return np.array(sharpe_boots)


def jobson_korkie_test(returns1: pd.Series, returns2: pd.Series) -> Tuple[float, float]:
    """
    Jobson-Korkie test for Sharpe ratio equality with Memmel correction.
    
    Parameters
    ----------
    returns1 : pd.Series
        First return series
    returns2 : pd.Series
        Second return series
        
    Returns
    -------
    tuple
        (t-statistic, p-value)
    """
    n = len(returns1)
    
    # Calculate Sharpe ratios
    sharpe1 = calculate_sharpe(returns1)
    sharpe2 = calculate_sharpe(returns2)
    
    # Calculate moments
    mu1 = returns1.mean() * 252
    mu2 = returns2.mean() * 252
    sigma1 = returns1.std() * np.sqrt(252)
    sigma2 = returns2.std() * np.sqrt(252)
    
    # Correlation
    rho = returns1.corr(returns2)
    
    # Jobson-Korkie variance with Memmel correction
    var_diff = (1 / n) * (
        2 - 2 * rho * (sigma1 / sigma2) + 
        (sharpe1 ** 2) / 2 + 
        (sharpe2 ** 2) / 2 - 
        sharpe1 * sharpe2 * rho
    )
    
    # Test statistic
    if var_diff > 0:
        t_stat = (sharpe1 - sharpe2) / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = 0.0
        p_value = 1.0
    
    return t_stat, p_value


def newey_west_tstat(series: pd.Series, lags: int) -> Tuple[float, float, float]:
    """
    Calculate Newey-West t-statistic for mean of a series.
    
    Parameters
    ----------
    series : pd.Series
        Data series
    lags : int
        Number of lags for HAC correction
        
    Returns
    -------
    tuple
        (mean, standard_error, t-statistic)
    """
    n = len(series)
    mean_val = series.mean()
    
    # Demean
    residuals = series - mean_val
    
    # Variance components
    gamma_0 = (residuals ** 2).mean()
    
    nw_var = gamma_0
    for j in range(1, lags + 1):
        gamma_j = (residuals.iloc[j:].values * residuals.iloc[:-j].values).mean()
        weight = 1 - j / (lags + 1)
        nw_var += 2 * weight * gamma_j
    
    nw_std = np.sqrt(nw_var / n)
    t_stat = mean_val / nw_std if nw_std > 0 else 0.0
    
    return mean_val, nw_std, t_stat


def compute_regime_from_vix(vix: pd.Series, window: int = 21, 
                           low_threshold: float = 13, 
                           high_threshold: float = 22) -> pd.Series:
    """
    Compute VIX regime classification.
    
    Parameters
    ----------
    vix : pd.Series
        VIX daily close
    window : int
        SMA window
    low_threshold : float
        Low regime threshold
    high_threshold : float
        High regime threshold
        
    Returns
    -------
    pd.Series
        Regime series: 0.5%, 2%, or 5% TE targets
    """
    # Compute VIX SMA
    vix_sma = vix.rolling(window=window, min_periods=window).mean()
    
    # Lag by one day (use yesterday's signal for today's regime)
    vix_sma_lagged = vix_sma.shift(1)
    
    # Classify regime
    regime = pd.Series(0.02, index=vix.index)  # Default to 2%
    regime[vix_sma_lagged < low_threshold] = 0.005  # Low: 0.5%
    regime[vix_sma_lagged > high_threshold] = 0.05  # High: 5%
    
    return regime


def compute_spread_volatility(spx_returns: pd.Series, agg_returns: pd.Series,
                              window: int = 63, floor: float = 0.001) -> pd.Series:
    """
    Compute annualized spread volatility with lag.
    
    Parameters
    ----------
    spx_returns : pd.Series
        SPX daily returns
    agg_returns : pd.Series
        AGG daily returns
    window : int
        Rolling window (63 trading days)
    floor : float
        Minimum volatility floor
        
    Returns
    -------
    pd.Series
        Annualized spread volatility (lagged)
    """
    # Compute spread
    spread = spx_returns - agg_returns
    
    # Rolling std (on lagged spread to avoid look-ahead)
    spread_lagged = spread.shift(1)
    spread_vol_daily = spread_lagged.rolling(window=window, min_periods=window).std()
    
    # Annualize
    spread_vol_ann = spread_vol_daily * np.sqrt(252)
    
    # Apply floor
    spread_vol_ann = spread_vol_ann.clip(lower=floor)
    
    return spread_vol_ann


def compute_active_weight(te_target: pd.Series, spread_vol: pd.Series, 
                         max_theta: float = 0.25) -> pd.Series:
    """
    Compute active weight (theta) from TE target and spread volatility.
    
    Parameters
    ----------
    te_target : pd.Series
        Target tracking error (annualized)
    spread_vol : pd.Series
        Spread volatility (annualized)
    max_theta : float
        Maximum active weight
        
    Returns
    -------
    pd.Series
        Active weight theta
    """
    theta = te_target / spread_vol
    theta = theta.clip(lower=0, upper=max_theta)
    return theta


def compute_portfolio_weights(theta: pd.Series, base_equity_weight: float = 0.70) -> Tuple[pd.Series, pd.Series]:
    """
    Compute portfolio weights from active weight.
    
    Parameters
    ----------
    theta : pd.Series
        Active weight
    base_equity_weight : float
        Base equity weight (0.70 for 70/30 benchmark)
        
    Returns
    -------
    tuple
        (equity_weight, bond_weight) series
    """
    equity_weight = base_equity_weight + theta
    bond_weight = (1 - base_equity_weight) - theta
    
    return equity_weight, bond_weight


def compute_portfolio_returns(equity_weight: pd.Series, bond_weight: pd.Series,
                             equity_returns: pd.Series, bond_returns: pd.Series) -> pd.Series:
    """
    Compute portfolio returns from weights and asset returns.
    
    Parameters
    ----------
    equity_weight : pd.Series
        Equity weight (t-1)
    bond_weight : pd.Series
        Bond weight (t-1)
    equity_returns : pd.Series
        Equity returns (t)
    bond_returns : pd.Series
        Bond returns (t)
        
    Returns
    -------
    pd.Series
        Portfolio returns
    """
    # Use lagged weights
    equity_weight_lagged = equity_weight.shift(1)
    bond_weight_lagged = bond_weight.shift(1)
    
    portfolio_returns = (equity_weight_lagged * equity_returns + 
                        bond_weight_lagged * bond_returns)
    
    return portfolio_returns


def compute_realized_te(active_returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Compute rolling realized tracking error.
    
    Parameters
    ----------
    active_returns : pd.Series
        Active returns (portfolio - benchmark)
    window : int
        Rolling window
        
    Returns
    -------
    pd.Series
        Annualized realized TE
    """
    te_daily = active_returns.rolling(window=window, min_periods=window).std()
    te_ann = te_daily * np.sqrt(252)
    return te_ann


def create_monthly_rebalance_weights(returns: pd.Series, 
                                     base_equity_weight: float = 0.70) -> Tuple[pd.Series, pd.Series]:
    """
    Create monthly-rebalanced benchmark weights.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns (with DatetimeIndex)
    base_equity_weight : float
        Target equity weight
        
    Returns
    -------
    tuple
        (equity_weight, bond_weight) time series
    """
    equity_weight = pd.Series(base_equity_weight, index=returns.index)
    bond_weight = pd.Series(1 - base_equity_weight, index=returns.index)
    
    # Identify month-end dates
    month_ends = returns.resample('M').last().index
    
    # For each period, reset to target weight at month start
    current_eq_weight = base_equity_weight
    current_bond_weight = 1 - base_equity_weight
    
    for i, date in enumerate(returns.index):
        if i == 0:
            equity_weight.iloc[i] = base_equity_weight
            bond_weight.iloc[i] = 1 - base_equity_weight
        else:
            # Check if we just passed a month-end (rebalance at close of month-end)
            prev_month = returns.index[i-1].to_period('M')
            curr_month = date.to_period('M')
            
            if curr_month > prev_month:
                # Start of new month: reset to target
                current_eq_weight = base_equity_weight
                current_bond_weight = 1 - base_equity_weight
            
            equity_weight.iloc[i] = current_eq_weight
            bond_weight.iloc[i] = current_bond_weight
    
    return equity_weight, bond_weight


def performance_summary(returns: pd.Series, name: str = "Portfolio") -> dict:
    """
    Generate comprehensive performance summary.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    name : str
        Portfolio name
        
    Returns
    -------
    dict
        Performance metrics
    """
    cagr = calculate_cagr(returns)
    vol = calculate_volatility(returns)
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(returns)
    cagr_maxdd = cagr / max_dd if max_dd > 0 else 0
    
    return {
        'name': name,
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'cagr_maxdd': cagr_maxdd,
        'total_return': (1 + returns).prod() - 1,
        'n_periods': len(returns)
    }


if __name__ == "__main__":
    # Test functions
    np.random.seed(42)
    test_returns = pd.Series(np.random.randn(252 * 5) * 0.01 + 0.0003, 
                            index=pd.date_range('2020-01-01', periods=252*5, freq='B'))
    
    print("Performance Summary:")
    print(performance_summary(test_returns, "Test Portfolio"))
    
    print("\nBootstrap Sharpe CI:")
    boot_sharpes = block_bootstrap(test_returns, n_iterations=1000)
    print(f"Mean: {boot_sharpes.mean():.3f}")
    print(f"95% CI: [{np.percentile(boot_sharpes, 2.5):.3f}, {np.percentile(boot_sharpes, 97.5):.3f}]")
