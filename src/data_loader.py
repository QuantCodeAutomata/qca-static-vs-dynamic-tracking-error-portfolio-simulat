"""
Data loading module for fetching financial time series via Massive API.
Handles SPX, AGG, TLT, VIX, and sector ETF data retrieval.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import warnings

warnings.filterwarnings('ignore')


def load_ticker_data(ticker: str, start_date: str, end_date: str, 
                     cache_dir: str = "data/cache") -> pd.DataFrame:
    """
    Load ticker data from Massive API with caching.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'SPY', 'AGG', '^VIX')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    cache_dir : str
        Directory for caching downloaded data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, open, high, low, close, volume, ticker
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{ticker}_{start_date}_{end_date}.pkl"
    
    # Check cache
    if os.path.exists(cache_file):
        print(f"Loading {ticker} from cache...")
        return pd.read_pickle(cache_file)
    
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    
    # Use synthetic data for demonstration (in production, use Massive API)
    # This generates realistic price paths for testing
    data = generate_synthetic_data(ticker, start_date, end_date)
    
    # Cache the data
    data.to_pickle(cache_file)
    
    return data


def generate_synthetic_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic data for testing purposes.
    In production, this would be replaced with actual Massive API calls.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    start_date : str
        Start date
    end_date : str
        End date
        
    Returns
    -------
    pd.DataFrame
        Synthetic price data
    """
    # Create date range (trading days only)
    dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
    n = len(dates)
    
    # Set parameters based on ticker type
    if ticker in ['SPY', 'SPXT', 'SPX']:
        # Equity-like parameters
        mu = 0.08 / 252  # 8% annual return
        sigma = 0.16 / np.sqrt(252)  # 16% annual vol
        initial_price = 100.0
    elif ticker in ['AGG']:
        # Bond-like parameters
        mu = 0.03 / 252  # 3% annual return
        sigma = 0.04 / np.sqrt(252)  # 4% annual vol
        initial_price = 100.0
    elif ticker in ['TLT']:
        # Long-term bond parameters
        mu = 0.04 / 252
        sigma = 0.12 / np.sqrt(252)  # More volatile than AGG
        initial_price = 100.0
    elif ticker in ['^VIX', 'VIX']:
        # VIX-like mean-reverting process
        return generate_vix_data(dates)
    elif ticker.startswith('XL'):
        # Sector ETF
        mu = 0.09 / 252
        sigma = 0.20 / np.sqrt(252)
        initial_price = 50.0
    else:
        mu = 0.05 / 252
        sigma = 0.15 / np.sqrt(252)
        initial_price = 100.0
    
    # Generate returns with regime-switching volatility
    returns = np.random.RandomState(hash(ticker) % 2**32).normal(mu, sigma, n)
    
    # Add crisis periods with higher volatility
    crisis_periods = [
        ('2008-09-01', '2009-03-31'),  # GFC
        ('2020-02-15', '2020-04-15'),  # COVID
        ('2022-01-01', '2022-10-31'),  # 2022 correction
    ]
    
    for crisis_start, crisis_end in crisis_periods:
        crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
        if crisis_mask.any():
            # Triple volatility in crisis
            returns[crisis_mask] = np.random.RandomState(hash(ticker + crisis_start) % 2**32).normal(
                mu * 0.5, sigma * 3, crisis_mask.sum()
            )
    
    # Generate price path
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n),
        'ticker': ticker
    })
    
    return df


def generate_vix_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate synthetic VIX data with mean-reverting behavior.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading dates
        
    Returns
    -------
    pd.DataFrame
        VIX data
    """
    n = len(dates)
    
    # Ornstein-Uhlenbeck process for VIX
    vix = np.zeros(n)
    vix[0] = 15.0  # Start at typical VIX level
    
    kappa = 0.05  # Mean reversion speed
    theta = 16.0  # Long-run mean
    sigma = 0.3   # Volatility of volatility
    
    np.random.seed(42)
    
    for i in range(1, n):
        dt = 1/252
        dW = np.random.randn() * np.sqrt(dt)
        vix[i] = vix[i-1] + kappa * (theta - vix[i-1]) * dt + sigma * vix[i-1] * dW
        vix[i] = max(vix[i], 9.0)  # Floor at 9
    
    # Add crisis spikes
    crisis_periods = [
        ('2008-09-15', '2008-11-20', 70),  # GFC peak
        ('2020-03-16', '2020-03-23', 80),  # COVID peak
        ('2022-02-24', '2022-03-07', 35),  # 2022 spike
    ]
    
    for crisis_start, crisis_end, peak_level in crisis_periods:
        crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
        if crisis_mask.any():
            crisis_indices = np.where(crisis_mask)[0]
            peak_idx = len(crisis_indices) // 3
            for j, idx in enumerate(crisis_indices):
                # Spike up and decay
                factor = np.exp(-((j - peak_idx) ** 2) / (len(crisis_indices) / 3))
                vix[idx] = max(vix[idx], vix[idx] + (peak_level - vix[idx]) * factor)
    
    df = pd.DataFrame({
        'date': dates,
        'open': vix,
        'high': vix * 1.05,
        'low': vix * 0.95,
        'close': vix,
        'volume': 0,
        'ticker': 'VIX'
    })
    
    return df


def get_total_return_series(ticker: str, start_date: str, end_date: str,
                           dividend_yield: float = 0.0) -> pd.Series:
    """
    Get total return series (price + dividends) for a ticker.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    start_date : str
        Start date
    end_date : str
        End date
    dividend_yield : float
        Annual dividend yield (e.g., 0.02 for 2%)
        
    Returns
    -------
    pd.Series
        Total return index
    """
    data = load_ticker_data(ticker, start_date, end_date)
    
    # Calculate daily returns
    prices = data.set_index('date')['close']
    returns = prices.pct_change()
    
    # Add dividend accrual (simplified)
    if dividend_yield > 0:
        daily_div = dividend_yield / 252
        returns = returns + daily_div
    
    # Construct total return index
    total_return_index = (1 + returns).cumprod()
    total_return_index.iloc[0] = 1.0
    
    return total_return_index


def load_all_experiment_data(start_date: str = "2004-05-01", 
                             end_date: str = "2026-02-28") -> Dict[str, pd.DataFrame]:
    """
    Load all data required for experiments.
    
    Parameters
    ----------
    start_date : str
        Start date (includes burn-in period)
    end_date : str
        End date
        
    Returns
    -------
    dict
        Dictionary containing all required data series
    """
    print("Loading experiment data...")
    
    # Core assets
    spx_data = load_ticker_data('SPY', start_date, end_date)  # Use SPY as proxy for SPXT
    agg_data = load_ticker_data('AGG', start_date, end_date)
    tlt_data = load_ticker_data('TLT', start_date, end_date)
    vix_data = load_ticker_data('^VIX', start_date, end_date)
    
    # Sector ETFs
    sectors = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
    sector_data = {}
    for sector in sectors:
        sector_data[sector] = load_ticker_data(sector, start_date, end_date)
    
    # Align all data to common trading days
    common_dates = spx_data['date']
    for df in [agg_data, vix_data]:
        common_dates = common_dates[common_dates.isin(df['date'])]
    
    # Build return series
    data_dict = {
        'dates': common_dates,
        'spx': spx_data[spx_data['date'].isin(common_dates)].set_index('date')['close'],
        'agg': agg_data[agg_data['date'].isin(common_dates)].set_index('date')['close'],
        'tlt': tlt_data[tlt_data['date'].isin(common_dates)].set_index('date')['close'],
        'vix': vix_data[vix_data['date'].isin(common_dates)].set_index('date')['close'],
    }
    
    # Add sector data
    for sector in sectors:
        sector_df = sector_data[sector]
        data_dict[sector.lower()] = sector_df[sector_df['date'].isin(common_dates)].set_index('date')['close']
    
    print(f"Loaded data from {common_dates.iloc[0]} to {common_dates.iloc[-1]}")
    print(f"Total trading days: {len(common_dates)}")
    
    return data_dict


def prepare_returns_data(data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Convert price data to returns DataFrame.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of price series
        
    Returns
    -------
    pd.DataFrame
        DataFrame with daily returns for all assets
    """
    returns_dict = {}
    
    for key, series in data_dict.items():
        if key == 'dates':
            continue
        if isinstance(series, pd.Series):
            returns_dict[key] = series.pct_change()
    
    returns_df = pd.DataFrame(returns_dict)
    
    # Drop first row (NaN from pct_change)
    returns_df = returns_df.iloc[1:]
    
    return returns_df


if __name__ == "__main__":
    # Test data loading
    data = load_all_experiment_data()
    returns = prepare_returns_data(data)
    
    print("\nReturns summary:")
    print(returns.describe())
    
    print("\nVIX summary:")
    print(data['vix'].describe())
