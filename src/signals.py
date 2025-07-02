"""Signal engineering utilities for orderbook-style alpha generation."""
import pandas as pd
import numpy as np
from typing import Optional

def queue_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """Compute (bid - ask) / (bid + ask)."""
    denom = bid_size + ask_size
    return (bid_size - ask_size) / denom.replace(0, pd.NA)

def vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute VWAP deviation signal: (close - VWAP) / VWAP
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
        window: Rolling window for VWAP calculation
    
    Returns:
        Series with VWAP deviation signal
    """
    # Typical price weighted by volume
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    
    return (df['close'] - vwap) / vwap

def price_momentum(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Compute price momentum over specified window.
    
    Args:
        df: DataFrame with 'close' column
        window: Lookback window for momentum calculation
    
    Returns:
        Series with momentum signal
    """
    return df['close'].pct_change(window)

def volume_surge(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
    """
    Detect volume surges relative to rolling average.
    
    Args:
        df: DataFrame with 'volume' column
        window: Window for volume average
        threshold: Multiple of average volume to trigger signal
    
    Returns:
        Binary series indicating volume surges
    """
    avg_volume = df['volume'].rolling(window=window).mean()
    return (df['volume'] / avg_volume > threshold).astype(int)

def volatility_signal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute realized volatility signal using high-low range.
    
    Args:
        df: DataFrame with 'high', 'low' columns
        window: Window for volatility calculation
    
    Returns:
        Series with normalized volatility signal
    """
    hl_ratio = (df['high'] - df['low']) / df['low']
    vol = hl_ratio.rolling(window=window).std()
    return (vol - vol.rolling(window=window*2).mean()) / vol.rolling(window=window*2).std()

def mean_reversion_signal(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Simple mean reversion signal based on price deviation from moving average.
    
    Args:
        df: DataFrame with 'close' column  
        window: Window for moving average
    
    Returns:
        Series with mean reversion signal (negative when price above MA)
    """
    ma = df['close'].rolling(window=window).mean()
    return -(df['close'] - ma) / ma  # Negative for mean reversion

def rsi_signal(df: pd.DataFrame, window: int = 14, overbought: float = 70, oversold: float = 30) -> pd.Series:
    """
    RSI-based signal for mean reversion trading.
    
    Args:
        df: DataFrame with 'close' column
        window: RSI calculation window
        overbought: RSI level considered overbought
        oversold: RSI level considered oversold
    
    Returns:
        Series with RSI signal (-1 for overbought, +1 for oversold, 0 otherwise)
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    signal = pd.Series(0, index=df.index)
    signal[rsi > overbought] = -1  # Sell signal when overbought
    signal[rsi < oversold] = 1     # Buy signal when oversold
    
    return signal

def combine_signals(signals: dict, weights: Optional[dict] = None) -> pd.Series:
    """
    Combine multiple signals with optional weights.
    
    Args:
        signals: Dictionary of signal name -> signal series
        weights: Optional dictionary of signal name -> weight
    
    Returns:
        Combined signal series
    """
    if weights is None:
        weights = {name: 1.0 for name in signals.keys()}
    
    combined = pd.Series(0.0, index=list(signals.values())[0].index)
    total_weight = sum(weights.values())
    
    for name, signal in signals.items():
        weight = weights.get(name, 0.0)
        combined += (weight / total_weight) * signal.fillna(0)
    
    return combined