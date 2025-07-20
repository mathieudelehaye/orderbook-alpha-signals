"""Simple vectorised intraday back‑tester with performance metrics."""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def backtest_intraday(df: pd.DataFrame, signal: pd.Series, cost_bp: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """
    Vectorized backtest for intraday trading signals.
    
    Args:
        df: DataFrame with OHLCV data, must contain 'close' column
        signal: Trading signal series (1 = long, -1 = short, 0 = neutral)
        cost_bp: Trading cost in basis points (e.g., 1.0 = 1bp = 0.01%)
    
    Returns:
        Tuple of (equity_curve, strategy_returns)
    """
    # Calculate returns
    returns = df['close'].pct_change().fillna(0.0)
    
    # Apply signal with 1-period lag (realistic trading assumption)
    position = signal.shift(1).fillna(0.0)
    
    # Strategy returns before costs
    strategy_returns = returns * position
    
    # Apply trading costs
    if cost_bp > 0:
        # Calculate position changes (trades)
        trades = position.diff().abs().fillna(0.0)
        transaction_costs = trades * cost_bp * 1e-4  # Convert bp to decimal
        strategy_returns -= transaction_costs
    
    # Calculate equity curve
    equity_curve = (1 + strategy_returns).cumprod()
    
    return equity_curve, strategy_returns

def calculate_metrics(equity_curve: pd.Series, strategy_returns: pd.Series, 
                     benchmark_returns: pd.Series = None) -> Dict[str, Any]:
    """
    Calculate comprehensive trading performance metrics.
    
    Args:
        equity_curve: Cumulative equity curve
        strategy_returns: Strategy returns series
        benchmark_returns: Optional benchmark returns for comparison
    
    Returns:
        Dictionary of performance metrics
    """
    # Basic returns metrics
    total_return = equity_curve.iloc[-1] - 1.0
    n_periods = len(strategy_returns)
    
    # Annualized metrics (assuming minute data, ~252*390 minutes per year)
    periods_per_year = 252 * 390  # Trading minutes per year
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    # Volatility
    returns_std = strategy_returns.std()
    annualized_vol = returns_std * np.sqrt(periods_per_year)
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Maximum drawdown
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calmar ratio
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0
    
    # Win rate and profit factor
    positive_returns = strategy_returns[strategy_returns > 0]
    negative_returns = strategy_returns[strategy_returns < 0]
    
    win_rate = len(positive_returns) / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(strategy_returns[strategy_returns != 0]),
        'avg_return_per_trade': strategy_returns[strategy_returns != 0].mean() if len(strategy_returns[strategy_returns != 0]) > 0 else 0
    }
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        benchmark_total = (1 + benchmark_returns).prod() - 1
        benchmark_vol = benchmark_returns.std() * np.sqrt(periods_per_year)
        benchmark_sharpe = (benchmark_total * periods_per_year / n_periods) / benchmark_vol if benchmark_vol > 0 else 0
        
        metrics['benchmark_return'] = benchmark_total
        metrics['excess_return'] = total_return - benchmark_total
        metrics['information_ratio'] = (strategy_returns - benchmark_returns).mean() / (strategy_returns - benchmark_returns).std() * np.sqrt(periods_per_year)
    
    return metrics

def backtest_limited_trades(df: pd.DataFrame, signal: pd.Series, cost_bp: float = 0.0, 
                           max_trades_per_day: int = 10, signal_threshold: float = 0.3) -> Tuple[pd.Series, pd.Series]:
    """
    Enhanced backtest with trade frequency limits and signal filtering.
    
    Args:
        df: DataFrame with OHLCV data, must contain 'close' column
        signal: Trading signal series (1 = long, -1 = short, 0 = neutral)
        cost_bp: Trading cost in basis points (e.g., 1.0 = 1bp = 0.01%)
        max_trades_per_day: Maximum number of trades allowed per day
        signal_threshold: Only trade when |signal| > threshold (filters weak signals)
    
    Returns:
        Tuple of (equity_curve, strategy_returns)
    """
    # Calculate returns
    returns = df['close'].pct_change().fillna(0.0)
    
    # Filter signals by threshold - only trade on strong signals
    filtered_signal = signal.copy()
    filtered_signal[abs(signal) < signal_threshold] = 0.0
    
    # Create position series with trade limits
    position = pd.Series(0.0, index=filtered_signal.index)
    current_position = 0.0
    
    # Track trades per day
    daily_trades = {}
    
    for i, (timestamp, sig) in enumerate(filtered_signal.items()):
        if i == 0:
            continue
            
        # Get the date for trade limiting
        current_date = timestamp.date()
        
        # Initialize daily trade counter
        if current_date not in daily_trades:
            daily_trades[current_date] = 0
        
        # Determine desired position
        desired_position = sig
        
        # Check if we need to change position
        if desired_position != current_position:
            # Check if we can make another trade today
            if daily_trades[current_date] < max_trades_per_day:
                current_position = desired_position
                daily_trades[current_date] += 1
        
        position.iloc[i] = current_position
    
    # Apply 1-period lag (realistic trading assumption)
    position = position.shift(1).fillna(0.0)
    
    # Strategy returns before costs
    strategy_returns = returns * position
    
    # Apply trading costs
    if cost_bp > 0:
        # Calculate position changes (trades)
        trades = position.diff().abs().fillna(0.0)
        transaction_costs = trades * cost_bp * 1e-4  # Convert bp to decimal
        strategy_returns -= transaction_costs
    
    # Calculate equity curve
    equity_curve = (1 + strategy_returns).cumprod()
    
    return equity_curve, strategy_returns

def backtest(df: pd.DataFrame, signal: pd.Series, cost_bp: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """
    Legacy function name for backwards compatibility.
    """
    return backtest_intraday(df, signal, cost_bp)