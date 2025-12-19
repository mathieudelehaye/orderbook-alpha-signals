"""Simple vectorised intraday back‑tester with performance metrics."""

import numpy as np
import pandas as pd
from dataclasses import astuple, dataclass
from typing import Any, Dict, Tuple, Optional


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity_curve: pd.Series
    strategy_returns: pd.Series
    position: pd.Series
    num_trades: int
    min_time_between_trades: Optional[pd.Timedelta]


def filter_trades(sig: pd.Series, max_trades: int, threshold: int) -> Tuple[pd.Series, pd.Timedelta]:
    # Filter signals by threshold - only trade on strong signals
    filtered_signal = sig.copy()
    filtered_signal[abs(sig) < threshold] = 0.0

    # Create position series with trade limits
    position = pd.Series(0.0, index=filtered_signal.index)
    current_position = 0.0

    # Track trades per day
    daily_trades = {}

    # Track minimum time between position changes
    last_change_time = None
    min_time_between_changes = None

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
            if daily_trades[current_date] < max_trades:
                # Track time between changes
                if last_change_time is not None:
                    time_delta = timestamp - last_change_time
                    if min_time_between_changes is None or time_delta < min_time_between_changes:
                        min_time_between_changes = time_delta
                
                last_change_time = timestamp
                current_position = desired_position
                daily_trades[current_date] += 1

        position.iloc[i] = current_position

    return position, min_time_between_changes

def backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bp: float = 0.0,
    max_trades_per_day: int = None,
    signal_threshold: float = None,
) -> BacktestResult:
    """
    Backtest with possibly trade frequency limits and signal filtering.

    Args:
        df: DataFrame with OHLCV data, must contain 'close' column
        signal: Trading signal series (1 = long, -1 = short, 0 = neutral)
        cost_bp: Trading cost in basis points (e.g., 1.0 = 1bp = 0.01%)
        max_trades_per_day: if given, maximum number of trades allowed per day
        signal_threshold: if given, only trade when |signal| > threshold (filters weak signals)

    Returns:
        BacktestResult object containing equity_curve, strategy_returns, position, num_trades, min_time_between_trades
    """
    # Calculate returns
    returns = df["close"].pct_change().fillna(0.0)

    # Possibly limit the number of signal positions and apply signal with 1-period lag (realistic trading assumption):
    if max_trades_per_day and signal_threshold:
        position, min_time_between_changes = filter_trades(signal, max_trades_per_day, signal_threshold)
        position = position.shift(1).fillna(0.0)
    else: 
        position = signal.shift(1).fillna(0.0)
        min_time_between_changes = None

    # Strategy returns before costs
    strategy_returns = returns * position

    # Calculate number of trades (position changes)
    position_changes = position.diff().abs()
    num_trades = (position_changes > 0).sum()

    # Apply trading costs
    if cost_bp > 0:
        # Calculate position changes (trades)
        trades = position.diff().abs().fillna(0.0)
        transaction_costs = trades * cost_bp * 1e-4  # Convert bp to decimal
        strategy_returns -= transaction_costs

    # Calculate equity curve
    equity_curve = (1 + strategy_returns).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        position=position,
        num_trades=num_trades,
        min_time_between_trades=min_time_between_changes
    )


def calculate_metrics(
    results : BacktestResult,
    benchmark_returns: pd.Series = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive trading performance metrics.

    Args:
        results: Results from backtesting
        benchmark_returns: Optional benchmark returns for comparison

    Returns:
        Dictionary of performance metrics
    """
    equity_curve, strategy_returns, _, num_trades, _ = astuple(results)

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

    win_rate = (
        len(positive_returns) / len(strategy_returns[strategy_returns != 0])
        if len(strategy_returns[strategy_returns != 0]) > 0
        else 0
    )
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": num_trades,
        "avg_return_per_trade": (
            strategy_returns[strategy_returns != 0].mean()
            if len(strategy_returns[strategy_returns != 0]) > 0
            else 0
        ),
    }

    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        benchmark_total = (1 + benchmark_returns).prod() - 1

        metrics["benchmark_return"] = benchmark_total
        metrics["excess_return"] = total_return - benchmark_total
        metrics["information_ratio"] = (
            (strategy_returns - benchmark_returns).mean()
            / (strategy_returns - benchmark_returns).std()
            * np.sqrt(periods_per_year)
        )

    return metrics