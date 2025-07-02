"""Simple vectorised intraday back‑tester."""
import pandas as pd
from typing import Tuple

def backtest(df: pd.DataFrame, signal: pd.Series, cost_bp: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    returns = df['close'].pct_change().fillna(0.0)
    strat = returns * signal.shift(1).fillna(0.0)
    if cost_bp:
        trades = signal.diff().abs().fillna(0.0)
        strat -= trades * cost_bp * 1e-4
    equity = (1 + strat).cumprod()
    return equity, strat