"""Signal engineering utilities."""
import pandas as pd

def queue_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """Compute (bid - ask) / (bid + ask)."""
    denom = bid_size + ask_size
    return (bid_size - ask_size) / denom.replace(0, pd.NA)