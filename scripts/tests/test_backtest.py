import sys
from pathlib import Path

import pandas as pd

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import backtest


def test_backtest_len():
    df = pd.DataFrame({"close": [1, 1.01, 1.02]})
    sig = pd.Series([0, 1, 0])
    eq, _ = backtest(df, sig)
    assert len(eq) == 3
