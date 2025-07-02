import pandas as pd
from src.backtest import backtest

def test_backtest_len():
    df = pd.DataFrame({'close':[1,1.01,1.02]})
    sig = pd.Series([0,1,0])
    eq,_ = backtest(df, sig)
    assert len(eq)==3