# Alpha Orderbook Signals

Code-only research toolkit for engineering orderbook-style signals and backtesting intraday alpha strategies. This repository contains **no raw market data** — users fetch data with their own API credentials.

## 🚀 Features

- **Fetch 1-minute OHLCV bars** from Polygon.io REST API
- **Engineer orderbook-style signals** (queue imbalance, VWAP deviation, momentum, etc.)
- **Vectorized backtesting** with realistic transaction costs
- **Comprehensive performance metrics** (Sharpe, max drawdown, win rate, etc.)
- **Jupyter notebook walkthrough** for complete workflow demonstration

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd orderbook-alpha-signals
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up API credentials:**
```bash
cp .env.example .env
# Edit .env and add your Polygon.io API key
```

Get your free API key at [polygon.io](https://polygon.io/)

## 📊 Quick Start

### 1. Fetch Market Data
```bash
python fetch_data.py --symbol AAPL --start 2023-01-01 --end 2023-12-31
```

### 2. Run the Example Notebook
Open `notebooks/01_signal_backtest.ipynb` in Jupyter and run all cells to see:
- Data loading and preprocessing
- Signal engineering (VWAP deviation, momentum, etc.)
- Backtesting with transaction costs
- Performance visualization and metrics

### 3. Use the Python API
```python
import pandas as pd
from src.signals import vwap_deviation, mean_reversion_signal
from src.backtest import backtest_intraday, calculate_metrics

# Load your data
df = pd.read_csv('data/AAPL_2023-01-01_2023-12-31_1min.csv')

# Generate signals
vwap_signal = vwap_deviation(df, window=20)
mean_rev_signal = mean_reversion_signal(df, window=10)

# Backtest
equity_curve, returns = backtest_intraday(df, vwap_signal, cost_bp=1.0)

# Calculate metrics
metrics = calculate_metrics(equity_curve, returns)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

## 🔧 Available Signals

| Signal Function | Description |
|----------------|-------------|
| `queue_imbalance()` | Bid-ask imbalance ratio |
| `vwap_deviation()` | Price deviation from VWAP |
| `price_momentum()` | Rolling price momentum |
| `volume_surge()` | Volume spike detection |
| `volatility_signal()` | Realized volatility signal |
| `mean_reversion_signal()` | Mean reversion based on moving average |
| `rsi_signal()` | RSI-based overbought/oversold signals |
| `combine_signals()` | Weighted combination of multiple signals |

## 📈 Backtesting Features

- **Vectorized execution** for fast performance
- **Realistic transaction costs** (configurable basis points)
- **Signal lag modeling** (1-period lag by default)
- **Comprehensive metrics**:
  - Total & annualized returns
  - Sharpe ratio & volatility
  - Maximum drawdown & Calmar ratio
  - Win rate & profit factor
  - Information ratio (vs benchmark)

## 📁 Project Structure

```
orderbook-alpha-signals/
├── fetch_data.py                    # Data fetching CLI tool
├── src/
│   ├── signals.py                   # Signal engineering functions
│   └── backtest.py                  # Backtesting engine
├── notebooks/
│   └── 01_signal_backtest.ipynb     # Example walkthrough
├── tests/
│   └── test_backtest.py             # Unit tests
├── data/                            # CSV files (git-ignored)
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
└── README.md                        # This file
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## ⚖️ License & Data Usage

**Important:** This repository contains **code only**. No market data is included.

- Market data © Polygon.io — check their terms of service for usage rights
- Users are responsible for their own API key and data usage compliance
- Code is provided for educational and research purposes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 References

- [Polygon.io API Documentation](https://polygon.io/docs)
- Academic papers on orderbook imbalance and microstructure signals
- Quantitative trading literature on signal engineering and backtesting

---

**Disclaimer:** This tool is for research and educational purposes. Past performance does not guarantee future results. Always test strategies thoroughly before live trading.
