"""Risk management utilities for trading strategies."""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Comprehensive risk management for trading strategies.

    Features:
    - Position sizing based on volatility
    - Stop-loss and take-profit levels
    - Maximum drawdown protection
    - Portfolio heat limits
    - Dynamic position scaling
    """

    def __init__(
        self,
        max_position_size: float = 1.0,
        max_portfolio_heat: float = 0.02,  # 2% max risk per trade
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,  # 4% take profit (2:1 R/R)
        volatility_window: int = 20,
        max_drawdown_limit: float = 0.10,  # 10% max drawdown
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    ):
        self.max_position_size = max_position_size
        self.max_portfolio_heat = max_portfolio_heat
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.volatility_window = volatility_window
        self.max_drawdown_limit = max_drawdown_limit
        self.risk_free_rate = risk_free_rate

        # Risk state tracking
        self.current_drawdown = 0.0
        self.peak_equity = 1.0
        self.is_risk_off = False

    def calculate_position_size(
        self,
        signal_strength: float,
        current_price: float,
        volatility: float,
        current_equity: float = 1.0,
    ) -> float:
        """
        Calculate position size based on signal strength, volatility, and risk parameters.

        Args:
            signal_strength: Signal strength [-1, 1]
            current_price: Current asset price
            volatility: Current volatility estimate
            current_equity: Current portfolio equity

        Returns:
            Position size as fraction of portfolio
        """
        try:
            # Base position size from signal strength
            base_size = abs(signal_strength) * self.max_position_size

            # Volatility-adjusted sizing (Kelly-like)
            if volatility > 0:
                # Risk-adjusted position size
                risk_per_dollar = self.stop_loss_pct
                expected_return = abs(signal_strength) * self.take_profit_pct

                if risk_per_dollar > 0:
                    kelly_fraction = expected_return / (risk_per_dollar * volatility)
                    # Cap Kelly fraction to prevent over-leverage
                    kelly_fraction = min(kelly_fraction, 1.0)
                    vol_adjusted_size = base_size * kelly_fraction
                else:
                    vol_adjusted_size = base_size
            else:
                vol_adjusted_size = base_size

            # Apply portfolio heat limit
            max_size_by_heat = self.max_portfolio_heat / self.stop_loss_pct
            heat_adjusted_size = min(vol_adjusted_size, max_size_by_heat)

            # Apply drawdown scaling
            if self.current_drawdown > self.max_drawdown_limit * 0.5:
                # Reduce position size when in significant drawdown
                drawdown_scalar = max(
                    0.1, 1.0 - (self.current_drawdown / self.max_drawdown_limit)
                )
                heat_adjusted_size *= drawdown_scalar
                logger.warning(
                    "Reducing position size due to drawdown: %f" % self.current_drawdown
                )

            # Risk-off mode check
            if self.is_risk_off:
                heat_adjusted_size *= 0.5
                logger.info("Risk-off mode: halving position size")

            return max(0.0, min(heat_adjusted_size, self.max_position_size))

        except Exception as e:
            logger.error("Error calculating position size: %s" % e)
            return 0.0

    def apply_stop_loss_take_profit(
        self, df: pd.DataFrame, positions: pd.Series, entry_prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Apply stop-loss and take-profit rules to positions.

        Args:
            df: OHLCV DataFrame
            positions: Position series
            entry_prices: Entry price for each position

        Returns:
            Tuple of (adjusted_positions, exit_signals)
        """
        try:
            adjusted_positions = positions.copy()
            exit_signals = pd.Series(0, index=positions.index)

            for i in range(1, len(positions)):
                current_pos = positions.iloc[i - 1]
                current_price = df["close"].iloc[i]

                if current_pos != 0 and not pd.isna(entry_prices.iloc[i - 1]):
                    entry_price = entry_prices.iloc[i - 1]

                    if current_pos > 0:  # Long position
                        # Stop loss
                        if current_price <= entry_price * (1 - self.stop_loss_pct):
                            adjusted_positions.iloc[i] = 0
                            exit_signals.iloc[i] = -1
                            logger.debug(
                                "Long stop loss triggered at %.2f" % current_price
                            )
                        # Take profit
                        elif current_price >= entry_price * (1 + self.take_profit_pct):
                            adjusted_positions.iloc[i] = 0
                            exit_signals.iloc[i] = 1
                            logger.debug(
                                "Long take profit triggered at %.2f" % current_price
                            )

                    elif current_pos < 0:  # Short position
                        # Stop loss
                        if current_price >= entry_price * (1 + self.stop_loss_pct):
                            adjusted_positions.iloc[i] = 0
                            exit_signals.iloc[i] = 1
                            logger.debug(
                                "Short stop loss triggered at %.2f" % current_price
                            )
                        # Take profit
                        elif current_price <= entry_price * (1 - self.take_profit_pct):
                            adjusted_positions.iloc[i] = 0
                            exit_signals.iloc[i] = -1
                            logger.debug(
                                "Short take profit triggered at %.2f" % current_price
                            )

            return adjusted_positions, exit_signals

        except Exception as e:
            logger.error("Error applying stop-loss/take-profit: %s" % e)
            return positions, pd.Series(0, index=positions.index)

    def update_risk_state(self, current_equity: float) -> None:
        """Update risk management state based on current equity."""
        try:
            # Track peak equity
            self.peak_equity = max(self.peak_equity, current_equity)

            # Calculate current drawdown
            self.current_drawdown = (
                self.peak_equity - current_equity
            ) / self.peak_equity

            # Activate risk-off mode if max drawdown exceeded
            if self.current_drawdown >= self.max_drawdown_limit:
                self.is_risk_off = True
                logger.warning(
                    "Risk-off mode activated. Drawdown: %.2f%%"
                    % (self.current_drawdown * 100)
                )

            # Deactivate risk-off mode when drawdown recovers
            elif (
                self.is_risk_off
                and self.current_drawdown < self.max_drawdown_limit * 0.5
            ):
                self.is_risk_off = False
                logger.info("Risk-off mode deactivated")

        except Exception as e:
            logger.error("Error updating risk state: %s" % e)

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Return series
            confidence_level: VaR confidence level
            method: 'historical' or 'parametric'

        Returns:
            VaR value
        """
        try:
            if len(returns) == 0:
                return 0.0

            if method == "historical":
                return float(returns.quantile(1 - confidence_level))

            if method == "parametric":
                from scipy import stats

                mean_return = returns.mean()
                std_return = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                return float(mean_return + z_score * std_return)

            raise ValueError("Unknown VaR method: %s" % method)

        except Exception as e:
            logger.error("Error calculating VaR: %s" % e)
            return 0.0

    def calculate_expected_shortfall(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            var = self.calculate_var(returns, confidence_level)
            return float(returns[returns <= var].mean())
        except Exception as e:
            logger.error("Error calculating Expected Shortfall: %s" % e)
            return 0.0

    def get_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Get comprehensive risk metrics."""
        try:
            return {
                "var_95": self.calculate_var(returns, 0.95),
                "var_99": self.calculate_var(returns, 0.99),
                "expected_shortfall_95": self.calculate_expected_shortfall(
                    returns, 0.95
                ),
                "expected_shortfall_99": self.calculate_expected_shortfall(
                    returns, 0.99
                ),
                "current_drawdown": self.current_drawdown,
                "max_drawdown_limit": self.max_drawdown_limit,
                "is_risk_off": self.is_risk_off,
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
            }
        except Exception as e:
            logger.error("Error calculating risk metrics: %s" % e)
            return {}


def apply_risk_management(
    df: pd.DataFrame, signals: pd.Series, risk_manager: Optional[RiskManager] = None
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Apply comprehensive risk management to trading signals.

    Args:
        df: OHLCV DataFrame
        signals: Raw trading signals
        risk_manager: RiskManager instance

    Returns:
        Tuple of (risk_adjusted_positions, equity_curve, risk_metrics)
    """
    if risk_manager is None:
        risk_manager = RiskManager()

    try:
        # Calculate volatility
        returns = df["close"].pct_change()
        volatility = returns.rolling(risk_manager.volatility_window).std()

        # Initialize arrays
        positions = pd.Series(0.0, index=signals.index)
        equity_curve = pd.Series(1.0, index=signals.index)
        entry_prices = pd.Series(np.nan, index=signals.index)

        # Apply risk management
        for i in range(1, len(signals)):
            current_signal = signals.iloc[i]
            current_price = df["close"].iloc[i]
            current_vol = (
                volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 0.02
            )
            current_equity = equity_curve.iloc[i - 1]

            # Update risk state
            risk_manager.update_risk_state(current_equity)

            # Calculate position size
            if abs(current_signal) > 0.1:  # Only trade on significant signals
                position_size = risk_manager.calculate_position_size(
                    current_signal, current_price, current_vol, current_equity
                )
                positions.iloc[i] = np.sign(current_signal) * position_size

                # Record entry price for new positions
                if positions.iloc[i - 1] == 0:
                    entry_prices.iloc[i] = current_price
                else:
                    entry_prices.iloc[i] = entry_prices.iloc[i - 1]
            else:
                positions.iloc[i] = positions.iloc[i - 1]
                entry_prices.iloc[i] = entry_prices.iloc[i - 1]

        # Apply stop-loss and take-profit
        positions, exit_signals = risk_manager.apply_stop_loss_take_profit(
            df, positions, entry_prices
        )

        # Calculate equity curve with risk management
        strategy_returns = returns * positions.shift(1)
        equity_curve = (1 + strategy_returns.fillna(0)).cumprod()

        # Get risk metrics
        risk_metrics = risk_manager.get_risk_metrics(strategy_returns.dropna())
        risk_metrics["total_trades"] = (positions.diff() != 0).sum()
        risk_metrics["exit_signals"] = (exit_signals != 0).sum()

        return positions, equity_curve, risk_metrics

    except Exception as e:
        logger.error("Error in risk management application: %s" % e)
        return signals, pd.Series(1.0, index=signals.index), {}
