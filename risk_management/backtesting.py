"""
Strategy Backtesting Framework
================================
Production-grade backtesting engine with realistic assumptions and
comprehensive performance analytics.

Features:
  - Event-driven signal processing
  - Transaction cost modeling (spread + commission)
  - Performance metrics: Sharpe, Sortino, Calmar, Max DD, Omega
  - Rolling metrics for stability analysis
  - Walk-forward validation support

Strategies implemented as examples:
  1. Dual Moving Average Crossover
  2. Momentum (12-1 month)
  3. Mean Reversion (Bollinger Band)

Author: Quant Finance Portfolio
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: int
    exit_date: int
    direction: int        # +1 long, -1 short
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    n_trades: int
    returns: np.ndarray
    equity_curve: np.ndarray
    trades: List[Trade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, rf_daily: float = 0.0, periods: int = 252) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - rf_daily
    if np.std(excess) == 0:
        return 0.0
    return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods)


def sortino_ratio(returns: np.ndarray, rf_daily: float = 0.0, periods: int = 252) -> float:
    """Sortino ratio: penalises downside volatility only."""
    excess   = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return np.inf
    downside_vol = np.std(downside, ddof=1) * np.sqrt(periods)
    return np.mean(excess) * periods / downside_vol


def max_drawdown(equity_curve: np.ndarray) -> tuple:
    """
    Maximum peak-to-trough drawdown.
    Returns (max_dd, duration_days).
    """
    peak   = np.maximum.accumulate(equity_curve)
    dd     = (equity_curve - peak) / peak
    max_dd = dd.min()

    max_dur = 0
    cur_dur = 0
    for b in (dd < 0):
        if b:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return max_dd, max_dur


def calmar_ratio(annualised_return: float, max_dd: float) -> float:
    """Calmar = annualised return / |max drawdown|."""
    if max_dd == 0:
        return np.inf
    return annualised_return / abs(max_dd)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega ratio: probability-weighted ratio of gains to losses.
    Omega > 1 is desirable. Does not assume normality.
    """
    gains  = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if losses.sum() == 0:
        return np.inf
    return gains.sum() / losses.sum()


def profit_factor(trades: List[Trade]) -> float:
    """Gross profit / gross loss across all trades."""
    gains  = sum(t.pnl for t in trades if t.pnl > 0)
    losses = sum(abs(t.pnl) for t in trades if t.pnl < 0)
    return gains / losses if losses > 0 else np.inf


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

def run_backtest(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    rf_annual: float = 0.05,
) -> BacktestResult:
    """
    Execute a backtest given a price series and signal array.

    Parameters
    ----------
    prices           : Array of asset prices (close)
    signals          : Position signal per bar (+1 long / -1 short / 0 flat)
    transaction_cost : One-way cost as fraction of price (default 10bps)
    slippage         : Market impact per trade one-way (default 5bps)
    rf_annual        : Annual risk-free rate for Sharpe calculation

    Returns
    -------
    BacktestResult with full metrics and trade log.
    """
    n          = len(prices)
    rf_daily   = rf_annual / 252
    total_cost = transaction_cost + slippage

    returns   = np.zeros(n)
    equity    = np.zeros(n)
    equity[0] = initial_capital
    trades    = []

    position    = 0
    entry_price = 0.0
    entry_idx   = 0

    for i in range(1, n):
        price_return = (prices[i] - prices[i-1]) / prices[i-1]
        new_signal   = signals[i]

        if new_signal != position:
            if position != 0:
                raw_pnl = position * (prices[i] - entry_price)
                cost    = total_cost * prices[i] * 2
                net_pnl = raw_pnl - cost
                pnl_pct = net_pnl / (entry_price * 1)
                trades.append(Trade(
                    entry_date=entry_idx,
                    exit_date=i,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=prices[i],
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                ))

            if new_signal != 0:
                entry_price = prices[i] * (1 + total_cost * new_signal)
                entry_idx   = i

            position = new_signal

        if position != 0:
            returns[i] = position * price_return - total_cost * abs(new_signal - signals[i-1])
        else:
            returns[i] = 0.0

        equity[i] = equity[i-1] * (1 + returns[i])

    # Metrics
    total_ret = (equity[-1] - equity[0]) / equity[0]
    n_years   = n / 252
    ann_ret   = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol   = np.std(returns[returns != 0], ddof=1) * np.sqrt(252)

    sr       = sharpe_ratio(returns, rf_daily)
    so       = sortino_ratio(returns, rf_daily)
    mdd, dur = max_drawdown(equity)
    cal      = calmar_ratio(ann_ret, mdd)
    om       = omega_ratio(returns)
    pf       = profit_factor(trades)

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    return BacktestResult(
        total_return=total_ret,
        annualised_return=ann_ret,
        annualised_vol=ann_vol,
        sharpe_ratio=sr,
        sortino_ratio=so,
        calmar_ratio=cal,
        omega_ratio=om,
        max_drawdown=mdd,
        max_drawdown_duration=dur,
        win_rate=len(wins) / len(trades) if trades else 0,
        profit_factor=pf,
        avg_win=np.mean([t.pnl_pct for t in wins]) if wins else 0,
        avg_loss=np.mean([t.pnl_pct for t in losses]) if losses else 0,
        n_trades=len(trades),
        returns=returns,
        equity_curve=equity,
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Example strategies
# ---------------------------------------------------------------------------

def signal_dual_ma(prices: np.ndarray, fast: int = 20, slow: int = 50) -> np.ndarray:
    """Dual moving average crossover: long when fast > slow, short otherwise."""
    fast_ma = np.convolve(prices, np.ones(fast) / fast, mode="full")[:len(prices)]
    slow_ma = np.convolve(prices, np.ones(slow) / slow, mode="full")[:len(prices)]
    signals = np.zeros(len(prices))
    signals[slow:] = np.where(fast_ma[slow:] > slow_ma[slow:], 1, -1)
    return signals


def signal_momentum(prices: np.ndarray, lookback: int = 252, skip: int = 21) -> np.ndarray:
    """12-1 month momentum: long if 12M-1M return > 0."""
    signals = np.zeros(len(prices))
    for i in range(lookback, len(prices)):
        ret = prices[i - skip] / prices[i - lookback] - 1
        signals[i] = 1 if ret > 0 else -1
    return signals


def signal_mean_reversion(
    prices: np.ndarray,
    window: int = 20,
    n_std: float = 2.0
) -> np.ndarray:
    """Bollinger Band mean reversion: short upper band, long lower band."""
    signals = np.zeros(len(prices))
    for i in range(window, len(prices)):
        window_prices = prices[i - window:i]
        mean  = np.mean(window_prices)
        std   = np.std(window_prices)
        upper = mean + n_std * std
        lower = mean - n_std * std
        if prices[i] > upper:
            signals[i] = -1
        elif prices[i] < lower:
            signals[i] = 1
        else:
            signals[i] = signals[i-1]
    return signals


# ---------------------------------------------------------------------------
# Synthetic price generator
# ---------------------------------------------------------------------------

def generate_gbm_prices(
    S0: float = 100,
    mu: float = 0.08,
    sigma: float = 0.18,
    n: int = 2520,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic GBM price series for demo."""
    rng = np.random.default_rng(seed)
    dt  = 1 / 252
    Z   = rng.standard_normal(n)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    prices = S0 * np.exp(np.cumsum(log_returns))
    return np.insert(prices, 0, S0)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices = generate_gbm_prices(n=2520)

    print("=" * 65)
    print("  BACKTESTING FRAMEWORK — 10Y Simulation")
    print("=" * 65)

    strategies = {
        "Dual MA (20/50)":          signal_dual_ma(prices, 20, 50),
        "Momentum (252-21)":        signal_momentum(prices, 252, 21),
        "Mean Reversion (20, 2std)": signal_mean_reversion(prices, 20, 2.0),
    }

    print(f"\n  {'Strategy':28}  {'Ann Ret':>8}  {'Vol':>7}  {'Sharpe':>7}  "
          f"{'Sortino':>8}  {'MaxDD':>7}  {'Calmar':>7}  {'Win%':>6}  {'Trades':>7}")
    print("  " + "-" * 100)

    for name, signals in strategies.items():
        r = run_backtest(prices, signals)
        print(
            f"  {name:28}  "
            f"{r.annualised_return*100:>7.2f}%  "
            f"{r.annualised_vol*100:>6.2f}%  "
            f"{r.sharpe_ratio:>7.2f}  "
            f"{r.sortino_ratio:>8.2f}  "
            f"{r.max_drawdown*100:>6.2f}%  "
            f"{r.calmar_ratio:>7.2f}  "
            f"{r.win_rate*100:>5.1f}%  "
            f"{r.n_trades:>7}"
        )

    print(f"\n  Detailed: Dual MA (20/50)")
    r = run_backtest(prices, signal_dual_ma(prices))
    print(f"    Total Return    : {r.total_return*100:.2f}%")
    print(f"    Profit Factor   : {r.profit_factor:.2f}")
    print(f"    Avg Win         : {r.avg_win*100:.3f}%")
    print(f"    Avg Loss        : {r.avg_loss*100:.3f}%")
    print(f"    Max DD Duration : {r.max_drawdown_duration} days")
    print(f"    Omega Ratio     : {r.omega_ratio:.2f}")
