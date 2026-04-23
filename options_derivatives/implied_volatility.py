"""
Implied Volatility Solver
==========================
Recovers the market-implied volatility from observed option prices.

Two solvers implemented:
  1. Newton-Raphson  — fast quadratic convergence near the solution
  2. Brent's method  — robust bracketing fallback, guaranteed convergence

Industry context:
  IV is the central quantity in options markets. Traders quote in vol,
  not price. Building a vol surface from market quotes is the foundation
  of any derivatives desk workflow.

Author: Quant Finance Portfolio
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Literal, Optional
from options_derivatives.black_scholes import bs_price


# ---------------------------------------------------------------------------
# Intrinsic value & boundary checks
# ---------------------------------------------------------------------------

def intrinsic_value(S, K, T, r, option_type="call") -> float:
    """Lower bound (intrinsic value) of an option."""
    disc = np.exp(-r * T)
    if option_type == "call":
        return max(S - K * disc, 0.0)
    return max(K * disc - S, 0.0)


def validate_price(market_price, S, K, T, r, option_type) -> None:
    """Raise if market_price violates no-arbitrage bounds."""
    lb = intrinsic_value(S, K, T, r, option_type)
    ub = S if option_type == "call" else K * np.exp(-r * T)
    if market_price < lb - 1e-6:
        raise ValueError(f"Price {market_price:.4f} below intrinsic value {lb:.4f}.")
    if market_price > ub + 1e-6:
        raise ValueError(f"Price {market_price:.4f} above theoretical upper bound {ub:.4f}.")


# ---------------------------------------------------------------------------
# Newton-Raphson solver
# ---------------------------------------------------------------------------

def iv_newton_raphson(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    sigma0: float = 0.30,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Newton-Raphson implied volatility solver.

    Iteration: sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

    Parameters
    ----------
    market_price : Observed option price to invert
    sigma0       : Initial volatility guess (default 30%)
    tol          : Convergence tolerance on price error
    max_iter     : Maximum iterations before returning None

    Returns
    -------
    Implied volatility (float) or None if no convergence.
    """
    validate_price(market_price, S, K, T, r, option_type)
    sigma = sigma0

    for i in range(max_iter):
        res   = bs_price(S, K, T, r, sigma, q=q, option_type=option_type)
        price = res.price
        vega  = res.vega * 100

        error = price - market_price

        if abs(error) < tol:
            return sigma

        if abs(vega) < 1e-12:
            return None

        sigma -= error / vega
        sigma  = max(sigma, 1e-6)

    return None


# ---------------------------------------------------------------------------
# Brent's method solver
# ---------------------------------------------------------------------------

def iv_brent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    sigma_low: float = 1e-4,
    sigma_high: float = 10.0,
    tol: float = 1e-8,
) -> Optional[float]:
    """
    Brent's method implied volatility solver.

    Guaranteed to converge within [sigma_low, sigma_high] if a root exists.
    Slower than Newton-Raphson but robust — used as fallback or primary for
    deep ITM/OTM options where vega is tiny.

    Returns
    -------
    Implied volatility (float) or None if no root in bracket.
    """
    validate_price(market_price, S, K, T, r, option_type)

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, q=q, option_type=option_type).price - market_price

    f_low  = objective(sigma_low)
    f_high = objective(sigma_high)

    if f_low * f_high > 0:
        return None

    return brentq(objective, sigma_low, sigma_high, xtol=tol)


# ---------------------------------------------------------------------------
# Combined solver — NR first, Brent as fallback
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    verbose: bool = False,
) -> float:
    """
    Robust implied volatility calculation.

    Strategy:
      1. Attempt Newton-Raphson (fast, O(5-10) iterations typical)
      2. Fall back to Brent's method if NR fails to converge
      3. Raise ValueError if neither method finds a solution

    Parameters
    ----------
    market_price : float — Observed option mid-price
    S, K, T, r   : Standard BSM inputs
    q            : Continuous dividend yield
    option_type  : 'call' or 'put'
    verbose      : Print convergence info

    Returns
    -------
    Implied volatility as a decimal (e.g. 0.2543 = 25.43%)
    """
    iv = iv_newton_raphson(market_price, S, K, T, r, q, option_type)

    if iv is not None:
        if verbose:
            print(f"  [NR ] Converged → IV = {iv*100:.4f}%")
        return iv

    iv = iv_brent(market_price, S, K, T, r, q, option_type)

    if iv is not None:
        if verbose:
            print(f"  [Brent] Converged → IV = {iv*100:.4f}%")
        return iv

    raise ValueError("Implied volatility solver failed to converge. Check inputs.")


# ---------------------------------------------------------------------------
# Volatility smile from a chain of strikes
# ---------------------------------------------------------------------------

def vol_smile(
    market_prices: list,
    strikes: list,
    S: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
) -> list:
    """
    Compute IV for each (strike, price) pair — builds the vol smile.

    Returns list of (strike, moneyness, iv) tuples.
    """
    results = []
    for K, price in zip(strikes, market_prices):
        try:
            iv = implied_volatility(price, S, K, T, r, q, option_type)
            moneyness = np.log(K / S)
            results.append((K, moneyness, iv))
        except Exception:
            results.append((K, np.log(K / S), None))
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  IMPLIED VOLATILITY SOLVER")
    print("  Newton-Raphson + Brent's Method")
    print("=" * 55)

    S, T, r = 100, 0.25, 0.05
    true_sigma = 0.23

    print(f"\n  Setup: S={S}, T={T}y, r={r*100}%, true σ={true_sigma*100}%")
    print(f"  {'Strike':>8}  {'Type':>5}  {'Market Px':>10}  {'IV (NR)':>10}  {'IV (Brent)':>12}  {'Error':>10}")
    print("  " + "-" * 62)

    for K in [90, 95, 100, 105, 110]:
        for opt in ["call", "put"]:
            market_px = bs_price(S, K, T, r, true_sigma, option_type=opt).price
            iv_nr     = iv_newton_raphson(market_px, S, K, T, r, option_type=opt)
            iv_br     = iv_brent(market_px, S, K, T, r, option_type=opt)
            err       = abs((iv_nr or 0) - true_sigma) * 100
            print(f"  {K:>8}  {opt:>5}  {market_px:>10.4f}  "
                  f"{(iv_nr or 0)*100:>9.4f}%  {(iv_br or 0)*100:>11.4f}%  {err:>9.2e}%")

    print("\n  Vol Smile (call options, varying strikes)")
    print(f"  {'Strike':>8}  {'Moneyness':>10}  {'IV':>8}")
    print("  " + "-" * 32)
    strikes   = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    smile_ivs = [0.35, 0.30, 0.27, 0.24, 0.22, 0.21, 0.21, 0.22, 0.23]
    prices    = [bs_price(S, K, T, r, iv, option_type="call").price
                 for K, iv in zip(strikes, smile_ivs)]
    smile     = vol_smile(prices, strikes, S, T, r, option_type="call")
    for K, moneyness, iv in smile:
        if iv:
            print(f"  {K:>8}  {moneyness:>10.4f}  {iv*100:>7.2f}%")
