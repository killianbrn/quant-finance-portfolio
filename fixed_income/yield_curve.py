"""
Yield Curve Construction & Bootstrapping
==========================================
Build a zero-coupon yield curve from observable market instruments.

Instruments used (in order of maturity):
  - Deposit rates     (O/N to 1Y)
  - Interest Rate Swaps (2Y to 30Y)

Curve models:
  1. Bootstrap         — exact fit to market quotes
  2. Nelson-Siegel     — parsimonious 4-parameter smooth curve

Derived quantities:
  - Zero coupon rates  r(0, T)
  - Discount factors   P(0, T) = exp(-r(0,T) * T)
  - Forward rates      f(T1, T2) = [r(T2)*T2 - r(T1)*T1] / (T2 - T1)

Author: Quant Finance Portfolio
"""

import numpy as np
from scipy.optimize import minimize, brentq
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketQuote:
    tenor_years: float
    rate: float          # As decimal, e.g. 0.045 = 4.5%
    instrument: str      # 'deposit' or 'swap'


@dataclass
class CurvePoint:
    tenor: float
    zero_rate: float
    discount_factor: float
    forward_rate_1y: float


# ---------------------------------------------------------------------------
# Discount factor / zero rate conversions
# ---------------------------------------------------------------------------

def zero_to_df(zero_rate: float, T: float) -> float:
    """Continuously compounded zero rate to discount factor."""
    return np.exp(-zero_rate * T)


def df_to_zero(df: float, T: float) -> float:
    """Discount factor to continuously compounded zero rate."""
    if T <= 0 or df <= 0:
        raise ValueError("T and df must be positive.")
    return -np.log(df) / T


# ---------------------------------------------------------------------------
# Bootstrapping (exact)
# ---------------------------------------------------------------------------

def bootstrap_curve(quotes: List[MarketQuote]) -> List[Tuple[float, float]]:
    """
    Bootstrap a zero curve from sorted market quotes.

    For deposit instruments: direct conversion.
    For swaps: solve for the terminal discount factor given previous DFs.

    Returns list of (tenor, zero_rate) tuples.
    """
    quotes = sorted(quotes, key=lambda q: q.tenor_years)
    curve  = []   # list of (tenor, df, zero_rate)

    for quote in quotes:
        T = quote.tenor_years

        if quote.instrument == "deposit":
            df = 1.0 / (1.0 + quote.rate * T)
            z  = df_to_zero(df, T)
            curve.append((T, df, z))

        elif quote.instrument == "swap":
            freq = 2
            dt   = 1.0 / freq
            c    = quote.rate / freq

            known_T  = [p[0] for p in curve]
            known_df = [p[1] for p in curve]

            def interp_df(t):
                """Linear interpolation on zero rates."""
                if t <= known_T[0]:
                    return known_df[0] ** (t / known_T[0])
                if t >= known_T[-1]:
                    z_last = -np.log(known_df[-1]) / known_T[-1]
                    return np.exp(-z_last * t)
                z_known  = [-np.log(df) / ti for ti, df in zip(known_T, known_df)]
                z_interp = np.interp(t, known_T, z_known)
                return np.exp(-z_interp * t)

            n_periods  = int(T * freq)
            pv_coupons = sum(
                c * interp_df(dt * i)
                for i in range(1, n_periods)
            )
            df_T = (1.0 - pv_coupons) / (1.0 + c)
            df_T = max(df_T, 1e-6)
            z_T  = df_to_zero(df_T, T)
            curve.append((T, df_T, z_T))

    return [(t, z) for t, df, z in curve]


# ---------------------------------------------------------------------------
# Nelson-Siegel model
# ---------------------------------------------------------------------------

def nelson_siegel(
    T: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float
) -> np.ndarray:
    """
    Nelson-Siegel yield curve:
      r(T) = beta0
           + beta1 * [(1 - e^{-T/tau}) / (T/tau)]
           + beta2 * [(1 - e^{-T/tau}) / (T/tau) - e^{-T/tau}]

    Interpretation:
      beta0 : long-run level
      beta1 : short-term component (slope)
      beta2 : medium-term component (hump or trough)
      tau   : decay factor
    """
    x     = T / tau
    load1 = (1 - np.exp(-x)) / x
    load2 = load1 - np.exp(-x)
    return beta0 + beta1 * load1 + beta2 * load2


def fit_nelson_siegel(tenors: np.ndarray, zero_rates: np.ndarray) -> dict:
    """Fit Nelson-Siegel to observed zero rates via least squares."""
    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0 or b0 <= 0:
            return 1e10
        fitted = nelson_siegel(tenors, b0, b1, b2, tau)
        return np.sum((fitted - zero_rates) ** 2)

    result = minimize(
        objective,
        x0=[0.05, -0.02, 0.01, 2.0],
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
    )
    b0, b1, b2, tau = result.x
    return {
        "beta0": b0,
        "beta1": b1,
        "beta2": b2,
        "tau": tau,
        "success": result.success
    }


# ---------------------------------------------------------------------------
# Forward rates
# ---------------------------------------------------------------------------

def forward_rate(zero1: float, T1: float, zero2: float, T2: float) -> float:
    """
    Implied forward rate between T1 and T2:
      f(T1, T2) = [r2*T2 - r1*T1] / (T2 - T1)
    Continuously compounded.
    """
    if T2 <= T1:
        raise ValueError("T2 must be greater than T1.")
    return (zero2 * T2 - zero1 * T1) / (T2 - T1)


def build_forward_curve(tenors, zero_rates) -> List[Tuple[float, float]]:
    """Compute 1Y x 1Y forward rates along the curve."""
    forwards = []
    for i in range(len(tenors) - 1):
        f = forward_rate(zero_rates[i], tenors[i], zero_rates[i+1], tenors[i+1])
        mid_tenor = (tenors[i] + tenors[i+1]) / 2
        forwards.append((mid_tenor, f))
    return forwards


# ---------------------------------------------------------------------------
# Sample market data (approximate USD rates)
# -------------------------
