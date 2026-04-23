"""
Monte Carlo Option Pricer
==========================
Simulation-based pricing for path-independent and path-dependent options.

Options implemented:
  1. European Call/Put          — benchmark against Black-Scholes
  2. Asian (Average Price)      — arithmetic & geometric average
  3. Barrier (Up-and-Out/In)    — knock-out and knock-in variants

Variance reduction techniques:
  - Antithetic variates         — reduces variance ~50% at zero cost
  - Control variate (BS price)  — exploits known analytical solution

Author: Quant Finance Portfolio
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from options_derivatives.black_scholes import bs_price


# ---------------------------------------------------------------------------
# Core result container
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    price: float
    std_error: float
    conf_interval_95: tuple
    n_simulations: int
    n_steps: int


# ---------------------------------------------------------------------------
# GBM path generator
# ---------------------------------------------------------------------------

def simulate_gbm(
    S: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int,
    n_steps: int,
    q: float = 0.0,
    antithetic: bool = True,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths under risk-neutral measure.

    dS = (r - q) * S * dt + sigma * S * dW

    Exact discretisation (no Euler error):
      S(t+dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    antithetic : If True, doubles n_sims by mirroring Z to -Z
                 (variance reduction: keeps mean, halves variance)

    Returns
    -------
    np.ndarray of shape (n_sims [*2 if antithetic], n_steps+1)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt    = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_sims, n_steps))

    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)

    log_returns = drift + vol * Z
    log_paths   = np.cumsum(log_returns, axis=1)
    log_paths   = np.concatenate([np.zeros((log_paths.shape[0], 1)), log_paths], axis=1)

    return S * np.exp(log_paths)


# ---------------------------------------------------------------------------
# 1. European Option (MC with CV)
# ---------------------------------------------------------------------------

def mc_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    n_sims: int = 100_000,
    n_steps: int = 252,
    control_variate: bool = True,
    seed: int = 42,
) -> MCResult:
    """
    Price a European option via Monte Carlo.

    Control Variate technique:
      Since E[BS_price] = BS_price analytically, we can write:
        price_CV = price_MC - beta * (MC_payoff_underlying - BS_price)
      This dramatically reduces standard error when beta is close to 1.
    """
    paths = simulate_gbm(S, r, sigma, T, n_sims, n_steps, q=q, seed=seed)
    S_T   = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    disc_payoffs = np.exp(-r * T) * payoffs

    if control_variate:
        expected_ST  = S * np.exp((r - q) * T)
        disc_ST      = np.exp(-r * T) * S_T
        beta         = np.cov(disc_payoffs, disc_ST)[0, 1] / np.var(disc_ST)
        disc_payoffs = disc_payoffs - beta * (disc_ST - np.exp(-r * T) * expected_ST)

    price = np.mean(disc_payoffs)
    se    = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))

    return MCResult(
        price=price,
        std_error=se,
        conf_interval_95=(price - 1.96 * se, price + 1.96 * se),
        n_simulations=len(disc_payoffs),
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# 2. Asian Option (Average Price)
# ---------------------------------------------------------------------------

def mc_asian(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    average: Literal["arithmetic", "geometric"] = "arithmetic",
    n_sims: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
) -> MCResult:
    """
    Price an Asian (average price) option.

    Payoff:
      Call: max(A(S) - K, 0)
      Put:  max(K - A(S), 0)
    where A(S) is either arithmetic or geometric average of the path.
    """
    paths = simulate_gbm(S, r, sigma, T, n_sims, n_steps, q=q, seed=seed)

    if average == "arithmetic":
        avg_S = np.mean(paths[:, 1:], axis=1)
    else:
        avg_S = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

    if option_type == "call":
        payoffs = np.maximum(avg_S - K, 0)
    else:
        payoffs = np.maximum(K - avg_S, 0)

    disc_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(disc_payoffs)
    se    = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))

    return MCResult(
        price=price,
        std_error=se,
        conf_interval_95=(price - 1.96 * se, price + 1.96 * se),
        n_simulations=len(disc_payoffs),
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# 3. Barrier Option
# ---------------------------------------------------------------------------

def mc_barrier(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"] = "up-and-out",
    option_type: Literal["call", "put"] = "call",
    rebate: float = 0.0,
    q: float = 0.0,
    n_sims: int = 200_000,
    n_steps: int = 252,
    seed: int = 42,
) -> MCResult:
    """
    Price a single-barrier option.

    Barrier types:
      up-and-out   : knock out if S crosses barrier from below
      up-and-in    : only active if S crosses barrier from below
      down-and-out : knock out if S crosses barrier from above
      down-and-in  : only active if S crosses barrier from above

    Parameters
    ----------
    barrier : Barrier level H
    rebate  : Cash amount paid if option is knocked out (default 0)
    """
    paths = simulate_gbm(S, r, sigma, T, n_sims, n_steps, q=q, seed=seed)
    S_T   = paths[:, -1]

    if option_type == "call":
        vanilla_payoff = np.maximum(S_T - K, 0)
    else:
        vanilla_payoff = np.maximum(K - S_T, 0)

    if barrier_type == "up-and-out":
        breached = np.any(paths >= barrier, axis=1)
        payoffs  = np.where(breached, rebate, vanilla_payoff)
    elif barrier_type == "up-and-in":
        breached = np.any(paths >= barrier, axis=1)
        payoffs  = np.where(breached, vanilla_payoff, rebate)
    elif barrier_type == "down-and-out":
        breached = np.any(paths <= barrier, axis=1)
        payoffs  = np.where(breached, rebate, vanilla_payoff)
    elif barrier_type == "down-and-in":
        breached = np.any(paths <= barrier, axis=1)
        payoffs  = np.where(breached, vanilla_payoff, rebate)
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type}")

    disc_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(disc_payoffs)
    se    = np.std(disc_payoffs) / np.sqrt(len(disc_payoffs))

    return MCResult(
        price=price,
        std_error=se,
        conf_interval_95=(price - 1.96 * se, price + 1.96 * se),
        n_simulations=len(disc_payoffs),
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_study(S, K, T, r, sigma, option_type="call") -> list:
    """Show how MC price converges to BS as n_sims increases."""
    bs_ref = bs_price(S, K, T, r, sigma, option_type=option_type).price
    results = []
    for n in [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]:
        res = mc_european(S, K, T, r, sigma, option_type=option_type, n_sims=n, seed=42)
        results.append({
            "n_sims":   n,
            "mc_price": res.price,
            "bs_price": bs_ref,
            "error":    abs(res.price - bs_ref),
            "se":       res.std_error,
        })
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print("=" * 60)
    print("  MONTE CARLO OPTION PRICER")
    print("=" * 60)

    bs_ref = bs_price(S, K, T, r, sigma, option_type="call").price
    mc_eu  = mc_european(S, K, T, r, sigma, option_type="call", n_sims=100_000)
    print(f"\n  European Call")
    print(f"    BS (analytical) : {bs_ref:.4f}")
    print(f"    MC (100k sims)  : {mc_eu.price:.4f}  +-{mc_eu.std_error:.4f}")
    print(f"    95% CI          : [{mc_eu.conf_interval_95[0]:.4f}, {mc_eu.conf_interval_95[1]:.4f}]")

    mc_asian_arith = mc_asian(S, K, T, r, sigma, option_type="call", average="arithmetic")
    mc_asian_geo   = mc_asian(S, K, T, r, sigma, option_type="call", average="geometric")
    print(f"\n  Asian Call (Arithmetic avg) : {mc_asian_arith.price:.4f}")
    print(f"  Asian Call (Geometric  avg) : {mc_asian_geo.price:.4f}")
    print(f"  [Asian always < European — averaging reduces vol]")

    mc_uao = mc_barrier(S, K, T, r, sigma, barrier=120, barrier_type="up-and-out")
    mc_uai = mc_barrier(S, K, T, r, sigma, barrier=120, barrier_type="up-and-in")
    print(f"\n  Barrier Call (H=120)")
    print(f"    Up-and-Out : {mc_uao.price:.4f}")
    print(f"    Up-and-In  : {mc_uai.price:.4f}")
    print(f"    Sum        : {mc_uao.price + mc_uai.price:.4f}  (should be close to vanilla {bs_ref:.4f})")

    print(f"\n  Convergence Study")
    print(f"  {'N Sims':>10}  {'MC Price':>10}  {'Error':>10}  {'Std Err':>10}")
    print("  " + "-" * 45)
    for row in convergence_study(S, K, T, r, sigma):
        print(f"  {row['n_sims']:>10,}  {row['mc_price']:>10.4f}  "
              f"{row['error']:>10.4f}  {row['se']:>10.4f}")
