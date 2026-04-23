"""
Binomial Tree Option Pricer — Cox-Ross-Rubinstein (CRR)
=========================================================
Lattice-based pricing supporting European and American exercise.

Key advantage over Black-Scholes:
  American options cannot be priced analytically — the binomial tree
  handles early exercise optimally via backward induction.

CRR parameterisation:
  u = exp(sigma * sqrt(dt))   — up factor
  d = 1 / u                   — down factor (recombining tree)
  p = (exp((r-q)*dt) - d) / (u - d)  — risk-neutral probability

Author: Quant Finance Portfolio
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal
from options_derivatives.black_scholes import bs_price


@dataclass
class BinomialResult:
    price: float
    delta: float
    gamma: float
    theta: float
    exercise_type: str
    n_steps: int


# ---------------------------------------------------------------------------
# CRR tree builder and pricer
# ---------------------------------------------------------------------------

def binomial_crr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    exercise: Literal["european", "american"] = "european",
    n_steps: int = 500,
) -> BinomialResult:
    """
    Price a vanilla option using the CRR binomial tree.

    Parameters
    ----------
    exercise : 'european' (no early exercise) or 'american' (early exercise allowed)
    n_steps  : Number of time steps (higher = more accurate, slower)
               Convergence is O(1/N) — 500 steps gives ~4 decimal places

    Algorithm
    ---------
    1. Build terminal stock prices:    S_T[j] = S * u^j * d^(N-j)
    2. Compute terminal payoffs:       max(S_T - K, 0) for calls
    3. Backward induction:
       V[i,j] = e^{-r*dt} * (p*V[i+1,j+1] + (1-p)*V[i+1,j])
       For American: V[i,j] = max(V[i,j], intrinsic[i,j])
    4. Greeks via finite differences on the first two steps of the tree.
    """
    dt = T / n_steps
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    if not (0 < p < 1):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside (0,1). "
            "Reduce dt or check inputs."
        )

    # Terminal stock prices
    j   = np.arange(n_steps + 1)
    S_T = S * (u ** j) * (d ** (n_steps - j))

    # Terminal payoffs
    if option_type == "call":
        V = np.maximum(S_T - K, 0)
    else:
        V = np.maximum(K - S_T, 0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])

        if exercise == "american":
            j_i  = np.arange(i + 1)
            S_i  = S * (u ** j_i) * (d ** (i - j_i))
            if option_type == "call":
                intrinsic = np.maximum(S_i - K, 0)
            else:
                intrinsic = np.maximum(K - S_i, 0)
            V = np.maximum(V, intrinsic)

    price = V[0]

    # Greeks from first two tree steps
    S_u = S * u
    S_d = S * d

    def price_from(spot):
        return binomial_crr(
            spot, K, T - dt, r, sigma, q, option_type, exercise, n_steps - 1
        ).price

    V_u = price_from(S_u)
    V_d = price_from(S_d)

    delta = (V_u - V_d) / (S_u - S_d)
    gamma = (
        (V_u - price) / (S_u - S) - (price - V_d) / (S - S_d)
    ) / (0.5 * (S_u - S_d))

    V_mid_later = binomial_crr(
        S, K, T - 2 * dt, r, sigma, q, option_type, exercise, n_steps - 2
    ).price
    theta = (V_mid_later - price) / (2 * dt) / 365

    return BinomialResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        exercise_type=exercise,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Early exercise premium
# ---------------------------------------------------------------------------

def early_exercise_premium(S, K, T, r, sigma, q=0.0, option_type="put") -> dict:
    """
    Compute the early exercise premium for an American option.
    Always non-negative. Most significant for:
      - Deep ITM puts with high r
      - Deep ITM calls with high dividend yield q
    """
    american = binomial_crr(S, K, T, r, sigma, q, option_type, "american").price
    european = binomial_crr(S, K, T, r, sigma, q, option_type, "european").price
    return {
        "american": american,
        "european": european,
        "early_exercise_premium": american - european,
        "premium_pct": (american - european) / european * 100,
    }


# ---------------------------------------------------------------------------
# Convergence with step count
# ---------------------------------------------------------------------------

def convergence_vs_steps(S, K, T, r, sigma, option_type="call", exercise="european") -> list:
    """Price as a function of N — oscillates then converges to BS."""
    bs_ref = bs_price(S, K, T, r, sigma, option_type=option_type).price
    results = []
    for N in [10, 25, 50, 100, 200, 500]:
        res = binomial_crr(
            S, K, T, r, sigma,
            option_type=option_type,
            exercise=exercise,
            n_steps=N
        )
        results.append({
            "n_steps": N,
            "price":   res.price,
            "error":   abs(res.price - bs_ref),
        })
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02

    print("=" * 60)
    print("  BINOMIAL TREE PRICER — Cox-Ross-Rubinstein")
    print("=" * 60)

    bs_call = bs_price(S, K, T, r, sigma, q=q, option_type="call").price
    bs_put  = bs_price(S, K, T, r, sigma, q=q, option_type="put").price

    eu_call = binomial_crr(S, K, T, r, sigma, q=q, option_type="call", exercise="european")
    eu_put  = binomial_crr(S, K, T, r, sigma, q=q, option_type="put",  exercise="european")

    print(f"\n  European (N=500 steps, BS benchmark)")
    print(f"  {'':5}  {'Binomial':>10}  {'BS Analytic':>12}  {'Error':>10}")
    print(f"  Call   {eu_call.price:>10.4f}  {bs_call:>12.4f}  {abs(eu_call.price-bs_call):>10.2e}")
    print(f"  Put    {eu_put.price:>10.4f}  {bs_put:>12.4f}  {abs(eu_put.price-bs_put):>10.2e}")

    am_call = binomial_crr(S, K, T, r, sigma, q=q, option_type="call", exercise="american")
    am_put  = binomial_crr(S, K, T, r, sigma, q=q, option_type="put",  exercise="american")

    print(f"\n  American vs European")
    print(f"  {'':8}  {'European':>10}  {'American':>10}  {'EEP':>10}")
    print(f"  Call     {eu_call.price:>10.4f}  {am_call.price:>10.4f}  {am_call.price - eu_call.price:>10.4f}")
    print(f"  Put      {eu_put.price:>10.4f}  {am_put.price:>10.4f}  {am_put.price - eu_put.price:>10.4f}")

    print(f"\n  Greeks (American Put)")
    print(f"    Delta = {am_put.delta:.4f}")
    print(f"    Gamma = {am_put.gamma:.6f}")
    print(f"    Theta = {am_put.theta:.4f} per day")

    print(f"\n  Convergence (European Call, BS ref={bs_call:.4f})")
    print(f"  {'N Steps':>10}  {'Price':>10}  {'Error':>10}")
    for row in convergence_vs_steps(S, K, T, r, sigma):
        print(f"  {row['n_steps']:>10}  {row['price']:>10.4f}  {row['error']:>10.2e}")
