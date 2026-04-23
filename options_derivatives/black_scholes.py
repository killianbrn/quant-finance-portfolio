"""
Black-Scholes Option Pricer + Full Greeks
==========================================
Analytical pricing of European vanilla options under the Black-Scholes framework.

Model assumptions:
  - Constant volatility and risk-free rate
  - No dividends (continuous dividend yield q supported)
  - Log-normal underlying dynamics: dS = mu*S*dt + sigma*S*dW
  - Frictionless markets, no transaction costs

Author: Quant Finance Portfolio
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass
class BSMResult:
    """Container for Black-Scholes pricing output."""
    option_type: str
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    d1: float
    d2: float


# ---------------------------------------------------------------------------
# Black-Scholes analytical pricer
# ---------------------------------------------------------------------------

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> BSMResult:
    """
    Price a European vanilla option using the Black-Scholes-Merton formula.

    Parameters
    ----------
    S     : float  — Current spot price
    K     : float  — Strike price
    T     : float  — Time to expiry in years (e.g. 0.25 = 3 months)
    r     : float  — Continuously compounded risk-free rate (e.g. 0.05 = 5%)
    sigma : float  — Annualised implied volatility (e.g. 0.20 = 20%)
    q     : float  — Continuous dividend yield (default 0)
    option_type : 'call' or 'put'

    Returns
    -------
    BSMResult dataclass with price and all first-order Greeks.
    """
    if T <= 0:
        raise ValueError("Time to expiry T must be strictly positive.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be strictly positive.")
    if S <= 0 or K <= 0:
        raise ValueError("Spot S and strike K must be positive.")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    discount = np.exp(-r * T)
    forward  = S * np.exp(-q * T)

    if option_type == "call":
        price = forward * norm.cdf(d1) - K * discount * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        rho   = K * T * discount * norm.cdf(d2) / 100
    elif option_type == "put":
        price = K * discount * norm.cdf(-d2) - forward * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        rho   = -K * T * discount * norm.cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (
        -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * discount * norm.cdf(d2 if option_type == "call" else -d2)
        + q * forward * norm.cdf(d1 if option_type == "call" else -d1)
    ) / 365

    return BSMResult(
        option_type=option_type,
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        d1=d1,
        d2=d2,
    )


# ---------------------------------------------------------------------------
# Put-Call Parity verification
# ---------------------------------------------------------------------------

def put_call_parity_check(S, K, T, r, q=0.0) -> float:
    """
    Returns the put-call parity spread: C - P - (F - K*e^{-rT}).
    Should be ~0 for consistent pricing.
    """
    call = bs_price(S, K, T, r, sigma=0.20, q=q, option_type="call").price
    put  = bs_price(S, K, T, r, sigma=0.20, q=q, option_type="put").price
    forward_pv = S * np.exp(-q * T) - K * np.exp(-r * T)
    return call - put - forward_pv


# ---------------------------------------------------------------------------
# Greeks sensitivity table
# ---------------------------------------------------------------------------

def greeks_surface(S, K, T, r, sigmas, option_type="call"):
    """Compute Greeks across a range of volatilities."""
    results = {"sigma": [], "price": [], "delta": [], "gamma": [], "vega": [], "theta": []}
    for sigma in sigmas:
        res = bs_price(S, K, T, r, sigma, option_type=option_type)
        results["sigma"].append(round(sigma * 100, 1))
        results["price"].append(round(res.price, 4))
        results["delta"].append(round(res.delta, 4))
        results["gamma"].append(round(res.gamma, 6))
        results["vega"].append(round(res.vega, 4))
        results["theta"].append(round(res.theta, 4))
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

    print("=" * 55)
    print("  BLACK-SCHOLES PRICER — European Vanilla Options")
    print("=" * 55)
    print(f"\n  Inputs: S={S}, K={K}, T={T}y, r={r*100}%, σ={sigma*100}%\n")

    for opt in ["call", "put"]:
        res = bs_price(S, K, T, r, sigma, option_type=opt)
        print(f"  {'CALL' if opt=='call' else 'PUT ':4s}  │  Price: {res.price:7.4f}")
        print(f"         │  d1={res.d1:.4f}  d2={res.d2:.4f}")
        print(f"         │  Δ={res.delta:.4f}  Γ={res.gamma:.6f}")
        print(f"         │  ν={res.vega:.4f}  Θ={res.theta:.4f}  ρ={res.rho:.4f}")
        print()

    parity = put_call_parity_check(S, K, T, r)
    print(f"  Put-Call Parity spread: {parity:.2e}  ✓" if abs(parity) < 1e-10
          else f"  Put-Call Parity violation: {parity:.6f}  ✗")
