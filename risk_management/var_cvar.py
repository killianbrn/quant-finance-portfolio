"""
Value at Risk (VaR) & Conditional Value at Risk (CVaR)
=======================================================
Three methodologies for quantifying portfolio downside risk.

Methods:
  1. Historical Simulation  — non-parametric, preserves fat tails & skew
  2. Parametric (Variance-Covariance)  — Gaussian assumption, fast
  3. Monte Carlo Simulation  — flexible, handles nonlinearities

Measures:
  VaR(alpha, T)  : Max loss not exceeded with probability alpha over horizon T
  CVaR(alpha, T) : Expected loss given that loss exceeds VaR (Expected Shortfall)

CVaR is preferred by regulators (Basel III/IV) because:
  - Coherent risk measure (sub-additive, monotone, positive homogeneous)
  - Captures tail severity, not just tail threshold
  - VaR ignores the shape of the loss distribution beyond the quantile

Author: Quant Finance Portfolio
"""

import numpy as np
from scipy.stats import norm, t as t_dist
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    method: str
    confidence_level: float
    horizon_days: int
    var: float
    cvar: float
    var_pct: float
    cvar_pct: float


# ---------------------------------------------------------------------------
# 1. Historical Simulation
# ---------------------------------------------------------------------------

def var_historical(
    returns: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.99,
    horizon: int = 1,
) -> RiskResult:
    """
    Historical Simulation VaR & CVaR.

    Parameters
    ----------
    returns         : Array of daily P&L returns (as decimals)
    portfolio_value : Current portfolio NAV
    confidence      : VaR confidence level (e.g. 0.99 = 99%)
    horizon         : Risk horizon in trading days

    Notes
    -----
    - No distributional assumption — uses empirical loss distribution
    - Sensitive to the historical window chosen
    - Scale to multi-day horizon using sqrt(T) rule (assumes iid returns)
    """
    losses  = -returns
    var_1d  = np.percentile(losses, confidence * 100)
    cvar_1d = losses[losses >= var_1d].mean()

    var_h  = var_1d  * np.sqrt(horizon)
    cvar_h = cvar_1d * np.sqrt(horizon)

    return RiskResult(
        method="Historical Simulation",
        confidence_level=confidence,
        horizon_days=horizon,
        var=var_h * portfolio_value,
        cvar=cvar_h * portfolio_value,
        var_pct=var_h * 100,
        cvar_pct=cvar_h * 100,
    )


# ---------------------------------------------------------------------------
# 2. Parametric (Variance-Covariance)
# ---------------------------------------------------------------------------

def var_parametric(
    returns: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.99,
    horizon: int = 1,
    distribution: str = "normal",
    dof: Optional[float] = None,
) -> RiskResult:
    """
    Parametric VaR & CVaR.

    Assumes returns follow N(mu, sigma^2) or t(nu) distribution.

    Normal VaR:
      VaR = -(mu + sigma * z_alpha) * sqrt(T)
    where z_alpha = norm.ppf(1 - confidence)

    Normal CVaR (Expected Shortfall):
      ES = sigma * norm.pdf(z_alpha) / (1 - confidence) * sqrt(T)

    Parameters
    ----------
    distribution : 'normal' or 't' (Student-t for fat tails)
    dof          : Degrees of freedom for t-distribution
    """
    mu    = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    alpha = 1 - confidence

    if distribution == "normal":
        z     = norm.ppf(alpha)
        var1  = -(mu + sigma * z)
        cvar1 = sigma * norm.pdf(z) / alpha - mu

    elif distribution == "t":
        if dof is None:
            _, dof, _ = t_dist.fit(returns, floc=mu)
        z     = t_dist.ppf(alpha, df=dof)
        var1  = -(mu + sigma * z)
        cvar1 = (sigma / alpha) * t_dist.pdf(z, df=dof) * (dof + z**2) / (dof - 1) - mu

    else:
        raise ValueError("distribution must be 'normal' or 't'")

    var_h  = var1  * np.sqrt(horizon)
    cvar_h = cvar1 * np.sqrt(horizon)

    return RiskResult(
        method=f"Parametric ({distribution})",
        confidence_level=confidence,
        horizon_days=horizon,
        var=var_h * portfolio_value,
        cvar=cvar_h * portfolio_value,
        var_pct=var_h * 100,
        cvar_pct=cvar_h * 100,
    )


# ---------------------------------------------------------------------------
# 3. Monte Carlo VaR
# ---------------------------------------------------------------------------

def var_monte_carlo(
    mu: float,
    sigma: float,
    portfolio_value: float,
    confidence: float = 0.99,
    horizon: int = 10,
    n_sims: int = 100_000,
    seed: int = 42,
    fat_tails: bool = False,
    dof: float = 5.0,
) -> RiskResult:
    """
    Monte Carlo VaR & CVaR.

    Simulates portfolio P&L paths under Geometric Brownian Motion
    or Student-t shocks for fat-tail modeling.

    Parameters
    ----------
    mu, sigma   : Daily return mean and volatility
    horizon     : Multi-day horizon (simulates full path)
    fat_tails   : Use Student-t innovations instead of Gaussian
    dof         : Degrees of freedom for Student-t (lower = fatter tails)
    """
    rng = np.random.default_rng(seed)

    if fat_tails:
        Z = rng.standard_t(df=dof, size=(n_sims, horizon))
        Z = Z / np.sqrt(dof / (dof - 2))
    else:
        Z = rng.standard_normal((n_sims, horizon))

    daily_returns = mu + sigma * Z
    total_returns = np.prod(1 + daily_returns, axis=1) - 1
    losses        = -total_returns

    alpha = 1 - confidence
    var   = np.percentile(losses, confidence * 100)
    cvar  = losses[losses >= var].mean()

    return RiskResult(
        method=f"Monte Carlo ({'t-dist' if fat_tails else 'Normal'})",
        confidence_level=confidence,
        horizon_days=horizon,
        var=var * portfolio_value,
        cvar=cvar * portfolio_value,
        var_pct=var * 100,
        cvar_pct=cvar * 100,
    )


# ---------------------------------------------------------------------------
# Portfolio VaR — multi-asset with correlation
# ---------------------------------------------------------------------------

def portfolio_var_parametric(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.99,
    horizon: int = 1,
) -> RiskResult:
    """
    Parametric VaR for a multi-asset portfolio.

    Portfolio variance: sigma^2_p = w' * Sigma * w
    where Sigma is the covariance matrix of asset returns.

    Parameters
    ----------
    weights        : Array of portfolio weights (sum to 1)
    returns_matrix : (T x N) matrix of historical asset returns
    """
    cov_matrix   = np.cov(returns_matrix.T)
    port_sigma   = np.sqrt(weights @ cov_matrix @ weights)
    port_returns = returns_matrix @ weights
    port_mu      = np.mean(port_returns)

    alpha = 1 - confidence
    z     = norm.ppf(alpha)
    var1  = -(port_mu + port_sigma * z)
    cvar1 = port_sigma * norm.pdf(z) / alpha - port_mu

    var_h  = var1  * np.sqrt(horizon)
    cvar_h = cvar1 * np.sqrt(horizon)

    return RiskResult(
        method="Portfolio Parametric",
        confidence_level=confidence,
        horizon_days=horizon,
        var=var_h * portfolio_value,
        cvar=cvar_h * portfolio_value,
        var_pct=var_h * 100,
        cvar_pct=cvar_h * 100,
    )


# ---------------------------------------------------------------------------
# Backtesting VaR — Kupiec test
# ---------------------------------------------------------------------------

def kupiec_test(
    returns: np.ndarray,
    var_series: np.ndarray,
    confidence: float = 0.99
) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.
    Tests whether observed VaR breach rate matches expected rate.

    H0: p = 1 - confidence (model correctly specified)
    Under H0: LR = -2 * ln(L0/L1) ~ chi^2(1)
    """
    from scipy.stats import chi2

    alpha    = 1 - confidence
    losses   = -returns
    breaches = losses > var_series
    n        = len(returns)
    x        = breaches.sum()
    p_hat    = x / n

    if p_hat == 0 or p_hat == 1:
        return {
            "breaches": x,
            "expected": int(n * alpha),
            "lr_stat":  np.nan,
            "p_value":  np.nan
        }

    lr = -2 * (
        x * np.log(alpha / p_hat) +
        (n - x) * np.log((1 - alpha) / (1 - p_hat))
    )
    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "n_observations":    n,
        "breaches_observed": x,
        "breaches_expected": round(n * alpha),
        "breach_rate":       p_hat,
        "expected_rate":     alpha,
        "lr_statistic":      lr,
        "p_value":           p_value,
        "reject_h0":         p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n   = 756
    returns = (
        0.0003 + 0.012 * rng.standard_normal(n)
        + 0.003 * rng.standard_t(df=4, size=n)
    )
    portfolio_value = 10_000_000

    print("=" * 65)
    print("  VALUE AT RISK & CONDITIONAL VALUE AT RISK")
    print("  Portfolio: $10M | 3Y daily returns | 99% confidence")
    print("=" * 65)

    mu_daily    = np.mean(returns)
    sigma_daily = np.std(returns)
    print(f"\n  Return stats: mu={mu_daily*100:.4f}%/day, sigma={sigma_daily*100:.4f}%/day")
    print(f"  Annualised:   mu={mu_daily*252*100:.2f}%, sigma={sigma_daily*np.sqrt(252)*100:.2f}%\n")

    results = [
        var_historical(returns, portfolio_value, 0.99, 1),
        var_historical(returns, portfolio_value, 0.99, 10),
        var_parametric(returns, portfolio_value, 0.99, 1, "normal"),
        var_parametric(returns, portfolio_value, 0.99, 1, "t"),
        var_monte_carlo(mu_daily, sigma_daily, portfolio_value, 0.99, 1,  fat_tails=False),
        var_monte_carlo(mu_daily, sigma_daily, portfolio_value, 0.99, 10, fat_tails=True),
    ]

    print(f"  {'Method':35}  {'H':>3}  {'VaR ($)':>12}  {'CVaR ($)':>12}  {'VaR %':>7}")
    print("  " + "-" * 78)
    for r in results:
        print(f"  {r.method:35}  {r.horizon_days:>3}d  "
              f"${r.var:>11,.0f}  ${r.cvar:>11,.0f}  {r.var_pct:>6.2f}%")

    print(f"\n  Kupiec Test (rolling 1D 99% historical VaR)")
    rolling_var  = np.array([
        np.percentile(-returns[max(0, i-252):i], 99)
        for i in range(252, n)
    ])
    test_returns = returns[252:]
    kt = kupiec_test(test_returns, rolling_var, 0.99)
    print(f"    Breaches: {kt['breaches_observed']} observed vs {kt['breaches_expected']} expected")
    print(f"    LR stat : {kt['lr_statistic']:.4f}  p-value: {kt['p_value']:.4f}")
    print(f"    Model {'REJECTED' if kt['reject_h0'] else 'ACCEPTED'} at 5% level")
