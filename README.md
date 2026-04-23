# Quant Finance Portfolio

A production-grade implementation of quantitative finance models across
derivatives pricing, fixed income, and risk management.

Built to demonstrate rigorous mathematical thinking and clean engineering —
the standard expected at derivatives desks, hedge funds, and digital asset firms.

---

## Structure

quant-finance-portfolio/
├── options_derivatives/
│   ├── black_scholes.py          # Analytical BSM pricer + full Greeks
│   ├── implied_volatility.py     # IV solver: Newton-Raphson + Brent
│   ├── monte_carlo.py            # MC pricing: European, Asian, Barrier
│   └── binomial_tree.py          # CRR lattice: European & American
├── fixed_income/
│   └── yield_curve.py            # Bootstrap + Nelson-Siegel curve fitting
├── risk_management/
│   ├── var_cvar.py               # VaR & CVaR: Historical, Parametric, MC
│   └── backtesting.py            # Strategy backtester + performance metrics
└── requirements.txt

---

## Modules

### Options & Derivatives

#### `black_scholes.py` — BSM Pricer + Greeks

Analytical pricing of European vanilla options under the Black-Scholes-Merton framework.

| Greek | Interpretation |
|-------|----------------|
| Delta (Δ) | P&L per $1 move in spot |
| Gamma (Γ) | Delta change per $1 spot move |
| Vega (ν) | P&L per 1% move in vol |
| Theta (Θ) | Daily time decay |
| Rho (ρ) | P&L per 1% rate move |

```python
from options_derivatives.black_scholes import bs_price

result = bs_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
print(result.price)   # 5.0666
print(result.delta)   # 0.5398
print(result.vega)    # 0.1974
```

#### `implied_volatility.py` — IV Solver

Inverts the BSM formula to recover market-implied volatility from observed option prices.

Two solvers with automatic fallback:
- **Newton-Raphson**: quadratic convergence, ~5-10 iterations typical
- **Brent's method**: bracketing fallback, guaranteed convergence

```python
from options_derivatives.implied_volatility import implied_volatility

iv = implied_volatility(market_price=5.50, S=100, K=100, T=0.25, r=0.05)
print(f"IV = {iv*100:.2f}%")   # IV = 21.43%
```

#### `monte_carlo.py` — MC Option Pricer

Simulation-based pricing for vanilla and exotic options.
Exact GBM discretisation — no Euler error.

| Option Type | Payoff | Notes |
|-------------|--------|-------|
| European | max(S_T - K, 0) | BS benchmark |
| Asian Arithmetic | max(Avg - K, 0) | No closed form |
| Asian Geometric | max(Geo - K, 0) | Closed form exists |
| Barrier Up-and-Out | Vanilla * 1{max S < H} | Cheaper than vanilla |
| Barrier Up-and-In | Vanilla * 1{max S >= H} | Out + In = Vanilla |

Variance reduction: antithetic variates + control variate.

#### `binomial_tree.py` — CRR Lattice

Cox-Ross-Rubinstein binomial tree for European and American options.
Handles early exercise optimally via backward induction.

```python
from options_derivatives.binomial_tree import binomial_crr

am_put = binomial_crr(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                      option_type="put", exercise="american")
eu_put = binomial_crr(S=100, K=100, T=1.0, r=0.05, sigma=0.20,
                      option_type="put", exercise="european")

print(f"Early Exercise Premium: {am_put.price - eu_put.price:.4f}")
```

---

### Fixed Income

#### `yield_curve.py` — Curve Construction

Build a zero-coupon yield curve from money market and swap quotes.

**Bootstrap** (exact fit): converts deposit rates and solves for swap discount factors iteratively.

**Nelson-Siegel** (smooth parametric fit):

r(T) = β0 + β1 * [(1 - e^{-T/τ}) / (T/τ)] + β2 * [(1 - e^{-T/τ}) / (T/τ) - e^{-T/τ}]

| Parameter | Economic Meaning |
|-----------|-----------------|
| β0 | Long-run yield level |
| β1 | Slope (short vs long rates) |
| β2 | Curvature / hump |
| τ | Decay speed |

---

### Risk Management

#### `var_cvar.py` — VaR & CVaR

Three methodologies for tail risk measurement at 99% confidence:

| Method | Assumptions | Strengths |
|--------|------------|-----------|
| Historical Simulation | None (empirical) | Captures real fat tails |
| Parametric Normal | Gaussian returns | Fast, analytical formula |
| Parametric Student-t | Fat-tailed returns | Better for financial data |
| Monte Carlo | User-specified | Handles nonlinearity |

CVaR (Expected Shortfall) is the regulatory standard under Basel III/IV.
Includes **Kupiec POF test** for VaR model backtesting.

#### `backtesting.py` — Strategy Backtest Engine

Event-driven backtester with transaction cost modeling.

| Metric | Interpretation |
|--------|----------------|
| Sharpe | Risk-adjusted return |
| Sortino | Penalises downside only |
| Calmar | Return per unit drawdown |
| Omega | Full distribution ratio |
| Profit Factor | Trade-level edge |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## References

- Black & Scholes (1973). *The Pricing of Options and Corporate Liabilities*
- Cox, Ross & Rubinstein (1979). *Option Pricing: A Simplified Approach*
- Nelson & Siegel (1987). *Parsimonious Modeling of Yield Curves*
- Rockafellar & Uryasev (2000). *Optimization of Conditional Value-at-Risk*
- Kupiec (1995). *Techniques for Verifying the Accuracy of Risk Management Models*
