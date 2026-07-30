"""
classical/markowitz.py -- Phase 2: classical mean-variance baselines.

Provides the reference portfolios the quantum result is measured against.

THREE BASELINES, AND WHY THERE ARE THREE
----------------------------------------
The quantum encoding (see ``hamiltonian.py`` section 5) can only express
equally weighted B-asset baskets -- for B = 2 that is just C(4,2) = 6 possible
portfolios. Continuous Markowitz optimises over the whole simplex. Reporting
only "quantum vs continuous Markowitz" would conflate two different questions:
whether the optimiser works, and whether the encoding is expressive. So:

1. ``max_sharpe``       Continuous long-only tangency portfolio. The textbook
                        baseline the brief asks for. Strictly more expressive
                        than the quantum encoding, so it should generally win
                        on Sharpe -- that is a statement about the ENCODING,
                        not about the optimiser.

2. ``min_variance``     Continuous long-only minimum-variance portfolio.
                        Reference point for the risk column.

3. ``best_equal_weight``Exhaustive search over the C(n,B) equal-weight baskets,
                        scored by the SAME QUBO cost the quantum circuit
                        minimises. This is the true optimum inside the quantum
                        method's own feasible set, so "quantum vs this" is the
                        honest test of whether the optimiser found the best
                        answer available to it.

Continuous problems are solved with ``scipy.optimize.minimize`` (SLSQP), which
handles the equality constraint ``sum(w) = 1`` and the box bounds ``0 <= w <= 1``
directly. Every solve is verified: convergence flag, constraint satisfaction,
and -- because SLSQP is a local method -- agreement across multiple starting
points.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from classical.hamiltonian import (
    DEFAULT_BUDGET,
    DEFAULT_RISK_AVERSION,
    objective,
    penalised_objective,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("markowitz")

DEFAULT_RISK_FREE = 0.05


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def portfolio_stats(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free: float = DEFAULT_RISK_FREE,
) -> dict:
    """Return, volatility, and Sharpe ratio for a weight vector."""
    w = np.asarray(w, dtype=float)
    ret = float(mu @ w)
    vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    sharpe = (ret - risk_free) / vol if vol > 1e-12 else 0.0
    return {
        "weights": w.tolist(),
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": float(sharpe),
    }


# --------------------------------------------------------------------------
# Continuous optimisation
# --------------------------------------------------------------------------

def _solve(
    fun: Callable[[np.ndarray], float],
    n: int,
    n_starts: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Minimise ``fun`` over the long-only simplex from several start points.

    SLSQP is a local method, so a single start can land on a local optimum.
    Restarting from the equal-weight point, each vertex, and random Dirichlet
    draws makes that visible: the diagnostics report how many starts reached
    the best value found.

    Returns:
        ``(best_weights, diagnostics)``.
    """
    rng = np.random.default_rng(seed)
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * n

    starts = [np.full(n, 1.0 / n)]
    starts.extend(np.eye(n))
    while len(starts) < n_starts:
        starts.append(rng.dirichlet(np.ones(n)))

    best_w, best_f, n_success, n_at_best = None, np.inf, 0, 0
    for x0 in starts:
        res = minimize(
            fun, x0, method="SLSQP", bounds=bounds,
            constraints=constraints, options={"maxiter": 500, "ftol": 1e-12},
        )
        if not res.success:
            continue
        n_success += 1
        if res.fun < best_f - 1e-10:
            best_f, best_w, n_at_best = float(res.fun), np.asarray(res.x, dtype=float), 1
        elif abs(res.fun - best_f) <= 1e-10:
            n_at_best += 1

    if best_w is None:
        raise RuntimeError("SLSQP failed to converge from every starting point.")

    # Clean tiny numerical negatives and renormalise so the reported weights
    # satisfy the constraints exactly rather than to solver tolerance.
    best_w = np.clip(best_w, 0.0, 1.0)
    best_w = best_w / best_w.sum()

    return best_w, {
        "starts_attempted": len(starts),
        "starts_converged": n_success,
        "starts_at_best_optimum": n_at_best,
        "best_objective": best_f,
    }


def min_variance_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free: float = DEFAULT_RISK_FREE,
) -> dict:
    """Long-only minimum-variance portfolio."""
    n = len(mu)
    w, diag = _solve(lambda w: float(w @ sigma @ w), n)
    out = portfolio_stats(w, mu, sigma, risk_free)
    out["solver"] = diag
    out["method"] = "min_variance (SLSQP, long-only, sum(w)=1)"
    return out


def max_sharpe_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free: float = DEFAULT_RISK_FREE,
) -> dict:
    """Long-only maximum-Sharpe (tangency) portfolio."""
    n = len(mu)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(w @ sigma @ w, 1e-18)))
        return -((float(mu @ w) - risk_free) / vol)

    w, diag = _solve(neg_sharpe, n)
    out = portfolio_stats(w, mu, sigma, risk_free)
    out["solver"] = diag
    out["method"] = "max_sharpe (SLSQP, long-only, sum(w)=1)"
    return out


def target_return_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: float,
    risk_free: float = DEFAULT_RISK_FREE,
) -> dict:
    """Minimum-variance portfolio achieving at least ``target`` expected return."""
    n = len(mu)
    constraints = (
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "ineq", "fun": lambda w: float(mu @ w - target)},
    )
    res = minimize(
        lambda w: float(w @ sigma @ w),
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"target-return solve failed: {res.message}")
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    out = portfolio_stats(w, mu, sigma, risk_free)
    out["method"] = f"target_return >= {target:.4f} (SLSQP)"
    return out


def efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_points: int = 40,
    risk_free: float = DEFAULT_RISK_FREE,
) -> list[dict]:
    """Sample the efficient frontier between the min-variance and max-return ends."""
    lo = portfolio_stats(min_variance_portfolio(mu, sigma, risk_free)["weights"], mu, sigma)["expected_return"]
    hi = float(np.max(mu))
    points = []
    for target in np.linspace(lo, hi, n_points):
        try:
            points.append(target_return_portfolio(mu, sigma, float(target), risk_free))
        except RuntimeError:
            continue
    return points


# --------------------------------------------------------------------------
# Discrete baseline (the quantum method's own feasible set)
# --------------------------------------------------------------------------

def best_equal_weight_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    tickers: Sequence[str],
    budget: int = DEFAULT_BUDGET,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    penalty: float = 0.0,
    risk_free: float = DEFAULT_RISK_FREE,
) -> dict:
    """Best equal-weight ``budget``-asset basket, by exhaustive search.

    Scored with the same QUBO objective the quantum circuit minimises, so this
    is exactly the answer a perfect optimiser would return under the Phase 2
    encoding. Only feasible baskets are considered, so the penalty term is
    identically zero and its value is irrelevant to the ranking.

    Returns:
        Stats for the winning basket, plus ``all_candidates`` ranked by cost.
    """
    n = len(mu)
    rows = []
    for bits in itertools.product((0, 1), repeat=n):
        if sum(bits) != budget:
            continue
        w = np.asarray(bits, dtype=float) / budget
        stats = portfolio_stats(w, mu, sigma, risk_free)
        stats["bitstring"] = "".join(str(b) for b in bits)
        stats["held"] = [tickers[i] for i, b in enumerate(bits) if b]
        stats["qubo_cost"] = penalised_objective(
            bits, mu, sigma, risk_aversion, budget, penalty
        )
        stats["objective_unpenalised"] = objective(bits, mu, sigma, risk_aversion)
        rows.append(stats)

    rows.sort(key=lambda r: r["qubo_cost"])
    best = dict(rows[0])
    best["method"] = f"best equal-weight B={budget} basket (exhaustive over {len(rows)})"
    best["all_candidates"] = rows
    return best


# --------------------------------------------------------------------------

def compute_baselines(
    mu: np.ndarray,
    sigma: np.ndarray,
    tickers: Sequence[str],
    budget: int = DEFAULT_BUDGET,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    risk_free: float = DEFAULT_RISK_FREE,
) -> dict:
    """Compute all classical baselines and persist them to ``data/``."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    baselines = {
        "tickers": list(tickers),
        "risk_free_rate": risk_free,
        "risk_aversion": risk_aversion,
        "budget": budget,
        "max_sharpe": max_sharpe_portfolio(mu, sigma, risk_free),
        "min_variance": min_variance_portfolio(mu, sigma, risk_free),
        "equal_weight_all": portfolio_stats(
            np.full(len(mu), 1.0 / len(mu)), mu, sigma, risk_free
        ),
        "best_equal_weight": best_equal_weight_portfolio(
            mu, sigma, tickers, budget, risk_aversion, 0.0, risk_free
        ),
    }

    path = DATA_DIR / "markowitz_baseline.json"
    path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    logger.info("Wrote classical baselines to %s", path)
    return baselines


def load_baselines() -> dict:
    """Load the baselines written by ``compute_baselines``."""
    path = DATA_DIR / "markowitz_baseline.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python classical/markowitz.py")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_baselines(baselines: dict, mu: np.ndarray, sigma: np.ndarray) -> bool:
    """Check every reported portfolio satisfies the constraints it claims."""
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        logger.info("    [%s] %s: %s", "PASS" if passed else "FAIL", name, detail)
        if not passed:
            ok = False

    for key in ("max_sharpe", "min_variance", "equal_weight_all", "best_equal_weight"):
        w = np.asarray(baselines[key]["weights"], dtype=float)
        check(f"{key}: sum(w)=1", abs(w.sum() - 1.0) < 1e-9, f"sum = {w.sum():.12f}")
        check(f"{key}: long-only", bool(np.all(w >= -1e-12)), f"min w = {w.min():.2e}")

        # Recompute the reported statistics from the weights.
        recomputed = portfolio_stats(w, mu, sigma, baselines["risk_free_rate"])
        drift = max(
            abs(recomputed["expected_return"] - baselines[key]["expected_return"]),
            abs(recomputed["volatility"] - baselines[key]["volatility"]),
        )
        check(f"{key}: stats consistent", drift < 1e-9, f"max drift = {drift:.2e}")

    # The tangency portfolio must not be beaten on Sharpe by the other,
    # more constrained, portfolios -- that would mean the solver failed.
    best_sharpe = baselines["max_sharpe"]["sharpe_ratio"]
    for key in ("min_variance", "equal_weight_all", "best_equal_weight"):
        other = baselines[key]["sharpe_ratio"]
        check(
            f"max_sharpe >= {key}",
            best_sharpe >= other - 1e-8,
            f"{best_sharpe:.6f} vs {other:.6f}",
        )

    solver = baselines["max_sharpe"]["solver"]
    check(
        "max_sharpe multi-start agreement",
        solver["starts_at_best_optimum"] >= 2,
        f"{solver['starts_at_best_optimum']}/{solver['starts_converged']} converged starts "
        "reached the best optimum",
    )
    return ok
