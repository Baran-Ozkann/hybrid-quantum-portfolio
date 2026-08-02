"""Rolling-window fold construction for the Phase 8 backtest.

Each fold is a complete, self-contained instance of the Phase 2 problem:
estimate mu and Sigma from a 250-day training window, build that window's own
Ising Hamiltonian, and then measure what the resulting portfolios actually
returned over the following 63 days.

WHAT THIS MEASURES. The fold machinery is entirely classical. It evaluates the
*objective* -- whether minimising a mean-variance Hamiltonian estimated on 250
days of history selects portfolios that do well in the next quarter. It says
nothing about whether VQE minimises that Hamiltonian correctly; that is Phases
4-9, and is measured separately in ``run_backtest.py``.

Conventions inherited from Phase 2, unchanged so the folds are comparable to
the main instance: annualisation by 252 trading days, simple (not log) returns
for mu and Sigma, rf = 5%, risk aversion q = 1.0, budget B = 2.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H              # noqa: E402
from classical.data_pipeline import (               # noqa: E402
    RISK_FREE_RATE, TRADING_DAYS, sharpe_ratio,
)
from backtest.fetch_snapshot import load_backtest_prices  # noqa: E402

logger = logging.getLogger("folds")

TRAIN_DAYS = 250     # matches the ~1-year estimation window of the main instance
TEST_DAYS = 63       # one quarter
STEP_DAYS = 63       # non-overlapping holding periods; training windows overlap

#: One-way transaction cost, charged on the absolute weight change per asset.
#: Selling 0.5 of A to buy 0.5 of B moves sum|dw| = 1.0 and costs 10 bps.
COST_PER_SIDE = 0.0010


@dataclass
class PortfolioOutcome:
    """One of the six feasible 2-asset portfolios, in and out of sample."""

    bitstring: str
    held: tuple[str, ...]
    weights: np.ndarray
    #: In-sample, from the training window's mu and Sigma.
    is_return: float
    is_vol: float
    is_sharpe: float
    #: The Ising energy the optimiser would be minimising for this state.
    energy: float
    #: Out-of-sample, realised over the 63-day holding period.
    oos_total_return: float
    oos_ann_return: float
    oos_ann_vol: float
    oos_sharpe: float


@dataclass
class Fold:
    """One rolling window: a Hamiltonian, and what happened next."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    mu: np.ndarray
    sigma: np.ndarray
    ham: H.PortfolioHamiltonian
    #: All six feasible portfolios, ordered by ascending energy.
    portfolios: list[PortfolioOutcome]
    #: Spectrum diagnostics over all 16 states.
    ground_bitstring: str
    ground_energy: float
    runner_up_bitstring: str
    runner_up_energy: float
    energy_gap: float
    spectral_width: float
    ground_feasible: bool
    runner_up_feasible: bool
    #: Realised returns of the reference arms over the same window.
    ew4_total_return: float = 0.0
    ew4_sharpe: float = 0.0
    single_asset: dict = field(default_factory=dict)

    @property
    def gap_fraction(self) -> float:
        """Degeneracy gap as a fraction of the spectral width."""
        return self.energy_gap / self.spectral_width if self.spectral_width else 0.0

    def by_bitstring(self, bits: str) -> PortfolioOutcome | None:
        for p in self.portfolios:
            if p.bitstring == bits:
                return p
        return None

    @property
    def is_best(self) -> PortfolioOutcome:
        """The in-sample winner -- what a perfect optimiser would select."""
        return min(self.portfolios, key=lambda p: p.energy)

    @property
    def oos_best(self) -> PortfolioOutcome:
        """The out-of-sample winner -- knowable only in hindsight."""
        return max(self.portfolios, key=lambda p: p.oos_sharpe)


def _realised(prices: pd.DataFrame, weights: np.ndarray,
              risk_free: float = RISK_FREE_RATE) -> tuple[float, float, float, float]:
    """Realised buy-and-hold performance of fixed initial ``weights``.

    Weights are set at the first bar and left to drift for the quarter, which
    is what actually happens between rebalances -- daily rebalancing would
    quietly add a rebalancing premium the strategy never earned.

    Returns:
        ``(total_return, annualised_return, annualised_vol, sharpe)``.
    """
    rel = prices.to_numpy() / prices.to_numpy()[0]     # each asset's growth factor
    value = rel @ np.asarray(weights, dtype=float)     # portfolio value path, V0 = 1
    total = float(value[-1] - 1.0)

    daily = np.diff(value) / value[:-1]
    ann_ret = float(daily.mean() * TRADING_DAYS)
    ann_vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 1e-12 else 0.0
    return total, ann_ret, ann_vol, float(sharpe)


def build_folds(prices: pd.DataFrame | None = None,
                train_days: int = TRAIN_DAYS,
                test_days: int = TEST_DAYS,
                step_days: int = STEP_DAYS,
                risk_free: float = RISK_FREE_RATE) -> list[Fold]:
    """Construct every rolling fold from the frozen backtest snapshot.

    Fold ``i`` estimates on returns ``[i*step, i*step + train_days)`` and holds
    over the following ``test_days`` bars. Training windows overlap by
    construction, so the folds are NOT independent -- see docs/backtest.md.
    """
    if prices is None:
        prices = load_backtest_prices()

    tickers = tuple(prices.columns)
    returns = prices.pct_change().dropna(how="any")
    n = len(returns)

    n_folds = (n - train_days - test_days) // step_days + 1
    if n_folds < 1:
        raise ValueError(
            f"{n} return rows is too few for {train_days}+{test_days} windows."
        )
    logger.info("Building %d folds from %d return rows (%d train / %d test, step %d).",
                n_folds, n, train_days, test_days, step_days)

    folds: list[Fold] = []
    for i in range(n_folds):
        a = i * step_days
        b = a + train_days
        c = b + test_days

        train = returns.iloc[a:b]
        # Price bars spanning the holding period, inclusive of the bar the
        # position is opened at. returns.index[k] corresponds to prices row k+1.
        test_prices = prices.iloc[b:c + 1]

        mu = train.mean().to_numpy() * TRADING_DAYS
        sigma = train.cov().to_numpy() * TRADING_DAYS
        sigma = 0.5 * (sigma + sigma.T)

        ham = H.build_hamiltonian(mu, sigma, tickers)
        states = H.enumerate_states(ham)             # ascending cost, all 16

        # enumerate_states sorts by penalised QUBO cost; the optimiser minimises
        # the Ising energy. They induce the same order (the map is affine), but
        # the gap must be measured in the units VQE actually sees.
        states.sort(key=lambda r: r["ising_energy"])
        spectrum = [s["ising_energy"] for s in states]
        width = float(spectrum[-1] - spectrum[0])

        outcomes: list[PortfolioOutcome] = []
        for s in states:
            if not s["feasible"]:
                continue
            w = H.decode_weights(s["bits"], ham.budget)
            total, ann_r, ann_v, shp = _realised(test_prices, w, risk_free)
            outcomes.append(PortfolioOutcome(
                bitstring=s["bitstring"],
                held=tuple(s["held"]),
                weights=w,
                is_return=float(np.dot(w, mu)),
                is_vol=float(np.sqrt(max(w @ sigma @ w, 0.0))),
                is_sharpe=float(sharpe_ratio(w, mu, sigma, risk_free)),
                energy=float(s["ising_energy"]),
                oos_total_return=total,
                oos_ann_return=ann_r,
                oos_ann_vol=ann_v,
                oos_sharpe=shp,
            ))
        outcomes.sort(key=lambda p: p.energy)

        ew4 = np.full(len(tickers), 1.0 / len(tickers))
        ew4_total, _, _, ew4_shp = _realised(test_prices, ew4, risk_free)

        singles = {}
        for k, t in enumerate(tickers):
            w = np.zeros(len(tickers))
            w[k] = 1.0
            tot, _, _, shp = _realised(test_prices, w, risk_free)
            singles[t] = {"total_return": tot, "sharpe": shp}

        folds.append(Fold(
            index=i,
            train_start=train.index[0], train_end=train.index[-1],
            test_start=test_prices.index[0], test_end=test_prices.index[-1],
            mu=mu, sigma=sigma, ham=ham,
            portfolios=outcomes,
            ground_bitstring=states[0]["bitstring"],
            ground_energy=float(states[0]["ising_energy"]),
            runner_up_bitstring=states[1]["bitstring"],
            runner_up_energy=float(states[1]["ising_energy"]),
            energy_gap=float(states[1]["ising_energy"] - states[0]["ising_energy"]),
            spectral_width=width,
            ground_feasible=bool(states[0]["feasible"]),
            runner_up_feasible=bool(states[1]["feasible"]),
            ew4_total_return=ew4_total,
            ew4_sharpe=ew4_shp,
            single_asset=singles,
        ))

    return folds


def turnover_cost(prev_weights: np.ndarray | None,
                  new_weights: np.ndarray,
                  cost_per_side: float = COST_PER_SIDE) -> float:
    """Cost of moving to ``new_weights``, as a fraction of portfolio value.

    ``sum |dw|`` already counts both the sale and the purchase, so multiplying
    by the one-way rate charges each side exactly once. The first fold pays for
    establishing the position from cash.
    """
    if prev_weights is None:
        return float(cost_per_side * np.abs(new_weights).sum())
    return float(cost_per_side * np.abs(np.asarray(new_weights) - np.asarray(prev_weights)).sum())


def drifted_weights(prices: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Where fixed initial ``weights`` end up after a buy-and-hold quarter.

    Rebalancing cost must be measured against the *drifted* position, not the
    original target, or a strategy that holds the same assets twice in a row
    would be charged for a trade it never made.
    """
    rel = prices.to_numpy()[-1] / prices.to_numpy()[0]
    ending = np.asarray(weights, dtype=float) * rel
    total = ending.sum()
    return ending / total if total > 0 else np.asarray(weights, dtype=float)
