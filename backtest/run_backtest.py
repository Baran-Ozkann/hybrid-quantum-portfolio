"""Phase 8 orchestrator: run every arm across the rolling folds.

READ THIS BEFORE INTERPRETING THE OUTPUT.

A backtest does not test the optimiser. It tests the objective. The arms are
arranged as a decomposition so the two cannot be confused:

    quantum          vs  best_equal_weight   ->  OPTIMISER error
                                                 (same objective, same feasible
                                                  set; any difference is VQE
                                                  failing to find the argmin)

    best_equal_weight vs equal_weight_4      ->  value of the OBJECTIVE
                                                 (a perfect optimiser against
                                                  naive diversification)

    both              vs single-asset        ->  context

``best_equal_weight`` is the *perfect optimiser*: it selects each fold's exact
in-sample argmin by brute-force enumeration. So if `best_equal_weight` does not
beat `equal_weight_4`, no optimiser could have helped on this data, and the
quantum arm's error rate is not what is costing anything.

The quantum arm uses CVaR alpha = 0.4 at k = 1, which Phase 9 measured at 30/30
single-shot on the main instance. That makes this arm cost 16 runs rather than
80, at the price of assuming a Phase 9 result transfers to per-fold Hamiltonians
-- an assumption this script checks and reports rather than presumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H                       # noqa: E402
from classical.data_pipeline import RISK_FREE_RATE, TRADING_DAYS  # noqa: E402
from backtest.folds import (                                 # noqa: E402
    COST_PER_SIDE, build_folds, drifted_weights, turnover_cost,
)
from training.optimization import TrainingConfig, train      # noqa: E402

logger = logging.getLogger("run_backtest")

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_JSON = LOGS_DIR / "backtest_results.json"

#: Phase 9's single-shot configuration.
CVAR_ALPHA = 0.4
QUANTUM_SEED = 42
EPOCHS = 200


def run_quantum_arm(folds, alpha: float = CVAR_ALPHA, seed: int = QUANTUM_SEED,
                    epochs: int = EPOCHS) -> list[dict]:
    """Train one VQE run per fold against that fold's own Hamiltonian."""
    out = []
    for f in folds:
        t0 = time.time()
        config = TrainingConfig(
            seed=seed, epochs=epochs, learning_rate=0.05,
            lr_schedule="constant", cvar_alpha=alpha, log_every=10 ** 9,
        )
        result = train(f.ham, config, risk_free=RISK_FREE_RATE)

        readout = result.mode_bitstring
        n_held = sum(int(b) for b in readout)
        feasible = n_held == f.ham.budget
        matched = readout == f.ground_bitstring

        logger.info(
            "fold %2d  readout %s  target %s  %-7s  P(gs)=%.4f  %.0fs",
            f.index, readout, f.ground_bitstring,
            "MATCH" if matched else ("infeas" if not feasible else "miss"),
            result.ground_state_prob, time.time() - t0,
        )
        out.append({
            "fold": f.index,
            "readout": readout,
            "target": f.ground_bitstring,
            "matched": bool(matched),
            "feasible": bool(feasible),
            "n_held": n_held,
            "ground_state_prob": float(result.ground_state_prob),
            "mode_probability": float(result.mode_probability),
            "final_energy": float(result.final_energy),
            "ground_energy": float(result.ground_energy),
            "energy_gap": float(f.energy_gap),
            "settle_epoch": int(result.settle_epoch),
            "n_saturated_epochs": int(result.n_saturated_epochs),
            "wall_time_s": float(result.wall_time_s),
        })
    return out


def simulate(folds, choose, cost_per_side: float = COST_PER_SIDE) -> dict:
    """Compound one strategy across the folds, net of transaction costs.

    Args:
        folds: The rolling folds, in order.
        choose: ``fold -> weight vector`` for the coming holding period.
        cost_per_side: One-way cost rate; see ``folds.turnover_cost``.

    Returns:
        Equity curve, per-fold detail, and annualised summary statistics.
    """
    equity = 1.0
    prev_target = None
    curve, per_fold, all_daily, total_cost = [], [], [], 0.0

    for f in folds:
        w = np.asarray(choose(f), dtype=float)
        prices = _test_prices(f)

        cost = turnover_cost(prev_target, w, cost_per_side)
        total_cost += cost
        equity *= (1.0 - cost)

        rel = prices.to_numpy() / prices.to_numpy()[0]
        value = rel @ w
        daily = np.diff(value) / value[:-1]
        all_daily.extend(daily.tolist())

        gross = float(value[-1])
        equity *= gross
        curve.append(equity)

        per_fold.append({
            "fold": f.index,
            "weights": w.tolist(),
            "held": [f.ham.tickers[i] for i, x in enumerate(w) if x > 0],
            "cost": cost,
            "gross_return": gross - 1.0,
            "equity_after": equity,
        })
        prev_target = drifted_weights(prices, w)

    daily = np.asarray(all_daily)
    ann_ret = float(daily.mean() * TRADING_DAYS)
    ann_vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    years = len(daily) / TRADING_DAYS
    return {
        "final_equity": equity,
        "total_return": equity - 1.0,
        "cagr": float(equity ** (1.0 / years) - 1.0) if years > 0 else 0.0,
        # Gross-of-cost Sharpe: costs hit the equity curve, not the daily series.
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 1e-12 else 0.0,
        "total_cost": total_cost,
        "equity_curve": curve,
        "per_fold": per_fold,
    }


_PRICES = None


def _test_prices(f):
    """Price bars over fold ``f``'s holding period, from the frozen snapshot.

    Cached: the loader verifies the SHA-256 on every call, and simulating seven
    arms over sixteen folds would otherwise re-hash the file 112 times.
    """
    global _PRICES
    if _PRICES is None:
        from backtest.fetch_snapshot import load_backtest_prices
        _PRICES = load_backtest_prices()
    return _PRICES.loc[f.test_start:f.test_end]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("optimization").setLevel(logging.WARNING)
    logging.getLogger("q_bridge").setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=CVAR_ALPHA)
    ap.add_argument("--seed", type=int, default=QUANTUM_SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--skip-quantum", action="store_true",
                    help="classical arms only (fast; for iterating on reporting)")
    args = ap.parse_args()

    folds = build_folds()
    logger.info("Built %d folds.", len(folds))

    quantum = [] if args.skip_quantum else run_quantum_arm(
        folds, args.alpha, args.seed, args.epochs
    )

    tickers = folds[0].ham.tickers
    n = len(tickers)

    def pick_best(f):
        return f.is_best.weights

    def pick_ew4(f):
        return np.full(n, 1.0 / n)

    def pick_quantum(f):
        row = quantum[f.index]
        bits = [int(c) for c in row["readout"]]
        if row["feasible"]:
            return H.decode_weights(bits, f.ham.budget)
        # An infeasible readout is not a portfolio. Phase 2's repair policy
        # decides what to hold; recording it separately keeps the arm honest.
        return H.decode_with_repair(bits, f.mu, f.sigma, f.ham.budget)[0]

    arms = {
        "best_equal_weight": simulate(folds, pick_best),
        "equal_weight_4": simulate(folds, pick_ew4),
    }
    if quantum:
        arms["quantum_cvar040"] = simulate(folds, pick_quantum)
    for k, t in enumerate(tickers):
        w = np.zeros(n)
        w[k] = 1.0
        arms[f"buy_hold_{t}"] = simulate(folds, lambda f, w=w: w)

    payload = {
        "config": {
            "train_days": 250, "test_days": 63, "step_days": 63,
            "n_folds": len(folds), "cost_per_side": COST_PER_SIDE,
            "risk_free": RISK_FREE_RATE,
            "cvar_alpha": args.alpha, "quantum_seed": args.seed,
            "epochs": args.epochs, "k_restarts": 1,
        },
        "folds": [{
            "index": f.index,
            "train_start": str(f.train_start.date()), "train_end": str(f.train_end.date()),
            "test_start": str(f.test_start.date()), "test_end": str(f.test_end.date()),
            "ground_bitstring": f.ground_bitstring,
            "runner_up_bitstring": f.runner_up_bitstring,
            "energy_gap": f.energy_gap,
            "spectral_width": f.spectral_width,
            "gap_fraction": f.gap_fraction,
            "ground_feasible": f.ground_feasible,
            "portfolios": [{
                "bitstring": p.bitstring, "held": list(p.held),
                "energy": p.energy,
                "is_return": p.is_return, "is_vol": p.is_vol, "is_sharpe": p.is_sharpe,
                "oos_total_return": p.oos_total_return,
                "oos_ann_vol": p.oos_ann_vol, "oos_sharpe": p.oos_sharpe,
            } for p in f.portfolios],
            "ew4_total_return": f.ew4_total_return,
            "ew4_sharpe": f.ew4_sharpe,
            "single_asset": f.single_asset,
        } for f in folds],
        "quantum": quantum,
        "arms": arms,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", RESULTS_JSON)

    print()
    print(f"{'arm':<24} {'total':>9} {'CAGR':>8} {'vol':>7} {'Sharpe':>8} {'cost':>7}")
    for name, a in arms.items():
        print(f"{name:<24} {a['total_return']:>+8.1%} {a['cagr']:>+7.1%} "
              f"{a['ann_vol']:>6.1%} {a['sharpe']:>+8.3f} {a['total_cost']:>6.2%}")
    if quantum:
        m = sum(q["matched"] for q in quantum)
        inf = sum(not q["feasible"] for q in quantum)
        print(f"\nquantum arm: matched fold argmin {m}/{len(quantum)}, "
              f"infeasible readouts {inf}")
    print("\nBACKTEST RUN: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
