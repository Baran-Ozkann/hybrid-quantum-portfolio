"""Does the budget penalty swamp the objective it is meant to constrain?

THE HYPOTHESIS. The Hamiltonian's energy scale is |E| ~ 604 while the top-two
FEASIBLE gap is ~0.06 -- a ratio of 1e-4. Both trained arms learned the
constraint essentially perfectly (feasible mass 1.0000) and got the allocation
wrong, and both drove the true optimum to exactly zero amplitude from a
random-parameter baseline of ~1e-3. If the gradient signal for satisfying the
constraint is four orders of magnitude larger than the signal for discriminating
among the 120 feasible states, that is exactly what one would expect.

WHY THIS DID NOT ARISE AT 4 ASSETS. There, the contest was between 1010 and
0110 -- both feasible. The penalty contributed identically to each and
cancelled out of the comparison entirely. Here lambda shapes the landscape the
feasible states live in, so it does not cancel.

This module answers, by enumeration and for free:

  1. what the true minimal lambda is (the sufficient bound used to build the
     Hamiltonian is conservative), and
  2. how the scale ratio, spectral width, and ground-state identity move as
     lambda varies.

The scale problem may be structural rather than a bad choice of lambda: level
sums run 0..18 against a budget of 4, so violations reach 14 and the penalty is
quadratic. 14^2 = 196 multiplies whatever lambda is chosen.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from varweight import encoding as enc              # noqa: E402
from varweight import hamiltonian as vh            # noqa: E402
from varweight.data import statistics              # noqa: E402

logger = logging.getLogger("probe_penalty")


def minimal_lambda(mu, sigma, risk_aversion=vh.DEFAULT_RISK_AVERSION,
                   budget=enc.BUDGET) -> tuple[float, float]:
    """Exact smallest lambda keeping the feasible optimum globally optimal.

    For an infeasible state with violation ``v >= 1``, feasibility of the
    optimum requires ``obj(s) + lambda v^2 > obj(best_feasible)``, so

        lambda > max over infeasible s of (obj(best_feasible) - obj(s)) / v^2

    The conservative bound in ``hamiltonian.sufficient_penalty`` uses ``v = 1``
    and the full objective range, which is much larger.

    Returns:
        ``(exact_minimum, conservative_bound_used_by_default)``.
    """
    L = enc.level_matrix()
    best_feas = np.inf
    worst_need = -np.inf
    for idx in range(enc.N_STATES):
        bits = np.asarray(enc.index_to_bits(idx), dtype=float)
        lv = L @ bits
        obj = vh.objective_from_levels(lv, mu, sigma, risk_aversion, budget)
        if int(lv.sum()) == budget:
            best_feas = min(best_feas, obj)

    for idx in range(enc.N_STATES):
        bits = np.asarray(enc.index_to_bits(idx), dtype=float)
        lv = L @ bits
        v = int(lv.sum()) - budget
        if v == 0:
            continue
        obj = vh.objective_from_levels(lv, mu, sigma, risk_aversion, budget)
        worst_need = max(worst_need, (best_feas - obj) / float(v * v))

    conservative = vh.sufficient_penalty(mu, sigma, risk_aversion, budget,
                                         scale=1.0)
    return float(max(worst_need, 0.0)), float(conservative)


def describe(mu, sigma, penalty: float) -> dict:
    """Spectrum diagnostics for one choice of lambda."""
    ham = vh.build_hamiltonian(mu, sigma, penalty=penalty)
    states = vh.enumerate_states(ham)
    E = np.array([s["ising_energy"] for s in states])
    feas = [s for s in states if s["feasible"]]
    fE = np.array([s["ising_energy"] for s in feas])
    return {
        "penalty": penalty,
        "ground_bitstring": states[0]["bitstring"],
        "ground_levels": states[0]["levels"],
        "ground_feasible": states[0]["feasible"],
        "gap": float(fE[1] - fE[0]),
        "full_width": float(E.max() - E.min()),
        "feasible_width": float(fE.max() - fE.min()),
        "abs_max": float(np.abs(E).max()),
        "gap_over_absmax": float((fE[1] - fE[0]) / np.abs(E).max()),
        "gap_over_feaswidth": float((fE[1] - fE[0]) / (fE.max() - fE.min())),
        "n_infeasible_below_best_feasible": int(
            sum(1 for s in states if not s["feasible"]
                and s["ising_energy"] < fE.min())
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.5, 1.0, 0.5, 0.25, 0.1, 0.05])
    args = ap.parse_args()

    mu, sigma, _ = statistics()
    exact, conservative = minimal_lambda(mu, sigma)

    print("=" * 88)
    print("PENALTY SCALE PROBE")
    print("=" * 88)
    print(f"\n  conservative bound (used by default, x1.5) : "
          f"{conservative:.6f}  -> lambda = {1.5*conservative:.6f}")
    print(f"  EXACT minimum lambda for a feasible ground  : {exact:.6f}")
    print(f"  the default is {1.5*conservative/exact:.1f}x larger than needed"
          if exact > 0 else "  (no infeasible state can ever win)")

    max_violation = enc.N_ASSETS * (enc.LEVELS_PER_ASSET - 1) - enc.BUDGET
    print(f"\n  level sums run 0..{enc.N_ASSETS*(enc.LEVELS_PER_ASSET-1)} "
          f"against budget {enc.BUDGET}, so violations reach {max_violation}")
    print(f"  the penalty is quadratic, so ANY lambda is multiplied by up to "
          f"{max_violation**2}")
    print(f"  => even at the exact minimum, the penalty spans "
          f"{exact*max_violation**2:.1f} in energy")

    print(f"\n{'lambda':>10} {'x min':>7} {'ground':>14} {'feas?':>6} "
          f"{'gap':>9} {'|E|max':>9} {'gap/|E|':>10} {'gap/feasW':>10} {'inf<feas':>9}")
    rows = []
    for sc in args.scales:
        lam = sc * conservative
        d = describe(mu, sigma, lam)
        rows.append(d)
        print(f"{lam:>10.4f} {lam/exact if exact else float('nan'):>7.2f} "
              f"{d['ground_bitstring']:>14} "
              f"{'yes' if d['ground_feasible'] else 'NO':>6} "
              f"{d['gap']:>9.6f} {d['abs_max']:>9.1f} "
              f"{d['gap_over_absmax']:>10.2e} {d['gap_over_feaswidth']:>10.2%} "
              f"{d['n_infeasible_below_best_feasible']:>9}")

    print("\n  Reading: 'gap' is the top-two FEASIBLE gap and is invariant in lambda")
    print("  (both states satisfy the budget, so the penalty cancels between them).")
    print("  What lambda changes is the SCALE the optimiser sees it against.")
    print("\n  4-asset reference: |E|max 11.64, gap 0.039209, gap/|E| 3.37e-03")
    print("\nPENALTY PROBE: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
