"""When does a run's readout stop changing? -- epoch-count convergence.

Every accuracy figure in this project is quoted at a fixed epoch budget, and
that budget is an arbitrary cut unless runs have stopped moving by the end of
it. Phase 7 found seed 456 crossing basins at epoch 176 of 200 and holding for
only 24 epochs, which means 200 is a cut sitting underneath a published number.

This tool answers three questions for a pair of arms:

  1. Does the headline rate move when the budget is extended?
  2. Which seeds change basin, and at which epoch?
  3. Is anything STILL crossing late in the extended window -- i.e. are there
     runs whose readout last changed after the shorter arm had already stopped?

(3) is the one that decides whether the longer budget is itself converged. If
runs are still crossing in the final stretch, the new number is no safer than
the old one and the honest statement is that neither is converged.

``settle_epoch`` is the epoch at which the readout LAST changed, deliberately
not the first epoch matching the final answer: a run oscillating
1010 -> 0110 -> 1010 would otherwise report an early settling epoch it never held.
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

from training.optimization import load_result  # noqa: E402

logger = logging.getLogger("settle_analysis")

SEEDS = [42, 123, 456] + list(range(1000, 1027))


def collect(tag: str | None, seeds=SEEDS) -> dict[int, dict]:
    out = {}
    for s in seeds:
        try:
            r = load_result(s, tag)
        except Exception:
            continue
        epochs = int(r.config.get("epochs", len(r.energy_history)))
        out[s] = {
            "readout": r.mode_bitstring,
            "correct": bool(r.found_ground_state),
            "settle": int(r.settle_epoch),
            "epochs": epochs,
            "held": epochs - int(r.settle_epoch) if r.settle_epoch >= 0 else epochs,
            "energy": float(r.final_energy),
            "p_ground": float(r.ground_state_prob),
        }
    return out


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="longer arm tag")
    ap.add_argument("--baseline", required=True, help="shorter arm tag")
    ap.add_argument("--late-window", type=int, default=None,
                    help="count settles after this epoch (default: baseline length)")
    args = ap.parse_args()

    base = collect(args.baseline)
    arm = collect(args.arm)
    shared = sorted(set(base) & set(arm))
    if not shared:
        print(f"No shared seeds between {args.baseline!r} and {args.arm!r}")
        return 1

    b_ep = max(v["epochs"] for v in base.values())
    a_ep = max(v["epochs"] for v in arm.values())
    late_after = args.late_window if args.late_window is not None else b_ep

    b_correct = sum(base[s]["correct"] for s in shared)
    a_correct = sum(arm[s]["correct"] for s in shared)

    print("=" * 78)
    print(f"SETTLE ANALYSIS   {args.baseline} ({b_ep} ep)  vs  {args.arm} ({a_ep} ep)")
    print("=" * 78)
    print(f"  correct: {b_correct}/{len(shared)} -> {a_correct}/{len(shared)}"
          f"   ({b_correct/len(shared):.1%} -> {a_correct/len(shared):.1%})")

    changed = [s for s in shared if base[s]["readout"] != arm[s]["readout"]]
    print(f"\n  Seeds whose READOUT changed between arms: {len(changed)}")
    if changed:
        print(f"    {'seed':>6} {'was':>6} {'now':>6} {'settle(base)':>13} "
              f"{'settle(arm)':>12} {'held':>6}  effect")
        for s in changed:
            eff = ("RECOVERED" if arm[s]["correct"] and not base[s]["correct"]
                   else "REGRESSED" if base[s]["correct"] and not arm[s]["correct"]
                   else "wrong->wrong")
            print(f"    {s:>6} {base[s]['readout']:>6} {arm[s]['readout']:>6} "
                  f"{base[s]['settle']:>13} {arm[s]['settle']:>12} "
                  f"{arm[s]['held']:>6}  {eff}")
    else:
        print("    (none -- every seed reached the same answer under both budgets)")

    print(f"\n  Settle epochs in {args.arm}:")
    settles = np.array([arm[s]["settle"] for s in shared])
    never = int((settles < 0).sum())
    moved = settles[settles >= 0]
    if len(moved):
        print(f"    median {np.median(moved):.0f}   max {moved.max()}   "
              f"never changed: {never}")
    print(f"    runs whose readout last changed AFTER epoch {late_after}: "
          f"{int((settles > late_after).sum())}")

    late = [s for s in shared if arm[s]["settle"] > late_after]
    if late:
        print(f"\n  *** STILL CROSSING LATE -- {args.arm} is NOT converged either ***")
        for s in late:
            print(f"    seed {s}: settled at {arm[s]['settle']} of {a_ep}, "
                  f"held {arm[s]['held']} epochs, "
                  f"readout {arm[s]['readout']} "
                  f"({'correct' if arm[s]['correct'] else 'WRONG'})")
    else:
        print(f"\n  No run's readout changed after epoch {late_after}.")
        tail = sorted(shared, key=lambda s: arm[s]["settle"], reverse=True)[:3]
        print(f"  Latest settles: " + ", ".join(
            f"seed {s} @ {arm[s]['settle']} (held {arm[s]['held']})" for s in tail))

    # The shortest hold is the fragility measure: a run that settled 5 epochs
    # before the end is one that could plausibly move again.
    holds = sorted((arm[s]["held"], s) for s in shared if arm[s]["settle"] >= 0)
    if holds:
        print(f"\n  Shortest holds in {args.arm} (fragility):")
        for h, s in holds[:5]:
            print(f"    seed {s:>5}: held {h:>4} epochs after settling at "
                  f"{arm[s]['settle']}  ({'correct' if arm[s]['correct'] else 'WRONG'})")

    print("\nSETTLE ANALYSIS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
