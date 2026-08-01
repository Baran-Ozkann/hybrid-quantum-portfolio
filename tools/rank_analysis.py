"""
tools/rank_analysis.py -- does CVaR concentrate toward the right answer, or
just produce noisy readouts that land right more often?

In the CVaR regime a "correct" run means only that the ground state is the
LARGEST of sixteen, and it can win with P = 0.32 from a near-diffuse
distribution. Success counting alone cannot distinguish two very different
situations:

  skill  -- the ground state is consistently near the top of the distribution,
            and failures are near misses where it sits second by a hair.
  luck   -- its rank scatters across the failures while successes are thin wins,
            i.e. the arm produces noisy readouts that happen to land right more
            often than the baseline.

The discriminator is the FULL RANK of the ground state in every run, and its
mean rank per arm -- a single number that is robust to the argmax being a thin
win, and the fairest cross-arm comparison available given that the CVaR arms
operate in a different regime from the plain-VQE arms.

Also reports the "correct AND resolvable at 4096 shots" criterion alongside raw
accuracy. On hardware shot noise is the binding constraint, so an exact-backend
accuracy that does not survive sampling is not an accuracy anyone can use.

Run::

    python tools/rank_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H          # noqa: E402
from quantum import q_bridge as qb              # noqa: E402
from training import optimization as opt        # noqa: E402

#: Shots the UI actually draws; the margin must be resolvable at this count.
SHOTS = 4096
#: Significance required of the top-two margin for a readout to count as
#: resolvable under sampling.
SIGMA_THRESHOLD = 5.0

ARMS = [
    ("plain (lr_constant_200)", "lr_constant_200"),
    ("CVaR a=0.6", "cvar_a060"),
    ("CVaR a=0.5", "cvar_a050"),
    ("CVaR a=0.4", "cvar_a040"),
    ("CVaR a=0.1", "cvar_a010"),
]
SEEDS = [42, 123, 456] + list(range(1000, 1027))


#: Multinomial draws per run for the shot-limited accuracy estimate.
N_TRIALS = 1000
RNG = np.random.default_rng(20260801)


def margin_sigma(p1: float, p2: float, shots: int = SHOTS) -> float:
    """Top-two margin expressed in standard deviations of shot noise.

    A useful diagnostic but an OPTIMISTIC one: it treats the readout as a
    two-way contest, whereas the sampled argmax competes against all 2^n
    outcomes at once. Use ``shot_limited_accuracy`` for the real figure.
    """
    var = (p1 + p2 - (p1 - p2) ** 2) / shots
    return (p1 - p2) / math.sqrt(max(var, 1e-18))


def shot_limited_accuracy(probs: np.ndarray, ground_idx: int,
                          shots: int = SHOTS, trials: int = N_TRIALS) -> float:
    """P(the sampled argmax is the ground state) at a finite shot count.

    Simulated rather than derived from a margin threshold, for two reasons.
    The argmax over sampled counts can lose to ANY of the competing outcomes,
    not just the runner-up -- and with fragmented mass across fourteen minor
    states that matters. And "robust" is a statement about an expectation, so
    it should be measured as one instead of read off a cutoff.

    Ties are shared rather than broken by index order, which would silently
    favour whichever state happens to sort first.
    """
    counts = RNG.multinomial(shots, probs, size=trials)
    top = counts.max(axis=1, keepdims=True)
    is_top = counts == top
    n_tied = is_top.sum(axis=1)
    ground_is_top = is_top[:, ground_idx]
    return float(np.sum(ground_is_top / n_tied) / trials)


def analyse(tag: str, ground_idx: int) -> dict | None:
    directory = opt.arm_dir(tag)
    if not list(directory.glob("trained_seed*.json")):
        return None

    rows = {}
    for s in SEEDS:
        try:
            r = opt.load_result(s, tag)
        except FileNotFoundError:
            continue
        p = np.asarray(r.final_probabilities, dtype=float)
        order = np.argsort(p)[::-1]
        # Rank of the ground state, 1 = most probable.
        rank = int(np.where(order == ground_idx)[0][0]) + 1
        p1, p2 = float(p[order[0]]), float(p[order[1]])
        sig = margin_sigma(p1, p2)
        rows[s] = {
            "rank": rank,
            "p_ground": float(p[ground_idx]),
            "margin": p1 - p2,
            "margin_sigma": sig,
            "correct": rank == 1,
            "resolvable": sig >= SIGMA_THRESHOLD,
            "robust": rank == 1 and sig >= SIGMA_THRESHOLD,
            "shot_accuracy": shot_limited_accuracy(p, ground_idx),
            "final_energy": r.final_energy,
        }
    if not rows:
        return None

    ranks = np.array([v["rank"] for v in rows.values()])
    correct = np.array([v["correct"] for v in rows.values()])
    robust = np.array([v["robust"] for v in rows.values()])
    shot_acc = np.array([v["shot_accuracy"] for v in rows.values()])
    return {
        "tag": tag, "n": len(rows), "rows": rows,
        "n_correct": int(correct.sum()),
        "n_robust": int(robust.sum()),
        "shot_accuracy_mean": float(shot_acc.mean()),
        "shot_accuracy_median": float(np.median(shot_acc)),
        "shot_accuracy_min": float(shot_acc.min()),
        "n_shot_below_0.9": int((shot_acc < 0.9).sum()),
        "n_shot_below_0.5": int((shot_acc < 0.5).sum()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "worst_rank": int(ranks.max()),
        "rank_hist": {int(k): int(v) for k, v in
                      zip(*np.unique(ranks, return_counts=True))},
        "rank_when_wrong": sorted(int(v["rank"]) for v in rows.values()
                                  if not v["correct"]),
    }


def main() -> int:
    ham = H.load_hamiltonian()
    spectrum = sorted(H.enumerate_states(ham), key=lambda r: r["ising_energy"])
    g_bits = spectrum[0]["bitstring"]
    ground_idx = qb.bitstring_to_index([int(c) for c in g_bits], ham.n_assets)

    results = {}
    for label, tag in ARMS:
        a = analyse(tag, ground_idx)
        if a:
            results[label] = a

    print("=" * 92)
    print(f"RANK ANALYSIS -- where does the ground state {g_bits} sit in each "
          f"run's distribution?")
    print("=" * 92)
    print(f"  rank 1 = most probable (a correct readout). Resolvable means the "
          f"top-two margin\n  exceeds {SIGMA_THRESHOLD:.0f} sigma of shot noise "
          f"at {SHOTS} shots.\n")

    print(f"  {'arm':<26}{'correct':>9}{'robust':>9}{'mean rank':>11}"
          f"{'median':>8}{'worst':>7}   rank when wrong")
    for label, a in results.items():
        print(f"  {label:<26}{a['n_correct']:>4}/{a['n']:<4}"
              f"{a['n_robust']:>4}/{a['n']:<4}{a['mean_rank']:>11.3f}"
              f"{a['median_rank']:>8.1f}{a['worst_rank']:>7}   "
              f"{a['rank_when_wrong']}")

    print("\n  Rank histogram (how many runs place the ground state at each rank):")
    all_ranks = sorted({r for a in results.values() for r in a["rank_hist"]})
    header = "".join(f"{r:>6}" for r in all_ranks)
    print(f"  {'arm':<26}{header}")
    for label, a in results.items():
        cells = "".join(f"{a['rank_hist'].get(r, 0):>6}" for r in all_ranks)
        print(f"  {label:<26}{cells}")

    # --- the two criteria side by side -----------------------------------
    print("\n" + "=" * 92)
    print("TWO CRITERIA -- the ranking depends on which is applied")
    print("=" * 92)
    print(f"  {'arm':<26}{'exact':>9}{'E[shot acc]':>13}{'median':>9}"
          f"{'min':>8}{'<0.9':>6}{'<0.5':>6}{'min margin':>12}")
    for label, a in results.items():
        margins = [v["margin"] for v in a["rows"].values() if v["correct"]]
        print(f"  {label:<26}{a['n_correct'] / a['n']:>8.1%}"
              f"{a['shot_accuracy_mean']:>13.1%}{a['shot_accuracy_median']:>9.1%}"
              f"{a['shot_accuracy_min']:>8.1%}"
              f"{a['n_shot_below_0.9']:>6}{a['n_shot_below_0.5']:>6}"
              f"{min(margins):>12.4f}")
    print(f"\n  E[shot acc] is the mean over runs of P(sampled argmax = ground "
          f"state) at {SHOTS}\n  shots, from {N_TRIALS} multinomial draws per "
          f"run. It competes against ALL {2 ** 4} outcomes,\n  so it is stricter "
          f"than a pairwise margin and needs no cutoff to be chosen.")

    # --- which seeds recover, and do the stable ones have better margins? --
    if "plain (lr_constant_200)" in results:
        base = results["plain (lr_constant_200)"]["rows"]
        rec = {}
        for label, a in results.items():
            if label.startswith("plain"):
                continue
            rec[label] = {
                "recovered": sorted(s for s in a["rows"]
                                    if a["rows"][s]["correct"] and not base[s]["correct"]),
                "lost": sorted(s for s in a["rows"]
                               if not a["rows"][s]["correct"] and base[s]["correct"]),
            }
        print("\n" + "=" * 92)
        print("WHICH RUNS RECOVER -- alpha changes the identity of the successes, "
              "not only their count")
        print("=" * 92)
        for label, d in rec.items():
            print(f"  {label:<26} recovered {d['recovered']}")
            print(f"  {'':<26} lost      {d['lost'] or '[]'}")

        cvar = [l for l in rec if "CVaR" in l]
        if len(cvar) >= 2:
            sets = {l: set(rec[l]["recovered"]) for l in cvar}
            core = set.intersection(*sets.values())
            union = set.union(*sets.values())
            print(f"\n  stable core (recovered in EVERY CVaR arm): {sorted(core)}")
            print(f"  fringe (recovered in some but not all)   : "
                  f"{sorted(union - core)}")
            for label in cvar:
                a = results[label]
                cm = [a["rows"][s]["margin"] for s in core if s in a["rows"]]
                fm = [a["rows"][s]["margin"] for s in (union - core)
                      if s in a["rows"] and a["rows"][s]["correct"]]
                if cm and fm:
                    print(f"    {label:<24} median margin  core {np.median(cm):.4f}  "
                          f"fringe {np.median(fm):.4f}")

    dest = PROJECT_ROOT / "logs" / "rank_analysis.json"
    dest.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "rows"} |
            {"rows": {str(s): r for s, r in v["rows"].items()}}
         for k, v in results.items()}, indent=2), encoding="utf-8")
    print(f"\n  written -> {dest.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
