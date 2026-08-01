"""
tools/compare_arms.py -- paired comparison between two experiment arms.

Every arm trains the SAME 30 seeds against the SAME frozen snapshot and
Hamiltonian, changing exactly one thing. Because the seeds are shared the
comparison is PAIRED, so it is analysed with McNemar's test on the discordant
pairs rather than as two independent proportions. At n = 30 the paired design is
the only thing giving the comparison any power, and it also answers a question
two independent rates cannot: does the new arm fix the *same* runs the baseline
got wrong, or a different set?

McNemar uses the EXACT BINOMIAL test, not the chi-square approximation: the
discordant count here is single digits, where the asymptotic version is
unreliable.

Usage::

    python tools/compare_arms.py --arm lr_cosine_200
    python tools/compare_arms.py --arm cvar_a0.50 --baseline lr_cosine_200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H          # noqa: E402
from quantum import q_bridge as qb              # noqa: E402
from training import optimization as opt        # noqa: E402
from tools.stability_report import wilson, selection_curve, K_VALUES  # noqa: E402

#: Fraction of the spectral width within which a run counts as "converged".
CONVERGED_TOL_FRACTION = 0.005


def load_arm(tag: str | None) -> dict[int, object]:
    """Load every saved run for an arm, keyed by seed."""
    directory = opt.arm_dir(tag)
    runs = {}
    for path in sorted(directory.glob("trained_seed*.json")):
        m = re.search(r"trained_seed(\d+)\.json$", path.name)
        if m:
            runs[int(m.group(1))] = opt.load_result(int(m.group(1)), tag)
    return runs


def readout_margins(runs: dict[int, object], ground_idx: int,
                    runner_idx: int) -> dict[int, dict]:
    """Per-run readout robustness, computed from the final distribution.

    Under the PLAIN objective these quantities are implied: runs collapse into
    the two lowest-energy states, so the readout margin is just
    ``|P(ground) - P(runner-up)|`` and correctness is equivalent to
    ``P(ground) > 0.5``.

    Under CVaR that equivalence breaks. CVaR is indifferent to how the upper
    ``1 - alpha`` of the distribution is arranged, so mass fragments and the
    argmax can win with well under 0.5 -- seed 1006 reads out correctly at
    P(ground) = 0.416. An argmax held at 0.42 against fifteen scattered
    competitors is a materially weaker result than the same argmax at 0.60
    against a single competitor, and it is the first thing that would degrade
    under finite shots. So the margin is measured rather than inferred.
    """
    out = {}
    for seed, r in runs.items():
        p = np.asarray(r.final_probabilities, dtype=float)
        order = np.argsort(p)[::-1]
        top, second = int(order[0]), int(order[1])
        out[seed] = {
            "p_ground": float(p[ground_idx]),
            "p_runner_up_state": float(p[runner_idx]),
            "p_mode": float(p[top]),
            "p_second": float(p[second]),
            "readout_margin": float(p[top] - p[second]),
            "two_state_mass": float(p[ground_idx] + p[runner_idx]),
        }
    return out


def arm_stats(runs: dict[int, object], ground_energy: float,
              spectral_width: float, gap: float,
              ground_idx: int = 10, runner_idx: int = 6) -> dict:
    """All per-arm statistics, in the same structure for every arm."""
    seeds = sorted(runs)
    energies = np.array([runs[s].final_energy for s in seeds])
    correct = np.array([runs[s].found_ground_state for s in seeds])
    gs_probs = np.array([runs[s].ground_state_prob for s in seeds])
    mode_probs = np.array([runs[s].mode_probability for s in seeds])
    modes = [runs[s].mode_bitstring for s in seeds]

    n = len(seeds)
    n_correct = int(correct.sum())
    rate, lo, hi = wilson(n_correct, n)

    cut = ground_energy + CONVERGED_TOL_FRACTION * spectral_width
    conv = energies <= cut
    n_conv = int(conv.sum())
    n_conv_correct = int((conv & correct).sum())
    conv_rate, conv_lo, conv_hi = wilson(n_conv_correct, n_conv) if n_conv else (0, 0, 1)

    # Does the empty band that made the Phase 7 partition threshold-independent
    # survive in this arm?
    conv_worst = float(energies[conv].max()) if n_conv else float("nan")
    fail_best = float(energies[~conv].min()) if n_conv < n else float("nan")
    band = (fail_best - conv_worst) if (0 < n_conv < n) else None

    # The argmax guarantee: p(ground) >= 0.5 forces the ground state to be the
    # mode, because the other 2^n - 1 states sum to at most 0.5. For any arm
    # whose objective saturates at or above 0.5 this fraction must EQUAL the
    # readout accuracy; a mismatch means the theory or the code is wrong.
    n_ge_half = int((gs_probs >= 0.5).sum())

    margins = readout_margins(runs, ground_idx, runner_idx)
    margin_vals = np.array([margins[s]["readout_margin"] for s in seeds])
    two_state = np.array([margins[s]["two_state_mass"] for s in seeds])

    by_energy = np.argsort(energies)
    curves = {
        "energy_with_replacement": selection_curve(by_energy, correct, True),
        "energy_without_replacement": selection_curve(by_energy, correct, False),
    }

    return {
        "seeds": seeds,
        "n": n,
        "n_correct": n_correct,
        "rate": rate,
        "wilson_95": [lo, hi],
        "attractors": dict(Counter(modes)),
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std(ddof=0)),
        "energy_spread": float(np.ptp(energies)),
        "energy_spread_over_gap": float(np.ptp(energies) / gap),
        "converged": {
            "cut": cut, "n": n_conv, "n_correct": n_conv_correct,
            "rate": conv_rate, "wilson_95": [conv_lo, conv_hi],
            "spread": float(np.ptp(energies[conv])) if n_conv else 0.0,
            "worst_energy": conv_worst,
        },
        "failed": {
            "n": n - n_conv,
            "n_correct": int((~conv & correct).sum()),
            "best_energy": fail_best,
        },
        "empty_band": band,
        "empty_band_over_gap": (band / gap) if band is not None else None,
        "p_ground_mean": float(gs_probs.mean()),
        "p_ground_median": float(np.median(gs_probs)),
        "n_p_ground_ge_half": n_ge_half,
        "argmax_guarantee_holds": n_ge_half == n_correct,
        "mode_prob_mean": float(mode_probs.mean()),
        "readout_margin_median": float(np.median(margin_vals)),
        "readout_margin_min": float(margin_vals.min()),
        "n_margin_below_0.05": int((margin_vals < 0.05).sum()),
        "two_state_mass_median": float(np.median(two_state)),
        "two_state_mass_min": float(two_state.min()),
        "n_two_state_below_0.5": int((two_state < 0.5).sum()),
        "margins": {int(s): margins[s] for s in seeds},
        "multi_start": curves,
        "per_seed": {
            int(s): {
                "final_energy": runs[s].final_energy,
                "ground_state_prob": runs[s].ground_state_prob,
                "mode_probability": runs[s].mode_probability,
                "mode_bitstring": runs[s].mode_bitstring,
                "correct": bool(runs[s].found_ground_state),
                "converged": bool(runs[s].final_energy <= cut),
            }
            for s in seeds
        },
    }


def mcnemar(base: dict, arm: dict) -> dict:
    """Exact-binomial McNemar test on the shared seeds."""
    shared = sorted(set(base["per_seed"]) & set(arm["per_seed"]))
    both = base_only = arm_only = neither = 0
    recovered, lost = [], []
    for s in shared:
        b = base["per_seed"][s]["correct"]
        a = arm["per_seed"][s]["correct"]
        if b and a:
            both += 1
        elif b and not a:
            base_only += 1
            lost.append(s)
        elif a and not b:
            arm_only += 1
            recovered.append(s)
        else:
            neither += 1

    discordant = base_only + arm_only
    if discordant:
        p = binomtest(min(base_only, arm_only), discordant, 0.5,
                      alternative="two-sided").pvalue
    else:
        p = 1.0
    return {
        "n_shared": len(shared),
        "both_correct": both,
        "baseline_only": base_only,
        "arm_only": arm_only,
        "neither": neither,
        "discordant": discordant,
        "p_value_exact_binomial": float(p),
        "recovered_seeds": recovered,
        "lost_seeds": lost,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired comparison of two arms")
    parser.add_argument("--arm", required=True, help="arm tag to evaluate")
    parser.add_argument("--baseline", default=None,
                        help="baseline arm tag (default: the committed baseline)")
    parser.add_argument("--json", action="store_true", help="dump full JSON")
    args = parser.parse_args()

    ham = H.load_hamiltonian()
    spectrum = sorted(H.enumerate_states(ham), key=lambda r: r["ising_energy"])
    ground_e = spectrum[0]["ising_energy"]
    width = spectrum[-1]["ising_energy"] - ground_e
    gap = spectrum[1]["ising_energy"] - ground_e

    base_runs = load_arm(args.baseline)
    arm_runs = load_arm(args.arm)
    if not arm_runs:
        print(f"No runs found for arm {args.arm!r}")
        return 1

    g_idx = qb.bitstring_to_index([int(c) for c in spectrum[0]["bitstring"]], 4)
    r_idx = qb.bitstring_to_index([int(c) for c in spectrum[1]["bitstring"]], 4)
    base = arm_stats(base_runs, ground_e, width, gap, g_idx, r_idx)
    arm = arm_stats(arm_runs, ground_e, width, gap, g_idx, r_idx)
    paired = mcnemar(base, arm)

    bl = args.baseline or "baseline"
    print("=" * 78)
    print(f"PAIRED COMPARISON   {bl}  vs  {args.arm}")
    print("=" * 78)
    print(f"  ground state {spectrum[0]['bitstring']} E={ground_e:.6f}, "
          f"gap {gap:.6f}, spectral width {width:.4f}")

    print(f"\n{'':32}{bl:>20}{args.arm:>20}")
    def row(label, a, b, fmt="{:.4f}"):
        print(f"  {label:<30}{fmt.format(a):>20}{fmt.format(b):>20}")
    row("runs", base["n"], arm["n"], "{:d}")
    row("reached ground state", base["n_correct"], arm["n_correct"], "{:d}")
    row("success rate", base["rate"], arm["rate"], "{:.1%}")
    # Built outside the f-string: nesting the same quote character inside an
    # f-string expression requires Python 3.12, and this project targets 3.11.
    base_ci = "[{:.1%}, {:.1%}]".format(*base["wilson_95"])
    arm_ci = "[{:.1%}, {:.1%}]".format(*arm["wilson_95"])
    print(f"  {'Wilson 95%':<30}{base_ci:>20}{arm_ci:>20}")
    row("mean final <H>", base["energy_mean"], arm["energy_mean"], "{:.6f}")
    row("energy spread", base["energy_spread"], arm["energy_spread"], "{:.6f}")
    row("spread / gap", base["energy_spread_over_gap"], arm["energy_spread_over_gap"], "{:.2f}x")
    row("converged (<=0.5%)", base["converged"]["n"], arm["converged"]["n"], "{:d}")
    row("  of which correct", base["converged"]["n_correct"], arm["converged"]["n_correct"], "{:d}")
    row("  converged accuracy", base["converged"]["rate"], arm["converged"]["rate"], "{:.1%}")
    row("  converged spread", base["converged"]["spread"], arm["converged"]["spread"], "{:.6f}")
    row("failed to optimise", base["failed"]["n"], arm["failed"]["n"], "{:d}")
    row("mean P(ground)", base["p_ground_mean"], arm["p_ground_mean"], "{:.4f}")
    row("runs with P(ground)>=0.5", base["n_p_ground_ge_half"], arm["n_p_ground_ge_half"], "{:d}")

    print("\n  Empty band between converged and failed groups:")
    for name, st in ((bl, base), (args.arm, arm)):
        if st["empty_band"] is None:
            print(f"    {name:<24} n/a (one group is empty)")
        else:
            print(f"    {name:<24} worst converged {st['converged']['worst_energy']:.6f}, "
                  f"best failed {st['failed']['best_energy']:.6f}, "
                  f"band {st['empty_band']:.6f} = {st['empty_band_over_gap']:.2f}x gap")

    print("\n  Argmax guarantee  (P(ground)>=0.5 must equal readout accuracy):")
    for name, st in ((bl, base), (args.arm, arm)):
        print(f"    {name:<24} P>=0.5 in {st['n_p_ground_ge_half']:>2} runs, "
              f"correct in {st['n_correct']:>2} -> "
              f"{'consistent' if st['argmax_guarantee_holds'] else 'MISMATCH'}")

    print("\n  Readout robustness (margin = P(mode) - P(second best)):")
    for name, st in ((bl, base), (args.arm, arm)):
        print(f"    {name:<24} margin median {st['readout_margin_median']:.4f}, "
              f"min {st['readout_margin_min']:.4f}, "
              f"{st['n_margin_below_0.05']} runs below 0.05")
    print("\n  Two-state mass  P(ground) + P(runner-up):")
    for name, st in ((bl, base), (args.arm, arm)):
        print(f"    {name:<24} median {st['two_state_mass_median']:.4f}, "
              f"min {st['two_state_mass_min']:.4f}, "
              f"{st['n_two_state_below_0.5']} runs below 0.5")

    print("\n  Attractors:")
    for name, st in ((bl, base), (args.arm, arm)):
        print(f"    {name:<24} {st['attractors']}")

    print("\n  Paired (McNemar, exact binomial):")
    print(f"    both correct        {paired['both_correct']}")
    print(f"    {bl} only{'':<8} {paired['baseline_only']}  (lost: {paired['lost_seeds']})")
    print(f"    {args.arm} only{'':<6} {paired['arm_only']}  (recovered: {paired['recovered_seeds']})")
    print(f"    neither             {paired['neither']}")
    print(f"    discordant          {paired['discordant']}")
    print(f"    exact p-value       {paired['p_value_exact_binomial']:.4f}")

    print("\n  Multi-start, keep lowest energy (with replacement):")
    print(f"    {'k':>4}{bl:>20}{args.arm:>20}")
    cb = {r["k"]: r for r in base["multi_start"]["energy_with_replacement"]}
    ca = {r["k"]: r for r in arm["multi_start"]["energy_with_replacement"]}
    for k in sorted(cb):
        print(f"    {k:>4}{cb[k]['selected']:>19.2%}{ca[k]['selected']:>19.2%}")

    out = {"baseline_tag": args.baseline, "arm_tag": args.arm,
           "baseline": base, "arm": arm, "paired": paired}
    dest = PROJECT_ROOT / "logs" / f"compare_{args.arm}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  written -> {dest.relative_to(PROJECT_ROOT)}")

    if args.json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
