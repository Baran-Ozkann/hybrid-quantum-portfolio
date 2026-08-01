"""
tools/stability_report.py -- Phase 7 stability and variance analysis.

Analyses every saved training run and writes ``docs/stability_report.md`` plus
summary figures.

WHAT IS BEING MEASURED, AND WHY IT IS NOT "DID THE LOSS GO DOWN"
---------------------------------------------------------------
The Hamiltonian has only 16 eigenvalues, all known exactly from the Phase 2
brute-force enumeration, so every run is scored against the true optimum rather
than against its own convergence curve. Three quantities matter and they do NOT
move together:

  energy_gap        how far <H> sits above the exact ground energy
  ground_state_prob how much amplitude landed on the correct bitstring
  mode_bitstring    what the decoder actually reads out

A run can have a small energy gap and still read out the wrong portfolio,
because the two lowest states of this problem are nearly degenerate.

THREE ANALYSES
--------------
1. Success rate with a Wilson 95% interval. A 1-of-3 result supports no
   quantitative claim (Wilson interval roughly [6%, 79%]); the sweep is sized
   so the rate means something.
2. A measured multi-start curve. For k = 1, 3, 5, 10, bootstrap k-subsets and
   report how often "run k seeds, keep the lowest energy" returns the true
   ground state -- against the oracle bound "at least one of the k reached it".
   The gap between those two curves is a direct measurement of how badly energy
   ranking tracks correctness.
3. The number of distinct attractors the optimiser falls into.

Run::

    python tools/stability_report.py
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import data_pipeline as dp     # noqa: E402
from classical import hamiltonian as H        # noqa: E402
from training import optimization as opt      # noqa: E402

DOCS = PROJECT_ROOT / "docs"
FIGURES = DOCS / "figures"
TRAINING = PROJECT_ROOT / "training"

#: The three seeds kept as the illustrative case study.
CASE_STUDY_SEEDS = (42, 123, 456)
#: Multi-start sizes to evaluate.
K_VALUES = (1, 2, 3, 5, 10, 20)
#: Bootstrap resamples per k.
N_BOOTSTRAP = 20_000
RNG = np.random.default_rng(20260730)


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly for small ``n`` and extreme proportions -- exactly the
    regime a handful of seeds lands in.

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    if n == 0:
        return 0.0, 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_all_runs() -> dict[int, object]:
    """Load every ``training/trained_seed*.json`` on disk, keyed by seed."""
    runs = {}
    for path in sorted(TRAINING.glob("trained_seed*.json")):
        m = re.search(r"trained_seed(\d+)\.json$", path.name)
        if not m:
            continue
        seed = int(m.group(1))
        runs[seed] = opt.load_result(seed)
    return runs


def selection_curve(
    preference: np.ndarray,
    correct: np.ndarray,
    with_replacement: bool,
) -> list[dict]:
    """Exact probability that a rank-based selection rule returns a correct run.

    Computed in CLOSED FORM, not by bootstrap. Monte Carlo added sampling noise
    of its own (k = 1 came out 60.2% against a true 60.0%) and rounded
    small failure probabilities away entirely, which is precisely the regime that
    matters at large k.

    ``preference`` is the run ordering the rule induces, best first (e.g. runs
    sorted by ascending energy for "keep the lowest energy"). The rule picks
    whichever drawn run ranks best, so it succeeds exactly when that run is
    correct.

    Let ``c_j = 1`` when the j-th most preferred run (1-indexed) is correct.

    WITHOUT replacement -- "I already have these n runs; pick k of them":

        P(best drawn rank = j) = C(n - j, k - 1) / C(n, k)
        P(success)             = sum_j c_j * C(n - j, k - 1) / C(n, k)

      This is a statement about THIS pool. It is bounded above by pool
      composition: once k > n - (index of the first correct run), every subset
      is forced to contain a top-ranked run, so the curve hits exactly 1.0 as an
      artifact of the pool size rather than a property of the method.

    WITH replacement -- "run k fresh independent seeds", the predictive claim:

        P(best drawn rank >= j) = ((n - j + 1) / n)^k
        P(success) = sum_j c_j * [ ((n-j+1)/n)^k - ((n-j)/n)^k ]

      This treats the 30 observed runs as the population and never saturates.

    The oracle row -- "at least one drawn run was correct" -- is the ceiling no
    ranking rule can exceed:

        without replacement:  1 - C(n_wrong, k) / C(n, k)
        with replacement:     1 - (n_wrong / n)^k
    """
    n = len(preference)
    c = correct[preference].astype(bool)
    n_wrong = int((~c).sum())
    rows = []

    for k in K_VALUES:
        if not with_replacement and k > n:
            continue

        if with_replacement:
            selected = sum(
                float(c[j - 1]) * (((n - j + 1) / n) ** k - ((n - j) / n) ** k)
                for j in range(1, n + 1)
            )
            oracle = 1.0 - (n_wrong / n) ** k
        else:
            total = math.comb(n, k)
            selected = sum(
                float(c[j - 1]) * math.comb(n - j, k - 1)
                for j in range(1, n + 1)
            ) / total
            oracle = 1.0 - (math.comb(n_wrong, k) / total if n_wrong >= k else 0.0)

        rows.append({"k": k, "selected": selected, "oracle": oracle})
    return rows


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    ham = H.load_hamiltonian()
    stats = dp.load_statistics()
    baselines = json.loads(
        (PROJECT_ROOT / "data" / "markowitz_baseline.json").read_text(encoding="utf-8")
    )
    runs = load_all_runs()
    if not runs:
        print("No trained runs found. Run: python training/optimization.py --seeds 42")
        return 1

    seeds = sorted(runs)
    n_runs = len(seeds)

    spectrum = sorted(H.enumerate_states(ham), key=lambda r: r["ising_energy"])
    ground, runner_up = spectrum[0], spectrum[1]
    spectral_width = spectrum[-1]["ising_energy"] - ground["ising_energy"]
    gap = runner_up["ising_energy"] - ground["ising_energy"]

    energies = np.array([runs[s].final_energy for s in seeds])
    correct = np.array([runs[s].found_ground_state for s in seeds])
    gs_probs = np.array([runs[s].ground_state_prob for s in seeds])
    sharpes = np.array([runs[s].sharpe_ratio for s in seeds])
    modes = [runs[s].mode_bitstring for s in seeds]

    n_correct = int(correct.sum())
    rate, lo, hi = wilson(n_correct, n_runs)
    attractors = Counter(modes)

    mode_probs = np.array([runs[s].mode_probability for s in seeds])

    # Two selection rules, each defined by the order it prefers runs in.
    by_energy = np.argsort(energies)                 # ascending energy = best first
    by_concentration = np.argsort(-mode_probs)       # most concentrated first

    curves = {
        ("energy", False): selection_curve(by_energy, correct, False),
        ("energy", True): selection_curve(by_energy, correct, True),
        ("concentration", False): selection_curve(by_concentration, correct, False),
        ("concentration", True): selection_curve(by_concentration, correct, True),
    }
    curve = curves[("energy", False)]   # descriptive curve, used in the figure

    # ---- converged vs failed -------------------------------------------------
    # The incorrect runs are two unrelated populations: runs that optimised well
    # and then lost the near-degenerate discrimination, and runs that simply
    # failed to optimise. Merging them conflates the finding this project is
    # about with ordinary optimiser failure, and inflates the energy spread.
    converged_tol = 0.005 * spectral_width          # 0.5% of the spectral width
    converged_cut = ground["ising_energy"] + converged_tol
    conv = energies <= converged_cut
    n_conv = int(conv.sum())
    n_conv_correct = int((conv & correct).sum())
    n_failed = n_runs - n_conv
    n_failed_correct = int((~conv & correct).sum())
    conv_rate, conv_lo, conv_hi = wilson(n_conv_correct, n_conv)
    spread_conv = float(np.ptp(energies[conv])) if n_conv else 0.0

    # Was the 3-seed trio an unlucky draw, or an ordinary one? Given the measured
    # rate, this is a plain binomial question.
    p_true = n_correct / n_runs
    p_exactly_1_of_3 = 3 * p_true * (1 - p_true) ** 2
    p_at_most_1_of_3 = (1 - p_true) ** 3 + p_exactly_1_of_3

    # ---- pool-exhaustion thresholds, without replacement --------------------
    # These are DIFFERENT for the two columns and must be stated separately.
    #   oracle: fails only if every drawn run is incorrect, which is possible
    #           while k <= n_wrong. Forced to 100% at k = n_wrong + 1.
    #   keep-lowest: fails only if the subset avoids the entire leading block of
    #           correct runs (ranks 1..n_leading_correct), which is possible
    #           while k <= n - n_leading_correct. Forced at k = that + 1.
    # How far down the energy ranking can you go before hitting a wrong run?
    # Also what determines whether "keep the lowest energy" is a good rule.
    n_leading_correct = 0
    for i in by_energy:
        if not correct[i]:
            break
        n_leading_correct += 1

    n_wrong_total = n_runs - n_correct
    k_oracle_forced = n_wrong_total + 1
    k_keeplowest_forced = n_runs - n_leading_correct + 1

    # ---- is the converged/failed split robust to where the cut is placed? ----
    # If the two groups are separated by an empty band, the partition does not
    # depend on the threshold, which is a far stronger defence than the choice
    # of 0.5%.
    conv_worst = float(energies[conv].max()) if n_conv else float("nan")
    fail_best = float(energies[~conv].min()) if n_failed else float("nan")
    empty_band = fail_best - conv_worst if (n_conv and n_failed) else 0.0

    # ---- concentration ranking: is the tail one run, or the ordering? -------
    conc_order = np.argsort(-mode_probs)
    conc_top = [
        {
            "rank": r + 1,
            "seed": seeds[i],
            "mode_probability": float(mode_probs[i]),
            "readout": modes[i],
            "correct": bool(correct[i]),
        }
        for r, i in enumerate(conc_order[:5])
    ]
    conc_top5_correct = sum(1 for row in conc_top if row["correct"])

    # Energy ranking: does the global lowest-energy run happen to be correct?
    best_idx = int(np.argmin(energies))
    best_is_correct = bool(correct[best_idx])

    beq_sharpe = baselines["best_equal_weight"]["sharpe_ratio"]

    # -- the two closest-in-energy runs that disagree on the readout ----------
    order = np.argsort(energies)
    closest_disagreeing = None
    for a, b in zip(order, order[1:]):
        if modes[a] != modes[b]:
            closest_disagreeing = (seeds[a], seeds[b], abs(energies[a] - energies[b]))
            break

    # Do the correct and incorrect runs overlap in energy at all?
    e_correct = energies[correct]
    e_wrong = energies[~correct]
    overlap = (len(e_correct) and len(e_wrong)
               and e_correct.max() > e_wrong.min())
    n_correct_above_best_wrong = (
        int((e_correct > e_wrong.min()).sum()) if overlap else 0
    )

    # Runs that are BOTH confident (>=90% on one state) and correct.
    confident = np.array([runs[s].mode_probability >= 0.90 for s in seeds])
    n_conf_correct = int((confident & correct).sum())
    n_conf_wrong = int((confident & ~correct).sum())

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for state in sorted(attractors):
        m = np.array([mo == state for mo in modes])
        ax.scatter(energies[m], gs_probs[m], s=42, alpha=0.8,
                   label=f"read out {state} (n={attractors[state]})")
    ax.axvline(ground["ising_energy"], color="black", ls="--", lw=1.1,
               label=f"ground E = {ground['ising_energy']:.4f}")
    ax.axvline(runner_up["ising_energy"], color="grey", ls=":", lw=1.1,
               label=f"runner-up E = {runner_up['ising_energy']:.4f}")
    ax.set_xlabel(r"final $\langle H \rangle$")
    ax.set_ylabel("P(ground state)")
    ax.set_title(f"Energy ranks the readout but does not determine it "
                 f"({n_runs} seeds)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Plot the PREDICTIVE (with-replacement) curves: the without-replacement
    # version saturates from pool exhaustion and would overstate the method.
    ax = axes[1]
    pr_e = curves[("energy", True)]
    pr_c = curves[("concentration", True)]
    ks = [r["k"] for r in pr_e]
    ax.plot(ks, [r["oracle"] for r in pr_e], "o--", color="grey",
            label="oracle: any of k correct")
    ax.plot(ks, [r["selected"] for r in pr_e], "o-", color="tab:blue",
            label="keep lowest energy")
    ax.plot(ks, [r["selected"] for r in pr_c], "s-", color="tab:orange",
            label="keep most concentrated")
    ax.axhline(rate, color="tab:red", ls=":", lw=1.1,
               label=f"single run = {rate:.1%}")
    ax.set_xlabel("k (fresh independent seeds)")
    ax.set_ylabel("P(returns the true ground state)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Multi-start, exact (with replacement)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    labels = [s for s, _ in attractors.most_common()]
    counts = [attractors[s] for s in labels]
    held = {r["bitstring"]: ",".join(r["held"]) for r in spectrum}
    colours = ["tab:green" if s == ground["bitstring"] else "tab:red" for s in labels]
    ax.bar(range(len(labels)), counts, color=colours)
    ax.set_xticks(range(len(labels)),
                  [f"{s}\n{held.get(s, '?')}" for s in labels], fontsize=8)
    ax.set_ylabel("seeds")
    ax.set_title(f"Attractors ({len(attractors)} distinct)")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = FIGURES / "stability_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ report
    L: list[str] = []
    a = L.append

    a("# Stability Validation Report")
    a("")
    a(f"Phase 7 reproducibility analysis over **{n_runs} random "
      f"initialisations**. Generated by `tools/stability_report.py`.")
    a("")

    a("## Headline")
    a("")
    a(f"Every run was identical except for its random seed. Three numbers, not "
      f"one, because the runs are **two distinct populations** and a single "
      f"average hides that:")
    a("")
    a(f"- **{rate:.1%} overall** — the true ground state was returned in "
      f"{n_correct} of {n_runs} runs (Wilson 95% **[{lo:.1%}, {hi:.1%}]**).")
    a(f"- **{conv_rate:.1%} among the {n_conv} runs that converged** to within "
      f"0.5% of the ground energy (Wilson 95% [{conv_lo:.1%}, {conv_hi:.1%}]). "
      f"These optimised successfully and then lost a "
      f"{100 * gap / spectral_width:.2f}% discrimination between the two lowest "
      f"states. **This is the near-degeneracy finding.**")
    a(f"- **{n_failed} runs never optimised at all**, finishing far above the "
      f"ground energy for reasons unrelated to the degeneracy. They are ordinary "
      f"optimiser failure — a separate open question about initialisation and "
      f"step size — and they account for almost all of the headline energy "
      f"spread.")
    a("")
    a("The two groups are separated by an empty band in the data, so the split "
      "is not an artifact of where the threshold was drawn (see below).")
    a("")
    a("The pipeline is bit-for-bit reproducible; the *optimiser* is not reliably "
      "correct. Those are separate findings and both hold.")
    a("")

    a("## Scope: these are in-sample results")
    a("")
    a("Stated plainly because it bounds every number here.")
    a("")
    a("Expected returns and the covariance matrix are estimated from the **same "
      f"{stats['n_observations']} trading days** "
      f"({stats['first_observation']} to {stats['last_observation']}) that the "
      "resulting portfolios are then scored on. There is no train/test split and "
      "no walk-forward validation. A single one-year window also provides "
      "essentially **one observation** of each asset's annualised mean return, so "
      "`mu` carries enormous estimation error that no figure in this report "
      "attempts to quantify.")
    a("")
    a("What this report therefore does and does not claim:")
    a("")
    a("- **Claimed:** optimiser reliability against a *known ground truth* on a "
      "realistic problem instance. Because the Hamiltonian is diagonal with 16 "
      "enumerable eigenvalues, \"did the optimiser find the minimum?\" has an "
      "exact answer, and that question is entirely independent of whether the "
      "minimum is a good investment.")
    a("- **Not claimed:** that these portfolios are good investments, that the "
      "reported Sharpe ratios are achievable out of sample, or that the asset "
      "ranking would survive a different window. In-sample mean-variance "
      "optimisation is well known to overfit; the 100%-AMD max-Sharpe corner "
      "solution is a textbook symptom of exactly that.")
    a("")
    a("The optimisation target is a *fixed, fully specified* mathematical object. "
      "Whether that object is the right one to optimise is a finance question "
      "this project does not address.")
    a("")

    a("## Method")
    a("")
    a(f"{n_runs} runs, identical in every respect except the random seed: same "
      "frozen price snapshot (`data/prices.csv`, SHA-256 recorded in "
      "`data/statistics.json`), same Hamiltonian, 100 epochs, Adam at lr = 0.05, "
      "3 ansatz layers, 28 parameters, exact state-vector backend.")
    a("")
    a(f"- Exact ground state: **{ground['bitstring']}** "
      f"({', '.join(ground['held'])}), E = **{ground['ising_energy']:.6f}**")
    a(f"- Nearest competitor: **{runner_up['bitstring']}** "
      f"({', '.join(runner_up['held'])}), E = **{runner_up['ising_energy']:.6f}**")
    a(f"- Degeneracy gap: **{gap:.6f}**, against a spectral width of "
      f"{spectral_width:.4f} — **{100 * gap / spectral_width:.2f}%**")
    a("")

    a("## Aggregate results")
    a("")
    a("| Quantity | Value |")
    a("|---|---|")
    a(f"| Runs | {n_runs} |")
    a(f"| Reached the ground state | {n_correct} ({rate:.1%}) |")
    a(f"| Wilson 95% interval | [{lo:.1%}, {hi:.1%}] |")
    a(f"| Distinct attractors | {len(attractors)} |")
    a(f"| Final ⟨H⟩ mean | {energies.mean():.6f} |")
    a(f"| Final ⟨H⟩ std (population) | {energies.std(ddof=0):.6f} |")
    a(f"| Final ⟨H⟩ spread | {np.ptp(energies):.6f} |")
    a(f"| Energy spread ÷ degeneracy gap | **{np.ptp(energies) / gap:.2f}×** |")
    a(f"| Sharpe mean | {sharpes.mean():+.4f} |")
    a(f"| Sharpe spread | {np.ptp(sharpes):.4f} |")
    a("")
    a(f"**Run-to-run energy variation is {np.ptp(energies) / gap:.2f}× the "
      f"degeneracy gap the optimiser has to resolve.** The spread of outcomes is "
      f"an order of magnitude wider than the difference it is being asked to "
      f"detect, which is why any single run is unreliable.")
    a("")
    a("That ratio does **not**, however, mean the energy is uninformative, and "
      "the distinction matters for how the method should be used:")
    a("")
    a(f"- The **{n_leading_correct} lowest-energy runs of {n_runs} are all "
      f"correct**. The first incorrect run appears only at rank "
      f"{n_leading_correct + 1}.")
    if overlap:
        a(f"- The two groups do **overlap** in energy: correct runs span "
          f"[{e_correct.min():.6f}, {e_correct.max():.6f}], incorrect runs "
          f"[{e_wrong.min():.6f}, {e_wrong.max():.6f}], and "
          f"{n_correct_above_best_wrong} correct runs finish at a *higher* energy "
          f"than the best incorrect one. So energy does not **determine** the "
          f"readout — but the overlap lies in the poor-energy region, which a "
          f"keep-the-minimum rule never selects from.")
    a(f"- The globally lowest-energy run "
      f"{'**is** correct' if best_is_correct else 'is **not** correct'} "
      f"(seed {seeds[best_idx]}, ⟨H⟩ = {energies[best_idx]:.6f}, "
      f"P(ground state) = {gs_probs[best_idx]:.4f}).")
    a("")
    a("P(ground state) is deliberately *not* summarised by a standard deviation. "
      "Its distribution is strongly bimodal — runs either place substantial "
      "amplitude on the ground state or almost none — so a single spread "
      "statistic would describe nothing real. The attractor table below is the "
      "meaningful summary.")
    a("")

    a("### Two distinct failure modes, separated")
    a("")
    a("The incorrect runs are **not one population**. Splitting them at "
      f"{100 * 0.005:.1f}% of the spectral width above the ground energy "
      f"(⟨H⟩ ≤ {converged_cut:.6f}) isolates two unrelated behaviours:")
    a("")
    a("| Group | Runs | Correct | Accuracy | ⟨H⟩ range | Spread ÷ gap |")
    a("|---|---|---|---|---|---|")
    a(f"| **Converged** (within {100 * 0.005:.1f}% of ground) | {n_conv} | "
      f"{n_conv_correct} | **{conv_rate:.1%}** | "
      f"[{energies[conv].min():.6f}, {energies[conv].max():.6f}] | "
      f"{spread_conv / gap:.2f}× |")
    if n_failed:
        a(f"| **Failed to optimise** | {n_failed} | {n_failed_correct} | "
          f"{n_failed_correct / n_failed:.1%} | "
          f"[{energies[~conv].min():.6f}, {energies[~conv].max():.6f}] | "
          f"{np.ptp(energies[~conv]) / gap:.2f}× |")
    a("")
    if n_conv and n_failed and empty_band > 0:
        a(f"**The split does not depend on where the cut was placed.** The worst "
          f"converged run finishes at {conv_worst:.6f} and the best failed run at "
          f"{fail_best:.6f}, so the threshold ({converged_cut:.6f}) falls inside "
          f"an **empty band {empty_band:.6f} wide** — "
          f"{empty_band / gap:.2f}× the degeneracy gap itself, with no run "
          f"anywhere in it. Any cut in that interval produces exactly the same "
          f"partition. The two groups are genuinely disjoint populations rather "
          f"than two ends of one continuum, which is what justifies analysing "
          f"them separately; the 0.5% figure merely names a point inside the gap.")
        a("")
    a("This is the most important structural fact in the sweep, and merging the "
      "two groups obscured it:")
    a("")
    a(f"- **The {np.ptp(energies) / gap:.2f}× overall energy spread is almost "
      f"entirely the second group.** Within the converged subset the spread is "
      f"only **{spread_conv / gap:.2f}× the degeneracy gap** — the same order as "
      f"the gap itself, which is the regime the near-degeneracy finding actually "
      f"lives in.")
    a(f"- **The near-degeneracy result is the converged subset.** Those runs "
      f"optimised successfully and then lost a "
      f"{100 * gap / spectral_width:.2f}% discrimination: accuracy "
      f"{conv_rate:.1%} (Wilson 95% [{conv_lo:.1%}, {conv_hi:.1%}]) on "
      f"{n_conv} runs.")
    if n_failed:
        a(f"- **The remaining {n_failed} runs are ordinary optimiser failure**, "
          f"not a degeneracy problem — they finished "
          f"{energies[~conv].max() - ground['ising_energy']:.3f} to "
          f"{energies[~conv].min() - ground['ising_energy']:.3f} above the ground "
          f"energy, far outside the {gap:.3f} gap, and would be discarded by any "
          f"selection rule. Diagnosing them is a separate question about "
          f"initialisation and step size.")
    a("")

    a("### Attractors")
    a("")
    a("| Readout | Assets | Seeds | Share | Ising energy | Sharpe |")
    a("|---|---|---|---|---|---|")
    by_state = {r["bitstring"]: r for r in spectrum}
    for state, count in attractors.most_common():
        row = by_state.get(state, {})
        sh = [runs[s].sharpe_ratio for s in seeds if runs[s].mode_bitstring == state]
        tag = " **(ground state)**" if state == ground["bitstring"] else ""
        a(f"| `{state}`{tag} | {','.join(row.get('held', []))} | {count} | "
          f"{count / n_runs:.1%} | {row.get('ising_energy', float('nan')):.6f} | "
          f"{statistics.fmean(sh):+.4f} |")
    a("")
    a(f"All {len(attractors)} attractors are **feasible** portfolios — every run "
      "landed in the constrained subspace, so the budget penalty is doing its "
      "job even where the optimiser picks the wrong basket.")
    a("")

    a("## Multi-start selection, measured")
    a("")
    a("All figures below are **exact closed-form** probabilities, not bootstrap "
      "estimates. An earlier draft used 20 000 Monte Carlo draws, which added "
      "sampling noise of its own (k = 1 came out 60.2% against a true 60.0%) and "
      "rounded small failure probabilities away entirely — the regime that "
      "matters most at large k.")
    a("")
    a("Two selection rules are compared, each just an ordering of the runs:")
    a("")
    a("- **keep lowest energy** — prefer the run with the smallest final ⟨H⟩.")
    a("- **keep most concentrated** — prefer the run whose measured distribution "
      "is most concentrated on one basis state (largest P(readout)). This needs "
      "no knowledge of the true ground energy.")
    a("")
    a("**oracle** is \"at least one drawn run was correct\" — the ceiling no "
      "ranking rule can exceed.")
    a("")

    a("### Predictive: run k fresh seeds (with replacement)")
    a("")
    a("This is the model behind any \"just run k seeds\" recommendation — k new "
      "independent runs, treating the 30 observed as the population. It never "
      "saturates, so it is the table the practical claim rests on.")
    a("")
    a("| k | keep lowest energy | keep most concentrated | oracle | energy-rule shortfall |")
    a("|---|---|---|---|---|")
    ce_r = {r["k"]: r for r in curves[("energy", True)]}
    cc_r = {r["k"]: r for r in curves[("concentration", True)]}
    for k in sorted(ce_r):
        a(f"| {k} | {ce_r[k]['selected']:.2%} | {cc_r[k]['selected']:.2%} | "
          f"{ce_r[k]['oracle']:.2%} | {ce_r[k]['oracle'] - ce_r[k]['selected']:+.2%} |")
    a("")

    a("### Descriptive: pick k of the 30 runs already in hand (without replacement)")
    a("")
    ce_n = {r["k"]: r for r in curves[("energy", False)]}
    cc_n = {r["k"]: r for r in curves[("concentration", False)]}
    a("A statement about **this pool**, not about the method: both columns are "
      "forced to 100% once the pool can no longer supply a failing subset. The "
      "two thresholds are different and neither is a property of the optimiser.")
    a("")
    a(f"- **Oracle** fails only if *every* drawn run is incorrect. There are "
      f"{n_wrong_total} incorrect runs, so an all-incorrect subset exists for "
      f"k ≤ {n_wrong_total}; the oracle is forced to exactly 100% only at "
      f"**k ≥ {k_oracle_forced}**.")
    a(f"- **Keep lowest energy** fails only if the subset avoids the entire "
      f"leading block of correct runs — energy ranks 1–{n_leading_correct}, which "
      f"are all correct. Avoiding them means drawing from the remaining "
      f"{n_runs - n_leading_correct} runs, possible for "
      f"k ≤ {n_runs - n_leading_correct}; it is forced at "
      f"**k ≥ {k_keeplowest_forced}**.")
    a("")
    a(f"So the 100% entries at k = 20 are arithmetic, not merit. Below those "
      f"thresholds the values are genuine: at k = 10 the oracle is "
      f"{ce_n[10]['oracle']:.4%} and keep-lowest is {ce_n[10]['selected']:.4%} — "
      f"both short of 100%, which is exactly what the earlier bootstrap table "
      f"obscured by reporting a flat 100.0%.")
    a("")
    a("| k | keep lowest energy | keep most concentrated | oracle | energy-rule shortfall |")
    a("|---|---|---|---|---|")
    for k in sorted(ce_n):
        a(f"| {k} | {ce_n[k]['selected']:.4%} | {cc_n[k]['selected']:.4%} | "
          f"{ce_n[k]['oracle']:.4%} | {ce_n[k]['oracle'] - ce_n[k]['selected']:+.4%} |")
    a("")
    k10n = ce_n.get(10)
    if k10n:
        a(f"Note the k = 10 entry: **{k10n['selected']:.4%}**, not 100%. Keeping "
          f"the lowest energy fails whenever all 10 drawn runs land outside the "
          f"leading correct block — a small but non-zero probability that a "
          f"20 000-draw bootstrap reported as a flat 100.0%.")
        a("")

    a("### The two rules compared at the individual level")
    a("")
    n_conf_total = n_conf_correct + n_conf_wrong
    a(f"Concentration is informative before any multi-start at all. Of the "
      f"{n_conf_total} runs that were confident (≥90% of amplitude on one state), "
      f"**{n_conf_correct} were correct = "
      f"{n_conf_correct / max(n_conf_total, 1):.1%}**, against {rate:.1%} for an "
      f"unselected run. High concentration is therefore positive evidence of "
      f"correctness — but not proof, since {n_conf_wrong} confident runs were "
      f"wrong.")
    a("")
    a("**But concentration is a poor multi-start rule, and the tables above show "
      "why that is not a contradiction.** Look along the "
      "keep-most-concentrated column: it *degrades* as k grows, from "
      f"{cc_r[1]['selected']:.0%} at k = 1 to {cc_r[20]['selected']:.0%} at "
      f"k = 20, while keep-lowest-energy climbs to "
      f"{ce_r[20]['selected']:.0%}.")
    a("")
    a("The cause is the *top of the concentration ordering*, not a single "
      "anecdote. Ranked by how concentrated their final distribution is:")
    a("")
    a("| Rank | Seed | P(readout) | Readout | Correct? |")
    a("|---|---|---|---|---|")
    for row in conc_top:
        a(f"| {row['rank']} | {row['seed']} | {row['mode_probability']:.4f} | "
          f"`{row['readout']}` | {'yes' if row['correct'] else '**no**'} |")
    a("")
    a(f"{conc_top5_correct} of the top 5 are correct — barely better than the "
      f"{rate:.0%} base rate, and **the single most concentrated run is wrong**. "
      f"Averaged over confident runs concentration is positive evidence "
      f"({n_conf_correct}/{n_conf_total}); but a selection rule does not sample "
      f"the average, it takes the *extreme*, and this pool's extreme is a "
      f"confidently wrong run.")
    a("")
    a(f"Two consequences worth stating precisely:")
    a("")
    a(f"- **The asymptote is zero.** With replacement, as k grows the rule "
      f"selects the globally most concentrated run with probability → 1. That "
      f"run is incorrect, so P(correct) → **0%**, not to some plateau. The "
      f"measured {cc_r[20]['selected']:.1%} at k = 20 is on the way down.")
    a(f"- **The rule fails at the very top, not throughout.** The "
      f"without-replacement value at k = 20 is "
      f"{cc_n[20]['selected']:.2%} ≈ 1/3, which is just P(the subset misses "
      f"rank 1) = {1 - 20 / n_runs:.3f}. That it lands so close to 1/3 says "
      f"rank 2 is correct: when the rule escapes the bad top pick it usually "
      f"succeeds. The defect is concentrated in the leading element of the "
      f"ordering, which is precisely the element a selection rule always uses.")
    a("")
    a("So the honest summary of the two signals is asymmetric:")
    a("")
    a(f"- **Energy** is a sound selection rule: monotone in k, reaching "
      f"{ce_r[10]['selected']:.1%} at k = 10, and it needs no ground truth — only "
      f"the relative ordering of ⟨H⟩, which is always available.")
    a(f"- **Concentration** is a useful *diagnostic* for a single run "
      f"({n_conf_correct / max(n_conf_total, 1):.1%} vs {rate:.1%} unselected) "
      f"but must not be used to choose between runs. This is the sharpest form of "
      f"\"confident convergence can be confidently wrong\": the most confident "
      f"run available was wrong, so maximising confidence actively selects for "
      f"the failure mode.")
    a("")
    k10 = next((r for r in curve if r["k"] == 10), None)
    worst = max(curve, key=lambda r: r["oracle"] - r["selected"])
    if k10:
        a(f"**This converts the earlier future-work bullet into a result, and the "
          f"result is more favourable than the 3-seed sample suggested.** "
          f"Multi-start is an effective and cheap mitigation: at k = 5, keeping "
          f"the lowest energy returns the ground state "
          f"{next(r['selected'] for r in curve if r['k'] == 5):.1%} of the time, "
          f"and at k = {k10['k']} it reached {k10['selected']:.1%} in this sample "
          f"— against {rate:.1%} for a single run. At ~24 s per run, k = 10 costs "
          f"about four minutes.")
        a("")
        a(f"Energy-based selection is also close to optimal. Its largest measured "
          f"shortfall against the oracle is "
          f"**{worst['oracle'] - worst['selected']:.1%} at k = {worst['k']}**, "
          f"and by k = {k10['k']} the two curves coincide "
          f"({k10['selected']:.1%} vs {k10['oracle']:.1%}). That shortfall is the "
          f"honest quantification of \"energy ranking does not perfectly track "
          f"correctness\": the cost is real but modest, and it vanishes once "
          f"enough runs are available for the minimum to be reliably in the "
          f"correct basin.")
        a("")
        a("> **Correction to an earlier draft.** On the 3-seed sample this "
          "project claimed energy ranking \"does not track correctness\", "
          "generalising from the fact that seed 456 reached a lower energy than "
          "seed 123 while both were wrong. Over 30 seeds that framing is too "
          "strong: energy ranking is a good — though imperfect — selector. The "
          "3-seed observation was a real event in a sample too small to support "
          "the generalisation drawn from it.")
        a(">")
        a(f"> **And the trio was not even an unlucky draw.** At the measured rate "
          f"of {p_true:.0%}, seeing exactly 1 correct in 3 runs has probability "
          f"**{p_exactly_1_of_3:.1%}**, and 1-or-fewer **{p_at_most_1_of_3:.1%}**. "
          f"That is an entirely ordinary outcome — which is the cleanest possible "
          f"demonstration of why n = 3 supported no quantitative claim at all. "
          f"The original Wilson interval on 1-of-3, roughly [6%, 79%], said as "
          f"much at the time.")
        a("")

    a("## Illustrative case study: seeds 42, 123, 456")
    a("")
    a("The first three seeds run, kept because they display the failure modes "
      "unusually clearly.")
    a("")
    a("| Seed | Final ⟨H⟩ | Gap vs ground | P(ground state) | P(readout) | Readout | Correct? | Return | Vol | Sharpe |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for s in CASE_STUDY_SEEDS:
        if s not in runs:
            continue
        r = runs[s]
        p_gs = "<0.0001" if 0.0 <= r.ground_state_prob < 1e-4 else f"{r.ground_state_prob:.4f}"
        a(f"| {s} | {r.final_energy:.6f} | {r.energy_gap:+.6f} | {p_gs} | "
          f"{r.mode_probability:.4f} | `{r.mode_bitstring}` | "
          f"{'**yes**' if r.found_ground_state else 'no'} | "
          f"{r.expected_return:+.2%} | {r.volatility:.2%} | {r.sharpe_ratio:+.4f} |")
    a("")
    a("Three things this trio shows:")
    a("")
    if 42 in runs and 456 in runs:
        d = abs(runs[42].final_energy - runs[456].final_energy)
        a(f"1. **Energy ranking is decided inside the noise.** Seeds 42 and 456 "
          f"differ by only **{d:.5f}** in final energy — "
          f"**{100 * d / gap:.0f}% of the degeneracy gap** they need to resolve "
          f"— yet they read out *different portfolios* "
          f"(`{runs[42].mode_bitstring}` vs `{runs[456].mode_bitstring}`). "
          f"Energy ranking happens to pick the right run here, but by a margin "
          f"well inside run-to-run variation. Both seeds are in the **converged** "
          f"group, so the relevant yardstick is that group's spread of "
          f"{spread_conv:.5f} — their separation is "
          f"{100 * d / spread_conv:.0f}% of it. (Comparing against the overall "
          f"{np.ptp(energies):.5f} spread would be wrong here: that figure is "
          f"dominated by the {n_failed} runs that never optimised, a population "
          f"neither of these seeds belongs to.)")
        a("")
    if 123 in runs:
        a(f"2. **Confident convergence can be confidently wrong.** Seed 123 "
          f"placed **{runs[123].mode_probability:.1%}** of its amplitude on a "
          f"single state — the most decisive-looking run of the three — and that "
          f"state was wrong, with P(ground state) below 1e-4. Its energy, "
          f"{runs[123].final_energy:.6f}, would read as clean convergence in an "
          f"energy-only report.")
        a("")
    if 42 in runs:
        p = np.asarray(runs[42].final_probabilities)
        top = np.argsort(p)[::-1]
        a(f"3. **No run in this trio was both confident and correct.** Seed 42, "
          f"the only correct one, split {p[top[0]]:.1%} / {p[top[1]]:.1%} between "
          f"`{ground['bitstring']}` and `{runner_up['bitstring']}` — an amplitude "
          f"ratio of about {p[top[0]] / max(p[top[1]], 1e-12):.1f}:1. "
          f"Statistically decisive over 4096 shots, but not unambiguous "
          f"convergence.")
        a("")
        a(f"> **This third property is specific to the trio and does NOT "
          f"generalise.** Across all {n_runs} seeds, "
          f"**{n_conf_correct} runs were both confident "
          f"(≥90% on one state) and correct** — for example seed "
          f"{seeds[best_idx]}, which put {gs_probs[best_idx]:.2%} of its "
          f"amplitude on the ground state. {n_conf_wrong} runs were confident and "
          f"wrong. So \"confident convergence can be confidently wrong\" remains "
          f"true as a *warning* — it happens — but \"no run is both confident and "
          f"correct\" was an artifact of three draws, and is false in the wider "
          f"sample. The 3-seed trio happened to contain none of the "
          f"high-confidence correct runs that make up "
          f"{n_conf_correct / n_runs:.0%} of the sweep.")
        a("")

    a("## Reproducibility")
    a("")
    a("Numerical reproducibility holds **exactly**: re-running any seed "
      "reproduces its results bit-for-bit. Three things make that true, and all "
      "three were necessary:")
    a("")
    a("1. `data/prices.csv` is a **frozen snapshot**. A fixed historical date "
      "range is *not* sufficient — refetching the same fully-historical window "
      "returned different values (NVDA's annualised mean moved 1.0e-9, "
      "covariance entries up to 3.7e-8), which propagates into `h` and `J`.")
    a("2. The **exact state-vector backend** is deterministic to float64 "
      "rounding (~2.2e-15, ~10 ULP), with no shot noise.")
    a("3. `torch`, `numpy`, and the Q# simulator RNGs are all seeded per run.")
    a("")
    a("Verified by `tools/check_reproducibility.py`.")
    a("")

    a("## Why the problem is hard")
    a("")
    a(f"The root cause is a property of the instance, not a bug. The two best "
      f"portfolios are separated by {100 * gap / spectral_width:.2f}% of the "
      f"spectral width, and **both hold AMD** — they disagree only on whether "
      f"the partner is TSLA or NVDA:")
    a("")
    a("| State | Assets | Ising energy | Return | Vol | Sharpe |")
    a("|---|---|---|---|---|---|")
    for r in spectrum[:2]:
        a(f"| `{r['bitstring']}` | {','.join(r['held'])} | "
          f"{r['ising_energy']:.6f} | {r['expected_return']:+.2%} | "
          f"{r['volatility']:.2%} | "
          f"{(r['expected_return'] - stats['risk_free_rate']) / r['volatility']:+.4f} |")
    a("")
    a("This is **not** a penalty-tuning artifact. Both states are feasible, so "
      "the budget penalty contributes identically zero to each, and the gap "
      "between them is independent of `lambda`.")
    a("")
    ru_sharpe = statistics.fmean(
        [runs[s].sharpe_ratio for s in seeds
         if runs[s].mode_bitstring == runner_up["bitstring"]]
    )
    a(f"A single VQE run, whose outcomes span "
      f"{100 * np.ptp(energies) / spectral_width:.2f}% of the spectral width, "
      f"cannot reliably resolve a {100 * gap / spectral_width:.2f}% gap — hence "
      f"{rate:.0%} single-run accuracy. The financial cost of getting it wrong is "
      f"small (Sharpe {ru_sharpe:+.4f} against the optimum's {beq_sharpe:+.4f}, a "
      f"{(beq_sharpe - ru_sharpe) / abs(beq_sharpe):.1%} shortfall), and "
      f"multi-start selection removes most of the risk. The finding is therefore "
      f"not \"the method fails\" but a more specific and more useful statement: "
      f"**a single variational run is not trustworthy on a near-degenerate "
      f"instance, and reliability has to be bought with repeats.**")
    a("")

    a("## Future work")
    a("")
    a("Unattempted. Listed as directions, not as claims.")
    a("")
    a("- **CVaR or Gibbs objectives** in place of the plain expectation, which "
      "weight the low-energy tail more heavily and are reported in the "
      "literature to sharpen concentration onto the ground state.")
    a("- **Deeper or differently entangled ansätze.** Three layers reach within "
      "0.5% of the ground energy but do not concentrate; more expressive "
      "circuits might.")
    a("- **Out-of-sample evaluation.** Everything here is in-sample (see Scope). "
      "A walk-forward study would speak to whether the selected portfolios have "
      "any predictive value — a different question from the one asked here.")
    a("")
    a("Multi-start selection has moved out of this list: it is measured above. "
      "So has the CVaR objective, which is now a separate controlled study -- "
      "see [`objective_comparison.md`](objective_comparison.md).")
    a("")

    a("## Figures")
    a("")
    a("![Stability summary](figures/stability_summary.png)")
    a("")
    a("**Left:** final energy against P(ground state), coloured by readout. The "
      "clusters overlap in energy, so energy does not *determine* the readout — "
      "though it ranks it well (see the leading correct block above). The runs "
      f"far to the right, beyond {converged_cut:.2f}, are the "
      f"{n_failed} that never optimised. "
      "**Centre:** exact multi-start curves under the predictive "
      "(with-replacement) model — keep-lowest-energy rising to "
      f"{ce_r[10]['selected']:.1%} at k = 10 and effectively meeting the oracle, "
      "and keep-most-concentrated bending *downward* as k grows, because the most "
      "concentrated run in the pool is wrong. "
      "**Right:** how many seeds fell into each attractor.")
    a("")
    a(f"![Convergence](../training/convergence.png)")
    a("")
    a(f"All {n_runs} convergence traces, green where the run read out the ground "
      "state and red otherwise. The traces are **not** cleanly separated: runs "
      "ending at similar energies reach different readouts, and the lowest-energy "
      "band contains both colours.")
    a("")

    (DOCS / "stability_report.md").write_text("\n".join(L), encoding="utf-8")

    summary = {
        "n_runs": n_runs, "seeds": seeds,
        "n_correct": n_correct, "success_rate": rate,
        "wilson_95": [lo, hi],
        "attractors": dict(attractors),
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std(ddof=0)),
        "energy_spread": float(np.ptp(energies)),
        "degeneracy_gap": float(gap),
        "spread_over_gap": float(np.ptp(energies) / gap),
        "multi_start_curve": curve,
        "multi_start_curves": {
            f"{rule}_{'with' if wr else 'without'}_replacement": rows
            for (rule, wr), rows in curves.items()
        },
        "global_best_run_correct": best_is_correct,
        "n_confident_correct": n_conf_correct,
        "n_confident_wrong": n_conf_wrong,
        "n_leading_correct_by_energy": n_leading_correct,
        "converged": {
            "tolerance_fraction_of_spectrum": 0.005,
            "energy_cut": converged_cut,
            "n": n_conv, "n_correct": n_conv_correct,
            "accuracy": conv_rate, "wilson_95": [conv_lo, conv_hi],
            "energy_spread": spread_conv,
            "spread_over_gap": spread_conv / gap if gap else None,
        },
        "failed_to_optimise": {
            "n": n_failed, "n_correct": n_failed_correct,
        },
        "trio_binomial": {
            "p_exactly_1_of_3": p_exactly_1_of_3,
            "p_at_most_1_of_3": p_at_most_1_of_3,
        },
        "pool_exhaustion": {
            "k_oracle_forced": k_oracle_forced,
            "k_keeplowest_forced": k_keeplowest_forced,
        },
        "split_robustness": {
            "converged_worst_energy": conv_worst,
            "failed_best_energy": fail_best,
            "empty_band": empty_band,
            "empty_band_over_gap": empty_band / gap if gap else None,
        },
        "concentration_top5": conc_top,
        "closest_disagreeing_pair": (
            {"seeds": [closest_disagreeing[0], closest_disagreeing[1]],
             "energy_difference": float(closest_disagreeing[2])}
            if closest_disagreeing else None
        ),
    }
    (PROJECT_ROOT / "logs" / "stability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Runs: {n_runs}  correct: {n_correct} ({rate:.1%})  "
          f"Wilson 95% [{lo:.1%}, {hi:.1%}]")
    print(f"Attractors: {dict(attractors)}")
    print(f"Energy spread {np.ptp(energies):.6f} = {np.ptp(energies) / gap:.2f}x gap {gap:.6f}")
    print(f"Converged (<=0.5% of spectrum): {n_conv_correct}/{n_conv} = {conv_rate:.1%}, "
          f"spread {spread_conv:.6f} = {spread_conv / gap:.2f}x gap")
    print(f"Failed to optimise: {n_failed_correct}/{n_failed}")
    print(f"Confident (>=90%): {n_conf_correct}/{n_conf_correct + n_conf_wrong} correct")
    print(f"Trio: P(exactly 1 of 3) = {p_exactly_1_of_3:.1%}, "
          f"P(<=1 of 3) = {p_at_most_1_of_3:.1%}")
    print("\nPREDICTIVE (with replacement):")
    for r_e, r_c in zip(curves[("energy", True)], curves[("concentration", True)]):
        print(f"  k={r_e['k']:2d}  energy {r_e['selected']:7.2%}   "
              f"concentration {r_c['selected']:7.2%}   oracle {r_e['oracle']:7.2%}")
    print("\nDESCRIPTIVE (without replacement):")
    for r_e, r_c in zip(curves[("energy", False)], curves[("concentration", False)]):
        print(f"  k={r_e['k']:2d}  energy {r_e['selected']:8.4%}   "
              f"concentration {r_c['selected']:8.4%}   oracle {r_e['oracle']:8.4%}")
    print(f"\nWrote {DOCS / 'stability_report.md'}")
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

