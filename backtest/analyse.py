"""Phase 8 analysis: the decomposition, the rank metrics, and the figures.

Reads ``logs/backtest_results.json`` and emits every number quoted in
``docs/backtest.md`` plus the figures. Nothing here re-runs the optimiser, so it
is cheap to iterate on.

THE DECOMPOSITION IS THE POINT. Three comparisons, deliberately ordered:

    1. quantum vs best_equal_weight  -> optimiser error, in realised money
    2. best_equal_weight vs EW4      -> whether the objective is worth optimising
    3. both vs single-asset          -> whether either beats doing nothing

If (2) is indistinguishable from zero, then (1) cannot matter, because there was
nothing to win by optimising more accurately. That ordering is what keeps a weak
backtest from being misread as a verdict on VQE.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("analyse")

RESULTS_JSON = PROJECT_ROOT / "logs" / "backtest_results.json"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
ANALYSIS_JSON = PROJECT_ROOT / "logs" / "backtest_analysis.json"

def _main_instance_gap_fraction() -> float:
    """Top-two gap of the Phase 1-9 instance, as a fraction of spectral width.

    Computed from ``data/prices.csv`` rather than hard-coded, so the reference
    line on the degeneracy plot cannot drift away from the instance it claims
    to describe. Currently 0.039209 / 16.560790 = 0.2368%.
    """
    from classical import hamiltonian as H
    ham = H.load_hamiltonian()
    e = sorted(s["ising_energy"] for s in H.enumerate_states(ham))
    return (e[1] - e[0]) / (e[-1] - e[0])


#: The Phase 1-9 main instance, for reference on the gap plot.
MAIN_INSTANCE_GAP_FRACTION = _main_instance_gap_fraction()


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval; correct at the k = 0 and k = n edges."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def minimum_detectable_effect(sd: float, n: int, power: float = 0.80,
                              alpha: float = 0.05) -> float:
    """Smallest paired difference a two-sided t-test could detect at ``power``.

    Stated in advance so that "indistinguishable" is read as a design property
    rather than as a disappointing outcome.
    """
    df = n - 1
    t_a = stats.t.ppf(1 - alpha / 2, df)
    t_b = stats.t.ppf(power, df)
    return float((t_a + t_b) * sd / np.sqrt(n))


def analyse(payload: dict) -> dict:
    folds = payload["folds"]
    arms = payload["arms"]
    quantum = payload.get("quantum") or []
    n = len(folds)
    out: dict = {"n_folds": n}

    # ---------------------------------------------------------------- ranks
    rhos = []
    for f in folds:
        is_s = [p["is_sharpe"] for p in f["portfolios"]]
        oos_s = [p["oos_sharpe"] for p in f["portfolios"]]
        rhos.append(stats.spearmanr(is_s, oos_s).statistic)
    rhos = np.asarray(rhos)
    tt = stats.ttest_1samp(rhos, 0.0)
    out["spearman"] = {
        "per_fold": rhos.tolist(),
        "mean": float(rhos.mean()), "median": float(np.median(rhos)),
        "sd": float(rhos.std(ddof=1)),
        "ci95": [float(x) for x in stats.t.interval(
            0.95, n - 1, loc=rhos.mean(), scale=stats.sem(rhos))],
        "t": float(tt.statistic), "p": float(tt.pvalue),
        "n_positive": int((rhos > 0).sum()),
    }

    # ------------------------------------------------- in-sample top wins?
    hits = [f["portfolios"][0]["bitstring"] ==
            max(f["portfolios"], key=lambda p: p["oos_sharpe"])["bitstring"]
            for f in folds]
    k = sum(hits)
    lo, hi = clopper_pearson(k, n)
    out["is_top_is_oos_top"] = {
        "k": k, "n": n, "rate": k / n, "ci95": [lo, hi],
        "chance_rate": 1 / 6,
        "p_vs_chance": float(stats.binomtest(k, n, 1 / 6).pvalue),
    }

    # ------------------------------------- ground vs runner-up, per fold
    wins, d_is, d_oos, gapfrac = [], [], [], []
    for f in folds:
        by = {p["bitstring"]: p for p in f["portfolios"]}
        g, r = by.get(f["ground_bitstring"]), by.get(f["runner_up_bitstring"])
        if g is None or r is None:
            continue
        wins.append(g["oos_sharpe"] > r["oos_sharpe"])
        d_is.append(g["is_sharpe"] - r["is_sharpe"])
        d_oos.append(g["oos_sharpe"] - r["oos_sharpe"])
        gapfrac.append(f["gap_fraction"])
    kw, nw = sum(wins), len(wins)
    lo, hi = clopper_pearson(kw, nw)
    d_oos_arr = np.asarray(d_oos)
    tt2 = stats.ttest_1samp(d_oos_arr, 0.0)
    out["top_two_discrimination"] = {
        "k": kw, "n": nw, "rate": kw / nw, "ci95": [lo, hi],
        "p_vs_coinflip": float(stats.binomtest(kw, nw, 0.5).pvalue),
        "mean_d_is_sharpe": float(np.mean(d_is)),
        "mean_d_oos_sharpe": float(d_oos_arr.mean()),
        "sd_d_oos_sharpe": float(d_oos_arr.std(ddof=1)),
        "t": float(tt2.statistic), "p": float(tt2.pvalue),
        "spearman_dis_doos": float(stats.spearmanr(d_is, d_oos).statistic),
        "spearman_dis_doos_p": float(stats.spearmanr(d_is, d_oos).pvalue),
        "spearman_gap_doos": float(stats.spearmanr(gapfrac, d_oos).statistic),
        "mde_80pct": minimum_detectable_effect(float(d_oos_arr.std(ddof=1)), nw),
    }

    # --------------------- is the §4 result specific to tight gaps?
    # A tempting narrowing: the headline was measured on an unusually tight gap,
    # so perhaps the objective separates portfolios fine when the gap is wide.
    # Split at the median gap and check. Reported whichever way it points --
    # picking the more flattering of the sign test and the mean would be exactly
    # the error §4 warns about.
    med_gap = float(np.median([f["gap_fraction"] for f in folds]))
    halves = {}
    for name, keep in (("tight", lambda g: g < med_gap),
                       ("loose", lambda g: g >= med_gap)):
        sel = [f for f in folds if keep(f["gap_fraction"])]
        w, dd = [], []
        for f in sel:
            by = {p["bitstring"]: p for p in f["portfolios"]}
            g, r = by.get(f["ground_bitstring"]), by.get(f["runner_up_bitstring"])
            if g is None or r is None:
                continue
            w.append(g["oos_sharpe"] > r["oos_sharpe"])
            dd.append(g["oos_sharpe"] - r["oos_sharpe"])
        kk, nn = sum(w), len(w)
        lo, hi = clopper_pearson(kk, nn)
        halves[name] = {
            "k": kk, "n": nn, "rate": kk / nn if nn else 0.0, "ci95": [lo, hi],
            "mean_d_oos": float(np.mean(dd)),
            "gap_min": float(min(f["gap_fraction"] for f in sel)),
            "gap_max": float(max(f["gap_fraction"] for f in sel)),
        }
    fisher_p = float(stats.fisher_exact([
        [halves["tight"]["k"], halves["tight"]["n"] - halves["tight"]["k"]],
        [halves["loose"]["k"], halves["loose"]["n"] - halves["loose"]["k"]],
    ])[1])
    gaps_all = [f["gap_fraction"] for f in folds]
    wins_all, doos_all = [], []
    for f in folds:
        by = {p["bitstring"]: p for p in f["portfolios"]}
        g, r = by.get(f["ground_bitstring"]), by.get(f["runner_up_bitstring"])
        wins_all.append(1.0 if g["oos_sharpe"] > r["oos_sharpe"] else 0.0)
        doos_all.append(g["oos_sharpe"] - r["oos_sharpe"])
    out["gap_median_split"] = {
        "median_gap_fraction": med_gap,
        **halves,
        "fisher_p": fisher_p,
        "spearman_gap_win": float(stats.spearmanr(gaps_all, wins_all).statistic),
        "spearman_gap_win_p": float(stats.spearmanr(gaps_all, wins_all).pvalue),
        "spearman_gap_doos": float(stats.spearmanr(gaps_all, doos_all).statistic),
        "spearman_gap_doos_p": float(stats.spearmanr(gaps_all, doos_all).pvalue),
    }

    # ---- how much of the mean d_oos rests on one fold?
    d_arr = np.asarray(doos_all)
    worst = int(np.argmax(np.abs(d_arr - d_arr.mean())))
    out["top_two_discrimination"]["mean_sensitivity"] = {
        "mean_all": float(d_arr.mean()),
        "sd_over_mean": float(d_arr.std(ddof=1) / abs(d_arr.mean())),
        "most_influential_fold": int(folds[worst]["index"]),
        "its_d_oos": float(d_arr[worst]),
        "mean_without_it": float(np.delete(d_arr, worst).mean()),
    }

    # ------------------------- what an optimiser error would have cost
    # The bridge between the two halves of the study. If VQE reads out the
    # runner-up instead of the ground state -- exactly the error Phase 7
    # measured at 33% -- this is the realised return it costs. Measured in
    # per-fold total return, the same unit as the decomposition below, so the
    # two are directly comparable.
    cost_of_error = []
    for f in folds:
        by = {p["bitstring"]: p for p in f["portfolios"]}
        g, r = by.get(f["ground_bitstring"]), by.get(f["runner_up_bitstring"])
        if g is None or r is None:
            continue
        cost_of_error.append(g["oos_total_return"] - r["oos_total_return"])
    ce = np.asarray(cost_of_error)
    tt3 = stats.ttest_1samp(ce, 0.0)
    out["cost_of_an_optimiser_error"] = {
        "mean": float(ce.mean()), "sd": float(ce.std(ddof=1)),
        "median": float(np.median(ce)),
        "n_positive": int((ce > 0).sum()), "n": int(len(ce)),
        "t": float(tt3.statistic), "p": float(tt3.pvalue),
        "ci95": [float(x) for x in stats.t.interval(
            0.95, len(ce) - 1, loc=ce.mean(), scale=stats.sem(ce))],
        "mde_80pct": minimum_detectable_effect(float(ce.std(ddof=1)), len(ce)),
    }

    # ------------------------------------------- the project's own pair
    w = l = t = 0
    pair_d = []
    for f in folds:
        by = {p["bitstring"]: p for p in f["portfolios"]}
        a, b = by.get("1010"), by.get("0110")
        if a is None or b is None:
            continue
        pair_d.append(a["oos_sharpe"] - b["oos_sharpe"])
        if a["oos_sharpe"] > b["oos_sharpe"]:
            w += 1
        elif a["oos_sharpe"] < b["oos_sharpe"]:
            l += 1
        else:
            t += 1
    lo, hi = clopper_pearson(w, w + l + t)
    out["tsla_amd_vs_nvda_amd"] = {
        "wins": w, "losses": l, "ties": t, "rate": w / (w + l + t),
        "ci95": [lo, hi],
        "p_vs_coinflip": float(stats.binomtest(w, w + l + t, 0.5).pvalue),
        "mean_d_oos_sharpe": float(np.mean(pair_d)),
    }

    # --------------------------------------------------- degeneracy gaps
    gf = np.asarray([f["gap_fraction"] for f in folds])
    out["degeneracy"] = {
        "gap_fraction_per_fold": gf.tolist(),
        "median": float(np.median(gf)), "min": float(gf.min()), "max": float(gf.max()),
        "main_instance": MAIN_INSTANCE_GAP_FRACTION,
        "n_tighter_than_main": int((gf < MAIN_INSTANCE_GAP_FRACTION).sum()),
        "distinct_ground_states": sorted({f["ground_bitstring"] for f in folds}),
        "ground_state_counts": {
            b: sum(1 for f in folds if f["ground_bitstring"] == b)
            for b in sorted({f["ground_bitstring"] for f in folds})
        },
    }

    # ------------------------------------------------- the decomposition
    def per_fold_returns(arm):
        return np.asarray([r["gross_return"] for r in arms[arm]["per_fold"]])

    def paired(a, b):
        da = per_fold_returns(a) - per_fold_returns(b)
        identical = not np.any(da != 0)
        # A t-test on an all-zero difference vector is 0/0. That is not a
        # failure -- it means the two arms made the same choice on every fold,
        # which is a stronger statement than any p-value. Report it as such.
        if identical:
            tstat = pval = wilcoxon = float("nan")
        else:
            tt = stats.ttest_1samp(da, 0.0)
            tstat, pval = float(tt.statistic), float(tt.pvalue)
            wilcoxon = float(stats.wilcoxon(da).pvalue)
        return {
            "mean_diff": float(da.mean()), "sd": float(da.std(ddof=1)),
            "median_diff": float(np.median(da)),
            "n_positive": int((da > 0).sum()), "n": int(len(da)),
            "identical": bool(identical),
            "t": tstat, "p": pval, "wilcoxon_p": wilcoxon,
            "mde_80pct": minimum_detectable_effect(float(da.std(ddof=1)), len(da)),
        }

    out["arms"] = {
        name: {kk: a[kk] for kk in
               ("total_return", "cagr", "ann_vol", "sharpe", "total_cost")}
        for name, a in arms.items()
    }
    out["decomposition"] = {}
    if "quantum_cvar040" in arms:
        out["decomposition"]["optimiser_error"] = paired("quantum_cvar040",
                                                         "best_equal_weight")
    out["decomposition"]["objective_value"] = paired("best_equal_weight",
                                                     "equal_weight_4")
    best_single = max((kk for kk in arms if kk.startswith("buy_hold_")),
                      key=lambda kk: arms[kk]["sharpe"])
    out["decomposition"]["vs_best_single"] = paired("best_equal_weight", best_single)
    out["decomposition"]["best_single"] = best_single

    # --------------------------------------------------- quantum fidelity
    if quantum:
        matched = sum(q["matched"] for q in quantum)
        lo, hi = clopper_pearson(matched, len(quantum))
        out["quantum"] = {
            "matched": matched, "n": len(quantum),
            "rate": matched / len(quantum), "ci95": [lo, hi],
            "infeasible": sum(not q["feasible"] for q in quantum),
            "mean_ground_prob": float(np.mean([q["ground_state_prob"] for q in quantum])),
            "misses": [{"fold": q["fold"], "readout": q["readout"],
                        "target": q["target"], "gap": q["energy_gap"]}
                       for q in quantum if not q["matched"]],
        }
    return out


def make_figures(payload: dict, res: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    folds = payload["folds"]
    arms = payload["arms"]

    # ---- 1. equity curves
    fig, ax = plt.subplots(figsize=(9, 5))
    # The quantum arm matched the perfect optimiser on every fold, so its curve
    # sits exactly on top. Drawn dashed over solid so the coincidence reads as
    # a result rather than as a missing series.
    styles = {
        "best_equal_weight": dict(
            label="best_equal_weight (perfect optimiser)",
            lw=3.4, color="#4C72B0", ls="-", marker="o", ms=4, zorder=2),
        "equal_weight_4": dict(
            label="equal-weight all 4",
            lw=2, color="#DD8452", ls="-", marker="o", ms=3, zorder=2),
        "quantum_cvar040": dict(
            label="quantum (CVaR a=0.4, k=1) - identical, 16/16",
            lw=1.7, color="#55A868", ls=(0, (4, 3)), marker="", zorder=3),
    }
    for name, style in styles.items():
        if name not in arms:
            continue
        curve = [1.0] + arms[name]["equity_curve"]
        ax.plot(range(len(curve)), curve, **style)
    for name in arms:
        if name.startswith("buy_hold_"):
            curve = [1.0] + arms[name]["equity_curve"]
            ax.plot(range(len(curve)), curve, alpha=0.35, ls="--", lw=1,
                    label=name.replace("buy_hold_", ""))
    ax.set_yscale("log")
    ax.set_xlabel("fold (63-day holding periods)")
    ax.set_ylabel("equity, log scale (start = 1.0)")
    ax.set_title("Phase 8: net-of-cost equity, 16 quarterly folds\n"
                 "selection bias makes the level uninformative; "
                 "the comparison between arms is the content", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "backtest_equity.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # ---- 2. in-sample vs out-of-sample Sharpe
    fig, ax = plt.subplots(figsize=(6.5, 6))
    xs, ys = [], []
    for f in folds:
        for p_ in f["portfolios"]:
            xs.append(p_["is_sharpe"])
            ys.append(p_["oos_sharpe"])
    ax.scatter(xs, ys, alpha=0.55, s=26, edgecolor="none")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    sp = res["spearman"]
    ax.set_xlabel("in-sample Sharpe (250-day estimate)")
    ax.set_ylabel("realised Sharpe (next 63 days)")
    ax.set_title(f"96 portfolio-folds: the objective has little predictive content\n"
                 f"mean per-fold Spearman rho = {sp['mean']:+.3f} "
                 f"(p = {sp['p']:.2f}, n = {res['n_folds']} folds)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG_DIR / "backtest_is_vs_oos.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # ---- 3. per-fold degeneracy gap
    fig, ax = plt.subplots(figsize=(9, 4))
    gf = np.asarray(res["degeneracy"]["gap_fraction_per_fold"]) * 100
    ax.bar(range(len(gf)), gf, color="#4C72B0", alpha=0.85)
    ax.axhline(MAIN_INSTANCE_GAP_FRACTION * 100, color="crimson", ls="--", lw=1.6,
               label=f"main instance ({MAIN_INSTANCE_GAP_FRACTION*100:.2f}%)")
    for i, f in enumerate(folds):
        ax.text(i, gf[i] + 0.06, f["ground_bitstring"], ha="center", fontsize=6.5)
    ax.set_xlabel("fold")
    ax.set_ylabel("top-two gap, % of spectral width")
    n_tighter = res["degeneracy"]["n_tighter_than_main"]
    ax.set_title(
        f"The main instance is unusually hard, but its difficulty recurs\n"
        f"median {res['degeneracy']['median']:.2%} vs main {MAIN_INSTANCE_GAP_FRACTION:.2%}; "
        f"{n_tighter} of {len(gf)} folds are tighter  (label = that fold's ground state)",
        fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = FIG_DIR / "backtest_degeneracy.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    return paths


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not RESULTS_JSON.exists():
        print(f"Missing {RESULTS_JSON}. Run: python backtest/run_backtest.py")
        return 1
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    res = analyse(payload)
    ANALYSIS_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")

    figs = make_figures(payload, res)

    sp = res["spearman"]
    td = res["top_two_discrimination"]
    pr = res["tsla_amd_vs_nvda_amd"]
    dg = res["degeneracy"]

    print(f"\n=== {res['n_folds']} folds ===")
    print(f"\n[1] Does in-sample Sharpe predict out-of-sample Sharpe?")
    print(f"    per-fold Spearman rho : mean {sp['mean']:+.3f}  median {sp['median']:+.3f}"
          f"  sd {sp['sd']:.3f}")
    print(f"    95% CI                : [{sp['ci95'][0]:+.3f}, {sp['ci95'][1]:+.3f}]"
          f"   t = {sp['t']:+.2f}, p = {sp['p']:.3f}")
    print(f"    folds with rho > 0    : {sp['n_positive']}/{res['n_folds']}")

    it = res["is_top_is_oos_top"]
    print(f"\n[2] In-sample top is also out-of-sample top: {it['k']}/{it['n']}"
          f" = {it['rate']:.1%}  CI [{it['ci95'][0]:.1%}, {it['ci95'][1]:.1%}]")
    print(f"    chance = {it['chance_rate']:.1%}, p = {it['p_vs_chance']:.3f}")

    print(f"\n[3] Top-two discrimination (the question VQE is asked to resolve)")
    print(f"    ground beats runner-up OOS : {td['k']}/{td['n']} = {td['rate']:.1%}"
          f"  CI [{td['ci95'][0]:.1%}, {td['ci95'][1]:.1%}]  p = {td['p_vs_coinflip']:.3f}")
    print(f"    mean dSharpe  in-sample    : {td['mean_d_is_sharpe']:+.4f}")
    print(f"    mean dSharpe out-of-sample : {td['mean_d_oos_sharpe']:+.4f}"
          f"  (sd {td['sd_d_oos_sharpe']:.3f}, p = {td['p']:.3f})")
    print(f"    Spearman(dIS, dOOS)        : {td['spearman_dis_doos']:+.3f}"
          f"  (p = {td['spearman_dis_doos_p']:.3f})")
    print(f"    80%-power MDE at n={td['n']}     : {td['mde_80pct']:.3f} Sharpe"
          f"  vs in-sample edge {td['mean_d_is_sharpe']:.3f}")

    print(f"\n[4] 1010 (TSLA+AMD) vs 0110 (NVDA+AMD): "
          f"{pr['wins']}W {pr['losses']}L {pr['ties']}T = {pr['rate']:.1%}"
          f"  p = {pr['p_vs_coinflip']:.3f}")

    ms = td["mean_sensitivity"]
    print(f"\n[4b] Trust the sign test, not the mean")
    print(f"     sd / |mean| = {ms['sd_over_mean']:.2f}x  -- the mean is fragile")
    print(f"     drop fold {ms['most_influential_fold']} (d_oos "
          f"{ms['its_d_oos']:+.4f}) -> mean {ms['mean_all']:+.4f} becomes "
          f"{ms['mean_without_it']:+.4f}")

    gs = res["gap_median_split"]
    print(f"\n[4c] Is the result specific to tight gaps? Split at median "
          f"{gs['median_gap_fraction']:.2%}")
    for name in ("tight", "loose"):
        h = gs[name]
        print(f"     {name:<6} gap {h['gap_min']:.2%}-{h['gap_max']:.2%}  "
              f"{h['k']}/{h['n']} = {h['rate']:.1%}  "
              f"CI [{h['ci95'][0]:.1%}, {h['ci95'][1]:.1%}]  "
              f"mean d_oos {h['mean_d_oos']:+.4f}")
    print(f"     Fisher exact tight vs loose : p = {gs['fisher_p']:.3f}")
    print(f"     Spearman(gap, win)          : {gs['spearman_gap_win']:+.3f} "
          f"(p = {gs['spearman_gap_win_p']:.3f})")
    print(f"     Spearman(gap, d_oos)        : {gs['spearman_gap_doos']:+.3f} "
          f"(p = {gs['spearman_gap_doos_p']:.3f})")

    print(f"\n[5] Degeneracy gap across folds (% of spectral width)")
    print(f"    median {dg['median']:.2%}  min {dg['min']:.2%}  max {dg['max']:.2%}"
          f"   main instance {dg['main_instance']:.2%}")
    print(f"    folds tighter than the main instance: {dg['n_tighter_than_main']}")
    print(f"    distinct ground states across folds : {dg['ground_state_counts']}")

    if "quantum" in res:
        q = res["quantum"]
        print(f"\n[6] Quantum arm (CVaR a=0.4, k=1): matched {q['matched']}/{q['n']}"
              f" = {q['rate']:.1%}  CI [{q['ci95'][0]:.1%}, {q['ci95'][1]:.1%}]")
        print(f"    infeasible readouts: {q['infeasible']},"
              f" mean P(ground) = {q['mean_ground_prob']:.4f}")
        for m in q["misses"]:
            print(f"      fold {m['fold']:>2}: read {m['readout']} want {m['target']}"
                  f"  gap {m['gap']:.4f}")

    print(f"\n[7] Decomposition (per-fold gross returns, paired)")
    for key, label in (("optimiser_error", "quantum - best_equal_weight  (optimiser)"),
                       ("objective_value", "best_equal_weight - EW4      (objective)"),
                       ("vs_best_single", f"best_equal_weight - {res['decomposition']['best_single']}")):
        d = res["decomposition"].get(key)
        if d is None:
            continue
        if d.get("identical"):
            print(f"    {label:<44} IDENTICAL on all {d['n']} folds "
                  f"(difference exactly 0; no test applies)")
            continue
        print(f"    {label:<44} mean {d['mean_diff']:+.4f}  "
              f"p = {d['p']:.3f}  {d['n_positive']}/{d['n']} folds positive")
        print(f"    {'':<44} 80%-power MDE {d['mde_80pct']:+.4f}")

    ce = res["cost_of_an_optimiser_error"]
    print(f"\n[7b] What one optimiser error would have cost (ground - runner-up,")
    print(f"     per-fold total return -- same unit as [7])")
    print(f"     mean {ce['mean']:+.4f}  median {ce['median']:+.4f}  sd {ce['sd']:.4f}")
    print(f"     95% CI [{ce['ci95'][0]:+.4f}, {ce['ci95'][1]:+.4f}]"
          f"   t = {ce['t']:+.3f}, p = {ce['p']:.3f}")
    print(f"     folds where the ground state paid more: "
          f"{ce['n_positive']}/{ce['n']}")

    print(f"\n[8] Arms")
    print(f"    {'arm':<24} {'total':>9} {'CAGR':>8} {'vol':>7} {'Sharpe':>8}")
    for name, a in res["arms"].items():
        print(f"    {name:<24} {a['total_return']:>+8.1%} {a['cagr']:>+7.1%} "
              f"{a['ann_vol']:>6.1%} {a['sharpe']:>+8.3f}")

    print(f"\nFigures: {', '.join(p.name for p in figs)}")
    print(f"Wrote {ANALYSIS_JSON}")
    print("\nBACKTEST ANALYSIS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
