"""One-panel figure: what CVaR's alpha buys, and what it costs.

The Phase 9 headline is a TRADE, not a rate (objective_comparison.md section 1),
so a plot of accuracy alone would misreport it. Readout accuracy and mean energy
go on the same panel, on opposite axes, because the whole point is that they
move in opposite directions.

Sized for video: large type, minimal gridlines, five labelled points, and
alpha = 1.0 named as plain VQE so nobody has to infer that it is the baseline.
Reads the committed per-arm summaries; regenerable from a fresh clone.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARMS = [
    (0.1, "cvar_a010"),
    (0.4, "cvar_a040"),
    (0.5, "cvar_a050"),
    (0.6, "cvar_a060"),
    (1.0, "lr_constant_200"),
]
OUT = PROJECT_ROOT / "docs" / "figures" / "alpha_sweep.png"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval -- the one used throughout this project."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for alpha, tag in ARMS:
        d = json.loads((PROJECT_ROOT / "training" / "arms" / tag /
                        "training_summary.json").read_text(encoding="utf-8"))
        runs = d["runs"]
        n = len(runs)
        k = sum(bool(r["found_ground_state"]) for r in runs)
        lo, hi = wilson(k, n)
        rows.append({
            "alpha": alpha, "tag": tag, "k": k, "n": n, "rate": k / n,
            "lo": lo, "hi": hi,
            "mean_energy": float(np.mean([r["final_energy"] for r in runs])),
            "ground_energy": float(d["ground_energy"]),
        })
        print(f"  alpha={alpha:<4} {tag:<16} {k:>2}/{n}  {k/n:>6.1%}  "
              f"Wilson [{lo:.1%}, {hi:.1%}]  mean <H> {rows[-1]['mean_energy']:+.4f}")

    x = np.arange(len(rows))
    rate = np.array([r["rate"] for r in rows]) * 100
    lo = np.array([r["lo"] for r in rows]) * 100
    hi = np.array([r["hi"] for r in rows]) * 100
    energy = np.array([r["mean_energy"] for r in rows])
    ground = rows[0]["ground_energy"]
    labels = [f"{r['alpha']:g}" for r in rows]
    labels[-1] = "1.0"

    plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.4})
    fig, ax = plt.subplots(figsize=(13, 7.8))

    ACC, ENE = "#1f6fb4", "#c0392b"

    # --- accuracy, left axis
    ax.errorbar(x, rate, yerr=[rate - lo, hi - rate], fmt="o-", color=ACC,
                lw=3.4, ms=15, capsize=9, capthick=2.6, elinewidth=2.6,
                zorder=3, label="readout accuracy (Wilson 95%)")
    ax.set_ylabel("runs that found the ground state  (%)", color=ACC,
                  fontsize=19, labelpad=12)
    ax.tick_params(axis="y", labelcolor=ACC, labelsize=17, width=1.4, length=7)
    ax.set_ylim(18, 118)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=19)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("CVaR tail fraction  α", fontsize=19, labelpad=10)
    ax.grid(axis="y", alpha=0.22, lw=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)

    # --- mean energy, right axis
    ax2 = ax.twinx()
    ax2.plot(x, energy, "s--", color=ENE, lw=3.0, ms=13, zorder=3,
             label="mean final ⟨H⟩")
    ax2.axhline(ground, color=ENE, lw=1.6, ls=":", alpha=0.6)
    # Label the reference line INSIDE the axes on the left, where the red
    # series does not run -- on the right it collided with the axis label.
    ax2.text(x[0] - 0.38, ground, "ground energy", color=ENE, fontsize=15,
             va="top", ha="left", alpha=0.9)
    ax2.set_ylabel("mean final ⟨H⟩   (lower = better converged)",
                   color=ENE, fontsize=19, labelpad=14)
    ax2.tick_params(axis="y", labelcolor=ENE, labelsize=17, width=1.4, length=7)
    ax2.set_ylim(-6.4, 1.9)
    ax2.spines["top"].set_visible(False)

    # --- the two points that carry the story
    best = int(np.argmax(rate))
    ax.annotate(str(rows[best]["k"]) + "/" + str(rows[best]["n"]),
                (x[best], hi[best]), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=23, fontweight="bold",
                color=ACC)
    # Baseline callout goes ABOVE its point: below and right are occupied by
    # the error bar and the axis edge. Single line -- two lines needed an
    # escape that does not survive being written programmatically.
    ax.annotate("plain VQE (no CVaR)", (x[-1], hi[-1]),
                textcoords="offset points", xytext=(0, 34), ha="center",
                fontsize=17, color=ACC,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=2.2,
                                shrinkA=2, shrinkB=6))
    ax.annotate(str(rows[-1]["k"]) + "/" + str(rows[-1]["n"]),
                (x[-1], rate[-1]), textcoords="offset points",
                xytext=(-46, -6), ha="center", fontsize=20,
                fontweight="bold", color=ACC)

    ax.set_title("Changing the objective trades energy for a correct readout"
                 + chr(10) + "30 seeds per point, same seeds, 200 epochs",
                 fontsize=21, pad=16)

    # Legend below the axes: every in-plot region is occupied by one of the
    # two series or their error bars.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center",
              bbox_to_anchor=(0.5, -0.135), fontsize=17, ncol=2,
              frameon=False, handletextpad=0.8, columnspacing=3.0)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"\n  wrote {OUT.relative_to(PROJECT_ROOT)}")
    print("ALPHA SWEEP FIGURE: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
