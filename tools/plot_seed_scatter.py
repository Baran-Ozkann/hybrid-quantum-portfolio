"""Standalone 30-seed scatter, sized for video.

The same data as the left panel of ``stability_summary.png``, redrawn rather
than cropped: the panel is 731 px wide in that figure and would be soft when
upscaled, and its title runs past a one-third crop.

The point of the plot is that the x-axis (how well the run converged) does not
determine the colour (whether it got the right answer). Orange points sitting
far to the right, and blue points sitting hard against the ground-state line,
are both the story.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "video_shots" / "02_seed_scatter.png"

NL = chr(10)
LB, RB = chr(0x27e8), chr(0x27e9)

#: Runner-up energy, from brute-force enumeration of all 16 bitstrings.
RUNNER_UP = -4.881910


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads((ROOT / "training" / "training_summary.json")
                   .read_text(encoding="utf-8"))
    runs = d["runs"]
    ground = float(d["ground_energy"])
    E = np.array([r["final_energy"] for r in runs])
    P = np.array([r["ground_state_prob"] for r in runs])
    ok = np.array([r["mode_bitstring"] for r in runs]) == "1010"

    plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.4})
    fig, ax = plt.subplots(figsize=(12.5, 7.6))

    ax.axvline(ground, color="black", ls="--", lw=2.2, zorder=1)
    ax.axvline(RUNNER_UP, color="0.45", ls=":", lw=2.0, zorder=1)

    ax.scatter(E[ok], P[ok], s=210, c="#e8871a", edgecolor="white", lw=1.6,
               zorder=3, label="read out 1010  (correct, n=" + str(int(ok.sum())) + ")")
    ax.scatter(E[~ok], P[~ok], s=210, c="#1f6fb4", edgecolor="white", lw=1.6,
               zorder=3, label="read out 0110  (wrong,   n=" + str(int((~ok).sum())) + ")")

    # Rotated, alongside their lines: the two energies differ by 0.039, so
    # horizontal labels overlap each other at this scale.
    ax.text(ground - 0.004, 0.60, "ground state", rotation=90, fontsize=16,
            ha="right", va="center", color="black")
    ax.text(RUNNER_UP + 0.004, 0.60, "runner-up", rotation=90, fontsize=16,
            ha="left", va="center", color="0.35")

    ax.set_xlabel("final " + LB + "H" + RB + "   (lower = better converged)",
                  fontsize=20, labelpad=10)
    ax.set_ylabel("P(ground state)", fontsize=20, labelpad=10)
    ax.set_title("Converged does not mean correct" + NL
                 + "30 random seeds, identical settings, 100 epochs",
                 fontsize=22, pad=16)
    ax.set_ylim(-0.07, 1.10)
    ax.set_xlim(-4.947, -4.32)
    ax.tick_params(labelsize=17, width=1.4, length=7)
    ax.grid(alpha=0.2, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=17, loc="center right", framealpha=0.95, borderpad=0.8)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("  wrote " + str(OUT.relative_to(ROOT)))
    print("SEED SCATTER FIGURE: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
