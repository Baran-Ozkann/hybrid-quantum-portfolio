"""The headline backtest test, drawn for video.

Separate from ``docs/figures/backtest_discrimination.png``, which keeps its
per-fold numbers and the fold-13 callout: that figure serves a reader who can
study it, this one has about four seconds on screen.

So the fold numbers are dropped. Sixteen labelled points in one panel is
exactly where labels collide, and the numbers carry nothing a viewer can use at
that speed -- the colours and the scatter are the whole message.

What the plot says: if the objective were predictive, points would lie on a
rising line. They do not, and green/red is split almost evenly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "video_shots" / "04_backtest_discrimination.png"

NL = chr(10)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads((ROOT / "logs" / "backtest_results.json")
                         .read_text(encoding="utf-8"))
    dis, doos = [], []
    for f in payload["folds"]:
        by = {p["bitstring"]: p for p in f["portfolios"]}
        g, r = by.get(f["ground_bitstring"]), by.get(f["runner_up_bitstring"])
        if g is None or r is None:
            continue
        dis.append(g["is_sharpe"] - r["is_sharpe"])
        doos.append(g["oos_sharpe"] - r["oos_sharpe"])
    dis, doos = np.asarray(dis), np.asarray(doos)
    won = doos > 0
    rho = stats.spearmanr(dis, doos)
    p_bin = stats.binomtest(int(won.sum()), len(won), 0.5).pvalue

    plt.rcParams.update({"font.size": 17, "axes.linewidth": 1.4})
    fig, ax = plt.subplots(figsize=(12.5, 7.6))

    ax.axhspan(0, doos.max() * 1.25, color="#55A868", alpha=0.05, zorder=0)
    ax.axhline(0, color="black", lw=1.8, zorder=1)
    ax.axvline(0, color="black", lw=1.8, zorder=1)

    ax.scatter(dis[won], doos[won], s=260, c="#55A868", edgecolor="white",
               lw=1.8, zorder=3,
               label="objective's pick won   (" + str(int(won.sum())) + ")")
    ax.scatter(dis[~won], doos[~won], s=300, c="#C44E52", marker="X",
               edgecolor="white", lw=1.8, zorder=3,
               label="objective's pick lost   (" + str(int((~won).sum())) + ")")

    ax.set_xlabel("how much the objective preferred its pick"
                  + NL + "(in-sample " + chr(0x394) + " Sharpe)",
                  fontsize=20, labelpad=10)
    ax.set_ylabel("what that pick actually earned" + NL
                  + "(realised " + chr(0x394) + " Sharpe, next quarter)",
                  fontsize=20, labelpad=10)
    ax.set_title("A bigger in-sample edge does not predict a bigger real one"
                 + NL + "16 quarters   "
                 + chr(0x3c1) + " = " + format(rho.statistic, "+.3f")
                 + " (p = " + format(rho.pvalue, ".2f") + ")   "
                 + "pick wins " + str(int(won.sum())) + "/" + str(len(won))
                 + " (p = " + format(p_bin, ".2f") + ")",
                 fontsize=21, pad=16)
    ax.tick_params(labelsize=17, width=1.4, length=7)
    ax.grid(alpha=0.2, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(dis.min() - 0.10, dis.max() + 0.10)
    ax.set_ylim(doos.min() - 0.55, doos.max() + 0.70)
    ax.legend(fontsize=17, loc="upper left", framealpha=0.95, borderpad=0.8)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("  " + str(int(won.sum())) + "/" + str(len(won))
          + "  rho = " + format(rho.statistic, "+.3f")
          + "  p = " + format(rho.pvalue, ".3f"))
    print("  wrote " + str(OUT.relative_to(ROOT)))
    print("BACKTEST DISCRIMINATION FIGURE: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
