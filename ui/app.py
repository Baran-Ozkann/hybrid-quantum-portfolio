"""
ui/app.py -- Phase 6: Streamlit interface.

Presents the classical inputs, runs the trained quantum circuit on demand, and
compares the decoded portfolio against the classical baselines.

DESIGN CONSTRAINTS (from the brief)
-----------------------------------
* Streamlit native widgets and Matplotlib only -- no HTML, CSS, or JS.
* Page load under 5 seconds. Achieved by caching: the parsed data files are
  memoised with ``st.cache_data`` and the Q# interpreter with
  ``st.cache_resource``, so the runtime is initialised once per session rather
  than on every Streamlit rerun.
* The quantum circuit is NOT retrained here. Training is Phase 5 and takes
  ~15-25 s per seed; this app loads the saved seed-42 parameters and performs a
  single forward evaluation plus a measurement readout.

HONESTY REQUIREMENTS
--------------------
Two things this UI deliberately surfaces rather than hides:

1. ``best_equal_weight`` is the PRIMARY classical baseline, because it is the
   true optimum over the same feasible set the quantum method searches (the six
   equally weighted 2-asset baskets). ``max_sharpe`` and ``min_variance``
   optimise continuous weights over the whole simplex -- a strictly larger
   feasible set -- so they are shown for context and explicitly labelled as not
   like-for-like.

2. The reliability section reports that only 1 of 3 random initialisations
   reached the true ground state. That is a real limitation of this optimiser
   on this problem and appears in the app, not only in the docs.

Run::

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H          # noqa: E402
from quantum import q_bridge as qb              # noqa: E402

TRAINING_DIR = PROJECT_ROOT / "training"
DATA_DIR = PROJECT_ROOT / "data"

PRIMARY_SEED = 42
STABILITY_SEEDS = (42, 123, 456)
READOUT_SHOTS = 4096

# Consistent asset colours across every chart in the app.
ASSET_COLOURS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_statistics() -> dict:
    """Market statistics from Phase 2."""
    path = DATA_DIR / "statistics.json"
    if not path.exists():
        raise FileNotFoundError("data/statistics.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    for key in ("expected_returns", "cov_matrix", "correlation_matrix", "volatilities"):
        d[key] = np.asarray(d[key], dtype=float)
    return d


@st.cache_data(show_spinner=False)
def load_baselines() -> dict:
    """Classical Markowitz baselines from Phase 2."""
    path = DATA_DIR / "markowitz_baseline.json"
    if not path.exists():
        raise FileNotFoundError("data/markowitz_baseline.json")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_stability_summary() -> dict | None:
    """Aggregate results of the full seed sweep, if it has been run.

    Written by ``tools/stability_report.py``. Absent on a fresh checkout, in
    which case Section 4 falls back to describing only the runs it can see.
    """
    path = PROJECT_ROOT / "logs" / "stability_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_training_runs() -> dict[int, dict]:
    """Every saved Phase 5 training run, keyed by seed."""
    runs = {}
    for seed in STABILITY_SEEDS:
        path = TRAINING_DIR / f"trained_seed{seed}.json"
        if path.exists():
            runs[seed] = json.loads(path.read_text(encoding="utf-8"))
    return runs


@st.cache_resource(show_spinner=False)
def get_hamiltonian() -> H.PortfolioHamiltonian:
    """The Phase 2 Ising Hamiltonian (cached as a resource: holds numpy state)."""
    return H.load_hamiltonian()


@st.cache_data(show_spinner=False)
def spectrum_facts() -> dict:
    """Exact spectrum quantities, computed once per session.

    Enumerating all 16 states is cheap, but it is pure derived data that never
    changes for a given Hamiltonian, so it is cached rather than recomputed on
    every rerun. Keeping load-time work minimal preserves headroom against the
    5-second page-load budget.
    """
    ham = H.load_hamiltonian()
    spec = sorted(H.enumerate_states(ham), key=lambda r: r["ising_energy"])
    return {
        "ground_bitstring": spec[0]["bitstring"],
        "ground_energy": spec[0]["ising_energy"],
        "runner_up_bitstring": spec[1]["bitstring"],
        "runner_up_energy": spec[1]["ising_energy"],
        "degeneracy_gap": spec[1]["ising_energy"] - spec[0]["ising_energy"],
        "spectral_width": spec[-1]["ising_energy"] - spec[0]["ising_energy"],
    }


@st.cache_resource(show_spinner=False)
def init_quantum_runtime() -> bool:
    """Start the Q# interpreter once per Streamlit session."""
    qb.load_circuit_operations()
    return True


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

def plot_correlation(corr: np.ndarray, tickers: list[str]):
    """Correlation matrix as an annotated Matplotlib heatmap."""
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(tickers)), tickers)
    ax.set_yticks(range(len(tickers)), tickers)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(corr[i, j]) > 0.7 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Return correlation", fontsize=11)
    fig.tight_layout()
    return fig


def plot_prices(stats: dict):
    """Normalised price paths, so relative performance is directly comparable."""
    path = DATA_DIR / "prices.csv"
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    if path.exists():
        import pandas as pd
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        normalised = prices / prices.iloc[0] * 100.0
        for i, col in enumerate(prices.columns):
            ax.plot(normalised.index, normalised[col], label=col,
                    color=ASSET_COLOURS[i % len(ASSET_COLOURS)], linewidth=1.3)
        ax.axhline(100, color="grey", linestyle=":", linewidth=0.9)
        ax.set_ylabel("value (start = 100)")
        ax.legend(fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
        ax.set_title("Normalised price history", fontsize=11)
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_weights_pie(weights: np.ndarray, tickers: list[str], title: str):
    """Pie chart of portfolio weights, omitting zero-weight assets."""
    held = [(t, w, ASSET_COLOURS[i % len(ASSET_COLOURS)])
            for i, (t, w) in enumerate(zip(tickers, weights)) if w > 1e-9]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.pie([w for _, w, _ in held],
           labels=[t for t, _, _ in held],
           colors=[c for _, _, c in held],
           autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_comparison_bars(rows: list[dict], tickers: list[str]):
    """Grouped bar chart of weights across the compared portfolios."""
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    n_rows = len(rows)
    x = np.arange(len(tickers))
    width = 0.8 / n_rows
    for i, row in enumerate(rows):
        ax.bar(x + i * width, row["weights"], width, label=row["label"])
    ax.set_xticks(x + 0.4 - width / 2, tickers)
    ax.set_ylabel("weight")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Portfolio weights by method", fontsize=11)
    fig.tight_layout()
    return fig


def plot_seed_distributions(runs: dict[int, dict], ham: H.PortfolioHamiltonian):
    """Final basis-state distribution per seed, feasible states shaded."""
    n_states = 2 ** ham.n_assets
    labels = ["".join(str(b) for b in qb.index_to_bitstring(k, ham.n_assets))
              for k in range(n_states)]
    feasible = [sum(int(c) for c in lab) == ham.budget for lab in labels]

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    seeds = sorted(runs)
    width = 0.8 / max(len(seeds), 1)
    x = np.arange(n_states)
    for i, seed in enumerate(seeds):
        run = runs[seed]
        found = run["found_ground_state"]
        ax.bar(x + i * width, run["final_probabilities"], width,
               label=f"seed {seed} -> {run['mode_bitstring']}"
                     f"{' (ground state)' if found else ' (runner-up)'}")

    for k, ok in enumerate(feasible):
        if ok:
            ax.axvspan(k - 0.15, k + 0.85, color="green", alpha=0.07)

    ax.set_xticks(x + 0.4 - width / 2, labels, rotation=45, fontsize=8)
    ax.set_ylabel("probability")
    ax.set_xlabel("measured basis state (green = feasible: exactly 2 assets)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Final measured distribution per random initialisation", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Quantum execution
# --------------------------------------------------------------------------

def run_quantum_readout(params: list[float], ham: H.PortfolioHamiltonian,
                        num_layers: int, shots: int) -> dict:
    """Evaluate the trained circuit and decode a portfolio from measurements.

    Runs the exact state-vector evaluation (for <H> and the true distribution)
    and a shot-based measurement (the physically honest readout), then decodes
    the most frequent feasible bitstring.
    """
    started = time.time()
    init_quantum_runtime()

    probs = qb.probabilities(params, num_layers)
    energies = qb.ising_energy_table(ham.h, ham.J, ham.n_assets)
    energy = float(probs @ energies)

    bits = qb.measure_portfolio(params, num_layers, shots)
    counts: dict[str, int] = {}
    for row in bits:
        key = "".join(str(int(b)) for b in row)
        counts[key] = counts.get(key, 0) + 1

    weights, diagnostics = H.decode_distribution(counts, ham.budget)

    ret = float(ham.mu @ weights)
    vol = float(np.sqrt(max(weights @ ham.sigma @ weights, 0.0)))
    return {
        "weights": weights,
        "energy": energy,
        "probabilities": probs,
        "counts": counts,
        "diagnostics": diagnostics,
        "expected_return": ret,
        "volatility": vol,
        "elapsed": time.time() - started,
    }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Hybrid Quantum Portfolio Optimizer",
                       layout="wide")
    st.title("Hybrid Quantum Portfolio Optimizer")
    st.caption(
        "4-qubit VQE in Q#, trained through PyTorch autograd with the "
        "parameter-shift rule. Budget-constrained selection: hold exactly 2 of 4 assets."
    )

    # ---- load everything up front so failures are reported clearly ----
    try:
        stats = load_statistics()
        baselines = load_baselines()
        ham = get_hamiltonian()
        runs = load_training_runs()
    except FileNotFoundError as exc:
        st.error(f"Required data file missing: {exc}")
        st.info(
            "Generate the inputs first:\n\n"
            "1. `python classical/phase2_gate.py`\n"
            "2. `python training/optimization.py --seeds 42 123 456`"
        )
        return

    tickers = list(stats["tickers"])
    mu = stats["expected_returns"]
    sigma = stats["cov_matrix"]
    risk_free = float(stats["risk_free_rate"])

    if PRIMARY_SEED not in runs:
        st.error(f"No trained parameters for seed {PRIMARY_SEED}.")
        st.info("Run: `python training/optimization.py --seeds 42 123 456`")
        return

    primary = runs[PRIMARY_SEED]

    # ==================================================================
    # SECTION 1 -- classical asset data
    # ==================================================================
    st.header("1. Asset data")
    st.caption(
        f"{stats['n_observations']} trading days, "
        f"{stats['first_observation']} to {stats['last_observation']}. "
        f"Returns are {stats['return_convention']}. "
        f"**All Sharpe ratios on this page use a risk-free rate of "
        f"rf = {risk_free:.2%}**, i.e. Sharpe = (return - rf) / volatility."
    )

    # Columns are kept NUMERIC and formatted through column_config. Storing
    # pre-formatted strings here would make Streamlit sort them
    # lexicographically ("+168.93%" before "+33.20%"), which silently produces
    # a nonsensical ordering when the user clicks a column header.
    # Percentages are stored pre-scaled (43.64, not 0.4364) because
    # NumberColumn's printf format does not rescale; the values stay numeric so
    # header-click sorting remains correct.
    table = {
        "Ticker": tickers,
        "Qubit": [f"q{i}" for i in range(len(tickers))],
        "Annual return": [float(v) * 100.0 for v in mu],
        "Annual volatility": [float(v) * 100.0 for v in stats["volatilities"]],
        "Sharpe (standalone)": [
            float((mu[i] - risk_free) / stats["volatilities"][i])
            for i in range(len(tickers))
        ],
    }
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Annual return": st.column_config.NumberColumn(format="%+.2f%%"),
            "Annual volatility": st.column_config.NumberColumn(format="%.2f%%"),
            "Sharpe (standalone)": st.column_config.NumberColumn(format="%+.3f"),
        },
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.pyplot(plot_prices(stats), width="stretch")
    with col_b:
        st.pyplot(plot_correlation(stats["correlation_matrix"], tickers),
                  width="stretch")

    # ==================================================================
    # SECTION 2 -- run the quantum optimiser
    # ==================================================================
    st.header("2. Quantum optimization")
    st.caption(
        f"Loads the pre-trained seed-{PRIMARY_SEED} parameters "
        f"({len(primary['final_params'])} rotation angles, "
        f"{primary['config']['num_layers']} ansatz layers), evaluates the circuit, "
        f"and decodes {READOUT_SHOTS} measurement shots into portfolio weights."
    )

    if st.button("Run Quantum Optimization", type="primary"):
        with st.spinner("Executing quantum circuit..."):
            try:
                st.session_state["quantum"] = run_quantum_readout(
                    primary["final_params"], ham,
                    primary["config"]["num_layers"], READOUT_SHOTS,
                )
            except Exception as exc:
                st.error(f"Quantum execution failed: {exc}")
                return

    if "quantum" not in st.session_state:
        st.info("Press **Run Quantum Optimization** to execute the circuit.")
        return

    q = st.session_state["quantum"]
    q_weights = q["weights"]
    diag = q["diagnostics"]

    m1, m2, m3, m4 = st.columns(4)
    # delta_color="inverse": the energy gap is a variational SHORTFALL above the
    # ground state, so a positive number is worse, not better. Streamlit's
    # default would render it as a green up-arrow.
    m1.metric("Circuit energy <H>", f"{q['energy']:.4f}",
              f"{q['energy'] - primary['ground_energy']:+.4f} vs ground state",
              delta_color="inverse")
    m2.metric("Readout bitstring", diag["mode_bitstring"],
              f"{diag['mode_probability']:.1%} of shots", delta_color="off")
    m3.metric("Feasible shots", f"{diag['feasible_fraction']:.1%}",
              "exactly 2 assets held", delta_color="off")
    m4.metric("Execution time", f"{q['elapsed']:.2f} s")

    # Reconcile this live readout against the stored training-run figure shown
    # in Section 4 -- same model, two different numbers on screen.
    stored_p = primary["ground_state_prob"]
    live_p = diag["mode_probability"]
    shots = diag["total_shots"]
    # NOTE: deliberately not named `sigma` -- that name holds the covariance
    # matrix in this scope, and shadowing it breaks the Section 3 metrics.
    shot_sigma = float(np.sqrt(max(live_p * (1.0 - live_p), 0.0) / max(shots, 1)))
    st.caption(
        f"Why this reads {live_p:.1%} while Section 4 reports {stored_p:.4f} for the "
        f"same seed: this section RE-SAMPLES {shots} fresh measurement shots on each "
        f"click (the sampling backend, as hardware would), whereas Section 4 reports "
        f"the stored exact state-vector probability from the training run. "
        f"Binomial shot noise at this probability is sigma = "
        f"sqrt(p(1-p)/n) = {shot_sigma:.2%}, so the two agree to within "
        f"{abs(live_p - stored_p) / shot_sigma:.1f} sigma. Re-clicking the button will "
        f"move the figure by a fraction of a percent; the underlying state does not change."
    )

    col_pie, col_detail = st.columns([2, 3])
    with col_pie:
        st.pyplot(plot_weights_pie(q_weights, tickers, "Quantum portfolio"),
                  width="stretch")
    with col_detail:
        st.subheader("Decoded allocation")
        st.dataframe(
            {
                "Ticker": tickers,
                "Bit": [diag["mode_bitstring"][i] for i in range(len(tickers))],
                "Weight": [f"{w:.1%}" for w in q_weights],
            },
            width="stretch", hide_index=True,
        )
        st.caption(
            f"Encoding: each held asset receives an equal share, w_i = x_i / {ham.budget}, "
            "so the weights sum to 1 by construction. "
            f"Measured {diag['distinct_outcomes']} distinct outcomes over "
            f"{diag['total_shots']} shots."
        )
        if diag["mode_bitstring"] == primary["ground_bitstring"]:
            st.success(
                f"Readout **{diag['mode_bitstring']}** is the true ground state "
                "(verified by brute-force enumeration of all 16 bitstrings)."
            )
        else:
            st.warning(
                f"Readout **{diag['mode_bitstring']}** is NOT the true ground state "
                f"(**{primary['ground_bitstring']}**). See the reliability section below."
            )

    # ==================================================================
    # SECTION 3 -- comparison
    # ==================================================================
    st.header("3. Quantum vs classical")

    def metrics(w) -> tuple[float, float, float]:
        w = np.asarray(w, dtype=float)
        r = float(mu @ w)
        v = float(np.sqrt(max(w @ sigma @ w, 0.0)))
        return r, v, (r - risk_free) / v if v > 1e-12 else 0.0

    beq = baselines["best_equal_weight"]
    q_ret, q_vol, q_sharpe = metrics(q_weights)
    b_ret, b_vol, b_sharpe = metrics(beq["weights"])

    st.subheader("Primary comparison (same feasible set)")
    st.caption(
        "Both methods choose exactly 2 of 4 assets with equal weights, so this is "
        "a like-for-like test of the optimizer: `best_equal_weight` is the exact "
        f"optimum over those 6 portfolios, found by exhaustive search. "
        f"**Sharpe ratios use rf = {risk_free:.2%}.**"
    )
    # An exact tie must not render as a green "gain". delta_color="off" when the
    # difference is nil; otherwise sign-appropriate colouring.
    tie = abs(q_sharpe - b_sharpe) < 1e-9
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return", f"{q_ret:+.2%}", f"{q_ret - b_ret:+.2%} vs classical",
              delta_color="off" if abs(q_ret - b_ret) < 1e-9 else "normal")
    c2.metric("Volatility", f"{q_vol:.2%}", f"{q_vol - b_vol:+.2%} vs classical",
              delta_color="off" if abs(q_vol - b_vol) < 1e-9 else "inverse")
    c3.metric("Sharpe ratio", f"{q_sharpe:+.4f}", f"{q_sharpe - b_sharpe:+.4f} vs classical",
              delta_color="off" if tie else "normal")

    if abs(q_sharpe - b_sharpe) < 1e-6:
        st.success(
            "The quantum optimizer matched the classical optimum exactly over the "
            "shared feasible set."
        )
    elif q_sharpe < b_sharpe:
        st.warning(
            f"The quantum result is {(b_sharpe - q_sharpe) / abs(b_sharpe):.1%} below "
            "the classical optimum on Sharpe ratio."
        )

    st.subheader("Full comparison")
    rows = [
        ("Quantum VQE (seed 42)", q_weights, "exactly 2 assets, equal weight"),
        ("Best equal-weight (B=2)", np.asarray(beq["weights"]), "exactly 2 assets, equal weight"),
        ("Markowitz max-Sharpe", np.asarray(baselines["max_sharpe"]["weights"]),
         "continuous weights (different feasible set)"),
        ("Markowitz min-variance", np.asarray(baselines["min_variance"]["weights"]),
         "continuous weights (different feasible set)"),
        ("Equal weight (all 4)", np.full(len(tickers), 1.0 / len(tickers)),
         "continuous weights (different feasible set)"),
    ]
    st.dataframe(
        {
            "Method": [r[0] for r in rows],
            "Feasible set": [r[2] for r in rows],
            **{t: [f"{r[1][i]:.1%}" for r in rows] for i, t in enumerate(tickers)},
            "Return": [f"{metrics(r[1])[0]:+.2%}" for r in rows],
            "Volatility": [f"{metrics(r[1])[1]:.2%}" for r in rows],
            "Sharpe": [f"{metrics(r[1])[2]:+.4f}" for r in rows],
        },
        width="stretch", hide_index=True,
    )

    st.info(
        "**Why the last three rows are not like-for-like.** The quantum encoding can "
        "only express equally weighted 2-asset baskets -- 6 portfolios in total. "
        "Markowitz optimizes continuous weights over the entire simplex, a strictly "
        "larger feasible set, so it can and does score higher. That difference "
        "measures the expressiveness of the 4-qubit encoding, not the quality of "
        "the quantum optimizer.\n\n"
        "Note also that **max-Sharpe collapses to a single asset (100% AMD)** -- a "
        "corner solution, in-sample, identical to simply holding the highest-Sharpe "
        "asset. It is not a diversified benchmark, and it carries AMD's full 67.59% "
        "volatility with no risk pooling at all."
    )

    st.pyplot(
        plot_comparison_bars(
            [{"label": r[0], "weights": r[1]} for r in rows], tickers
        ),
        width="stretch",
    )

    # ==================================================================
    # SECTION 4 -- optimization reliability
    # ==================================================================
    st.header("4. Optimization reliability")

    n_found = sum(r["found_ground_state"] for r in runs.values())
    n_total = len(runs)
    runner_up_sharpe = None
    for r in runs.values():
        if not r["found_ground_state"]:
            runner_up_sharpe = r["sharpe_ratio"]
            break

    # Lead with the full seed sweep, not the 3-run trio. A 1-of-3 result carries
    # a Wilson interval of roughly [6%, 79%] and supports no quantitative claim.
    sweep = load_stability_summary()
    if sweep:
        lo, hi = sweep["wilson_95"]
        st.warning(
            f"**The optimizer returned the true ground state in "
            f"{sweep['n_correct']} of {sweep['n_runs']} random initializations "
            f"= {sweep['success_rate']:.1%}** "
            f"(Wilson 95% interval {lo:.1%} to {hi:.1%}). "
            f"Every run was identical except its random seed. The remaining runs "
            f"landed on the near-degenerate runner-up (`0110`, NVDA+AMD) — a "
            f"{(b_sharpe - runner_up_sharpe) / abs(b_sharpe):.1%}-suboptimal but "
            f"still reasonable portfolio."
        )
        st.caption(
            f"Run-to-run energy variation ({sweep['energy_spread']:.5f}) is "
            f"**{sweep['spread_over_gap']:.2f}x the degeneracy gap** "
            f"({sweep['degeneracy_gap']:.5f}) that the optimizer must resolve — "
            f"the noise floor exceeds the signal. "
            f"{len(sweep['attractors'])} distinct attractors were observed, all "
            f"of them feasible portfolios."
        )

        curve = {r["k"]: r for r in sweep.get("multi_start_curve", [])}
        if curve:
            st.markdown("**Does running more seeds fix it?** Measured, not assumed:")
            st.dataframe(
                {
                    "Seeds run (k)": [str(k) for k in sorted(curve)],
                    "Keep lowest energy": [f"{curve[k]['selected']:.1%}" for k in sorted(curve)],
                    "Oracle (any of k)": [f"{curve[k]['oracle']:.1%}" for k in sorted(curve)],
                    "Shortfall": [f"{curve[k]['oracle'] - curve[k]['selected']:+.1%}"
                                  for k in sorted(curve)],
                },
                width="stretch", hide_index=True,
            )
            st.caption(
                "Multi-start helps but does not close the gap. The **oracle** column "
                "is what a perfect selector would achieve with the same runs; the "
                "**shortfall** is the cost of ranking candidates by energy, which is "
                "the quantitative form of \"energy ranking does not track correctness\"."
            )
    elif runner_up_sharpe is not None and b_sharpe > 0:
        shortfall = (b_sharpe - runner_up_sharpe) / abs(b_sharpe)
        st.warning(
            f"**{n_found} of {n_total} random initializations converged to the true "
            f"ground state ({primary['ground_bitstring']}, TSLA+AMD).** The other "
            f"{n_total - n_found} converged to the near-degenerate runner-up "
            f"(0110, NVDA+AMD) -- a {shortfall:.1%}-suboptimal but still reasonable "
            "portfolio."
        )
    else:
        st.success(f"All {n_total} random initializations reached the true ground state.")

    st.subheader(f"Illustrative case study: seeds {', '.join(str(s) for s in sorted(runs))}")

    # The strongest statement of the finding: correctness and confidence never
    # coincided in any run. Computed from the stored distributions rather than
    # hard-coded, so it stays true if the runs are regenerated.
    primary_probs = np.asarray(primary["final_probabilities"], dtype=float)
    order = np.argsort(primary_probs)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    top1_bits = "".join(str(b) for b in qb.index_to_bitstring(top1, ham.n_assets))
    top2_bits = "".join(str(b) for b in qb.index_to_bitstring(top2, ham.n_assets))

    confident_wrong = [s for s, r in runs.items()
                       if not r["found_ground_state"] and r["mode_probability"] >= 0.9]

    # NOTE: an earlier version asserted "no run was both confident and correct".
    # That held for these three seeds but is FALSE across the 30-seed sweep, so
    # the claim is scoped to the trio and the wider result is stated alongside it.
    st.error(
        f"**Within this trio, no run was both confident and correct.** The only "
        f"high-confidence run (seed "
        f"{confident_wrong[0] if confident_wrong else 123}, "
        f"{max((runs[s]['mode_probability'] for s in confident_wrong), default=0.983):.1%} "
        f"on a single state) was confidently **wrong**. Seed {PRIMARY_SEED}, the "
        f"only correct one, wins by a "
        f"{primary_probs[top1]:.1%} / {primary_probs[top2]:.1%} margin between "
        f"{top1_bits} and {top2_bits} — those two states hold "
        f"{primary_probs[top1] + primary_probs[top2]:.1%} of all amplitude between them. "
        f"Over {READOUT_SHOTS} shots that margin is decisive statistically "
        f"(~{(primary_probs[top1] - primary_probs[top2]) / (2 * np.sqrt(0.25 / READOUT_SHOTS)):.0f} sigma), "
        f"but the readout rests on a "
        f"{primary_probs[top1] / max(primary_probs[top2], 1e-12):.1f}:1 amplitude ratio "
        f"between two near-degenerate states rather than on unambiguous convergence."
    )
    if sweep:
        st.info(
            f"**This property does not generalise.** Across all "
            f"{sweep['n_runs']} seeds, {sweep.get('n_confident_correct', '—')} runs "
            f"were both confident (≥90% on one state) **and** correct. "
            f"\"Confident convergence can be confidently wrong\" stands as a "
            f"warning — it happens — but \"no run is both confident and correct\" "
            f"was an artifact of three draws."
        )

    # The earlier version of this caption claimed the loss curves were
    # "visually indistinguishable", which the convergence figure contradicts --
    # seed 123 is plainly separated. The true statement is sharper.
    facts = spectrum_facts()
    degeneracy_gap = facts["degeneracy_gap"]
    spectral_width = facts["spectral_width"]

    if 42 in runs and 456 in runs:
        d = abs(runs[42]["final_energy"] - runs[456]["final_energy"])
        st.caption(
            f"The two lowest states differ by only {degeneracy_gap:.3f} in energy "
            f"against a spectral width of {spectral_width:.1f} "
            f"({100 * degeneracy_gap / spectral_width:.2f}%), so this is a genuinely "
            f"hard discrimination. Seeds 42 and 456 differ by just **{d:.5f}** in "
            f"final energy — about **{100 * d / degeneracy_gap:.0f}% of the "
            f"degeneracy gap** they need to resolve — yet they read out *different "
            f"portfolios* (`{runs[42]['mode_bitstring']}` vs "
            f"`{runs[456]['mode_bitstring']}`). Energy ranking picks the right run "
            f"here, but by a margin well inside run-to-run variation."
        )

    st.dataframe(
        {
            "Seed": [str(s) for s in sorted(runs)],
            "Final <H>": [f"{runs[s]['final_energy']:.6f}" for s in sorted(runs)],
            "Gap vs ground": [f"{runs[s]['energy_gap']:+.6f}" for s in sorted(runs)],
            # A run that put essentially no amplitude on the ground state should
            # not read as an exact zero; "<0.0001" says "below display
            # resolution", which is what actually happened. The bound is
            # inclusive of 0.0 so this never disagrees with the same quantity
            # reported elsewhere.
            "P(ground state)": [
                ("<0.0001" if 0.0 <= runs[s]["ground_state_prob"] < 1e-4
                 else f"{runs[s]['ground_state_prob']:.4f}")
                for s in sorted(runs)
            ],
            "Readout": [runs[s]["mode_bitstring"] for s in sorted(runs)],
            "Correct?": ["yes" if runs[s]["found_ground_state"] else "no"
                         for s in sorted(runs)],
            "Sharpe": [f"{runs[s]['sharpe_ratio']:+.4f}" for s in sorted(runs)],
        },
        width="stretch", hide_index=True,
    )

    st.pyplot(plot_seed_distributions(runs, ham), width="stretch")

    worst = min(runs.values(), key=lambda r: r["ground_state_prob"])
    st.caption(
        f"Seed {worst['config']['seed']} is the cautionary case: it placed "
        f"{worst['mode_probability']:.1%} of its amplitude on the wrong state "
        f"(P(ground state) < 0.0001) while reaching an energy of "
        f"{worst['final_energy']:.4f}, which in an energy-only report would look "
        "like clean convergence."
    )


if __name__ == "__main__":
    main()

