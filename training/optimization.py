"""
training/optimization.py -- Phase 5: variational optimisation loop.

Trains the Q# ansatz with PyTorch's Adam optimiser, driving gradients through
the parameter-shift bridge from Phase 4.

====================================================================
THE LOSS FUNCTION
====================================================================
    L(theta) = <H(theta)>  +  l2_coefficient * ||theta||^2

The brief asks for "<H> + regularisation terms (e.g. constraint penalties)".
The budget constraint is ALREADY inside H: the Phase 2 Hamiltonian carries the
term ``lambda * (sum_i x_i - B)^2`` folded into its Ising coefficients, so
adding a separate constraint penalty here would double-count it and silently
change the optimisation target.

``l2_coefficient`` therefore DEFAULTS TO ZERO, and is exposed only as a hook.
It is worth being explicit about why it is off: an L2 term on rotation angles
pulls every angle toward 0, and theta = 0 is exactly the state |0000> -- the
"hold nothing" portfolio, which is the WORST state in the spectrum
(cost +14.59 against the ground state's -1.25). Regularising the parameters
here would bias the optimiser toward the least useful answer, so switching it
on requires a reason.

====================================================================
WHAT SUCCESS MEANS
====================================================================
The Hamiltonian is diagonal with only 16 eigenvalues, all known exactly from
the Phase 2 brute-force enumeration. So convergence is not judged by "did the
loss go down" but against the exact optimum:

* ``energy_gap``          <H>_final minus the true ground-state energy.
* ``ground_state_prob``   Probability the final state assigns to the ground
                          state bitstring.
* ``mode_bitstring``      The most likely basis state -- what the portfolio
                          decoder will actually read out.

A VQE run can drive <H> close to the ground energy while leaving amplitude
spread over several basis states, so ``mode_bitstring`` is reported alongside
the energy rather than inferred from it.

NOTE ON THE NEAR-DEGENERACY: for this dataset the two lowest states are
``1010`` (TSLA+AMD, E = -4.9211) and ``0110`` (NVDA+AMD, E = -4.8819) -- a gap
of 0.039 against a total spectral width of ~16.6, i.e. 0.24%. Runs that settle
on ``0110`` are reported as such. That is a real property of this problem, not
a defect to be tuned away.

Usage::

    python training/optimization.py                      # seed 42
    python training/optimization.py --seeds 42 123 456   # multi-seed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless: no display required
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H          # noqa: E402
from quantum import q_bridge as qb              # noqa: E402
from training.gradient_bridge import (          # noqa: E402
    EnergyEvaluator,
    QuantumPortfolioLayer,
)

TRAINING_DIR = PROJECT_ROOT / "training"
DATA_DIR = PROJECT_ROOT / "data"

logger = logging.getLogger("optimization")


@dataclass
class TrainingConfig:
    """Hyperparameters for one optimisation run."""

    seed: int = 42
    epochs: int = 100
    learning_rate: float = 0.05      # mid-range of the 0.01-0.1 band
    num_layers: int = qb.DEFAULT_LAYERS
    init_scale: float = 0.1          # uniform(-0.1, 0.1) initial angles
    l2_coefficient: float = 0.0      # see module docstring
    log_every: int = 5
    readout_shots: int = 4096


@dataclass
class TrainingResult:
    """Everything produced by one run, in a serialisable form."""

    config: dict
    final_params: list[float]
    loss_history: list[float]
    energy_history: list[float]
    param_norm_history: list[float]
    grad_norm_history: list[float]
    final_energy: float
    ground_energy: float
    energy_gap: float
    ground_bitstring: str
    mode_bitstring: str
    mode_probability: float
    ground_state_prob: float
    found_ground_state: bool
    final_probabilities: list[float]
    decoded_weights: list[float]
    decoded_from: str
    sampled_diagnostics: dict
    expected_return: float
    volatility: float
    sharpe_ratio: float
    wall_time_s: float
    circuit_evaluations: int
    recycle_stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------

def train(
    ham: H.PortfolioHamiltonian,
    config: TrainingConfig,
    risk_free: float = 0.05,
) -> TrainingResult:
    """Run one Adam optimisation of the ansatz against ``ham``.

    Args:
        ham: The Phase 2 portfolio Hamiltonian.
        config: Hyperparameters, including the random seed.
        risk_free: Annualised risk-free rate, for the reported Sharpe ratio.

    Returns:
        A fully populated :class:`TrainingResult`.
    """
    started = time.time()

    # Seed every source of randomness that affects the run.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    qb.set_seed(config.seed)

    evaluator = EnergyEvaluator(
        ham.h, ham.J, num_layers=config.num_layers, backend="exact"
    )
    layer = QuantumPortfolioLayer(
        evaluator, seed=config.seed, init_scale=config.init_scale
    )
    optimiser = torch.optim.Adam([layer.theta], lr=config.learning_rate)

    # Exact reference values from the Phase 2 enumeration.
    spectrum = H.enumerate_states(ham)
    ground = min(spectrum, key=lambda r: r["ising_energy"])
    ground_energy = float(ground["ising_energy"])
    ground_bits = ground["bitstring"]
    energies_table = qb.ising_energy_table(ham.h, ham.J, ham.n_assets)

    logger.info("-" * 78)
    logger.info("TRAINING RUN -- seed %d", config.seed)
    logger.info("-" * 78)
    logger.info("  epochs=%d  lr=%.4f  layers=%d  params=%d  init=U(-%.2f, %.2f)  l2=%.4g",
                config.epochs, config.learning_rate, config.num_layers,
                evaluator.n_params, config.init_scale, config.init_scale,
                config.l2_coefficient)
    logger.info("  target ground state: %s (%s), E = %.6f",
                ground_bits, ", ".join(ground["held"]), ground_energy)
    logger.info("")
    logger.info("  %5s %12s %12s %11s %11s %9s %8s %6s",
                "epoch", "loss", "<H>", "|theta|", "|grad|", "lr", "P(gs)", "mode")

    loss_hist: list[float] = []
    energy_hist: list[float] = []
    pnorm_hist: list[float] = []
    gnorm_hist: list[float] = []

    for epoch in range(config.epochs + 1):
        optimiser.zero_grad(set_to_none=True)

        energy = layer()                       # forward: 1 circuit evaluation
        loss = energy
        if config.l2_coefficient:
            loss = loss + config.l2_coefficient * (layer.theta ** 2).sum()

        loss.backward()                        # backward: 2 * n_params evals

        grad_norm = float(layer.theta.grad.norm())
        param_norm = float(layer.theta.detach().norm())
        loss_val = float(loss.detach())
        energy_val = float(energy.detach())

        loss_hist.append(loss_val)
        energy_hist.append(energy_val)
        pnorm_hist.append(param_norm)
        gnorm_hist.append(grad_norm)

        if epoch % config.log_every == 0 or epoch == config.epochs:
            probs = layer.probabilities()
            gs_idx = qb.bitstring_to_index([int(c) for c in ground_bits], ham.n_assets)
            mode_idx = int(np.argmax(probs))
            mode_bits = "".join(str(b) for b in qb.index_to_bitstring(mode_idx, ham.n_assets))
            logger.info("  %5d %12.6f %12.6f %11.5f %11.5f %9.4f %8.4f %6s",
                        epoch, loss_val, energy_val, param_norm, grad_norm,
                        optimiser.param_groups[0]["lr"], probs[gs_idx], mode_bits)

        # The final iteration is measured, not stepped, so the reported
        # parameters are exactly those the metrics describe.
        if epoch < config.epochs:
            optimiser.step()

    # ------------------------------------------------------------- readout
    final_params = layer.theta.detach().cpu().numpy().copy()
    probs = qb.probabilities(final_params, config.num_layers)
    final_energy = float(probs @ energies_table)

    gs_idx = qb.bitstring_to_index([int(c) for c in ground_bits], ham.n_assets)
    ground_state_prob = float(probs[gs_idx])
    mode_idx = int(np.argmax(probs))
    mode_bits = "".join(str(b) for b in qb.index_to_bitstring(mode_idx, ham.n_assets))

    # Physically honest readout: draw shots the way hardware would, then decode
    # the most frequent FEASIBLE bitstring.
    bitstrings = qb.measure_portfolio(
        final_params, config.num_layers, config.readout_shots
    )
    counts: dict[str, int] = {}
    for row in bitstrings:
        key = "".join(str(int(b)) for b in row)
        counts[key] = counts.get(key, 0) + 1

    try:
        weights, sample_diag = H.decode_distribution(counts, ham.budget)
        decoded_from = "sampled feasible mode"
    except ValueError as exc:
        # No feasible outcome was measured: repair the exact mode rather than
        # silently reporting nothing.
        logger.warning("Sampled decoding failed (%s); repairing the exact mode.", exc)
        weights, repaired = H.decode_with_repair(
            [int(c) for c in mode_bits], ham.mu, ham.sigma, ham.budget, ham.risk_aversion
        )
        sample_diag = {"error": str(exc), "repaired": repaired}
        decoded_from = "repaired exact mode"

    ret = float(ham.mu @ weights)
    vol = float(np.sqrt(max(weights @ ham.sigma @ weights, 0.0)))
    sharpe = (ret - risk_free) / vol if vol > 1e-12 else 0.0

    elapsed = time.time() - started

    logger.info("")
    logger.info("  final <H>            : %.6f", final_energy)
    logger.info("  ground-state energy  : %.6f  (%s)", ground_energy, ground_bits)
    logger.info("  energy gap           : %+.6f", final_energy - ground_energy)
    logger.info("  P(ground state)      : %.4f", ground_state_prob)
    logger.info("  most likely state    : %s  (p = %.4f)", mode_bits, probs[mode_idx])
    logger.info("  decoded weights      : %s",
                "  ".join(f"{t}={w:.1%}" for t, w in zip(ham.tickers, weights)))
    logger.info("  decoded from         : %s", decoded_from)
    logger.info("  return / vol / Sharpe: %+.4f / %.4f / %+.4f", ret, vol, sharpe)
    logger.info("  wall time            : %.1f s (%d circuit evaluations)",
                elapsed, evaluator.n_evaluations)

    return TrainingResult(
        config=asdict(config),
        final_params=final_params.tolist(),
        loss_history=loss_hist,
        energy_history=energy_hist,
        param_norm_history=pnorm_hist,
        grad_norm_history=gnorm_hist,
        final_energy=final_energy,
        ground_energy=ground_energy,
        energy_gap=final_energy - ground_energy,
        ground_bitstring=ground_bits,
        mode_bitstring=mode_bits,
        mode_probability=float(probs[mode_idx]),
        ground_state_prob=ground_state_prob,
        found_ground_state=mode_bits == ground_bits,
        final_probabilities=probs.tolist(),
        decoded_weights=weights.tolist(),
        decoded_from=decoded_from,
        sampled_diagnostics=sample_diag,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=float(sharpe),
        wall_time_s=elapsed,
        circuit_evaluations=evaluator.n_evaluations,
        recycle_stats=qb.recycle_stats(),
    )


# --------------------------------------------------------------------------
# Persistence and plotting
# --------------------------------------------------------------------------

def save_result(result: TrainingResult) -> Path:
    """Write one run's parameters and history to ``training/``."""
    seed = result.config["seed"]
    path = TRAINING_DIR / f"trained_seed{seed}.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    logger.info("  saved -> %s", path.name)
    return path


def load_result(seed: int) -> TrainingResult:
    """Load a saved run by seed."""
    path = TRAINING_DIR / f"trained_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python training/optimization.py --seeds {seed}"
        )
    return TrainingResult(**json.loads(path.read_text(encoding="utf-8")))


def plot_convergence(results: Sequence[TrainingResult], ham: H.PortfolioHamiltonian) -> Path:
    """Plot convergence for every run against the exact spectrum.

    Three panels: energy vs epoch (with the exact ground state and the
    near-degenerate runner-up marked), gradient norm, and parameter norm.
    """
    spectrum = sorted(H.enumerate_states(ham), key=lambda r: r["ising_energy"])
    ground, runner_up = spectrum[0], spectrum[1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    colours = plt.get_cmap("tab10")

    # With a large seed sweep, one labelled line per run is unreadable. Colour
    # by OUTCOME instead -- did the run reach the ground state -- since that is
    # the distinction the figure exists to show.
    many = len(results) > 6
    lw = 0.9 if many else 1.6
    alpha = 0.5 if many else 1.0

    def style(res, i):
        if many:
            reached = res.mode_bitstring == ground["bitstring"]
            return ("tab:green" if reached else "tab:red"), None
        return colours(i), (f"seed {res.config['seed']} (final {res.final_energy:.4f}, "
                            f"mode {res.mode_bitstring})")

    ax = axes[0]
    for i, res in enumerate(results):
        colour, label = style(res, i)
        ax.plot(res.energy_history, color=colour, linewidth=lw, alpha=alpha, label=label)
    ax.axhline(ground["ising_energy"], color="black", linestyle="--", linewidth=1.2,
               label=f"ground state {ground['bitstring']} = {ground['ising_energy']:.4f}")
    ax.axhline(runner_up["ising_energy"], color="grey", linestyle=":", linewidth=1.2,
               label=f"runner-up {runner_up['bitstring']} = {runner_up['ising_energy']:.4f}")
    if many:
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        handles = [Line2D([], [], color="tab:green", lw=2),
                   Line2D([], [], color="tab:red", lw=2)] + handles
        labels = [f"read out {ground['bitstring']} (ground state)",
                  "read out another state"] + labels
        ax.legend(handles, labels, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel(r"$\langle H \rangle$")
    ax.set_title(f"VQE convergence vs exact spectrum ({len(results)} seeds)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, res in enumerate(results):
        colour, _ = style(res, i)
        ax.plot(res.grad_norm_history, color=colour, linewidth=lw, alpha=alpha)
    ax.set_ylabel(r"$\|\nabla_\theta \langle H \rangle\|_2$")
    ax.set_yscale("log")
    ax.set_title("Gradient norm")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for i, res in enumerate(results):
        colour, _ = style(res, i)
        ax.plot(res.param_norm_history, color=colour, linewidth=lw, alpha=alpha)
    ax.set_ylabel(r"$\|\theta\|_2$")
    ax.set_xlabel("epoch")
    ax.set_title("Parameter norm")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = TRAINING_DIR / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Convergence plot -> %s", path)
    return path


def plot_final_distribution(results: Sequence[TrainingResult],
                            ham: H.PortfolioHamiltonian) -> Path:
    """Bar chart of each run's final basis-state distribution."""
    n_states = 2 ** ham.n_assets
    labels = ["".join(str(b) for b in qb.index_to_bitstring(k, ham.n_assets))
              for k in range(n_states)]
    feasible = [sum(int(c) for c in lab) == ham.budget for lab in labels]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / max(len(results), 1)
    x = np.arange(n_states)
    for i, res in enumerate(results):
        ax.bar(x + i * width, res.final_probabilities, width,
               label=f"seed {res.config['seed']}")

    for k, ok in enumerate(feasible):
        if ok:
            ax.axvspan(k - 0.15, k + 0.85, color="green", alpha=0.06)

    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("probability")
    ax.set_xlabel("basis state (shaded = feasible, exactly 2 assets held)")
    ax.set_title("Final measured distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = TRAINING_DIR / "final_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Distribution plot -> %s", path)
    return path


# --------------------------------------------------------------------------

def run_all(seeds: Sequence[int], config: TrainingConfig) -> list[TrainingResult]:
    """Train once per seed, saving and summarising every run."""
    ham = H.load_hamiltonian()

    logger.info("=" * 78)
    logger.info("PHASE 5 -- VARIATIONAL OPTIMISATION")
    logger.info("=" * 78)
    logger.info("Assets : %s", ", ".join(ham.tickers))
    logger.info("Budget : hold exactly %d of %d", ham.budget, ham.n_assets)
    logger.info("Seeds  : %s", list(seeds))

    results = []
    for seed in seeds:
        cfg = TrainingConfig(**{**asdict(config), "seed": seed})
        res = train(ham, cfg)
        save_result(res)
        results.append(res)

    plot_convergence(results, ham)
    plot_final_distribution(results, ham)

    # ------------------------------------------------------------- summary
    logger.info("")
    logger.info("=" * 78)
    logger.info("SUMMARY ACROSS SEEDS")
    logger.info("=" * 78)
    logger.info("  %6s %12s %12s %9s %8s %6s %8s %s",
                "seed", "final <H>", "gap", "P(gs)", "mode", "found", "Sharpe", "weights")
    for res in results:
        logger.info("  %6d %12.6f %+12.6f %9.4f %8s %6s %+8.4f %s",
                    res.config["seed"], res.final_energy, res.energy_gap,
                    res.ground_state_prob, res.mode_bitstring,
                    "yes" if res.found_ground_state else "NO",
                    res.sharpe_ratio,
                    " ".join(f"{t}={w:.0%}" for t, w in zip(ham.tickers, res.decoded_weights)))

    energies = np.array([r.final_energy for r in results])
    logger.info("")
    logger.info("  final <H>: mean %.6f, std %.6f, spread %.6f",
                energies.mean(), energies.std(ddof=0), float(np.ptp(energies)))
    agree = len({r.mode_bitstring for r in results}) == 1
    logger.info("  all seeds agree on the readout state: %s", "yes" if agree else "NO")
    n_found = sum(r.found_ground_state for r in results)
    logger.info("  seeds reaching the true ground state: %d/%d", n_found, len(results))

    summary_path = TRAINING_DIR / "training_summary.json"
    summary_path.write_text(json.dumps({
        "seeds": list(seeds),
        "ground_bitstring": results[0].ground_bitstring,
        "ground_energy": results[0].ground_energy,
        "runs": [
            {
                "seed": r.config["seed"],
                "final_energy": r.final_energy,
                "energy_gap": r.energy_gap,
                "ground_state_prob": r.ground_state_prob,
                "mode_bitstring": r.mode_bitstring,
                "found_ground_state": r.found_ground_state,
                "decoded_weights": r.decoded_weights,
                "expected_return": r.expected_return,
                "volatility": r.volatility,
                "sharpe_ratio": r.sharpe_ratio,
                "wall_time_s": r.wall_time_s,
            }
            for r in results
        ],
        "mean_final_energy": float(energies.mean()),
        "std_final_energy": float(energies.std(ddof=0)),
        "all_seeds_agree": agree,
        "n_found_ground_state": int(n_found),
    }, indent=2), encoding="utf-8")
    logger.info("  summary -> %s", summary_path)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 variational optimisation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="random seeds to train (default: 42)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--layers", type=int, default=qb.DEFAULT_LAYERS)
    parser.add_argument("--l2", type=float, default=0.0)
    args = parser.parse_args()

    qb.configure_logging("optimization.log")
    config = TrainingConfig(
        epochs=args.epochs, learning_rate=args.lr,
        num_layers=args.layers, l2_coefficient=args.l2,
    )
    run_all(args.seeds, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
