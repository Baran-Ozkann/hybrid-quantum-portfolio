"""Step 3: a single plain-VQE run, to size what comes next.

Deliberately minimal. ``training/optimization.py`` is built around the 4-asset
``PortfolioHamiltonian`` and carries the arm/tag/summary machinery; adapting it
before knowing whether 12 qubits trains at all would be committing to a design
on the strength of an assumption. This runs the same Adam-on-parameter-shift
loop with nothing else attached, reports timing, and stops.

If the pipeline works, generalising ``train()`` is the next step -- not this
file growing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum import q_bridge as qb                                # noqa: E402
from training.cvar import CVaRLoss, CVaRObjective                 # noqa: E402
from training.gradient_bridge import (                            # noqa: E402
    EnergyEvaluator, QuantumCircuitFunction,
)
from varweight import encoding as enc                             # noqa: E402
from varweight import hamiltonian as vh                           # noqa: E402
from varweight.data import statistics                             # noqa: E402

logger = logging.getLogger("train_probe")
LOGS = PROJECT_ROOT / "logs"
LOGS.mkdir(exist_ok=True)


def run(layers: int, epochs: int, seed: int, lr: float, log_every: int,
        entangler: str = "linear", penalty: float | None = None,
        cvar_alpha: float | None = None) -> dict:
    mu, sigma, _ = statistics()
    ham = vh.build_hamiltonian(mu, sigma, penalty=penalty)
    n = enc.N_QUBITS

    states = vh.enumerate_states(ham)
    ground = states[0]
    runner = states[1]
    feas_states = [s for s in states if s["feasible"]]
    feas_E = np.array([s["ising_energy"] for s in feas_states])
    feas_width = float(feas_E.max() - feas_E.min())
    energies = qb.ising_energy_table(ham.h, ham.J, n)
    ground_idx = ground["index"]

    ev = EnergyEvaluator(h=ham.h, J=ham.J, num_layers=layers, num_qubits=n,
                         entangler=entangler)
    np_ = ev.n_params

    rng = np.random.default_rng(seed)
    theta = torch.tensor(rng.uniform(-0.1, 0.1, np_), dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)

    print(f"  {n} qubits, {layers} layers -> {np_} parameters")
    print(f"  cost per epoch: 1 forward + {2*np_} backward = {2*np_+1} evaluations")
    print(f"  ground state {ground['bitstring']} levels {ground['levels']} "
          f"E = {ground['ising_energy']:.6f}")
    print(f"  runner-up    {runner['bitstring']} levels {runner['levels']} "
          f"E = {runner['ising_energy']:.6f}\n")

    cvar_obj = None
    if cvar_alpha is not None and cvar_alpha < 1.0:
        cvar_obj = CVaRObjective(ev, energies, cvar_alpha)
        print(f"  objective: CVaR alpha = {cvar_alpha} "
              f"(saturates once P(ground) >= {cvar_alpha})\n")

    history, mode_history = [], []
    t0 = time.time()
    epoch_times = []
    for epoch in range(epochs):
        te = time.time()
        opt.zero_grad()
        if cvar_obj is None:
            loss = QuantumCircuitFunction.apply(theta, ev)
        else:
            # Freeze the tail decomposition, then differentiate the reweighted
            # observable through it -- the weights must not be recomputed inside
            # the shifted evaluations.
            tail = cvar_obj.linearise(theta.detach().numpy())
            loss = CVaRLoss.apply(theta, cvar_obj, tail)
        loss.backward()
        opt.step()
        epoch_times.append(time.time() - te)

        e = float(loss.item())
        history.append(e)
        probs = qb.probabilities(theta.detach().numpy(), layers, n, entangler)
        mode = int(np.argmax(probs))
        mode_history.append(mode)
        if epoch % log_every == 0 or epoch == epochs - 1:
            bits = enc.index_to_bits(mode)
            lv = enc.bits_to_levels(bits)
            print(f"    epoch {epoch:>4}  <H> = {e:>12.6f}  "
                  f"readout {enc.bitstring(bits)} levels {lv.tolist()} "
                  f"sum {int(lv.sum())}  P = {probs[mode]:.4f}  "
                  f"P(ground) = {probs[ground_idx]:.4f}  "
                  f"[{epoch_times[-1]:.2f}s]")
    wall = time.time() - t0

    probs = qb.probabilities(theta.detach().numpy(), layers, n, entangler)
    mode = int(np.argmax(probs))
    bits = enc.index_to_bits(mode)
    levels = enc.bits_to_levels(bits)
    feasible = int(levels.sum()) == ham.budget
    correct = mode == ground_idx

    # Where does the true ground state rank in the final distribution?
    #
    # NOT via argsort. At 4096 states the ground-state probability can underflow
    # to exactly 0.0, and then thousands of states tie at zero; argsort orders
    # ties by index, so the "rank" it reports is a deterministic artefact that
    # looks like a measurement. Two entirely different runs both reported rank
    # 3903 that way. Count strict inequality instead and report the tie block.
    p_ground = float(probs[ground_idx])
    strictly_better = int((probs > p_ground).sum())
    rank = strictly_better + 1              # best rank consistent with ties
    n_tied = int((probs == p_ground).sum())
    underflowed = p_ground == 0.0
    feas_mass = float(sum(probs[s["index"]] for s in states if s["feasible"]))

    print(f"\n  wall time        : {wall:.1f}s over {epochs} epochs "
          f"({np.mean(epoch_times):.2f}s/epoch, "
          f"{np.mean(epoch_times)/(2*np_+1)*1000:.1f} ms/evaluation)")
    # Position within the FEASIBLE band, not the full spectrum. The full
    # spectrum is penalty-dominated, so "N x the gap" against it conflates
    # "picked the wrong feasible portfolio" with "has mass on illegal states".
    above_ground = history[-1] - ground["ising_energy"]
    feas_position = above_ground / feas_width if feas_width else float("nan")
    print(f"  final <H>        : {history[-1]:.6f}  "
          f"(ground {ground['ising_energy']:.6f}, "
          f"gap to ground {above_ground:+.6f})")
    print(f"  penalty lambda   : {ham.penalty:.6f}")
    print(f"  feasible band    : [{feas_E.min():.6f}, {feas_E.max():.6f}] "
          f"width {feas_width:.6f}")
    print(f"  position in band : {feas_position:>7.1%}"
          + ("   <- OUTSIDE the feasible band; mass sits on illegal states"
             if feas_position > 1.0 else ""))
    if feasible:
        e_rank = [s["bitstring"] for s in feas_states].index(
            enc.bitstring(bits)) + 1
        print(f"  readout is feasible-energy rank {e_rank} of {len(feas_states)}")
    print(f"  readout          : {enc.bitstring(bits)} levels {levels.tolist()} "
          f"-> {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    if feasible:
        w = levels / ham.budget
        print("                     " + ", ".join(
            f"{t} {x:.0%}" for t, x in zip(ham.tickers, w) if x > 0))
    print(f"  correct          : {correct}")
    print(f"  P(readout)       : {probs[mode]:.4f}")
    print(f"  P(ground state)  : {p_ground:.3e}"
          + ("  [UNDERFLOWED TO EXACTLY ZERO]" if underflowed else ""))
    print(f"  states strictly above it: {strictly_better} of {len(probs)}"
          + (f"   (tied with it: {n_tied} -- rank is a range, not a number)"
             if n_tied > 1 else ""))
    print(f"  feasible mass    : {feas_mass:.4f}  "
          f"(uniform baseline {120/4096:.4f})")
    print(f"  circuit evals    : {ev.n_evaluations}")

    settle = -1
    for i in range(1, len(mode_history)):
        if mode_history[i] != mode_history[i - 1]:
            settle = i
    print(f"  readout last changed at epoch {settle} of {epochs}"
          + (f" (held {epochs-settle})" if settle >= 0 else " (never changed)"))

    return {
        "n_qubits": n, "layers": layers, "n_params": np_, "epochs": epochs,
        "seed": seed, "lr": lr,
        "wall_time_s": wall, "s_per_epoch": float(np.mean(epoch_times)),
        "ms_per_eval": float(np.mean(epoch_times) / (2 * np_ + 1) * 1000),
        "final_energy": history[-1],
        "ground_energy": ground["ising_energy"],
        "ground_bitstring": ground["bitstring"],
        "ground_levels": ground["levels"],
        "readout": enc.bitstring(bits), "readout_levels": levels.tolist(),
        "feasible": feasible, "correct": correct,
        "p_readout": float(probs[mode]), "p_ground": p_ground,
        "ground_states_strictly_above": strictly_better,
        "ground_tied_with": n_tied,
        "ground_prob_underflowed": underflowed,
        "feasible_mass": feas_mass,
        "penalty": float(ham.penalty),
        "feasible_width": feas_width,
        "position_in_feasible_band": feas_position,
        "settle_epoch": settle, "circuit_evaluations": ev.n_evaluations,
        "energy_history": history,
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--entangler", default="linear", choices=["linear", "ring"])
    ap.add_argument("--penalty", type=float, default=None,
                    help="override lambda; default is the exact minimum x1.85")
    ap.add_argument("--cvar-alpha", type=float, default=None,
                    help="CVaR tail fraction; omit for plain VQE")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    print("=" * 78)
    print(f"VARIABLE-WEIGHT TRAINING PROBE -- {args.layers} layers, "
          f"{args.epochs} epochs, seed {args.seed}, {args.entangler} entangler"
          + (f", lambda {args.penalty}" if args.penalty else "")
          + (f", CVaR alpha {args.cvar_alpha}" if args.cvar_alpha else ""))
    print("=" * 78)
    res = run(args.layers, args.epochs, args.seed, args.lr, args.log_every,
              args.entangler, args.penalty, args.cvar_alpha)
    res["entangler"] = args.entangler
    res["cvar_alpha"] = args.cvar_alpha

    tag = args.tag or (f"{args.entangler}_L{args.layers}_e{args.epochs}"
                       f"_s{args.seed}"
                       + (f"_lam{args.penalty:g}" if args.penalty else "")
                       + (f"_cvar{args.cvar_alpha:g}" if args.cvar_alpha else ""))
    dest = LOGS / f"varweight_probe_{tag}.json"
    dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  written -> {dest.relative_to(PROJECT_ROOT)}")
    print("\nVARWEIGHT PROBE: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
