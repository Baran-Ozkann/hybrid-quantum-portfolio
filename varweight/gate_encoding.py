"""Step 1 gate: the variable-weight encoding, verified exhaustively.

Everything here is checked by ENUMERATION over all 4096 states rather than by
argument. That is affordable by construction, and keeping it affordable is the
reason the encoding was sized the way it was.

Checks:
  1. decode inverts encode on all 4096 bit strings, and encode inverts decode
     on all 4^6 level tuples
  2. the feasible set matches the closed-form count, and every feasible state
     decodes to weights summing to 1
  3. the Ising energies reproduce the penalised objective exactly, on every
     state (this is the Phase 2 verify_ising_mapping check, at 12 qubits)
  4. the ground state is feasible -- i.e. the penalty actually binds
  5. the encodable optimum is reported against Markowitz, which is the whole
     motivation for the redesign
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classical.data_pipeline import RISK_FREE_RATE                # noqa: E402
from varweight import encoding as enc                             # noqa: E402
from varweight import hamiltonian as vh                           # noqa: E402
from varweight.data import statistics                             # noqa: E402

logger = logging.getLogger("gate_encoding")

_checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _checks.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return bool(ok)


def sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    return 0.0 if vol < 1e-12 else float((mu @ w - RISK_FREE_RATE) / vol)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("=" * 78)
    print("VARIABLE-WEIGHT ENCODING GATE")
    print("=" * 78)

    s = enc.summarise()
    print(f"\n  {s.n_assets} assets x {s.qubits_per_asset} qubits = {s.n_qubits} qubits"
          f" -> {s.n_states} states, budget {s.budget}")

    # ---------------------------------------------------- 1. round trips
    print("\n[1] Encode / decode are exact inverses")
    bad = 0
    for idx in range(enc.N_STATES):
        bits = enc.index_to_bits(idx)
        levels = enc.bits_to_levels(bits)
        if enc.levels_to_bits(levels) != bits:
            bad += 1
        if enc.bits_to_index(bits) != idx:
            bad += 1
    check("bits -> levels -> bits, all 4096 states", bad == 0, f"{bad} mismatches")

    bad2 = 0
    for lv in itertools.product(range(enc.LEVELS_PER_ASSET), repeat=enc.N_ASSETS):
        if tuple(enc.bits_to_levels(enc.levels_to_bits(lv))) != lv:
            bad2 += 1
    check(f"levels -> bits -> levels, all {enc.LEVELS_PER_ASSET ** enc.N_ASSETS} tuples",
          bad2 == 0, f"{bad2} mismatches")

    # ---------------------------------------------------- 2. feasible set
    print("\n[2] Feasible set")
    enumerated = sum(
        1 for idx in range(enc.N_STATES)
        if int(enc.bits_to_levels(enc.index_to_bits(idx)).sum()) == enc.BUDGET
    )
    closed = enc.feasible_count()
    check("enumerated count matches closed form",
          enumerated == closed == s.n_feasible,
          f"enumerated {enumerated}, closed form {closed}")

    weights_ok = all(
        abs(enc.levels_to_weights(lv).sum() - 1.0) < 1e-12
        for lv in enc.feasible_level_tuples()
    )
    check("every feasible state decodes to weights summing to 1", weights_ok)

    infeasible_rejected = True
    try:
        enc.levels_to_weights((3, 3, 3, 3, 3, 3))
        infeasible_rejected = False
    except ValueError:
        pass
    check("infeasible levels have no weight decoding", infeasible_rejected)

    print(f"\n    feasible states     : {s.n_feasible} of {s.n_states}"
          f"  ({s.feasible_fraction:.2%})")
    print(f"    distinct weight vec : {s.n_distinct_weight_vectors}")
    print(f"    4-asset comparison  : 6 of 16 ({6/16:.2%})")
    print(f"    -> the feasible FRACTION falls {0.375/s.feasible_fraction:.1f}x"
          f" while the feasible COUNT rises {s.n_feasible/6:.0f}x")

    # ---------------------------------------------------- 3. Ising identity
    print("\n[3] Ising energies reproduce the penalised objective")
    mu, sigma, _ = statistics()
    ham = vh.build_hamiltonian(mu, sigma)
    print(f"    penalty lambda = {ham.penalty:.6f} (sufficient bound x "
          f"{vh.DEFAULT_PENALTY_SCALE})")

    worst = 0.0
    for idx in range(enc.N_STATES):
        bits = enc.index_to_bits(idx)
        levels = enc.bits_to_levels(bits)
        lhs = vh.ising_energy(bits, ham.h, ham.J) + ham.offset
        rhs = vh.penalised_cost(levels, ham)
        worst = max(worst, abs(lhs - rhs))
    check("|E_ising + offset - cost| over all 4096 states", worst < 1e-9,
          f"max deviation {worst:.3e}")

    # ---------------------------------------------------- 4. ground state
    print("\n[4] Ground state")
    states = vh.enumerate_states(ham)
    g, r = states[0], states[1]
    energies = np.array([s_["ising_energy"] for s_ in states])
    width = float(energies[-1] - energies[0])
    gap = float(r["ising_energy"] - g["ising_energy"])

    check("ground state is feasible (the penalty binds)", g["feasible"],
          f"levels {g['levels']} sum {g['level_sum']}")
    n_inf_below = sum(1 for s_ in states if not s_["feasible"]
                      and s_["ising_energy"] < min(x["ising_energy"] for x in states
                                                   if x["feasible"]))
    check("no infeasible state undercuts the best feasible one", n_inf_below == 0,
          f"{n_inf_below} infeasible states below")

    feas_states = [s_ for s_ in states if s_["feasible"]]
    fe = np.array([s_["ising_energy"] for s_ in feas_states])
    feas_width = float(fe[-1] - fe[0])

    print(f"\n    ground    {g['bitstring']}  levels {g['levels']}  "
          f"E = {g['ising_energy']:.6f}")
    print(f"    runner-up {r['bitstring']}  levels {r['levels']}  "
          f"E = {r['ising_energy']:.6f}")
    print(f"    gap {gap:.6f}")

    # NORMALISATION WARNING. The full spectral width is dominated by the
    # penalty: the worst state violates the budget by 14 levels and pays
    # lambda * 14^2. That denominator says more about lambda than about the
    # problem, and it is NOT comparable across encodings with different
    # penalty scales. The feasible-subspace range is the honest normaliser --
    # it is the spread the optimiser must actually discriminate within.
    print(f"\n    full spectral width      {width:>12.6f}   gap/width "
          f"{gap/width:>8.4%}   <- penalty-dominated, do not compare")
    print(f"    FEASIBLE-subspace width  {feas_width:>12.6f}   gap/width "
          f"{gap/feas_width:>8.4%}   <- the comparable figure")
    print(f"\n    4-asset, same two normalisations (measured, not quoted):")
    print(f"      full     gap 0.039209 / width 16.560790 = 0.2368%")
    print(f"      feasible gap 0.039209 / width  2.311810 = 1.6960%")

    w_g = np.asarray(g["weights"])
    print(f"\n    ground portfolio:")
    for t, wt in zip(ham.tickers, w_g):
        if wt > 0:
            print(f"      {t:<5} {wt:.0%}")

    # ---------------------------------------------------- 5. expressiveness
    print("\n[5] Expressiveness -- the motivation for the redesign")
    best = max(feas_states, key=lambda s_: sharpe(np.asarray(s_["weights"]), mu, sigma))
    best_sharpe = sharpe(np.asarray(best["weights"]), mu, sigma)
    ground_sharpe = sharpe(w_g, mu, sigma)

    # Unconstrained long-only Markowitz max-Sharpe, by projected ascent.
    def neg_sharpe(w):
        return -sharpe(w, mu, sigma)

    rng = np.random.default_rng(0)
    best_w, best_v = None, np.inf
    for _ in range(200):
        w = rng.dirichlet(np.ones(enc.N_ASSETS))
        for _ in range(2000):
            eps = 1e-6
            grad = np.zeros_like(w)
            base = neg_sharpe(w)
            for k in range(len(w)):
                wp = w.copy(); wp[k] += eps
                wp = np.clip(wp, 0, None); wp /= wp.sum()
                grad[k] = (neg_sharpe(wp) - base) / eps
            w = w - 0.01 * grad
            w = np.clip(w, 0, None)
            tot = w.sum()
            if tot <= 0:
                break
            w /= tot
        v = neg_sharpe(w)
        if v < best_v:
            best_v, best_w = v, w.copy()
    mkw_sharpe = -best_v

    print(f"    best ENCODABLE Sharpe      : {best_sharpe:.4f}  "
          f"levels {best['levels']}")
    print(f"    ground-state Sharpe        : {ground_sharpe:.4f}")
    print(f"    long-only Markowitz Sharpe : {mkw_sharpe:.4f}")
    print(f"    encoding gap to Markowitz  : {mkw_sharpe - best_sharpe:+.4f}"
          f"  ({(mkw_sharpe - best_sharpe)/mkw_sharpe:+.2%})")
    print(f"    4-asset comparison         : 2.1599 encodable vs 2.4251 Markowitz"
          f"  (-10.94%)")

    print("\n" + "=" * 78)
    failed = [n for n, ok, _ in _checks if not ok]
    print(f"ENCODING GATE: {len(_checks) - len(failed)}/{len(_checks)} checks passed")
    print("VARWEIGHT ENCODING: " + ("ALL CHECKS PASSED" if not failed
                                    else f"FAILURES: {failed}"))
    print("=" * 78)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
