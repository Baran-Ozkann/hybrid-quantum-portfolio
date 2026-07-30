"""
classical/phase2_gate.py -- Phase 2 validation gate.

Runs the full classical pipeline end to end and asserts every property the
later phases depend on:

  1. Market data fetched, statistics well formed (delegated to data_pipeline).
  2. QUBO -> Ising algebra exact across all 2^n bitstrings.
  3. The Ising coefficients agree with what the Q# circuit actually computes.
  4. The brute-force ground state is feasible (holds exactly B assets).
  5. The budget penalty is genuinely binding.
  6. Encoding round-trips, and weights sum to 1.
  7. Classical Markowitz baselines converge and satisfy their constraints.

Run::

    python classical/phase2_gate.py
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import data_pipeline as dp                      # noqa: E402
from classical import hamiltonian as H                         # noqa: E402
from classical import markowitz as mk                          # noqa: E402
from quantum import q_bridge as qb                             # noqa: E402

logger = logging.getLogger("phase2_gate")


def _fmt_weights(w, tickers) -> str:
    return "  ".join(f"{t}={v:6.1%}" for t, v in zip(tickers, w))


def run_gate() -> bool:
    passed = True

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        logger.info("  [%s] %s: %s", "PASS" if ok else "FAIL", name, detail)
        if not ok:
            passed = False

    logger.info("=" * 70)
    logger.info("PHASE 2 VALIDATION GATE")
    logger.info("=" * 70)

    # ---------------------------------------------------------------- data
    logger.info("\n[1] Market data pipeline")
    stats = dp.run_pipeline()
    tickers = tuple(stats["tickers"])
    mu = stats["expected_returns"]
    sigma = stats["cov_matrix"]
    check("canonical order", tickers == dp.TICKERS, f"{tickers}")

    # --------------------------------------------------------- Hamiltonian
    logger.info("\n[2] Hamiltonian construction")
    ham = H.build_hamiltonian(mu, sigma, tickers,
                              risk_aversion=H.DEFAULT_RISK_AVERSION,
                              budget=H.DEFAULT_BUDGET)
    logger.info("    risk aversion q = %.4f", ham.risk_aversion)
    logger.info("    budget        B = %d", ham.budget)
    logger.info("    penalty  lambda = %.6f (auto)", ham.penalty)
    logger.info("    offset          = %.6f", ham.offset)
    logger.info("    h = [%s]", ", ".join(f"{v:+.6f}" for v in ham.h))
    for coeff, (i, j) in zip(ham.J, ham.pairs):
        logger.info("    J[%s,%s] = %+.6f", tickers[i], tickers[j], coeff)

    # ------------------------------------------------ QUBO <-> Ising exact
    logger.info("\n[3] QUBO -> Ising algebra, all 2^n states")
    ok, worst = H.verify_ising_mapping(ham)
    check("exact mapping", ok,
          f"max |E_ising + offset - C_qubo| = {worst:.3e} over "
          f"{2 ** ham.n_assets} bitstrings")

    # ------------------------------- Ising coefficients match the Q# circuit
    # The Python energy table and the Q# shot loop are independent
    # implementations of the same Hamiltonian. If they disagree, training
    # would optimise something other than the portfolio problem.
    logger.info("\n[4] Python energy table vs Q# circuit")
    table = qb.ising_energy_table(ham.h, ham.J, ham.n_assets)
    worst_tbl = 0.0
    for bits in itertools.product((0, 1), repeat=ham.n_assets):
        idx = qb.bitstring_to_index(bits, ham.n_assets)
        worst_tbl = max(worst_tbl, abs(table[idx] - H.ising_energy(bits, ham.h, ham.J)))
    check("energy tables agree", worst_tbl < 1e-12,
          f"max deviation = {worst_tbl:.3e} (q_bridge vs hamiltonian)")

    # Drive the actual Q# sampler at a basis state and confirm the energy it
    # reports matches the table. Ry(pi) on a qubit sets that bit to 1.
    try:
        qb.set_seed(4242)
        target_bits = (1, 0, 1, 0)
        params = [0.0] * qb.n_params(ham.n_assets, 1)
        # Undo CNOT-ladder propagation: to realise bits b, the Ry angles must
        # be the XOR-difference along the ladder, since CNOT(q_i, q_i+1)
        # flips q_{i+1} whenever q_i is 1.
        prev = 0
        for q in range(ham.n_assets):
            want = target_bits[q] ^ prev
            params[q] = np.pi * want
            prev = target_bits[q]
        probs = qb.probabilities(params, num_layers=1)
        got_idx = int(np.argmax(probs))
        want_idx = qb.bitstring_to_index(target_bits, ham.n_assets)
        check("basis-state preparation", got_idx == want_idx and probs[got_idx] > 0.999,
              f"prepared {qb.index_to_bitstring(got_idx)}, wanted {list(target_bits)}, "
              f"p={probs[got_idx]:.6f}")

        e_qsharp = qb.expectation_sampled(params, ham.h, ham.J, num_layers=1, num_shots=256)
        e_table = H.ising_energy(target_bits, ham.h, ham.J)
        check("Q# energy at basis state", abs(e_qsharp - e_table) < 1e-9,
              f"Q# = {e_qsharp:.9f}, table = {e_table:.9f}")
    except Exception as exc:
        logger.exception("Q# cross-check failed.")
        check("Q# energy at basis state", False, repr(exc))

    # ------------------------------------------------------- ground state
    logger.info("\n[5] Brute-force ground state (all 16 bitstrings)")
    spectrum = H.enumerate_states(ham)
    logger.info("    %-6s %-5s %-22s %10s %10s %9s %9s",
                "bits", "held", "assets", "QUBO cost", "Ising E", "return", "vol")
    for row in spectrum:
        assets = ",".join(row["held"]) or "(none)"
        ret = f"{row['expected_return']:+.4f}" if row["feasible"] else "     -"
        vol = f"{row['volatility']:.4f}" if row["feasible"] else "     -"
        logger.info("    %-6s %-5d %-22s %10.6f %10.6f %9s %9s",
                    row["bitstring"], row["n_held"], assets,
                    row["qubo_cost"], row["ising_energy"], ret, vol)

    ground = spectrum[0]
    check("ground state feasible", ground["feasible"],
          f"{ground['bitstring']} holds {ground['n_held']} assets "
          f"({', '.join(ground['held'])}), cost = {ground['qubo_cost']:.6f}")

    # The Ising ground state must be the same state, or the circuit is
    # minimising a different problem than the QUBO describes.
    ising_min = min(spectrum, key=lambda r: r["ising_energy"])
    check("Ising and QUBO agree on minimiser",
          ising_min["bitstring"] == ground["bitstring"],
          f"Ising min = {ising_min['bitstring']}, QUBO min = {ground['bitstring']}")

    # ------------------------------------------------------- penalty check
    logger.info("\n[6] Budget penalty is binding")
    feasible = [r for r in spectrum if r["feasible"]]
    infeasible = [r for r in spectrum if not r["feasible"]]
    best_feas = min(r["qubo_cost"] for r in feasible)
    best_infeas = min(r["qubo_cost"] for r in infeasible)
    check("feasible beats infeasible", best_feas < best_infeas,
          f"best feasible = {best_feas:.6f} < best infeasible = {best_infeas:.6f} "
          f"(margin {best_infeas - best_feas:.6f})")

    # Without the penalty, would the unconstrained optimum have been feasible?
    # Reporting this makes clear whether the constraint is doing real work.
    raw_best = min(spectrum, key=lambda r: r["objective_unpenalised"])
    logger.info("    unpenalised optimum would be %s (%d assets) -- penalty %s",
                raw_best["bitstring"], raw_best["n_held"],
                "IS binding" if raw_best["n_held"] != ham.budget else "is not binding here")

    # -------------------------------------------------------- encoding
    logger.info("\n[7] Encoding: bitstring -> weights")
    n_feasible_checked = 0
    for row in feasible:
        w = H.decode_weights(row["bits"], ham.budget)
        if abs(w.sum() - 1.0) > 1e-12:
            check("weights sum to 1", False, f"{row['bitstring']} -> sum {w.sum()}")
            break
        n_feasible_checked += 1
    else:
        check("weights sum to 1", True,
              f"all {n_feasible_checked} feasible bitstrings decode to sum(w)=1")

    try:
        H.decode_weights([1, 1, 1, 0], ham.budget)
        check("infeasible rejected", False, "decode_weights accepted a 3-asset string")
    except ValueError:
        check("infeasible rejected", True, "decode_weights raised on a 3-asset string")

    w_rep, repaired = H.decode_with_repair([1, 1, 1, 0], mu, sigma, ham.budget)
    check("repair policy", repaired and abs(w_rep.sum() - 1.0) < 1e-12,
          f"111 0 repaired to weights {np.round(w_rep, 4).tolist()}")

    # Mode decoding of a synthetic distribution should pick the feasible mode
    # even when an infeasible outcome is more frequent overall.
    counts = {"1110": 50, "1010": 40, "0110": 10}
    w_mode, diag = H.decode_distribution(counts, ham.budget)
    check("distribution decoding", diag["mode_bitstring"] == "1010",
          f"chose {diag['mode_bitstring']} "
          f"(feasible fraction {diag['feasible_fraction']:.2f})")

    # ------------------------------------------------------- Markowitz
    logger.info("\n[8] Classical Markowitz baselines")
    baselines = mk.compute_baselines(mu, sigma, tickers,
                                     budget=ham.budget,
                                     risk_aversion=ham.risk_aversion,
                                     risk_free=stats["risk_free_rate"])
    for key in ("max_sharpe", "min_variance", "equal_weight_all", "best_equal_weight"):
        b = baselines[key]
        logger.info("    %-18s ret=%+.4f vol=%.4f sharpe=%+.4f | %s",
                    key, b["expected_return"], b["volatility"], b["sharpe_ratio"],
                    _fmt_weights(b["weights"], tickers))
    if not mk.validate_baselines(baselines, mu, sigma):
        passed = False

    # ---------------------------------------- encoding-space consistency
    # The best equal-weight basket found by exhaustive search must be the same
    # basket as the Hamiltonian's ground state; they are two routes to one
    # answer, and disagreement would mean the Hamiltonian encodes the wrong
    # objective.
    logger.info("\n[9] Ground state agrees with exhaustive equal-weight search")
    check("ground state == best equal-weight basket",
          baselines["best_equal_weight"]["bitstring"] == ground["bitstring"],
          f"Hamiltonian ground state = {ground['bitstring']}, "
          f"exhaustive best = {baselines['best_equal_weight']['bitstring']}")

    H.save_hamiltonian(ham, spectrum)

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 VALIDATION: %s", "ALL TESTS PASSED" if passed else "FAILURES PRESENT")
    logger.info("=" * 70)
    return passed


if __name__ == "__main__":
    qb.configure_logging("phase2_gate.log")
    try:
        sys.exit(0 if run_gate() else 1)
    except dp.DataFetchError as exc:
        logger.error("DATA FETCH FAILED -- stopping as instructed: %s", exc)
        sys.exit(2)
