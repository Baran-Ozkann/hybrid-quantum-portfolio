"""
tools/verify_cvar_gradient.py -- Phase 9 gate 1.

Validates the CVaR reweighted-observable identity BEFORE any sweep is run, the
same way the plain parameter-shift rule was validated in Phase 4.

Three distinct claims are checked:

  1. VALUE identity.    <O_w> + E_m  ==  CVaR computed directly from the
                        definition, at many random points and every alpha.
  2. GRADIENT identity. Parameter-shift on the frozen reweighted observable ==
                        central differences of the true CVaR value, which
                        re-derives the tail at every perturbed point.
  3. alpha = 1 reduces exactly to the plain mean energy, so the CVaR code path
                        is a strict generalisation of the existing optimiser.

Also probes the degenerate regime (m = 0 -> zero gradient) so its onset is a
measured fact rather than a surprise mid-sweep.

Run::

    python tools/verify_cvar_gradient.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H                     # noqa: E402
from quantum import q_bridge as qb                         # noqa: E402
from training.gradient_bridge import EnergyEvaluator       # noqa: E402
from training import cvar as C                             # noqa: E402

ALPHAS = (0.05, 0.1, 0.25, 0.5, 1.0)
PASSED = True


def check(name: str, ok: bool, detail: str) -> None:
    global PASSED
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        PASSED = False


def main() -> int:
    ham = H.load_hamiltonian()
    energies = qb.ising_energy_table(ham.h, ham.J, ham.n_assets)
    ev = EnergyEvaluator(ham.h, ham.J, num_layers=qb.DEFAULT_LAYERS, backend="exact")
    n = ev.n_params
    n_states = 2 ** ham.n_assets
    rng = np.random.default_rng(9)

    print("=" * 74)
    print("PHASE 9 GATE 1 -- CVaR reweighted-observable identity")
    print("=" * 74)
    print(f"  {n_states} basis states, {n} parameters, "
          f"alpha floor = 1/{n_states} = {C.alpha_floor(n_states):.4f}")

    points: dict[str, np.ndarray] = {
        "near |0000> (as initialised)": rng.uniform(-0.1, 0.1, n),
        "random uniform(-pi, pi) #1": rng.uniform(-math.pi, math.pi, n),
        "random uniform(-pi, pi) #2": rng.uniform(-math.pi, math.pi, n),
    }
    try:
        from training.optimization import load_result
        points["trained (seed 42)"] = np.asarray(load_result(42).final_params)
    except Exception:
        pass

    # ------------------------------------------------------------ 1. value
    print("\n[1] Value identity:  <O_w> + E_m  ==  CVaR from the definition")
    worst_val = 0.0
    for alpha in ALPHAS:
        for label, p in points.items():
            probs = qb.probabilities(p, qb.DEFAULT_LAYERS)
            obj = C.CVaRObjective(ev, energies, alpha)
            worst_val = max(
                worst_val,
                abs(obj.value(p) - C.cvar_value_direct(probs, energies, alpha)),
            )
    check("value identity", worst_val < 1e-12,
          f"max |identity - direct| = {worst_val:.3e} over "
          f"{len(ALPHAS)}x{len(points)} combinations")

    # Guard against the specific mistake of reporting <O_w> as the loss.
    print("\n[2] <O_w> alone is NOT the loss")
    p = points["random uniform(-pi, pi) #1"]
    obj = C.CVaRObjective(ev, energies, 0.25)
    tail = obj.linearise(p)
    bare = float(qb.probabilities(p, qb.DEFAULT_LAYERS) @ tail.weights)
    full = obj.value(p, tail)
    check("constant term matters", bare <= 0.0 and abs(full - bare) > 1.0,
          f"<O_w> = {bare:.6f} (always <= 0), CVaR = {full:.6f}, "
          f"E_m = {tail.boundary_energy:.6f}")

    # -------------------------------------------------------- 3. gradient
    print("\n[3] Gradient identity: parameter-shift on frozen O_w vs central "
          "differences of the true CVaR")
    worst_grad = 0.0
    for alpha in ALPHAS:
        for label, p in points.items():
            obj = C.CVaRObjective(ev, energies, alpha)
            tail = obj.linearise(p)
            if tail.degenerate:
                continue    # covered in [5]
            comps = [0, n // 4, n // 2, 3 * n // 4, n - 1]
            ps = obj.gradient(p, tail)
            fd = obj.finite_difference(p, eps=1e-6, components=comps)
            diff = max(abs(ps[i] - fd[i]) for i in comps)
            worst_grad = max(worst_grad, diff)
            print(f"      alpha={alpha:<5} {label[:32]:<34} "
                  f"m={tail.m:<3} max|PS-FD| = {diff:.3e}")
    check("gradient identity", worst_grad < 1e-4,
          f"max |parameter-shift - finite-difference| = {worst_grad:.3e}")

    # ------------------------------------------- 4. alpha = 1 reduces to VQE
    print("\n[4] alpha = 1 reduces exactly to plain VQE")
    worst_reduce = worst_grad_reduce = 0.0
    for label, p in points.items():
        obj = C.CVaRObjective(ev, energies, 1.0)
        worst_reduce = max(worst_reduce, abs(obj.value(p) - ev.energy(p)))
        tail = obj.linearise(p)
        g_cvar = obj.gradient(p, tail)
        g_plain = ev.parameter_shift_gradient(p)
        worst_grad_reduce = max(worst_grad_reduce,
                                float(np.abs(g_cvar - g_plain).max()))
    check("alpha=1 value", worst_reduce < 1e-12,
          f"max |CVaR(1.0) - <H>| = {worst_reduce:.3e}")
    check("alpha=1 gradient", worst_grad_reduce < 1e-12,
          f"max |grad difference| = {worst_grad_reduce:.3e}")

    # ------------------------------------------------ 5. degenerate regime
    print("\n[5] Degenerate regime (m = 0 -> identically zero gradient)")
    rows = []
    for alpha in ALPHAS:
        for label, p in points.items():
            obj = C.CVaRObjective(ev, energies, alpha)
            tail = obj.linearise(p)
            if tail.degenerate:
                rows.append((alpha, label,
                             float(np.abs(obj.gradient(p, tail)).max())))
    if rows:
        for alpha, label, gmax in rows:
            print(f"      alpha={alpha:<5} {label[:34]:<36} |grad|_inf = {gmax:.3e}")
        check("degenerate gradient is exactly zero",
              all(g == 0.0 for _, _, g in rows),
              f"{len(rows)} degenerate combinations, all zero")
    else:
        print("      (none at these points)")

    print("\n[6] Onset condition: p(lowest-energy state) >= alpha")
    lowest_idx = int(np.argmin(energies))
    for label, p in points.items():
        p_low = float(qb.probabilities(p, qb.DEFAULT_LAYERS)[lowest_idx])
        degen = [a for a in ALPHAS if p_low >= a]
        print(f"      {label[:34]:<36} p(ground) = {p_low:.4f} -> "
              f"degenerate for alpha in {degen if degen else 'none'}")

    print("\n" + "=" * 74)
    print(f"PHASE 9 GATE 1: {'ALL CHECKS PASSED' if PASSED else 'FAILURES PRESENT'}")
    print("=" * 74)
    return 0 if PASSED else 1


if __name__ == "__main__":
    qb.configure_logging("cvar_gate.log")
    sys.exit(main())
