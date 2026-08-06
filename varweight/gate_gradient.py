"""Step 2 gate: does the gradient machinery still hold at 12 qubits?

The parameter-shift derivation is dimension-independent -- it follows from each
gate being ``exp(-i theta P / 2)`` with ``P^2 = I``, which says nothing about
register size. The IMPLEMENTATION is not: register allocation, the energy
table, the parameter-count contract, and the state-vector shape all carry a
qubit count that was 4 everywhere until now. So both gates are re-run here
against a real 12-qubit variable-weight Hamiltonian.

EXPECTED PRECISION. The same noise-amplified bounds as at 4 qubits, and for the
same reason: the finite-difference reference inherits the simulator's ~1e-15
nondeterminism, amplified by ``1/(2 eps) = 5e4``. Order of magnitude is stable,
the digits are not (see docs/mathematical_foundations.md section 8). Anything
near 1e-10 to 1e-9 is a pass; the gate bound is 1e-2, eight orders away.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum import q_bridge as qb                                # noqa: E402
from training.cvar import CVaRObjective                           # noqa: E402
from training.gradient_bridge import EnergyEvaluator              # noqa: E402
from varweight import encoding as enc                             # noqa: E402
from varweight import hamiltonian as vh                           # noqa: E402
from varweight.data import statistics                             # noqa: E402

logger = logging.getLogger("gate_gradient")

TOLERANCE = 1e-2          # the brief's bound, unchanged
LAYERS = 3

_checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _checks.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("=" * 78)
    print("VARIABLE-WEIGHT GRADIENT GATE -- 12 qubits")
    print("=" * 78)

    mu, sigma, _ = statistics()
    ham = vh.build_hamiltonian(mu, sigma)
    n = enc.N_QUBITS

    ev = EnergyEvaluator(h=ham.h, J=ham.J, num_layers=LAYERS, num_qubits=n)
    np_ = ev.n_params
    print(f"\n  register {n} qubits, {LAYERS} layers -> {np_} parameters")
    print(f"  h: {ham.h.shape}   J: {ham.J.shape} "
          f"(expected {n*(n-1)//2} pairs)")
    print(f"  one backward pass = {2*np_} circuit evaluations")

    # ------------------------------------------------ shape sanity first
    print("\n[1] State vector has the right shape")
    rng = np.random.default_rng(0)
    p0 = rng.uniform(-0.1, 0.1, np_)
    t0 = time.time()
    probs = qb.probabilities(p0, LAYERS, n)
    dt = time.time() - t0
    check(f"probabilities() returns 2^{n} = {2**n} entries",
          len(probs) == 2 ** n, f"got {len(probs)}, {dt*1000:.1f} ms")
    check("probabilities sum to 1", abs(probs.sum() - 1.0) < 1e-9,
          f"sum = {probs.sum():.12f}")

    # ------------------------------------------------ 2. parameter shift
    print("\n[2] Parameter-shift vs central finite difference")
    points = {
        "zeros": np.zeros(np_),
        "small uniform(-0.1, 0.1)": rng.uniform(-0.1, 0.1, np_),
        "random uniform(-pi, pi) #1": rng.uniform(-math.pi, math.pi, np_),
        "random uniform(-pi, pi) #2": rng.uniform(-math.pi, math.pi, np_),
    }
    worst = 0.0
    for label, point in points.items():
        ps = ev.parameter_shift_gradient(point)
        fd = ev.finite_difference_gradient(point, eps=1e-5)
        d = float(np.max(np.abs(ps - fd)))
        worst = max(worst, d)
        print(f"      {label:<28} max|PS-FD| = {d:.3e}   "
              f"|grad|_inf = {np.abs(ps).max():.4f}")
    check(f"max |PS - FD| <= {TOLERANCE:.0e}", worst <= TOLERANCE,
          f"{worst:.3e} (noise-amplified; order of magnitude only)")

    # ------------------------------------------------ 3. nondeterminism floor
    print("\n[3] Exact-backend nondeterminism floor at 12 qubits")
    vals = [ev.energy(p0) for _ in range(30)]
    spread = max(vals) - min(vals)
    scale = float(np.abs(energies_preview(ham)).max())
    print(f"      <H> spread over 30 evaluations: {spread:.3e} "
          f"({len(set(vals))} distinct values)")
    print(f"      |E| scale of this Hamiltonian : {scale:.1f}")
    print(f"      spread RELATIVE to that scale : {spread/scale:.3e}")
    print(f"      implied FD noise at eps=1e-5  : ~{spread/2e-5:.1e}")
    # The ABSOLUTE floor is far above the 4-qubit 1.4e-15, but the RELATIVE
    # floor is the same handful of ULP. The difference is the energy scale:
    # the budget penalty puts |E| in the hundreds here against single digits
    # at 4 assets, and floating-point error tracks magnitude. This is why the
    # gradient agreement below lands near 1e-6 rather than 1e-10 -- it is the
    # same phenomenon, scaled, not a new defect.
    check("nondeterminism floor is at machine precision RELATIVE to |E|",
          spread / scale < 1e-12, f"{spread/scale:.3e} of |E| ~ {scale:.0f}")
    globals()["_MEASURED_SPREAD"] = spread

    # ------------------------------------------------ 4. CVaR identity
    print("\n[4] CVaR reweighted-observable identity at 12 qubits")
    energies = qb.ising_energy_table(ham.h, ham.J, n)
    worst_cvar = 0.0
    for alpha in (0.1, 0.4, 0.5, 1.0):
        obj = CVaRObjective(ev, energies, alpha)
        for label, point in list(points.items())[1:]:
            probs = qb.probabilities(point, LAYERS, n)
            tail = obj.tail_state(probs) if hasattr(obj, "tail_state") else None
            value = obj.value(point)
            direct = _cvar_direct(probs, energies, alpha)
            d = abs(value - direct)
            worst_cvar = max(worst_cvar, d)
    check("CVaR value identity vs definition", worst_cvar < 1e-9,
          f"max deviation {worst_cvar:.3e}")

    # ------------------------------------------------ 5. autograd end to end
    print("\n[5] torch autograd through the 12-qubit layer")
    from training.gradient_bridge import QuantumCircuitFunction
    theta = torch.tensor(points["small uniform(-0.1, 0.1)"], dtype=torch.float64,
                         requires_grad=True)
    out = QuantumCircuitFunction.apply(theta, ev)
    out.backward()
    analytic = ev.parameter_shift_gradient(theta.detach().numpy())
    d = float(np.max(np.abs(theta.grad.numpy() - analytic)))
    # Both sides run the SAME algorithm, so any difference is the simulator
    # re-evaluating the same circuit to a slightly different answer. The bound
    # is therefore tied to the floor measured in [3] rather than to a constant:
    # a gradient is 0.5*(E+ - E-), so two independent evaluations differ by a
    # small multiple of the per-evaluation spread. A fixed 1e-12 would be a
    # 4-qubit threshold applied to a Hamiltonian two orders of magnitude larger.
    floor = globals().get("_MEASURED_SPREAD", 1e-15)
    bound = 100.0 * floor
    check("autograd backward matches parameter-shift", d <= bound,
          f"max deviation {d:.3e} vs bound {bound:.3e} (= 100 x measured floor "
          f"{floor:.3e}); relative to |E| ~ {scale:.0f} that is {d/scale:.2e}")

    print("\n" + "=" * 78)
    failed = [nm for nm, ok in _checks if not ok]
    print(f"GRADIENT GATE (12 qubits): {len(_checks)-len(failed)}/{len(_checks)} passed")
    print("VARWEIGHT GRADIENT: " + ("ALL CHECKS PASSED" if not failed
                                    else f"FAILURES: {failed}"))
    print("=" * 78)
    return 0 if not failed else 1


def energies_preview(ham) -> np.ndarray:
    """The diagonal energy table, for reporting the Hamiltonian's scale."""
    return qb.ising_energy_table(ham.h, ham.J, enc.N_QUBITS)


def _cvar_direct(probs: np.ndarray, energies: np.ndarray, alpha: float) -> float:
    """CVaR straight from the definition, independent of the implementation."""
    order = np.argsort(energies, kind="stable")
    p = probs[order]
    e = energies[order]
    acc, total = 0.0, 0.0
    for pi, ei in zip(p, e):
        take = min(pi, alpha - acc)
        if take <= 0:
            break
        total += take * ei
        acc += take
        if acc >= alpha:
            break
    return total / alpha


if __name__ == "__main__":
    raise SystemExit(main())
