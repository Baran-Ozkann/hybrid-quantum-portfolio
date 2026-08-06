"""Which qubits can the ansatz actually correlate, at a given depth?

The entangler is a LINEAR CNOT ladder, applied sequentially within each layer:
``CNOT(q0,q1); CNOT(q1,q2); ...; CNOT(q10,q11)``. Two things follow, and they
are not symmetric:

  * FORWARD (low index -> high index) information travels the whole register
    inside a single layer, because each CNOT's target becomes the next CNOT's
    control.
  * BACKWARD (high -> low) it does not. A CNOT writes to its target, so
    influence in the reverse direction accrues roughly one qubit per layer.

That asymmetry matters here because the budget constraint ``sum(level) = 4``
couples all six assets symmetrically -- the ansatz has to represent
correlations in both directions, and at 12 qubits a depth that was ample for 4
may not reach.

This measures it instead of arguing it: perturb one layer-0 Ry angle and see
which qubit marginals move. A qubit whose marginal is unmoved by qubit j's
angle cannot be correlated with it by this circuit at this depth.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum import q_bridge as qb                # noqa: E402
from varweight import encoding as enc             # noqa: E402

logger = logging.getLogger("lightcone")


def marginals(params: np.ndarray, layers: int, n: int,
              entangler: str = "linear") -> np.ndarray:
    """P(qubit q = 1) for each qubit."""
    probs = qb.probabilities(params, layers, n, entangler)
    idx = np.arange(len(probs))
    out = np.empty(n)
    for q in range(n):
        bit = (idx >> (n - 1 - q)) & 1
        out[q] = float(probs[bit == 1].sum())
    return out


def influence_map(layers: int, n: int, seed: int = 0,
                  delta: float = 0.35, entangler: str = "linear") -> np.ndarray:
    """``M[j, q]`` = how much qubit q's marginal moves when angle j is nudged.

    Perturbs the layer-0 Ry angle of qubit j. A generic base point is used
    rather than all-zeros: at zero angles the state is a product state and
    every CNOT acts trivially, which would understate connectivity.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.6, 0.6, qb.n_params(n, layers))
    ref = marginals(base, layers, n, entangler)
    M = np.zeros((n, n))
    for j in range(n):
        p = base.copy()
        p[j] += delta                     # layer-0 Ry for qubit j
        M[j] = np.abs(marginals(p, layers, n, entangler) - ref)
    return M


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 6])
    ap.add_argument("--threshold", type=float, default=1e-6)
    ap.add_argument("--entangler", default="linear", choices=sorted(qb.ENTANGLERS))
    args = ap.parse_args()

    n = enc.N_QUBITS
    ladder = "linear CNOT ladder" if args.entangler == "linear" \
        else "CNOT ladder CLOSED INTO A RING"
    print("=" * 78)
    print(f"ENTANGLER REACH -- {n} qubits, {ladder}")
    print("=" * 78)
    print("\n  Rows = perturbed qubit j, columns = qubit whose marginal moved.")
    print(f"  '#' = moved by more than {args.threshold:g}, '.' = unmoved.\n")

    for L in args.layers:
        M = influence_map(L, n, entangler=args.entangler)
        reached = M > args.threshold
        print(f"  --- {L} layer{'s' if L != 1 else ''} "
              f"({qb.n_params(n, L)} params) ---")
        print("      q  " + "".join(f"{q%10}" for q in range(n)))
        for j in range(n):
            row = "".join("#" if reached[j, q] else "." for q in range(n))
            print(f"      {j:>2} {row}")

        # Backward reach: how far below j does influence extend?
        back = []
        for j in range(n):
            lows = [q for q in range(j) if reached[j, q]]
            back.append(j - min(lows) if lows else 0)
        fwd = []
        for j in range(n):
            highs = [q for q in range(j + 1, n) if reached[j, q]]
            fwd.append(max(highs) - j if highs else 0)
        density = reached.sum() / (n * n)
        print(f"      forward reach  max {max(fwd):>2}  (need {n-1} to span)")
        print(f"      backward reach max {max(back):>2}  (need {n-1} to span)")
        print(f"      pairs connected {reached.sum():>3}/{n*n}  ({density:.1%})")

        # Asset-level: can every asset pair influence each other?
        apa = np.zeros((enc.N_ASSETS, enc.N_ASSETS), dtype=bool)
        for a in range(enc.N_ASSETS):
            for b in range(enc.N_ASSETS):
                qa = [a * 2, a * 2 + 1]
                qb_ = [b * 2, b * 2 + 1]
                apa[a, b] = any(reached[x, y] for x in qa for y in qb_)
        print(f"      asset pairs connected {apa.sum():>2}/{enc.N_ASSETS**2}"
              f"  ({apa.sum()/enc.N_ASSETS**2:.0%})"
              f"   symmetric: {bool((apa == apa.T).all())}\n")

    print("=" * 78)
    print("LIGHTCONE PROBE: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
