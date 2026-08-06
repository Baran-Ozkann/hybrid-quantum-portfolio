"""Can the ansatz reach every basis state at all?

Both 600-epoch runs converged with P(ground state) EXACTLY 0.0 -- not small,
zero. That has two very different explanations and they need separating before
anything else in this track is interpreted:

  (a) the optimiser drove the amplitude to zero, or
  (b) the ansatz cannot represent that state from |0...0>, at any parameters.

If (b), no entangler choice, depth, or epoch budget will ever find it, and the
encoding/ansatz pairing is broken rather than merely hard. That is a structural
question about the circuit, answerable at random parameters in seconds -- no
training required.

Method: sample random parameter vectors, count basis states whose probability
is exactly zero, and check whether the same states are zero every time. A state
that is zero at every random point is unreachable; one that is zero only
sometimes is a coincidence of that point.
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
from varweight import hamiltonian as vh           # noqa: E402
from varweight.data import statistics             # noqa: E402

logger = logging.getLogger("probe_support")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--entangler", default="linear", choices=sorted(qb.ENTANGLERS))
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    n = enc.N_QUBITS
    mu, sigma, _ = statistics()
    ham = vh.build_hamiltonian(mu, sigma)
    states = vh.enumerate_states(ham)
    ground_idx = states[0]["index"]

    print("=" * 78)
    print(f"ANSATZ SUPPORT -- {n} qubits, {args.layers} layers, "
          f"{args.entangler} entangler")
    print("=" * 78)

    rng = np.random.default_rng(12345)
    zero_masks = []
    for k in range(args.samples):
        # Wide range: if a state is reachable at all, a generic point finds it.
        p = rng.uniform(-np.pi, np.pi, qb.n_params(n, args.layers))
        probs = qb.probabilities(p, args.layers, n, args.entangler)
        zero_masks.append(probs == 0.0)
        print(f"  sample {k}: exact zeros {int((probs == 0.0).sum()):>5} / {2**n}"
              f"   P(ground) = {probs[ground_idx]:.3e}"
              f"   min nonzero = {probs[probs > 0].min():.3e}")

    always_zero = np.all(np.array(zero_masks), axis=0)
    n_always = int(always_zero.sum())
    print(f"\n  states with zero probability at EVERY sampled point: "
          f"{n_always} / {2**n}")
    print(f"  ground state ({states[0]['bitstring']}, levels "
          f"{states[0]['levels']}) always zero: {bool(always_zero[ground_idx])}")

    if n_always:
        feas_always = sum(1 for s in states
                          if s["feasible"] and always_zero[s["index"]])
        print(f"  of those, feasible: {feas_always} / {sum(1 for s in states if s['feasible'])}")
        idxs = np.nonzero(always_zero)[0][:8]
        print("  examples: " + ", ".join(
            enc.bitstring(enc.index_to_bits(int(i))) for i in idxs))

    verdict = ("REACHABLE -- the zeros in training were driven there by the "
               "optimiser" if not always_zero[ground_idx] else
               "UNREACHABLE -- the ansatz cannot represent the ground state")
    print(f"\n  VERDICT: ground state is {verdict}")
    print("\nSUPPORT PROBE: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
