"""QUBO / Ising Hamiltonian for the variable-weight encoding.

The Phase 2 objective is unchanged in form:

    C(w) = -mu' w  +  q * w' Sigma w  +  lambda * (sum(level) - B)^2

What changes is that ``w`` is no longer a 0/1 selection over assets but
``w = L b / B``, where ``L`` is the encoding matrix from ``encoding.py`` and
``b`` is the 12-bit register. Substituting keeps the whole thing quadratic in
``b``, so it is still a QUBO and still maps to an Ising Hamiltonian by the
Phase 2 route -- with 66 coupling terms instead of 6.

Building it by substitution rather than by expanding indices by hand is
deliberate: the 4-asset version was hand-derived and verified exhaustively,
and at 12 qubits an index slip would be far harder to see. Here the only
hand-written step is the matrix algebra below, and the gate checks it against
a direct evaluation of ``C(w)`` on all 4096 states.

PENALTY WEIGHT. The budget is enforced through the objective, so lambda must
be large enough that no infeasible state can undercut the best feasible one.
Any infeasible state has ``|sum(level) - B| >= 1`` and therefore pays at least
``lambda``. So

    lambda > (max objective - min objective) over all states

is sufficient, where "objective" means the return + risk part alone. The gate
verifies the consequence -- that the ground state is feasible -- rather than
trusting the bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from varweight.encoding import (
    BUDGET, N_ASSETS, N_QUBITS, N_STATES, TICKERS,
    bits_to_levels, index_to_bits, level_matrix,
)

#: Risk-aversion weight on the variance term. Matches Phase 2.
DEFAULT_RISK_AVERSION = 1.0
#: Safety multiplier on the EXACT minimal penalty (see ``minimal_penalty``).
#: 1.85 keeps the feasible optimum globally optimal with margin -- the measured
#: cliff is at 0.92x, where an infeasible state undercuts it -- while keeping
#: the energy scale small enough that the optimiser can still see the ~0.06 gap
#: between the top two feasible states.
DEFAULT_PENALTY_SCALE = 1.85


def upper_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """``(i, j)`` with ``i < j`` in the row-major order circuit.qs expects."""
    return tuple((i, j) for i in range(n - 1) for j in range(i + 1, n))


def objective_from_levels(levels: Sequence[int], mu: np.ndarray,
                          sigma: np.ndarray, risk_aversion: float,
                          budget: int = BUDGET) -> float:
    """Return + risk part only, with ``w = levels / budget``.

    Evaluated for INFEASIBLE levels too -- the weights simply do not sum to 1
    there. That is intentional: the penalty term is what makes such states
    expensive, and the gate needs the unpenalised part on every state to check
    the sufficient bound on lambda.
    """
    w = np.asarray(levels, dtype=float) / float(budget)
    return float(-mu @ w + risk_aversion * (w @ sigma @ w))


def _objective_table(mu: np.ndarray, sigma: np.ndarray,
                     risk_aversion: float, budget: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Unpenalised objective and level-sum violation for all 4096 states."""
    L = level_matrix()
    obj = np.empty(N_STATES, dtype=float)
    viol = np.empty(N_STATES, dtype=int)
    for idx in range(N_STATES):
        lv = L @ np.asarray(index_to_bits(idx), dtype=float)
        obj[idx] = objective_from_levels(lv, mu, sigma, risk_aversion, budget)
        viol[idx] = int(lv.sum()) - budget
    return obj, viol


def minimal_penalty(mu: np.ndarray, sigma: np.ndarray,
                    risk_aversion: float = DEFAULT_RISK_AVERSION,
                    budget: int = BUDGET) -> float:
    """EXACT smallest lambda keeping the feasible optimum globally optimal.

    An infeasible state with violation ``v`` pays ``lambda * v^2``, so the
    binding requirement is

        lambda > max over infeasible s of  (obj(best feasible) - obj(s)) / v_s^2

    and the ``v^2`` in the denominator matters: violations here reach 14, so a
    bound assuming ``v = 1`` is up to 196x too strict.

    WHY THIS REPLACED THE OLD BOUND. The previous version used
    ``scale * (max obj - min obj)``, i.e. the ``v = 1`` worst case over the
    whole objective range. Measured on this instance it returned 3.697 against
    a true minimum of 0.1335 -- **27.7x too large** -- and that was not free:
    at lambda = 3.697 the optimiser drove the true ground state to EXACTLY zero
    amplitude, while at 0.2465 the same run left it at 4.9e-04 (rank 5 of 4096)
    for a cost of 0.6 percentage points of feasible mass. An over-large penalty
    buys almost no feasibility and actively expels the state it is protecting.
    """
    obj, viol = _objective_table(mu, sigma, risk_aversion, budget)
    feasible = viol == 0
    if not feasible.any():
        raise ValueError("no feasible state exists for this budget")
    best_feasible = float(obj[feasible].min())
    infeasible = ~feasible
    need = (best_feasible - obj[infeasible]) / (viol[infeasible].astype(float) ** 2)
    return float(max(need.max(), 0.0))


def sufficient_penalty(mu: np.ndarray, sigma: np.ndarray,
                       risk_aversion: float = DEFAULT_RISK_AVERSION,
                       budget: int = BUDGET,
                       scale: float = DEFAULT_PENALTY_SCALE) -> float:
    """The exact minimum, times a safety factor.

    Kept as the name ``build_qubo`` calls so the change is one edit, but the
    quantity is now the exact bound rather than the loose one.
    """
    return float(scale * minimal_penalty(mu, sigma, risk_aversion, budget))


@dataclass(frozen=True)
class VarWeightHamiltonian:
    """Ising Hamiltonian for the variable-weight problem, plus its provenance."""

    h: np.ndarray                  # (N_QUBITS,)
    J: np.ndarray                  # (66,), in upper_pairs order
    offset: float
    pairs: tuple[tuple[int, int], ...]
    tickers: tuple[str, ...]
    mu: np.ndarray
    sigma: np.ndarray
    risk_aversion: float
    budget: int
    penalty: float

    @property
    def n_qubits(self) -> int:
        return len(self.h)


def build_qubo(mu: np.ndarray, sigma: np.ndarray,
               risk_aversion: float = DEFAULT_RISK_AVERSION,
               budget: int = BUDGET,
               penalty: float | None = None
               ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """QUBO coefficients over the 12 bits.

    With ``l = L b`` and ``w = l / B``::

        -mu' w          = -(1/B) (L' mu)' b                        [linear]
         q  w' S w      =  (q/B^2) b' (L' S L) b                   [quadratic]
         lam (1'l - B)^2 = lam[ (v'b)^2 - 2B v'b + B^2 ],  v = L' 1

    Using ``b_a^2 = b_a`` folds each quadratic diagonal into the linear part.

    Returns:
        ``(q_diag, q_off, const, penalty)`` with ``q_off`` strictly upper
        triangular, matching the Phase 2 signature.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if mu.shape != (N_ASSETS,):
        raise ValueError(f"mu must have shape ({N_ASSETS},), got {mu.shape}")
    if sigma.shape != (N_ASSETS, N_ASSETS):
        raise ValueError(f"sigma must be ({N_ASSETS}, {N_ASSETS}), got {sigma.shape}")

    if penalty is None:
        penalty = sufficient_penalty(mu, sigma, risk_aversion, budget)

    L = level_matrix()                       # (6, 12)
    v = L.sum(axis=0)                        # (12,) = L' 1, the level weight of each bit
    B = float(budget)

    # Quadratic form b' M b, symmetric.
    M = (risk_aversion / (B * B)) * (L.T @ sigma @ L) + penalty * np.outer(v, v)
    # Linear part.
    c = -(1.0 / B) * (L.T @ mu) - 2.0 * penalty * B * v
    const = penalty * B * B

    q_diag = c + np.diag(M)
    q_off = np.zeros((N_QUBITS, N_QUBITS), dtype=float)
    for i in range(N_QUBITS - 1):
        for j in range(i + 1, N_QUBITS):
            q_off[i, j] = 2.0 * M[i, j]

    return q_diag, q_off, float(const), float(penalty)


def qubo_to_ising(q_diag: np.ndarray, q_off: np.ndarray, const: float
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO coefficients to Ising ``(h, J, offset)`` via ``x = (1-s)/2``.

    Identical algebra to ``classical/hamiltonian.py``; repeated here rather
    than imported so the variable-weight track does not silently inherit a
    change made for the 4-asset problem.
    """
    n = len(q_diag)
    pairs = upper_pairs(n)
    sym = q_off + q_off.T

    h = -0.5 * np.asarray(q_diag, dtype=float) - 0.25 * sym.sum(axis=1)
    J = np.array([0.25 * q_off[i, j] for i, j in pairs], dtype=float)
    offset = (
        float(const)
        + 0.5 * float(np.sum(q_diag))
        + 0.25 * float(np.sum(q_off))
    )
    return h, J, offset


def build_hamiltonian(mu: np.ndarray, sigma: np.ndarray,
                      tickers: Sequence[str] = TICKERS,
                      risk_aversion: float = DEFAULT_RISK_AVERSION,
                      budget: int = BUDGET,
                      penalty: float | None = None) -> VarWeightHamiltonian:
    """Build the complete Ising Hamiltonian for the variable-weight problem."""
    q_diag, q_off, const, penalty_used = build_qubo(
        mu, sigma, risk_aversion, budget, penalty
    )
    h, J, offset = qubo_to_ising(q_diag, q_off, const)
    return VarWeightHamiltonian(
        h=h, J=J, offset=offset,
        pairs=upper_pairs(N_QUBITS),
        tickers=tuple(tickers),
        mu=np.asarray(mu, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        risk_aversion=risk_aversion,
        budget=budget,
        penalty=penalty_used,
    )


def penalised_cost(levels: Sequence[int], ham: VarWeightHamiltonian) -> float:
    """The full objective, evaluated directly from levels.

    This is the independent reference the gate compares the Ising energies
    against -- it never touches ``h`` or ``J``.
    """
    base = objective_from_levels(levels, ham.mu, ham.sigma,
                                 ham.risk_aversion, ham.budget)
    violation = float(np.sum(levels) - ham.budget)
    return base + ham.penalty * violation * violation


def ising_energy(bits: Sequence[int], h: np.ndarray, J: np.ndarray) -> float:
    """Ising energy of a bitstring, excluding the offset.

    ``bit 0 -> s = +1``, ``bit 1 -> s = -1`` (the circuit.qs convention).
    """
    s = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    e = float(h @ s)
    for coeff, (i, j) in zip(J, upper_pairs(len(s))):
        e += float(coeff) * s[i] * s[j]
    return e


def enumerate_states(ham: VarWeightHamiltonian) -> list[dict]:
    """Evaluate all 4096 states, sorted by ascending Ising energy.

    Tractable by design: keeping brute-force ground truth affordable is the
    constraint the whole encoding was chosen around.
    """
    rows: list[dict] = []
    for idx in range(N_STATES):
        bits = index_to_bits(idx)
        levels = bits_to_levels(bits)
        total = int(levels.sum())
        feasible = total == ham.budget
        row = {
            "index": idx,
            "bits": bits,
            "bitstring": "".join(str(b) for b in bits),
            "levels": levels.tolist(),
            "level_sum": total,
            "feasible": feasible,
            "ising_energy": ising_energy(bits, ham.h, ham.J),
            "cost": penalised_cost(levels, ham),
        }
        if feasible:
            w = levels.astype(float) / float(ham.budget)
            row["weights"] = w.tolist()
            row["held"] = [ham.tickers[i] for i, lv in enumerate(levels) if lv > 0]
            row["expected_return"] = float(ham.mu @ w)
            row["volatility"] = float(np.sqrt(max(w @ ham.sigma @ w, 0.0)))
        rows.append(row)
    rows.sort(key=lambda r: r["ising_energy"])
    return rows
