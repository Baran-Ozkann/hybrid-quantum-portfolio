"""
classical/hamiltonian.py -- Phase 2: QUBO / Ising problem formulation.

Turns the mean-variance portfolio problem into a diagonal Ising Hamiltonian
that the Phase 3 quantum circuit can minimise, and defines the encoding that
turns measured bitstrings back into portfolio weights.

====================================================================
1. DECISION VARIABLES
====================================================================
One binary variable per asset:

    x_i = 1  -> asset i is HELD          x_i = 0  -> asset i is NOT held

with i indexed by ``data_pipeline.TICKERS`` order, and qubit i carrying x_i.

====================================================================
2. OBJECTIVE
====================================================================
Budget-constrained mean-variance selection, minimised over x in {0,1}^n:

    C(x) = q * x' S x  -  mu' x  +  lambda * (sum_i x_i - B)^2
           \\_________/    \\____/    \\______________________/
             risk          return     budget penalty (B = 2)

``q``       risk-aversion weight; larger q favours lower-variance baskets.
``lambda``  penalty enforcing "hold exactly B assets" (see section 4).

Because x_i is binary, x_i^2 = x_i, so every quadratic diagonal term collapses
into the linear part. Expanding gives the QUBO coefficients:

    Q_ii  = q * S_ii - mu_i + lambda * (1 - 2B)
    Q_ij  = 2 * q * S_ij + 2 * lambda            (i < j)
    const = lambda * B^2

====================================================================
3. QUBO -> ISING
====================================================================
Substitute x_i = (1 - s_i) / 2 with spin s_i in {+1, -1}. This matches the
measurement convention in circuit.qs -- ``Zero -> s = +1`` and ``One -> s = -1``
-- so a qubit measured as One means "asset held", i.e. x_i = 1.

Using x_i x_j = (1 - s_i - s_j + s_i s_j) / 4:

    h_i    = -Q_ii / 2  -  (1/4) * sum_{j != i} Q_ij
    J_ij   =  Q_ij / 4                                   (i < j)
    offset =  const  +  (1/2) sum_i Q_ii  +  (1/4) sum_{i<j} Q_ij

so that, exactly,

    C(x) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset.

``h`` and ``J`` are what get handed to the quantum circuit; ``offset`` is a
constant shift the circuit never sees. It is tracked here so reported energies
can be converted back into true portfolio costs. This identity is not taken on
faith -- ``verify_ising_mapping`` checks it against all 2^n bitstrings.

====================================================================
4. CHOOSING THE PENALTY
====================================================================
Any infeasible x incurs at least 1 * lambda (the penalty is a squared integer
deviation). So the constraint is guaranteed active whenever

    lambda > best_feasible_objective - min_over_all_x objective

Setting ``lambda = penalty_scale * (max_obj - min_obj)`` with
``penalty_scale >= 1`` is sufficient, since ``best_feasible <= max_obj``.
``auto_penalty`` computes this from the actual objective spectrum, and the
Phase 2 gate then confirms by brute force that the true ground state really
does hold exactly B assets. Wildly over-large penalties are avoided because
they squash the energy differences that actually matter toward numerical noise.

====================================================================
5. ENCODING: BITSTRING -> PORTFOLIO WEIGHTS
====================================================================
A feasible bitstring holds exactly B assets; each held asset receives an equal
share of capital:

    w_i = x_i / B          =>  sum_i w_i = 1 exactly.

For B = 2 the reachable portfolios are the C(4,2) = 6 equally weighted pairs.
This is a deliberately coarse but completely unambiguous encoding -- it is what
a 4-qubit binary register can express. Its consequence for benchmarking is
important and is handled explicitly: the quantum method is NOT free to choose
continuous weights, so comparing it against unconstrained continuous Markowitz
compares different feasible sets. ``markowitz.py`` therefore also reports the
best achievable equal-weight B-asset portfolio, which is the true optimum
within the quantum encoding's own search space.

Infeasible bitstrings (|x| != B) have no valid weight decoding;
``decode_weights`` rejects them and callers choose a documented repair policy.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("hamiltonian")

#: Number of assets to hold (budget constraint).
DEFAULT_BUDGET = 2
#: Risk-aversion weight on the variance term.
DEFAULT_RISK_AVERSION = 1.0
#: Multiplier applied to the theoretically sufficient penalty bound.
DEFAULT_PENALTY_SCALE = 1.5


# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioHamiltonian:
    """A fully specified Ising Hamiltonian plus the problem it came from.

    Attributes:
        h: Linear Ising coefficients, shape (n,).
        J: Quadratic coefficients for pairs ``(i, j), i < j``, in row-major
            upper-triangular order -- the order circuit.qs expects.
        offset: Constant energy shift; ``C(x) = E_ising(s) + offset``.
        pairs: The ``(i, j)`` index pairs matching ``J``, for traceability.
        tickers: Asset names, positionally aligned to ``h`` and ``mu``.
        mu: Annualised expected returns.
        sigma: Annualised covariance matrix.
        risk_aversion: The ``q`` weight on the variance term.
        budget: Number of assets to hold, ``B``.
        penalty: The ``lambda`` actually used.
    """

    h: np.ndarray
    J: np.ndarray
    offset: float
    pairs: tuple[tuple[int, int], ...]
    tickers: tuple[str, ...]
    mu: np.ndarray
    sigma: np.ndarray
    risk_aversion: float
    budget: int
    penalty: float

    @property
    def n_assets(self) -> int:
        return len(self.h)

    def to_json(self) -> dict:
        """Serialisable form (numpy arrays become nested lists)."""
        return {
            "h": self.h.tolist(),
            "J": self.J.tolist(),
            "offset": float(self.offset),
            "pairs": [list(p) for p in self.pairs],
            "tickers": list(self.tickers),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "risk_aversion": float(self.risk_aversion),
            "budget": int(self.budget),
            "penalty": float(self.penalty),
        }


def upper_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """Upper-triangular index pairs in the order circuit.qs expects ``jCoeffs``."""
    return tuple((i, j) for i in range(n - 1) for j in range(i + 1, n))


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------

def objective(
    x: Sequence[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
) -> float:
    """Unpenalised objective ``q * x' S x - mu' x`` for a binary selection."""
    xv = np.asarray(x, dtype=float)
    return float(risk_aversion * (xv @ sigma @ xv) - mu @ xv)


def penalised_objective(
    x: Sequence[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    budget: int,
    penalty: float,
) -> float:
    """Full QUBO cost including the budget penalty."""
    xv = np.asarray(x, dtype=float)
    return objective(xv, mu, sigma, risk_aversion) + penalty * (xv.sum() - budget) ** 2


def auto_penalty(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    scale: float = DEFAULT_PENALTY_SCALE,
) -> float:
    """Smallest sufficient budget penalty, times a safety ``scale``.

    Uses the spread of the unpenalised objective over all 2^n selections; see
    section 4 of the module docstring for why that bound is sufficient.
    """
    n = len(mu)
    values = [objective(x, mu, sigma, risk_aversion)
              for x in itertools.product((0, 1), repeat=n)]
    spread = float(max(values) - min(values))
    # Guard the degenerate case of a completely flat objective.
    return scale * max(spread, 1e-6)


# --------------------------------------------------------------------------
# QUBO construction and Ising conversion
# --------------------------------------------------------------------------

def build_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    budget: int = DEFAULT_BUDGET,
    penalty: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build QUBO coefficients for the budget-constrained problem.

    Returns:
        ``(q_diag, q_off, const, penalty)`` where ``q_diag`` has shape (n,),
        ``q_off`` is the strictly-upper-triangular matrix of pair coefficients,
        and ``const`` is the constant term ``lambda * B^2``.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    if sigma.shape != (n, n):
        raise ValueError(f"sigma must be ({n}, {n}), got {sigma.shape}")
    if not 0 <= budget <= n:
        raise ValueError(f"budget must lie in [0, {n}], got {budget}")

    if penalty is None:
        penalty = auto_penalty(mu, sigma, risk_aversion)

    # Diagonal: variance self-term, expected return, and the linear part of the
    # expanded penalty. x_i^2 = x_i is what folds these together.
    q_diag = risk_aversion * np.diag(sigma) - mu + penalty * (1 - 2 * budget)

    q_off = np.zeros((n, n), dtype=float)
    for i, j in upper_pairs(n):
        q_off[i, j] = 2.0 * risk_aversion * sigma[i, j] + 2.0 * penalty

    const = penalty * budget ** 2
    return q_diag, q_off, float(const), float(penalty)


def qubo_to_ising(
    q_diag: np.ndarray,
    q_off: np.ndarray,
    const: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO coefficients to Ising ``(h, J, offset)``.

    Applies ``x_i = (1 - s_i) / 2``; see section 3 of the module docstring.

    Returns:
        ``(h, J, offset)`` with ``J`` flattened in ``upper_pairs`` order.
    """
    n = len(q_diag)
    pairs = upper_pairs(n)

    # Symmetrised off-diagonal lookup, so each h_i can sum over all partners.
    sym = q_off + q_off.T

    h = -0.5 * np.asarray(q_diag, dtype=float) - 0.25 * sym.sum(axis=1)
    J = np.array([0.25 * q_off[i, j] for i, j in pairs], dtype=float)
    offset = (
        float(const)
        + 0.5 * float(np.sum(q_diag))
        + 0.25 * float(sum(q_off[i, j] for i, j in pairs))
    )
    return h, J, offset


def build_hamiltonian(
    mu: np.ndarray,
    sigma: np.ndarray,
    tickers: Sequence[str],
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    budget: int = DEFAULT_BUDGET,
    penalty: float | None = None,
) -> PortfolioHamiltonian:
    """Build the complete Ising Hamiltonian for the portfolio problem."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q_diag, q_off, const, penalty_used = build_qubo(
        mu, sigma, risk_aversion, budget, penalty
    )
    h, J, offset = qubo_to_ising(q_diag, q_off, const)
    return PortfolioHamiltonian(
        h=h, J=J, offset=offset,
        pairs=upper_pairs(len(mu)),
        tickers=tuple(tickers),
        mu=mu, sigma=sigma,
        risk_aversion=risk_aversion, budget=budget, penalty=penalty_used,
    )


# --------------------------------------------------------------------------
# Energies and enumeration
# --------------------------------------------------------------------------

def ising_energy(bits: Sequence[int], h: np.ndarray, J: np.ndarray) -> float:
    """Ising energy of a bitstring under ``(h, J)``, excluding the offset.

    ``bit 0 -> s = +1``, ``bit 1 -> s = -1`` (the circuit.qs convention).
    """
    s = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    n = len(s)
    e = float(h @ s)
    for coeff, (i, j) in zip(J, upper_pairs(n)):
        e += float(coeff) * s[i] * s[j]
    return e


def enumerate_states(ham: PortfolioHamiltonian) -> list[dict]:
    """Evaluate every one of the 2^n bitstrings.

    Returns:
        A list of dicts sorted by ascending QUBO cost, each containing the
        bitstring, its Ising energy, the reconstructed cost, feasibility, and
        -- for feasible states -- portfolio weights, return, and volatility.
    """
    n = ham.n_assets
    rows: list[dict] = []
    for bits in itertools.product((0, 1), repeat=n):
        cost = penalised_objective(
            bits, ham.mu, ham.sigma, ham.risk_aversion, ham.budget, ham.penalty
        )
        e_ising = ising_energy(bits, ham.h, ham.J)
        held = int(sum(bits))
        feasible = held == ham.budget
        row: dict = {
            "bits": list(bits),
            "bitstring": "".join(str(b) for b in bits),
            "held": [ham.tickers[i] for i, b in enumerate(bits) if b],
            "n_held": held,
            "feasible": feasible,
            "qubo_cost": cost,
            "ising_energy": e_ising,
            "ising_plus_offset": e_ising + ham.offset,
            "objective_unpenalised": objective(bits, ham.mu, ham.sigma, ham.risk_aversion),
        }
        if feasible:
            w = decode_weights(bits, ham.budget)
            row["weights"] = w.tolist()
            row["expected_return"] = float(ham.mu @ w)
            row["volatility"] = float(np.sqrt(w @ ham.sigma @ w))
        rows.append(row)
    rows.sort(key=lambda r: r["qubo_cost"])
    return rows


def brute_force_ground_state(ham: PortfolioHamiltonian) -> dict:
    """Exact minimiser of the penalised cost, by enumerating all 2^n states.

    Tractable only because n = 4. This is the reference the quantum optimiser
    is judged against -- with 16 states the true optimum is known exactly, so
    "did VQE find the ground state?" has an unambiguous answer.
    """
    return enumerate_states(ham)[0]


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def decode_weights(bits: Sequence[int], budget: int = DEFAULT_BUDGET) -> np.ndarray:
    """Decode a feasible bitstring into portfolio weights summing to 1.

    Equal capital across held assets: ``w_i = x_i / B``.

    Raises:
        ValueError: if the bitstring does not hold exactly ``budget`` assets.
            Infeasible strings have no defined decoding; the caller must apply
            an explicit repair policy (see ``decode_with_repair``).
    """
    x = np.asarray(bits, dtype=float)
    held = int(x.sum())
    if held != budget:
        raise ValueError(
            f"Bitstring {''.join(str(int(b)) for b in bits)} holds {held} assets, "
            f"expected exactly {budget}; it has no valid weight decoding."
        )
    return x / float(budget)


def decode_with_repair(
    bits: Sequence[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    budget: int = DEFAULT_BUDGET,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
) -> tuple[np.ndarray, bool]:
    """Decode a bitstring, repairing it to feasibility if necessary.

    Repair policy, applied only when ``|x| != budget``: choose the feasible
    bitstring at minimum Hamming distance from ``bits``, breaking ties by
    lowest unpenalised objective. Deterministic, and it never silently invents
    an allocation the measurement did not support.

    Returns:
        ``(weights, was_repaired)``.
    """
    x = np.asarray(bits, dtype=int)
    if int(x.sum()) == budget:
        return decode_weights(x, budget), False

    n = len(x)
    best, best_key = None, None
    for cand in itertools.product((0, 1), repeat=n):
        if sum(cand) != budget:
            continue
        key = (int(np.abs(np.asarray(cand) - x).sum()),
               objective(cand, mu, sigma, risk_aversion))
        if best_key is None or key < best_key:
            best, best_key = cand, key
    assert best is not None, "no feasible bitstring exists for this budget"
    return decode_weights(best, budget), True


def decode_distribution(
    counts: dict,
    budget: int = DEFAULT_BUDGET,
) -> tuple[np.ndarray, dict]:
    """Decode a measured bitstring distribution by feasible-mode selection.

    Takes the most frequently observed FEASIBLE bitstring and decodes it. This
    is the standard VQE/QAOA readout: the variational state concentrates
    amplitude on the minimising basis state, so the mode is the answer, while
    infeasible outcomes are leakage the budget penalty is meant to suppress.

    Args:
        counts: Mapping of bitstring (``"1010"`` or a tuple of ints) to count.
        budget: Required number of held assets.

    Returns:
        ``(weights, diagnostics)`` where diagnostics reports the chosen
        bitstring, its share of shots, and the total feasible fraction.

    Raises:
        ValueError: if the distribution is empty, or no feasible bitstring was
            ever observed.
    """
    norm: dict[tuple[int, ...], int] = {}
    for key, count in counts.items():
        bits = tuple(int(c) for c in key) if isinstance(key, str) else tuple(int(b) for b in key)
        norm[bits] = norm.get(bits, 0) + int(count)

    total = sum(norm.values())
    if total == 0:
        raise ValueError("Empty measurement distribution.")

    feasible = {b: c for b, c in norm.items() if sum(b) == budget}
    if not feasible:
        raise ValueError(
            f"No feasible bitstring (exactly {budget} assets held) was measured "
            f"across {total} shots. The optimiser has not concentrated on the "
            "constrained subspace; inspect the penalty weight."
        )

    # Deterministic tie-break on the bitstring itself, so repeated runs on the
    # same distribution cannot disagree.
    mode = max(feasible, key=lambda b: (feasible[b], tuple(-v for v in b)))
    diagnostics = {
        "mode_bitstring": "".join(str(b) for b in mode),
        "mode_count": feasible[mode],
        "mode_probability": feasible[mode] / total,
        "feasible_fraction": sum(feasible.values()) / total,
        "total_shots": total,
        "distinct_outcomes": len(norm),
    }
    return decode_weights(mode, budget), diagnostics


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify_ising_mapping(ham: PortfolioHamiltonian, tol: float = 1e-9) -> tuple[bool, float]:
    """Check ``E_ising(s(x)) + offset == C_qubo(x)`` for all 2^n bitstrings.

    The QUBO->Ising algebra is easy to get subtly wrong (a factor of two on the
    off-diagonal, a dropped constant), and such an error would not crash
    anything -- it would silently optimise a different problem. This checks the
    identity exhaustively rather than trusting the derivation.

    Returns:
        ``(ok, max_abs_error)``.
    """
    worst = 0.0
    for row in enumerate_states(ham):
        worst = max(worst, abs(row["ising_plus_offset"] - row["qubo_cost"]))
    return worst <= tol, worst


def save_hamiltonian(ham: PortfolioHamiltonian, states: list[dict] | None = None) -> Path:
    """Persist the Hamiltonian (and optionally the full spectrum) to ``data/``."""
    payload = ham.to_json()
    if states is not None:
        payload["spectrum"] = states
    path = DATA_DIR / "hamiltonian.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote Hamiltonian to %s", path)
    return path


def load_hamiltonian() -> PortfolioHamiltonian:
    """Load the Hamiltonian previously written by ``save_hamiltonian``."""
    path = DATA_DIR / "hamiltonian.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python classical/hamiltonian.py")
    d = json.loads(path.read_text(encoding="utf-8"))
    return PortfolioHamiltonian(
        h=np.asarray(d["h"], dtype=float),
        J=np.asarray(d["J"], dtype=float),
        offset=float(d["offset"]),
        pairs=tuple(tuple(p) for p in d["pairs"]),
        tickers=tuple(d["tickers"]),
        mu=np.asarray(d["mu"], dtype=float),
        sigma=np.asarray(d["sigma"], dtype=float),
        risk_aversion=float(d["risk_aversion"]),
        budget=int(d["budget"]),
        penalty=float(d["penalty"]),
    )
