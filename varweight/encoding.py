"""Bit <-> level <-> weight mapping for the variable-weight encoding.

LAYOUT. Each asset gets 2 qubits holding a level in {0, 1, 2, 3}:

    asset i  ->  qubits (2i, 2i+1),  level_i = 2 * bit[2i] + bit[2i+1]

so qubit 2i is the MOST significant bit of asset i's level. Qubit 0 is the
most significant bit of the state-vector index (the Phase 1 convention,
verified empirically there), so the state index reads as the concatenation of
the per-asset level bits in asset order.

BUDGET. The levels must sum to exactly ``BUDGET`` (4). Weights are then
``w_i = level_i / 4``, i.e. multiples of 25% summing to 100%.

    THE CONSTRAINT IS ENFORCED, NOT NORMALISED.

Normalising -- dividing by whatever the levels happen to sum to -- would make
every one of the 4096 states feasible, and would destroy the finding in
``docs/objective_comparison.md`` section 7: that a penalty-encoded constraint
stops binding on the part of the distribution CVaR assigns zero weight, so
small alpha produces illegal portfolios. That result is one of the more
transferable things this project has produced, and keeping a hard constraint
is what keeps it testable at the new scale.

The all-zero state (every level 0) is infeasible here, exactly as ``0000`` was
in the 4-asset encoding.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb
from typing import Iterator, Sequence

import numpy as np

#: Assets, in canonical column order. The first four are retained from
#: Phases 1-9 for continuity; see ``data.py`` for the selection rule and its
#: bias disclosure.
TICKERS: tuple[str, ...] = ("TSLA", "NVDA", "AMD", "COIN", "XOM", "JNJ")

N_ASSETS = len(TICKERS)
QUBITS_PER_ASSET = 2
LEVELS_PER_ASSET = 2 ** QUBITS_PER_ASSET      # 0, 1, 2, 3
N_QUBITS = N_ASSETS * QUBITS_PER_ASSET        # 12
N_STATES = 2 ** N_QUBITS                      # 4096

#: Levels must sum to exactly this. Weights are level/BUDGET.
BUDGET = 4

#: Bit significance within one asset's level, most significant first.
_BIT_VALUES = tuple(2 ** (QUBITS_PER_ASSET - 1 - k) for k in range(QUBITS_PER_ASSET))


def level_matrix() -> np.ndarray:
    """The linear map ``levels = L @ bits``, shape ``(N_ASSETS, N_QUBITS)``.

    Having the encoding as an explicit matrix is what lets the Hamiltonian be
    built by substitution rather than by hand-expanding index arithmetic.
    """
    L = np.zeros((N_ASSETS, N_QUBITS), dtype=float)
    for i in range(N_ASSETS):
        for k, v in enumerate(_BIT_VALUES):
            L[i, i * QUBITS_PER_ASSET + k] = v
    return L


def bits_to_levels(bits: Sequence[int]) -> np.ndarray:
    """Decode a 12-bit qubit-ordered string into 6 integer levels."""
    b = np.asarray(bits, dtype=int)
    if b.shape != (N_QUBITS,):
        raise ValueError(f"expected {N_QUBITS} bits, got {b.shape}")
    if not np.all((b == 0) | (b == 1)):
        raise ValueError("bits must be 0 or 1")
    return (level_matrix() @ b).astype(int)


def levels_to_bits(levels: Sequence[int]) -> list[int]:
    """Encode 6 integer levels into a 12-bit qubit-ordered string.

    Raises:
        ValueError: if any level is outside ``[0, LEVELS_PER_ASSET)``.
    """
    lv = np.asarray(levels, dtype=int)
    if lv.shape != (N_ASSETS,):
        raise ValueError(f"expected {N_ASSETS} levels, got {lv.shape}")
    if np.any(lv < 0) or np.any(lv >= LEVELS_PER_ASSET):
        raise ValueError(
            f"levels must lie in [0, {LEVELS_PER_ASSET}), got {lv.tolist()}"
        )
    bits: list[int] = []
    for value in lv:
        for k in range(QUBITS_PER_ASSET):
            shift = QUBITS_PER_ASSET - 1 - k
            bits.append(int(value >> shift) & 1)
    return bits


def is_feasible(levels: Sequence[int], budget: int = BUDGET) -> bool:
    """Do these levels satisfy the budget exactly?"""
    return int(np.sum(levels)) == budget


def levels_to_weights(levels: Sequence[int], budget: int = BUDGET) -> np.ndarray:
    """Portfolio weights ``w_i = level_i / budget``.

    Raises:
        ValueError: if the levels do not satisfy the budget. Infeasible states
            have no defined decoding -- the caller must apply an explicit
            repair policy, exactly as in the 4-asset encoding.
    """
    lv = np.asarray(levels, dtype=float)
    if int(lv.sum()) != budget:
        raise ValueError(
            f"levels {lv.astype(int).tolist()} sum to {int(lv.sum())}, "
            f"expected exactly {budget}; no valid weight decoding."
        )
    return lv / float(budget)


def index_to_bits(index: int) -> list[int]:
    """State-vector index -> qubit-ordered bits (qubit 0 = MSB)."""
    return [(index >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)]


def bits_to_index(bits: Sequence[int]) -> int:
    """Qubit-ordered bits -> state-vector index (qubit 0 = MSB)."""
    idx = 0
    for q in range(N_QUBITS):
        idx |= (int(bits[q]) & 1) << (N_QUBITS - 1 - q)
    return idx


def bitstring(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def feasible_level_tuples(budget: int = BUDGET,
                          n_assets: int = N_ASSETS,
                          levels: int = LEVELS_PER_ASSET) -> Iterator[tuple[int, ...]]:
    """Every level assignment summing to ``budget``, in lexicographic order."""
    for combo in itertools.product(range(levels), repeat=n_assets):
        if sum(combo) == budget:
            yield combo


def feasible_count(budget: int = BUDGET,
                   n_assets: int = N_ASSETS,
                   levels: int = LEVELS_PER_ASSET) -> int:
    """Closed form for the number of feasible states.

    Compositions of ``budget`` into ``n_assets`` non-negative parts is
    ``C(budget + n_assets - 1, n_assets - 1)``. Parts are capped at
    ``levels - 1``, so subtract by inclusion-exclusion over which parts exceed
    the cap.
    """
    cap = levels - 1
    total = 0
    for j in range(n_assets + 1):
        rem = budget - j * (cap + 1)
        if rem < 0:
            break
        sign = -1 if j % 2 else 1
        total += sign * comb(n_assets, j) * comb(rem + n_assets - 1, n_assets - 1)
    return total


@dataclass(frozen=True)
class EncodingSummary:
    """What the encoding can express, for the record and for docs."""

    n_assets: int
    qubits_per_asset: int
    n_qubits: int
    n_states: int
    budget: int
    n_feasible: int
    feasible_fraction: float
    n_distinct_weight_vectors: int

    def as_dict(self) -> dict:
        return {
            "n_assets": self.n_assets,
            "qubits_per_asset": self.qubits_per_asset,
            "n_qubits": self.n_qubits,
            "n_states": self.n_states,
            "budget": self.budget,
            "n_feasible": self.n_feasible,
            "feasible_fraction": self.feasible_fraction,
            "n_distinct_weight_vectors": self.n_distinct_weight_vectors,
        }


def summarise() -> EncodingSummary:
    """Counts describing the search space.

    ``n_distinct_weight_vectors`` equals ``n_feasible`` here because the level
    -> weight map is injective; it is reported separately so a future encoding
    with redundant representations cannot silently inflate the feasible count.
    """
    feas = list(feasible_level_tuples())
    weights = {tuple(np.asarray(t, dtype=float) / BUDGET) for t in feas}
    return EncodingSummary(
        n_assets=N_ASSETS,
        qubits_per_asset=QUBITS_PER_ASSET,
        n_qubits=N_QUBITS,
        n_states=N_STATES,
        budget=BUDGET,
        n_feasible=len(feas),
        feasible_fraction=len(feas) / N_STATES,
        n_distinct_weight_vectors=len(weights),
    )
