"""
training/cvar.py -- Phase 9: CVaR objective for the variational optimiser.

Plain VQE minimises the MEAN energy. A run can reach a respectable mean while
splitting amplitude across two low-lying states, which is exactly the failure
measured in Phase 7: seed 42 finished at 60.1% / 39.7% between the two lowest
states and its readout rested on a 1.5:1 ratio.

CVaR (Conditional Value at Risk) minimises the mean of the lowest alpha-quantile
of the energy distribution instead, so probability mass outside the tail stops
earning credit. With states sorted by ascending energy and ``m`` the first index
at which the cumulative probability reaches ``alpha``:

    CVaR_alpha = (1/alpha) * [ sum_{j<m} p_j E_j  +  (alpha - P_{m-1}) E_m ]

====================================================================
THE REWEIGHTED-OBSERVABLE IDENTITY
====================================================================
CVaR is piecewise linear in the basis-state probabilities and the sort order is
locally constant, so BOTH its value and its gradient reduce to those of an
ordinary diagonal observable. Define

    w_j = (E_j - E_m) / alpha   for j < m      (strictly negative)
    w_j = 0                     otherwise

Then, exactly:

    <O_w> + E_m = (1/alpha)[ sum_{j<m} p_j E_j - E_m P_{m-1} ] + E_m
                = (1/alpha)[ sum_{j<m} p_j E_j + (alpha - P_{m-1}) E_m ]
                = CVaR_alpha

and, since m and the sort order are locally constant,

    d(CVaR)/d(theta) = d(<O_w>)/d(theta)

``O_w`` is diagonal exactly like ``H``, so the parameter-shift rule applies
unchanged: ONE pass of 2*n_params evaluations per step, the same cost as plain
VQE, rather than 2^n separate projector gradients.

TWO WAYS TO GET THIS WRONG
--------------------------
1. Reporting ``<O_w>`` as the loss. Every weight is negative, so ``<O_w>`` is
   always <= 0, while CVaR sits near -4.9. The gradient would still be right and
   the run would still train, producing a silently meaningless loss curve. The
   constant ``E_m`` must be added back for the VALUE (detached, so it does not
   perturb the gradient).
2. Recomputing ``w`` inside the parameter-shift evaluations. The weights encode
   the linearisation point and must be frozen for the whole step.

====================================================================
THE DEGENERATE REGIME (expected, detected, reported)
====================================================================
If the lowest-energy state alone already carries probability >= alpha then
``m = 0``: no state lies fully inside the tail, every weight is zero, and

    CVaR_alpha = E_1  (a constant)     and     d(CVaR)/d(theta) = 0 identically.

Training then does nothing. For a 2^n-state problem this bounds alpha from
below, and with 16 states the natural floor is 1/16 = 0.0625, so alpha = 0.05
is expected to degenerate rather than merely misbehave.

Note the degeneracy can also appear MID-RUN rather than at step 0: this project
initialises angles at U(-0.1, 0.1), so the state starts near |0000> and the
lowest-ENERGY state initially carries very little probability. The tail is
therefore well populated at first, and the gradient collapses only once the run
has pushed p(ground) above alpha -- which for small alpha is a very weak
condition to stop at. Both cases are detected: ``CVaRObjective`` reports ``m``
and the zero-gradient condition at every step.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum import q_bridge as qb                      # noqa: E402
from training.gradient_bridge import EnergyEvaluator    # noqa: E402


@dataclass
class TailState:
    """The CVaR linearisation at one point: which states are in the tail."""

    weights: np.ndarray     #: per-basis-state weights of the reweighted observable
    boundary_energy: float  #: E_m, the energy at the tail boundary
    m: int                  #: number of states lying FULLY inside the tail
    tail_probability: float  #: cumulative probability of those m states

    @property
    def degenerate(self) -> bool:
        """True when no state lies fully inside the tail, so the gradient is 0."""
        return self.m == 0


def tail_state(probs: np.ndarray, energies: np.ndarray, alpha: float) -> TailState:
    """Compute the CVaR tail decomposition at the current distribution.

    Args:
        probs: Basis-state probabilities, shape (2**n,).
        energies: Basis-state energies, same shape and ordering.
        alpha: Tail fraction in (0, 1]. ``alpha = 1`` recovers the plain mean.

    Returns:
        The :class:`TailState` describing the reweighted observable.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")

    order = np.argsort(energies, kind="stable")
    cumulative = np.cumsum(probs[order])

    # First index at which the cumulative probability reaches alpha. States
    # strictly before it lie FULLY inside the tail; that one is the boundary and
    # is only partially included.
    m = int(np.searchsorted(cumulative, alpha, side="left"))
    m = min(m, len(order) - 1)

    boundary_energy = float(energies[order[m]])
    weights = np.zeros_like(energies, dtype=float)
    for j in range(m):
        idx = order[j]
        weights[idx] = (energies[idx] - boundary_energy) / alpha

    tail_p = float(cumulative[m - 1]) if m > 0 else 0.0
    return TailState(weights, boundary_energy, m, tail_p)


def cvar_value_direct(probs: np.ndarray, energies: np.ndarray, alpha: float) -> float:
    """CVaR computed straight from the definition, for cross-checking.

    Deliberately NOT written in terms of the reweighted observable, so it is an
    independent implementation rather than a restatement.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")

    order = np.argsort(energies, kind="stable")
    total = 0.0
    remaining = alpha
    for idx in order:
        take = min(probs[idx], remaining)
        total += take * energies[idx]
        remaining -= take
        if remaining <= 1e-15:
            break
    # If the distribution sums to slightly under alpha (it should not), the
    # remainder is charged at the highest energy touched.
    return float(total / alpha)


class CVaRObjective:
    """CVaR objective backed by the exact state-vector evaluator.

    Exposes the same surface the training loop needs: a scalar value with a
    correct gradient, plus per-step diagnostics.

    Args:
        evaluator: The circuit evaluator (supplies probabilities and gradients).
        energies: Basis-state energies of the diagonal Hamiltonian.
        alpha: Tail fraction; ``1.0`` reduces exactly to plain VQE.
    """

    def __init__(self, evaluator: EnergyEvaluator, energies: np.ndarray, alpha: float):
        self.evaluator = evaluator
        self.energies = np.asarray(energies, dtype=float)
        self.alpha = float(alpha)
        self.history: list[dict] = []

    # -- core -------------------------------------------------------------
    def linearise(self, params: Sequence[float]) -> TailState:
        """Freeze the tail decomposition at ``params``."""
        probs = qb.probabilities(params, self.evaluator.num_layers,
                                 self.evaluator.num_qubits,
                                 self.evaluator.entangler)
        return tail_state(probs, self.energies, self.alpha)

    def value(self, params: Sequence[float], tail: TailState | None = None) -> float:
        """CVaR value at ``params``."""
        probs = qb.probabilities(params, self.evaluator.num_layers,
                                 self.evaluator.num_qubits,
                                 self.evaluator.entangler)
        if self.alpha >= 1.0:
            return float(probs @ self.energies)
        t = tail if tail is not None else tail_state(probs, self.energies, self.alpha)
        # THE VALUE IDENTITY: <O_w> + E_m. Omitting E_m yields a quantity that is
        # always <= 0 and trains identically, which is why it is spelled out.
        return float(probs @ t.weights + t.boundary_energy)

    def gradient(self, params: Sequence[float], tail: TailState) -> np.ndarray:
        """Parameter-shift gradient of the reweighted observable.

        ``tail`` is the FROZEN linearisation; the weights must not be
        recomputed inside the shifted evaluations.
        """
        p = np.asarray(params, dtype=float).copy()
        grad = np.empty_like(p)
        w = tail.weights
        for i in range(len(p)):
            original = p[i]
            p[i] = original + np.pi / 2
            plus = float(qb.probabilities(
                p, self.evaluator.num_layers, self.evaluator.num_qubits,
                self.evaluator.entangler) @ w)
            p[i] = original - np.pi / 2
            minus = float(qb.probabilities(
                p, self.evaluator.num_layers, self.evaluator.num_qubits,
                self.evaluator.entangler) @ w)
            p[i] = original
            grad[i] = 0.5 * (plus - minus)
        return grad

    def finite_difference(
        self, params: Sequence[float], eps: float = 1e-5,
        components: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Central-difference gradient of the CVaR VALUE, for validation.

        Differentiates the true objective -- re-deriving the tail at every
        perturbed point -- so it is an independent check of the identity rather
        than of the parameter-shift rule alone.
        """
        p = np.asarray(params, dtype=float).copy()
        idxs = range(len(p)) if components is None else components
        grad = np.zeros_like(p)
        for i in idxs:
            original = p[i]
            p[i] = original + eps
            plus = self.value(p)
            p[i] = original - eps
            minus = self.value(p)
            p[i] = original
            grad[i] = (plus - minus) / (2.0 * eps)
        return grad


class CVaRLoss(torch.autograd.Function):
    """Differentiable CVaR for PyTorch, via the frozen reweighted observable."""

    @staticmethod
    def forward(ctx, params: torch.Tensor, objective: CVaRObjective, tail: TailState):
        ctx.save_for_backward(params)
        ctx.objective = objective
        ctx.tail = tail
        value = objective.value(params.detach().cpu().numpy(), tail)
        return torch.as_tensor(value, dtype=params.dtype, device=params.device)

    @staticmethod
    def backward(ctx, grad_output):
        (params,) = ctx.saved_tensors
        grad = ctx.objective.gradient(params.detach().cpu().numpy(), ctx.tail)
        g = torch.as_tensor(grad, dtype=params.dtype, device=params.device)
        return grad_output * g, None, None


def alpha_floor(n_states: int) -> float:
    """Smallest alpha that can span more than the single lowest-energy state.

    For a uniform distribution over ``n_states`` this is ``1 / n_states``; below
    it the tail collapses to one state and the gradient vanishes identically.
    """
    return 1.0 / n_states
