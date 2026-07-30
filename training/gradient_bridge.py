"""
training/gradient_bridge.py -- Phase 4: PyTorch <-> Q# autograd bridge.

Makes the Q# circuit differentiable inside PyTorch, so a standard optimiser can
train the ansatz without knowing anything about quantum mechanics.

====================================================================
THE PARAMETER-SHIFT RULE
====================================================================
For a gate ``U(theta) = exp(-i * theta * P / 2)`` with ``P`` a Pauli operator
(so ``P`` has eigenvalues +/-1), the derivative of any observable expectation
is given EXACTLY by

    d<H>/d(theta_i) = [ <H(theta_i + pi/2)> - <H(theta_i - pi/2)> ] / 2

This is not a finite-difference approximation. It is an algebraic identity:
because ``P^2 = I``, the expectation is a pure sinusoid in each parameter,
``<H>(theta) = A*cos(theta + phi) + C``, and evaluating that sinusoid at two
points a quarter period apart recovers its derivative exactly. There is no step
size and no truncation error -- the shift is a full pi/2, not an infinitesimal.

Two preconditions, both satisfied here:

* Every gate is ``Ry`` or ``Rz``, and Q# defines both as ``exp(-i*theta*P/2)``.
* Each of the 28 parameters drives exactly ONE gate. If a parameter were shared
  across gates the rule would need a sum over occurrences; ``circuit.qs``
  assigns each angle to a single rotation, so the plain two-term rule applies.

====================================================================
WHY THE EXACT BACKEND, NOT SAMPLING
====================================================================
Established in Phase 1: at 1024 shots the statistical error on a single
expectation is ~0.02, so a parameter-shift difference of two such estimates
carries error ~0.04 -- roughly 4x the 1e-2 gradient-validation tolerance, and
sampling error falls only as 1/sqrt(N). Reaching 1e-2 reliably would need
~10^5 shots per evaluation, i.e. ~5.6 million shots per gradient step.

The exact state-vector backend has no such error, is deterministic, and is
about 350x faster. It is used for all training. Sampling remains in use for the
Phase 6 measurement readout, where drawing shots is the physically honest
operation. See ``docs/mathematical_foundations.md``.

====================================================================
COST
====================================================================
One backward pass costs ``2 * n_params = 56`` circuit evaluations, versus 1 for
a forward pass. At ~0.5 ms per exact evaluation that is ~30 ms per gradient
step, so a 100-epoch run is a few seconds.

Usage::

    python training/gradient_bridge.py        # run the Phase 4 validation gate
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum import q_bridge as qb  # noqa: E402

logger = logging.getLogger("gradient_bridge")

#: The parameter-shift displacement. Exact for Pauli-generated rotations.
SHIFT = math.pi / 2


# --------------------------------------------------------------------------
# Energy evaluator
# --------------------------------------------------------------------------

@dataclass
class EnergyEvaluator:
    """Evaluates ``<H>`` and its parameter-shift gradient for a fixed Hamiltonian.

    Holds the Ising coefficients and circuit depth so the autograd layer only
    has to pass parameters around, and counts circuit evaluations so the cost
    of training is measurable rather than assumed.

    Attributes:
        h: Linear Ising coefficients, shape (n_qubits,).
        J: Quadratic coefficients in ``upper_pairs`` order.
        num_layers: Ansatz depth.
        backend: ``"exact"`` or ``"sampling"``.
        num_shots: Shots per evaluation when ``backend == "sampling"``.
    """

    h: np.ndarray
    J: np.ndarray
    num_layers: int = qb.DEFAULT_LAYERS
    backend: str = "exact"
    num_shots: int = qb.DEFAULT_SHOTS
    n_evaluations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.h = np.asarray(self.h, dtype=float)
        self.J = np.asarray(self.J, dtype=float)
        if self.backend not in ("exact", "sampling"):
            raise ValueError(f"backend must be 'exact' or 'sampling', got {self.backend!r}")
        # Precompute the diagonal energy table; it does not depend on the
        # parameters, so it is built once instead of per evaluation.
        self._energies = qb.ising_energy_table(self.h, self.J, qb.NUM_QUBITS)

    @property
    def n_params(self) -> int:
        """Number of ansatz angles this evaluator expects."""
        return qb.n_params(qb.NUM_QUBITS, self.num_layers)

    def energy(self, params: Sequence[float]) -> float:
        """Evaluate ``<H>`` at ``params``."""
        self.n_evaluations += 1
        if self.backend == "exact":
            probs = qb.probabilities(params, self.num_layers)
            return float(probs @ self._energies)
        return qb.expectation_sampled(
            params, self.h, self.J, self.num_layers, self.num_shots
        )

    def parameter_shift_gradient(self, params: Sequence[float]) -> np.ndarray:
        """Exact gradient of ``<H>`` via the parameter-shift rule.

        Costs ``2 * n_params`` circuit evaluations.

        Returns:
            Array of shape ``(n_params,)``.
        """
        p = np.asarray(params, dtype=float).copy()
        grad = np.empty_like(p)
        for i in range(len(p)):
            original = p[i]
            p[i] = original + SHIFT
            plus = self.energy(p)
            p[i] = original - SHIFT
            minus = self.energy(p)
            p[i] = original
            grad[i] = 0.5 * (plus - minus)
        return grad

    def finite_difference_gradient(
        self, params: Sequence[float], eps: float = 1e-5
    ) -> np.ndarray:
        """Central-difference gradient, for validating the parameter-shift rule.

        ``eps = 1e-5`` sits near the optimum for a float64 central difference:
        truncation error grows as ``eps^2`` while round-off grows as
        ``1/eps``, and both land around 1e-10 here. This is a VALIDATION tool
        only -- training never uses it.
        """
        p = np.asarray(params, dtype=float).copy()
        grad = np.empty_like(p)
        for i in range(len(p)):
            original = p[i]
            p[i] = original + eps
            plus = self.energy(p)
            p[i] = original - eps
            minus = self.energy(p)
            p[i] = original
            grad[i] = (plus - minus) / (2.0 * eps)
        return grad


# --------------------------------------------------------------------------
# autograd integration
# --------------------------------------------------------------------------

class QuantumCircuitFunction(torch.autograd.Function):
    """Differentiable ``<H>`` evaluation backed by the Q# simulator.

    Forward runs the circuit once; backward applies the parameter-shift rule.
    The evaluator travels on ``ctx`` rather than as a tensor because it is not
    differentiable state.
    """

    @staticmethod
    def forward(ctx, params: torch.Tensor, evaluator: EnergyEvaluator) -> torch.Tensor:
        """Compute ``<H>(params)`` as a 0-dim tensor.

        Args:
            params: 1-D tensor of rotation angles.
            evaluator: Supplies the Hamiltonian and circuit configuration.
        """
        if params.dim() != 1:
            raise ValueError(f"params must be 1-D, got shape {tuple(params.shape)}")
        if params.numel() != evaluator.n_params:
            raise ValueError(
                f"Expected {evaluator.n_params} parameters for "
                f"{evaluator.num_layers} layers, got {params.numel()}."
            )

        ctx.save_for_backward(params)
        ctx.evaluator = evaluator

        value = evaluator.energy(params.detach().cpu().numpy())
        return torch.as_tensor(value, dtype=params.dtype, device=params.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Chain ``grad_output`` through the parameter-shift gradient."""
        (params,) = ctx.saved_tensors
        evaluator: EnergyEvaluator = ctx.evaluator

        grad = evaluator.parameter_shift_gradient(params.detach().cpu().numpy())
        grad_t = torch.as_tensor(grad, dtype=params.dtype, device=params.device)

        # `evaluator` is a non-tensor argument, so its gradient slot is None.
        return grad_output * grad_t, None


def quantum_expectation(params: torch.Tensor, evaluator: EnergyEvaluator) -> torch.Tensor:
    """Functional wrapper around :class:`QuantumCircuitFunction`."""
    return QuantumCircuitFunction.apply(params, evaluator)


class QuantumPortfolioLayer(nn.Module):
    """``nn.Module`` holding the ansatz angles as a trainable parameter.

    ``forward()`` returns ``<H>`` for the current angles, so the layer plugs
    into an ordinary PyTorch training loop.

    Args:
        evaluator: The Hamiltonian and circuit configuration to optimise.
        init_params: Explicit initial angles; random if omitted.
        seed: Seed for the random initialisation.
        init_scale: Half-width of the uniform initialisation window.
            Small angles start the state near ``|0...0>`` with a well-defined
            gradient, and avoid the flat, high-symmetry region that large
            random angles can land in.
        dtype: ``float64`` by default -- the circuit is evaluated in double
            precision, and matching it keeps gradient checks meaningful.
    """

    def __init__(
        self,
        evaluator: EnergyEvaluator,
        init_params: Sequence[float] | None = None,
        seed: int | None = None,
        init_scale: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.evaluator = evaluator

        if init_params is not None:
            values = np.asarray(init_params, dtype=float)
            if values.shape != (evaluator.n_params,):
                raise ValueError(
                    f"init_params must have shape ({evaluator.n_params},), "
                    f"got {values.shape}."
                )
        else:
            rng = np.random.default_rng(seed)
            values = rng.uniform(-init_scale, init_scale, evaluator.n_params)

        self.theta = nn.Parameter(torch.tensor(values, dtype=dtype))

    def forward(self) -> torch.Tensor:
        """Return ``<H>`` for the current parameters as a 0-dim tensor."""
        return quantum_expectation(self.theta, self.evaluator)

    @torch.no_grad()
    def probabilities(self) -> np.ndarray:
        """Exact basis-state distribution for the current parameters."""
        return qb.probabilities(self.theta.detach().cpu().numpy(), self.evaluator.num_layers)

    def extra_repr(self) -> str:
        return (
            f"n_params={self.evaluator.n_params}, "
            f"layers={self.evaluator.num_layers}, "
            f"backend={self.evaluator.backend}"
        )


# --------------------------------------------------------------------------
# Phase 4 validation gate
# --------------------------------------------------------------------------

def _load_hamiltonian_or_synthetic():
    """Use the real Phase 2 Hamiltonian if present, else a synthetic stand-in.

    Validating the gradient rule does not require real market data -- but if
    the real Hamiltonian is available, testing against it is strictly better,
    because its coefficient scale is what training will actually see.
    """
    try:
        from classical.hamiltonian import load_hamiltonian
        ham = load_hamiltonian()
        logger.info("Using the Phase 2 Hamiltonian from data/hamiltonian.json.")
        return ham.h, ham.J, ham.tickers
    except Exception as exc:
        logger.warning("Falling back to a synthetic Hamiltonian (%s).", exc)
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([0.1, -0.05, 0.02, 0.08, -0.03, 0.06])
        return h, J, ("A", "B", "C", "D")


def run_gradient_validation(
    num_layers: int = qb.DEFAULT_LAYERS,
    n_random_points: int = 3,
    tolerance: float = 1e-2,
    seed: int = 0,
) -> bool:
    """Phase 4 gate: parameter-shift gradients must match finite differences.

    The brief sets the bar at a max absolute difference of ~1e-2, sized for
    simulator/shot noise. With the exact backend there is no shot noise, so the
    achieved agreement should be many orders of magnitude tighter; the loose
    bound is reported alongside the actual figure.

    Args:
        num_layers: Ansatz depth to test.
        n_random_points: How many random parameter vectors to test.
        tolerance: The gate threshold on max |parameter-shift - finite-diff|.
        seed: Seed for the random test points.

    Returns:
        True if every check passes.
    """
    passed = True

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        logger.info("  [%s] %s: %s", "PASS" if ok else "FAIL", name, detail)
        if not ok:
            passed = False

    logger.info("=" * 70)
    logger.info("PHASE 4 VALIDATION GATE -- PyTorch/Q# gradient bridge")
    logger.info("=" * 70)

    h, J, tickers = _load_hamiltonian_or_synthetic()
    evaluator = EnergyEvaluator(h, J, num_layers=num_layers, backend="exact")
    n = evaluator.n_params

    logger.info("Assets            : %s", ", ".join(tickers))
    logger.info("Circuit           : %d qubits, %d layers, %d parameters",
                qb.NUM_QUBITS, num_layers, n)
    logger.info("Backend           : %s", evaluator.backend)
    logger.info("h                 : [%s]", ", ".join(f"{v:+.6f}" for v in evaluator.h))
    logger.info("J                 : [%s]", ", ".join(f"{v:+.6f}" for v in evaluator.J))
    logger.info("Gate tolerance    : %.1e (max abs difference)", tolerance)

    # ------------------------------------------------ 1. shapes and dtypes
    logger.info("\n[1] Tensor shapes and dtypes")
    layer = QuantumPortfolioLayer(evaluator, seed=seed)
    energy = layer()
    check("forward shape", energy.shape == torch.Size([]),
          f"<H> is a {tuple(energy.shape)} tensor, dtype={energy.dtype}")
    check("parameter shape", layer.theta.shape == torch.Size([n]),
          f"theta shape {tuple(layer.theta.shape)}, dtype={layer.theta.dtype}")

    energy.backward()
    check("gradient shape", layer.theta.grad is not None
          and layer.theta.grad.shape == torch.Size([n]),
          f"theta.grad shape {tuple(layer.theta.grad.shape)}")

    # ------------------------------ 2. parameter-shift vs finite difference
    logger.info("\n[2] Parameter-shift vs central finite difference, all %d parameters", n)
    rng = np.random.default_rng(seed)

    test_points: list[tuple[str, np.ndarray]] = [
        ("zeros", np.zeros(n)),
        ("small uniform(-0.1, 0.1)", rng.uniform(-0.1, 0.1, n)),
    ]
    for k in range(n_random_points):
        test_points.append((f"random uniform(-pi, pi) #{k + 1}", rng.uniform(-math.pi, math.pi, n)))

    worst_overall = 0.0
    worst_rel_overall = 0.0
    for label, point in test_points:
        ps = evaluator.parameter_shift_gradient(point)
        fd = evaluator.finite_difference_gradient(point, eps=1e-5)

        abs_diff = np.abs(ps - fd)
        max_abs = float(abs_diff.max())
        argmax = int(abs_diff.argmax())
        scale = max(float(np.abs(ps).max()), 1e-12)
        max_rel = max_abs / scale

        worst_overall = max(worst_overall, max_abs)
        worst_rel_overall = max(worst_rel_overall, max_rel)

        check(
            f"gradient agreement [{label}]",
            max_abs <= tolerance,
            f"max|PS - FD| = {max_abs:.3e} at param {argmax} "
            f"(rel {max_rel:.2e}); |grad|_inf = {scale:.4f}, "
            f"|grad|_2 = {float(np.linalg.norm(ps)):.4f}",
        )

    logger.info("    worst absolute difference across all test points: %.3e", worst_overall)
    logger.info("    worst relative difference across all test points: %.3e", worst_rel_overall)
    check("overall agreement", worst_overall <= tolerance,
          f"{worst_overall:.3e} <= {tolerance:.1e}")

    # ---------------------------------- 3. per-parameter detail at one point
    logger.info("\n[3] Per-parameter detail (random point #1)")
    point = test_points[2][1]
    ps = evaluator.parameter_shift_gradient(point)
    fd = evaluator.finite_difference_gradient(point, eps=1e-5)
    logger.info("    %-5s %-8s %14s %14s %12s", "idx", "role", "param-shift", "finite-diff", "abs diff")
    for i in range(n):
        layer_idx, role = _param_role(i, qb.NUM_QUBITS, num_layers)
        logger.info("    %-5d %-8s %+14.9f %+14.9f %12.3e",
                    i, f"L{layer_idx}{role}", ps[i], fd[i], abs(ps[i] - fd[i]))

    # ------------------------------ 4. reproducibility floor of the backend
    # The simulator applies an arbitrary overall phase to the returned state
    # (documented in q_bridge). Probabilities are mathematically identical, but
    # they are computed from differently-phased amplitudes, so the final float64
    # rounding differs run to run. That makes repeated evaluations equal to
    # within a few ULP rather than bit-identical. Measured here so the figure is
    # evidence rather than assumption -- and so a REAL regression (a genuinely
    # noisy backend) would show up as this number growing by orders of magnitude.
    logger.info("\n[4] Reproducibility floor (global-phase rounding)")
    repeat_point = test_points[2][1]
    energies = np.array([evaluator.energy(repeat_point) for _ in range(50)])
    grads = np.array([evaluator.parameter_shift_gradient(repeat_point) for _ in range(10)])
    energy_spread = float(np.ptp(energies))
    grad_spread = float(np.ptp(grads, axis=0).max())
    logger.info("    <H> spread over 50 evaluations      : %.3e (%d distinct values)",
                energy_spread, len(set(energies.tolist())))
    logger.info("    gradient spread over 10 evaluations : %.3e", grad_spread)
    check("reproducible to float64 rounding", grad_spread < 1e-12,
          f"max component spread = {grad_spread:.3e}, i.e. ~{grad_spread / 2.22e-16:.0f} ULP "
          "-- round-off, not sampling noise")

    # ------------------------------------------ 5. PyTorch's own gradcheck
    # An independent implementation of the same comparison, written by the
    # PyTorch authors -- it exercises the autograd plumbing (dtypes, chaining,
    # the None slot for the evaluator) rather than just the numerics.
    #
    # nondet_tol is REQUIRED here and is not a way of loosening the accuracy
    # bar. gradcheck defaults to nondet_tol=0.0, demanding bit-identical
    # repeated backward passes; it explicitly reported "analytical gradient
    # matches numerical gradient" while failing only its reentrancy test. The
    # tolerance below sits ~3 orders of magnitude above the measured rounding
    # floor and ~6 below the 1e-6 accuracy tolerances, so gradient ERRORS are
    # still caught.
    logger.info("\n[5] torch.autograd.gradcheck")
    try:
        theta = torch.tensor(rng.uniform(-math.pi, math.pi, n),
                             dtype=torch.float64, requires_grad=True)
        ok = torch.autograd.gradcheck(
            lambda t: QuantumCircuitFunction.apply(t, evaluator),
            (theta,), eps=1e-6, atol=1e-6, rtol=1e-4,
            nondet_tol=1e-12, raise_exception=False,
        )
        check("gradcheck", bool(ok),
              "numerical Jacobian check (nondet_tol=1e-12 for the rounding floor above)")
    except Exception as exc:
        logger.exception("gradcheck raised.")
        check("gradcheck", False, repr(exc))

    # ------------------------------------------------ 5. exactness of the rule
    # The parameter-shift rule should be EXACT, not merely close. Richardson
    # extrapolation of the central difference converges to the true derivative
    # much faster than a single step, so agreement at ~1e-11 is strong evidence
    # the rule is exact rather than coincidentally near.
    logger.info("\n[6] Exactness check via Richardson extrapolation")
    point = test_points[3][1]
    ps = evaluator.parameter_shift_gradient(point)
    fd_h = evaluator.finite_difference_gradient(point, eps=1e-4)
    fd_h2 = evaluator.finite_difference_gradient(point, eps=5e-5)
    richardson = (4.0 * fd_h2 - fd_h) / 3.0
    worst_rich = float(np.abs(ps - richardson).max())
    check("parameter-shift is exact", worst_rich < 1e-8,
          f"max|PS - Richardson| = {worst_rich:.3e}")

    # --------------------------------------- 6. analytic sinusoid property
    # <H> must be a pure sinusoid of period 2*pi in every single parameter.
    # This is the actual premise the parameter-shift rule rests on, so testing
    # it directly checks the assumption rather than only its consequence.
    logger.info("\n[7] Single-parameter sinusoidal dependence")
    base = test_points[2][1].copy()
    worst_fit = 0.0
    for i in (0, n // 2, n - 1):
        thetas = np.linspace(-math.pi, math.pi, 25)
        values = []
        p = base.copy()
        for t in thetas:
            p[i] = t
            values.append(evaluator.energy(p))
        values = np.array(values)
        # Least-squares fit of A*cos(t) + B*sin(t) + C.
        design = np.stack([np.cos(thetas), np.sin(thetas), np.ones_like(thetas)], axis=1)
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = float(np.abs(design @ coeffs - values).max())
        worst_fit = max(worst_fit, residual)
        logger.info("    param %2d: A=%+.6f B=%+.6f C=%+.6f  max residual=%.3e",
                    i, *coeffs, residual)
    check("pure sinusoid in each parameter", worst_fit < 1e-9,
          f"max fit residual = {worst_fit:.3e}")

    # -------------------------------------- 7. gradient descent actually works
    # A gradient that validates numerically but points the wrong way would
    # still be useless. Ten plain descent steps must reduce the energy.
    logger.info("\n[8] Descent sanity check (10 steps, lr=0.05)")
    p = test_points[2][1].copy()
    e_start = evaluator.energy(p)
    for _ in range(10):
        p = p - 0.05 * evaluator.parameter_shift_gradient(p)
    e_end = evaluator.energy(p)
    check("energy decreases", e_end < e_start,
          f"<H>: {e_start:.6f} -> {e_end:.6f} (change {e_end - e_start:+.6f})")

    # --------------------------------------------- 8. chain rule through loss
    # Training minimises a function OF <H>, so the chain rule must hold through
    # a non-trivial downstream op, not just the bare expectation.
    logger.info("\n[9] Chain rule through a downstream operation")
    theta = torch.tensor(test_points[1][1], dtype=torch.float64, requires_grad=True)
    loss = quantum_expectation(theta, evaluator) ** 2
    loss.backward()
    analytic = 2.0 * evaluator.energy(test_points[1][1]) * \
        evaluator.parameter_shift_gradient(test_points[1][1])
    worst_chain = float(np.abs(theta.grad.numpy() - analytic).max())
    check("d(<H>^2)/dtheta", worst_chain < 1e-9,
          f"max deviation from 2*<H>*d<H> = {worst_chain:.3e}")

    logger.info("\nCircuit evaluations consumed by this gate: %d", evaluator.n_evaluations)
    logger.info("Backward-pass cost: %d evaluations (2 x %d parameters)", 2 * n, n)

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4 VALIDATION: %s", "ALL TESTS PASSED" if passed else "FAILURES PRESENT")
    logger.info("=" * 70)
    return passed


def _param_role(index: int, n_qubits: int, num_layers: int) -> tuple[int, str]:
    """Map a flat parameter index to ``(layer, role)`` for readable logging."""
    per_layer = 2 * n_qubits
    if index >= per_layer * num_layers:
        return num_layers, f"Ry q{index - per_layer * num_layers}"
    layer, within = divmod(index, per_layer)
    gate = "Ry" if within < n_qubits else "Rz"
    return layer, f"{gate} q{within % n_qubits}"


if __name__ == "__main__":
    qb.configure_logging("gradient_bridge.log")
    sys.exit(0 if run_gradient_validation() else 1)
