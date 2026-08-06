"""
quantum/q_bridge.py -- Python <-> Q# interoperability bridge.

Single point of contact between Python and the Q# runtime. Everything that
touches the quantum simulator goes through here; no other module imports
``qsharp`` directly.

INSTALLED API (verified, not assumed)
-------------------------------------
Probed on this machine rather than recalled from memory, as the package
interface has changed across releases::

    qsharp 1.30.0 / qdk 1.30.0

    qsharp.init(*, target_profile=..., project_root=..., ...) -> Config
    qsharp.eval(source: str, *, save_events=False) -> Any
    qsharp.run(entry_expr, shots, *args, seed=None, ...) -> List[Any]
    qsharp.dump_machine() -> StateDump      # .as_dense_state(), .qubit_count
    qsharp.set_quantum_seed(seed) / set_classical_seed(seed)

Three findings that shape this module:

1. ``import qsharp`` emits a DeprecationWarning directing users to ``qdk``.
   Both packages are installed (they depend on each other) but are DISTINCT
   module objects. We import ``from qdk import qsharp`` to stay on the
   supported path.

2. Neither package exposes ``__version__``. The previous implementation read
   ``mod.__version__`` and logged "unknown"; versions are resolved here with
   ``importlib.metadata.version`` instead.

3. ``StateDump`` exposes ``as_dense_state()`` (a length-2^n complex ndarray),
   NOT ``as_dict()``.

TWO EVALUATION BACKENDS
-----------------------
``exact``    Reads the full state vector via ``dump_machine()`` and contracts
             it against the diagonal Hamiltonian. Deterministic to ~1e-15 and
             ~350x faster than sampling (measured: 48 evaluations in 0.025 s
             vs 8.9 s). This is the default for training, because
             parameter-shift gradients at 1024 shots carry a statistical error
             of roughly 0.04 -- four times the 1e-2 Phase 4 validation
             tolerance, which no amount of tuning would fix.

``sampling`` Draws ``num_shots`` computational-basis measurements, exactly as
             hardware would. Used for the honest measurement-based readout in
             Phase 6 and as an independent cross-check on ``exact``.

Both are exposed; the choice is explicit at every call site.

A NOTE ON GLOBAL PHASE
----------------------
Repeated identical ``ApplyAnsatz`` calls return state vectors that differ by an
overall phase factor (the simulator does not canonicalise it). Verified that
the discrepancy is purely global: amplitude ratios between runs have modulus
spread ~1e-15 and argument spread ~1e-15. Probabilities agree to 1e-16 and
<H> to 4e-15. Global phase is physically unobservable, so this is harmless --
but any future code comparing raw amplitudes must account for it.

Usage::

    python quantum/q_bridge.py                  # Phase 1 bridge validation
    python quantum/q_bridge.py --test-circuit   # Phase 3 circuit validation
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Paths & logging
# --------------------------------------------------------------------------

QUANTUM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QUANTUM_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("q_bridge")


def configure_logging(logfile: str = "q_bridge.log", level: int = logging.INFO) -> None:
    """Attach stdout + file handlers to the root logger (idempotent).

    Called from ``__main__`` and by other phase scripts. Safe to call twice:
    existing handlers are cleared first so repeated invocations do not
    duplicate every log line.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    # The Windows console is cp1252 by default; Unicode in log text would raise
    # UnicodeEncodeError. Force UTF-8 on the stream where the platform allows.
    try:
        stream.stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    file_handler = logging.FileHandler(LOG_DIR / logfile, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)

    root.addHandler(stream)
    root.addHandler(file_handler)
    root.setLevel(level)


# --------------------------------------------------------------------------
# Circuit constants
# --------------------------------------------------------------------------

NUM_QUBITS = 4
DEFAULT_LAYERS = 3
DEFAULT_SHOTS = 4096

#: Qubit index -> asset. Mirrors the header of circuit.qs.
QUBIT_ASSETS = ("TSLA", "NVDA", "AMD", "COIN")

#: Upper-triangular (i, j) pairs in the row-major order that circuit.qs expects
#: for ``jCoeffs``: [J01, J02, J03, J12, J13, J23].
J_PAIRS = tuple(
    (i, j) for i in range(NUM_QUBITS - 1) for j in range(i + 1, NUM_QUBITS)
)


def n_params(num_qubits: int = NUM_QUBITS, num_layers: int = DEFAULT_LAYERS) -> int:
    """Number of ansatz angles: ``num_qubits * (2 * num_layers + 1)``.

    The ``+1`` is the trailing Ry layer. Mirrors ``AnsatzParameterCount`` in
    circuit.qs -- see that file for why the circuit ends on Ry and not Rz.
    """
    return num_qubits * (2 * num_layers + 1)


# --------------------------------------------------------------------------
# Runtime initialisation
# --------------------------------------------------------------------------

_qsharp = None          # the qdk.qsharp module, once imported
_initialised = False    # interpreter started
_sources_loaded: set[str] = set()
_scratch_allocated = False
#: Size of the currently allocated scratch register. Only one may be live at a
#: time -- see _ensure_scratch_register for why.
_scratch_size: int = 0

# --- interpreter recycling -------------------------------------------------
# Every qsharp.eval() of a NEW source string is compiled into the interpreter
# session, and the session retains it. Because each circuit evaluation
# interpolates fresh float literals, every call is a new string, so session
# state grows without bound and evaluation cost grows LINEARLY with the number
# of prior calls. Measured on this machine:
#
#     after   200 evals ->  3.6 ms/eval
#     after 1_000 evals -> 15.8 ms/eval
#     after 2_400 evals -> 39.9 ms/eval
#
# Left alone this makes training cost quadratic in the number of evaluations
# (a 100-epoch run needs ~5_700), so a few-second job would take minutes and
# keep degrading. Passing parameters natively would avoid it, but the Q#
# callables exposed by qsharp.eval("Name") are `GlobalCallable` objects that
# Python cannot invoke, and a qubit register cannot round-trip through Python.
#
# So the interpreter is periodically torn down and rebuilt. A rebuild costs
# ~100 ms (init + reloading circuit.qs), which amortises to well under 1 ms per
# evaluation at this interval and pins the steady-state cost near the 3.6 ms
# floor instead of letting it grow.
MAX_EVALS_BEFORE_RECYCLE = 250

_evals_since_recycle = 0
_recycle_count = 0
_last_seed: int | None = None

# --- thread affinity -------------------------------------------------------
# The Q# interpreter is a pyo3 object marked `unsendable`: it may only be
# touched from the thread that constructed it. Using it from another thread
# does not raise a Python exception -- it triggers a RUST PANIC:
#
#     pyo3_runtime.PanicException: qdk::interpreter::Interpreter is
#     unsendable, but sent to another thread
#
# which kills the calling thread outright. Streamlit runs every script rerun
# on a NEW ScriptRunner thread, so an app that survives its first interaction
# can panic on the second. The failure is thread-scheduling dependent, and
# therefore intermittent, which makes it far worse than a deterministic crash.
#
# Every call into the interpreter is therefore funnelled onto ONE dedicated
# worker thread that owns it for the lifetime of the process. Callers may live
# on any thread. The handoff costs tens of microseconds against a ~2.2 ms
# evaluation, so the overhead is immaterial even across a full training run.
_QSHARP_THREAD_PREFIX = "qsharp-owner"
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.RLock()
#: OS identity of the owning thread, captured from inside it at creation.
_owner_thread_id: int | None = None


def _get_executor() -> ThreadPoolExecutor:
    """The single-threaded executor that owns the Q# interpreter."""
    global _executor, _owner_thread_id
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=_QSHARP_THREAD_PREFIX
            )
            # Ask the worker for its own identity, so the fast-path predicate
            # compares against a value observed from inside that thread rather
            # than one inferred from a name.
            _owner_thread_id = _executor.submit(threading.get_ident).result()
    return _executor


def owner_thread_id() -> int | None:
    """Identity of the thread that owns the interpreter (None before first use)."""
    return _owner_thread_id


def _on_qsharp_thread(fn, *args, **kwargs):
    """Execute ``fn`` on the interpreter's owning thread and return its result.

    THE PREDICATE IS THREAD IDENTITY, NEVER WORKER AVAILABILITY.

    The inline branch is taken if and only if the CALLER IS ALREADY THE OWNER
    THREAD. It exists solely so nested calls -- ``probabilities`` ->
    ``statevector``, or ``_maybe_recycle`` -> ``reset_runtime`` -- do not
    deadlock waiting on a single-worker pool that the caller itself occupies.

    It must never become "the worker is busy, so run it here": that would
    execute interpreter code on a foreign thread and reintroduce the pyo3
    panic, and no amount of concurrent testing would reveal it, because the
    result would still be numerically correct right up until it panics.
    Contention is handled by BLOCKING on the future, which is what serialises
    concurrent callers (see ``docs/architecture.md`` for the throughput
    consequence).
    """
    if _owner_thread_id is not None and threading.get_ident() == _owner_thread_id:
        return fn(*args, **kwargs)
    return _get_executor().submit(fn, *args, **kwargs).result()


def _import_qsharp():
    """Import the Q# runtime, preferring ``qdk`` over the deprecated shim."""
    global _qsharp
    if _qsharp is not None:
        return _qsharp

    try:
        from qdk import qsharp as _mod  # supported path
        _qsharp = _mod
        logger.info("Q# runtime loaded via 'qdk' package (supported path).")
    except ImportError:
        try:
            import qsharp as _mod  # deprecated compatibility shim
            _qsharp = _mod
            logger.warning(
                "Loaded the deprecated 'qsharp' package; 'qdk' is preferred."
            )
        except ImportError as exc:
            raise ImportError(
                "Q# runtime unavailable. Install it with:  pip install qdk"
            ) from exc
    return _qsharp


def get_qsharp():
    """Return the initialised Q# module, starting the interpreter on first use.

    Always dispatched to the owning thread so the interpreter is CONSTRUCTED
    there -- creating it on a caller's thread is what binds it to the wrong one.
    """
    return _on_qsharp_thread(_get_qsharp_impl)


def _get_qsharp_impl():
    """Body of :func:`get_qsharp`; runs only on the interpreter's thread."""
    global _initialised
    mod = _import_qsharp()
    if not _initialised:
        mod.init()
        _initialised = True
        logger.info("Q# interpreter initialised on thread %s.",
                    threading.current_thread().name)
    return mod


def set_seed(seed: int | None) -> None:
    """Seed the simulator's quantum and classical RNGs.

    Affects the sampling backend only -- the exact backend is deterministic by
    construction. Used by the Phase 7 stability runs.
    """
    _on_qsharp_thread(_set_seed_impl, seed)


def _set_seed_impl(seed: int | None) -> None:
    """Body of :func:`set_seed`; runs only on the interpreter's thread."""
    global _last_seed
    qs = get_qsharp()
    qs.set_quantum_seed(seed)
    qs.set_classical_seed(seed)
    _last_seed = seed
    logger.info("Simulator seeded with %s.", seed)


def reset_runtime() -> None:
    """Tear down interpreter state so a fresh run starts clean.

    Q# session state (loaded sources, the scratch register) lives in the
    interpreter, so a re-``init()`` must invalidate our bookkeeping too;
    otherwise ``_sources_loaded`` would suppress reloads the new interpreter
    never received. Needed by the multi-seed stability harness.
    """
    _on_qsharp_thread(_reset_runtime_impl)


def _reset_runtime_impl() -> None:
    """Body of :func:`reset_runtime`; runs only on the interpreter's thread."""
    global _initialised, _sources_loaded, _scratch_allocated, _evals_since_recycle
    global _scratch_size
    mod = _import_qsharp()
    mod.init()
    _initialised = True
    _sources_loaded = set()
    _scratch_allocated = False
    _scratch_size = 0
    _evals_since_recycle = 0
    logger.debug("Q# interpreter reset.")


def eval_qs(source: str):
    """Evaluate Q# source on the interpreter's owning thread.

    The one supported way for other modules to run arbitrary Q#. Calling
    ``get_qsharp().eval(...)`` directly from an arbitrary thread panics the
    Rust runtime; see the thread-affinity note above.
    """
    return _on_qsharp_thread(lambda: _get_qsharp_impl().eval(source))


def _release_interpreter() -> None:
    """Drop every reference to the interpreter, on its OWNING thread.

    The unsendable rule applies to destruction as well as use: if the objects
    survive until CPython's final GC they are freed on the main thread and the
    runtime reports

        RuntimeError: qdk::interpreter::Interpreter is unsendable,
        but is being dropped on another thread

    Clearing ``qdk._interpreter._default_context`` here -- from inside the
    owning thread -- releases the references this module is responsible for.

    KNOWN RESIDUAL, harmless: qdk also retains ``GlobalCallable`` objects for
    each evaluated Q# operation in internal state this module cannot reach, so
    a handful of those messages may still print during CPython finalisation
    (one per loaded operation, plus one for the interpreter). They are emitted
    AFTER all work has completed, are not raised into Python, and affect no
    result. Suppressing them would require a change inside qdk itself.
    """
    global _qsharp, _initialised, _sources_loaded, _scratch_allocated, _scratch_size
    try:
        import qdk._interpreter as _qdk_interp
        _qdk_interp._default_context = None
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
    _qsharp = None
    _initialised = False
    _sources_loaded = set()
    _scratch_allocated = False
    _scratch_size = 0
    import gc
    gc.collect()


@atexit.register
def _shutdown_qsharp_thread() -> None:
    """Release the interpreter and stop its thread before the process exits."""
    global _executor, _owner_thread_id
    with _executor_lock:
        executor, _executor = _executor, None
        _owner_thread_id = None
    if executor is None:
        return
    try:
        executor.submit(_release_interpreter).result(timeout=10)
    except Exception:  # pragma: no cover - never block shutdown
        pass
    executor.shutdown(wait=True)


def _maybe_recycle() -> None:
    """Rebuild the interpreter once the session has grown large.

    See ``MAX_EVALS_BEFORE_RECYCLE`` for why this is necessary.

    Recycling is invisible to the exact backend, which is deterministic. For
    the sampling backend a rebuild would restart the RNG stream, so if a seed
    was set it is re-derived as ``seed + recycle_count``: the stream stays
    reproducible for a given starting seed but does not repeat the same draws
    after every rebuild.
    """
    global _evals_since_recycle, _recycle_count
    if _evals_since_recycle < MAX_EVALS_BEFORE_RECYCLE:
        return

    _recycle_count += 1
    logger.debug(
        "Recycling Q# interpreter after %d evaluations (rebuild #%d).",
        _evals_since_recycle, _recycle_count,
    )
    seed = _last_seed
    reset_runtime()
    load_circuit_operations()
    if seed is not None:
        qs = get_qsharp()
        derived = seed + _recycle_count
        qs.set_quantum_seed(derived)
        qs.set_classical_seed(derived)


def recycle_stats() -> dict:
    """Interpreter-recycling counters, for performance logging."""
    return {
        "evals_since_recycle": _evals_since_recycle,
        "recycle_count": _recycle_count,
        "max_evals_before_recycle": MAX_EVALS_BEFORE_RECYCLE,
    }


# --------------------------------------------------------------------------
# Q# source loading
# --------------------------------------------------------------------------

def load_qs_source(filename: str) -> None:
    """Evaluate a .qs file from ``quantum/`` into the interpreter session.

    Files are declared at top level (no ``namespace`` block), so their
    operations become callable by bare name. Loading is idempotent per session.
    """
    _on_qsharp_thread(_load_qs_source_impl, filename)


def _load_qs_source_impl(filename: str) -> None:
    """Body of :func:`load_qs_source`; runs only on the interpreter's thread."""
    if filename in _sources_loaded:
        return

    path = QUANTUM_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Q# source not found: {path}")

    qs = get_qsharp()
    source = path.read_text(encoding="utf-8")
    logger.info("Loading Q# source %s (%d bytes)...", filename, len(source))
    try:
        qs.eval(source)
    except Exception:
        logger.error("Failed to evaluate %s", path)
        raise
    _sources_loaded.add(filename)
    logger.info("Loaded %s.", filename)


def load_bridge_operations() -> None:
    """Load test_bridge.qs (Phase 1 validation operations)."""
    load_qs_source("test_bridge.qs")


def load_circuit_operations() -> None:
    """Load circuit.qs (Phase 3 ansatz + measurement operations)."""
    load_qs_source("circuit.qs")


# --------------------------------------------------------------------------
# Value marshalling
# --------------------------------------------------------------------------

def _fmt_double_array(values: Iterable[float]) -> str:
    """Format a float sequence as a Q# ``Double[]`` literal.

    ``repr`` of a Python float round-trips exactly, so no precision is lost.
    Non-finite values are rejected: Q# would accept the token but the ensuing
    simulator error is far harder to trace than a failure here.
    """
    out = []
    for v in values:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite value cannot be marshalled to Q#: {v!r}")
        out.append(repr(f))
    if not out:
        raise ValueError("Cannot marshal an empty array to Q# Double[].")
    return "[" + ", ".join(out) + "]"


def _check_params(params: Sequence[float], num_layers: int,
                  num_qubits: int = NUM_QUBITS) -> list[float]:
    """Validate ansatz parameter count against register size and layer depth."""
    p = [float(x) for x in np.asarray(params, dtype=float).ravel()]
    expected = n_params(num_qubits, num_layers)
    if len(p) != expected:
        raise ValueError(
            f"Expected {expected} parameters for {num_qubits} qubits and "
            f"{num_layers} layers, got {len(p)}."
        )
    return p


# --------------------------------------------------------------------------
# Hamiltonian energy table (diagonal -> closed form)
# --------------------------------------------------------------------------

def ising_energy_table(
    h_coeffs: Sequence[float],
    j_coeffs: Sequence[float],
    num_qubits: int = NUM_QUBITS,
) -> np.ndarray:
    """Energy of every computational basis state under the Ising Hamiltonian.

    ``H`` is diagonal in the Z basis, so it is fully described by its 2^n
    eigenvalues::

        E(z) = sum_i h_i * s_i  +  sum_{i<j} J_ij * s_i * s_j

    with spin convention ``bit 0 -> s = +1``, ``bit 1 -> s = -1``.

    BIT ORDERING: qubit 0 is the most significant bit of the basis index, so
    ``bit(q) = (index >> (num_qubits - 1 - q)) & 1``. Verified empirically in
    Phase 1 (see module docstring of circuit.qs).

    Returns:
        Array of shape ``(2**num_qubits,)``; entry ``k`` is the energy of basis
        state ``k``.
    """
    h = np.asarray(h_coeffs, dtype=float)
    j = np.asarray(j_coeffs, dtype=float)
    pairs = [(i, jj) for i in range(num_qubits - 1) for jj in range(i + 1, num_qubits)]

    if h.shape != (num_qubits,):
        raise ValueError(f"h_coeffs must have shape ({num_qubits},), got {h.shape}.")
    if j.shape != (len(pairs),):
        raise ValueError(f"j_coeffs must have shape ({len(pairs)},), got {j.shape}.")

    dim = 2 ** num_qubits
    idx = np.arange(dim)
    # spins[:, q] = +1 / -1 for qubit q across all basis states.
    spins = np.stack(
        [1.0 - 2.0 * ((idx >> (num_qubits - 1 - q)) & 1) for q in range(num_qubits)],
        axis=1,
    ).astype(float)

    energies = spins @ h
    for coeff, (a, b) in zip(j, pairs):
        energies += coeff * spins[:, a] * spins[:, b]
    return energies


def bitstring_to_index(bits: Sequence[int], num_qubits: int = NUM_QUBITS) -> int:
    """Convert a qubit-ordered bit list to a state-vector index (qubit 0 = MSB)."""
    idx = 0
    for q in range(num_qubits):
        idx |= (int(bits[q]) & 1) << (num_qubits - 1 - q)
    return idx


def index_to_bitstring(index: int, num_qubits: int = NUM_QUBITS) -> list[int]:
    """Convert a state-vector index to a qubit-ordered bit list (qubit 0 = MSB)."""
    return [(index >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]


# --------------------------------------------------------------------------
# Exact backend (state vector)
# --------------------------------------------------------------------------

#: Entangler variants. "linear" is the Phase 1-9 CNOT ladder; "ring" closes it
#: with one extra CNOT(q[n-1], q[0]) per layer. Same parameter count, so the
#: two are drop-in alternatives -- see circuit.qs for why the ring exists.
ENTANGLERS = {"linear": "ApplyAnsatz", "ring": "ApplyAnsatzRing"}


def _ansatz_operation(entangler: str) -> str:
    """Q# operation name for an entangler variant."""
    try:
        return ENTANGLERS[entangler]
    except KeyError:
        raise ValueError(
            f"unknown entangler {entangler!r}; expected one of {sorted(ENTANGLERS)}"
        ) from None


def _scratch_name(num_qubits: int) -> str:
    """Q# identifier of the scratch register for a given size."""
    return f"__scratch{int(num_qubits)}"


def _ensure_scratch_register(num_qubits: int = NUM_QUBITS) -> None:
    """Allocate the session-level scratch register used by the exact backend.

    The register is declared at interpreter TOP LEVEL so it outlives the
    ``eval`` that creates it and its state can be inspected by
    ``dump_machine()``. Allocating inside an operation would not work: Q#
    releases qubits at end of scope and requires them to be in |0> first, so
    the prepared state would be destroyed before Python could read it.

    EXACTLY ONE REGISTER MAY BE ALLOCATED AT A TIME. ``dump_machine`` reports
    the state of the WHOLE session, not of one register, so a second live
    register widens every returned vector. Measured: with a 4-qubit and a
    12-qubit register both allocated, ``as_dense_state()`` returns 65536
    amplitudes and a 4-qubit ansatz's support lands at indices up to 61440 --
    the idle qubits become the low-order bits. Nothing raises; the numbers are
    simply the wrong shape.

    So changing register size RESETS the interpreter. Callers must reload the
    circuit sources afterwards (``_statevector_impl`` does, in that order).
    Switching sizes repeatedly inside one process is therefore expensive by
    design; the 4-asset and variable-weight tracks are meant to run separately.
    """
    global _scratch_allocated, _scratch_size
    if _scratch_allocated and _scratch_size == int(num_qubits):
        return
    if _scratch_allocated and _scratch_size != int(num_qubits):
        logger.debug("Scratch register resize %d -> %d; resetting interpreter.",
                     _scratch_size, num_qubits)
        _reset_runtime_impl()
    qs = get_qsharp()
    qs.eval(f"use {_scratch_name(num_qubits)} = Qubit[{int(num_qubits)}];")
    _scratch_allocated = True
    _scratch_size = int(num_qubits)
    logger.debug("Allocated top-level scratch register of %d qubits.", num_qubits)


def statevector(
    params: Sequence[float],
    num_layers: int = DEFAULT_LAYERS,
    num_qubits: int = NUM_QUBITS,
    entangler: str = "linear",
) -> np.ndarray:
    """Prepare the ansatz and return its exact state vector.

    The register is reset before returning, so consecutive calls are
    independent.

    NOTE: the returned vector carries an arbitrary overall phase that varies
    between calls (see module docstring). Use magnitudes, not raw amplitudes.

    Args:
        num_qubits: Register size. Changing it between calls resets the
            interpreter -- see :func:`_ensure_scratch_register`.

    Returns:
        Complex array of shape ``(2**num_qubits,)``.
    """
    return _on_qsharp_thread(_statevector_impl, params, num_layers, num_qubits,
                             entangler)


def _statevector_impl(params: Sequence[float], num_layers: int,
                      num_qubits: int = NUM_QUBITS,
                      entangler: str = "linear") -> np.ndarray:
    """Body of :func:`statevector`; runs only on the interpreter's thread."""
    global _evals_since_recycle
    p = _check_params(params, num_layers, num_qubits)
    op = _ansatz_operation(entangler)
    _maybe_recycle()
    # Scratch FIRST: a size change resets the interpreter, which would discard
    # sources loaded before it.
    _ensure_scratch_register(num_qubits)
    load_circuit_operations()
    reg = _scratch_name(num_qubits)
    qs = get_qsharp()

    _evals_since_recycle += 1
    qs.eval(f"{op}({reg}, {_fmt_double_array(p)}, {int(num_layers)});")
    try:
        state = np.asarray(qs.dump_machine().as_dense_state(), dtype=complex)
    finally:
        # Always restore |0...0>, even if the read failed, so the session
        # is not left holding a dirty register.
        qs.eval(f"ResetAll({reg});")

    expected = 2 ** int(num_qubits)
    if len(state) != expected:
        raise RuntimeError(
            f"dump_machine returned {len(state)} amplitudes, expected {expected} "
            f"for {num_qubits} qubits. A second register is live in this "
            "session; see _ensure_scratch_register."
        )
    return state


def probabilities(
    params: Sequence[float],
    num_layers: int = DEFAULT_LAYERS,
    num_qubits: int = NUM_QUBITS,
    entangler: str = "linear",
) -> np.ndarray:
    """Exact computational-basis probability distribution of the ansatz.

    Returns:
        Real array of shape ``(2**num_qubits,)`` summing to 1.
    """
    probs = np.abs(statevector(params, num_layers, num_qubits, entangler)) ** 2
    total = probs.sum()
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise RuntimeError(f"State vector is not normalised: sum(p) = {total!r}")
    return probs / total


def expectation_exact(
    params: Sequence[float],
    h_coeffs: Sequence[float],
    j_coeffs: Sequence[float],
    num_layers: int = DEFAULT_LAYERS,
    num_qubits: int = NUM_QUBITS,
    entangler: str = "linear",
) -> float:
    """Exact <H> = sum_z p(z) E(z), with no shot noise.

    Legitimate because H is diagonal in the measurement basis: its expectation
    is a plain average of basis-state energies weighted by their probabilities.
    """
    probs = probabilities(params, num_layers, num_qubits, entangler)
    energies = ising_energy_table(h_coeffs, j_coeffs, num_qubits)
    return float(probs @ energies)


# --------------------------------------------------------------------------
# Sampling backend (shots)
# --------------------------------------------------------------------------

def expectation_sampled(
    params: Sequence[float],
    h_coeffs: Sequence[float],
    j_coeffs: Sequence[float],
    num_layers: int = DEFAULT_LAYERS,
    num_shots: int = DEFAULT_SHOTS,
) -> float:
    """Estimate <H> by measurement, the way real hardware would.

    Statistical error falls as 1/sqrt(num_shots); see
    ``expectation_standard_error`` to quantify it.
    """
    return _on_qsharp_thread(
        _expectation_sampled_impl, params, h_coeffs, j_coeffs, num_layers, num_shots
    )


def _expectation_sampled_impl(
    params: Sequence[float],
    h_coeffs: Sequence[float],
    j_coeffs: Sequence[float],
    num_layers: int,
    num_shots: int,
) -> float:
    """Body of :func:`expectation_sampled`; runs only on the interpreter's thread."""
    global _evals_since_recycle
    p = _check_params(params, num_layers)
    _maybe_recycle()
    load_circuit_operations()
    qs = get_qsharp()

    _evals_since_recycle += 1
    expr = (
        f"EvaluateHamiltonian({_fmt_double_array(p)}, "
        f"{_fmt_double_array(h_coeffs)}, {_fmt_double_array(j_coeffs)}, "
        f"{int(num_layers)}, {int(num_shots)})"
    )
    return float(qs.eval(expr))


def measure_portfolio(
    params: Sequence[float],
    num_layers: int = DEFAULT_LAYERS,
    num_shots: int = 1024,
) -> np.ndarray:
    """Draw computational-basis bitstrings from the ansatz.

    Returns:
        Integer array of shape ``(num_shots, NUM_QUBITS)``; column ``q`` is
        qubit ``q``'s outcome.
    """
    return _on_qsharp_thread(_measure_portfolio_impl, params, num_layers, num_shots)


def _measure_portfolio_impl(
    params: Sequence[float], num_layers: int, num_shots: int
) -> np.ndarray:
    """Body of :func:`measure_portfolio`; runs only on the interpreter's thread."""
    global _evals_since_recycle
    p = _check_params(params, num_layers)
    _maybe_recycle()
    load_circuit_operations()
    qs = get_qsharp()

    _evals_since_recycle += 1
    expr = (
        f"MeasurePortfolio({_fmt_double_array(p)}, {NUM_QUBITS}, "
        f"{int(num_layers)}, {int(num_shots)})"
    )
    raw = qs.eval(expr)
    arr = np.asarray(raw, dtype=int)
    if arr.shape != (num_shots, NUM_QUBITS):
        raise RuntimeError(
            f"MeasurePortfolio returned shape {arr.shape}, "
            f"expected {(num_shots, NUM_QUBITS)}."
        )
    return arr


def expectation_standard_error(
    params: Sequence[float],
    h_coeffs: Sequence[float],
    j_coeffs: Sequence[float],
    num_layers: int = DEFAULT_LAYERS,
    num_shots: int = DEFAULT_SHOTS,
) -> float:
    """Predicted standard error of a sampled <H> estimate, from the exact state.

    ``sqrt(Var[E] / num_shots)`` where ``Var[E]`` is computed against the exact
    distribution. Used to justify the choice of backend for gradients.
    """
    probs = probabilities(params, num_layers)
    energies = ising_energy_table(h_coeffs, j_coeffs, NUM_QUBITS)
    mean = probs @ energies
    var = probs @ (energies - mean) ** 2
    return float(math.sqrt(max(var, 0.0) / num_shots))


def expectation(
    params: Sequence[float],
    h_coeffs: Sequence[float],
    j_coeffs: Sequence[float],
    num_layers: int = DEFAULT_LAYERS,
    backend: str = "exact",
    num_shots: int = DEFAULT_SHOTS,
) -> float:
    """Dispatch <H> evaluation to the selected backend.

    Args:
        backend: ``"exact"`` (state vector, deterministic) or ``"sampling"``
            (shot-based, statistically noisy).
    """
    if backend == "exact":
        return expectation_exact(params, h_coeffs, j_coeffs, num_layers)
    if backend == "sampling":
        return expectation_sampled(
            params, h_coeffs, j_coeffs, num_layers, num_shots
        )
    raise ValueError(f"Unknown backend {backend!r}; use 'exact' or 'sampling'.")


# --------------------------------------------------------------------------
# Environment logging
# --------------------------------------------------------------------------

PACKAGES = (
    "qsharp", "qdk", "torch", "numpy", "pandas",
    "yfinance", "streamlit", "matplotlib", "scipy", "plotly",
)


def log_environment(save: bool = True) -> dict:
    """Capture interpreter, platform, and package versions for reproducibility.

    Versions come from ``importlib.metadata`` because neither ``qsharp`` nor
    ``qdk`` defines ``__version__`` -- reading that attribute is what produced
    the "unknown" entries in the earlier environment log.
    """
    env: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "num_qubits": NUM_QUBITS,
        "default_layers": DEFAULT_LAYERS,
        "ansatz_parameters": n_params(),
        "qubit_asset_map": dict(zip(range(NUM_QUBITS), QUBIT_ASSETS)),
    }

    packages: dict[str, str] = {}
    for name in PACKAGES:
        try:
            packages[name] = _pkg_version(name)
        except PackageNotFoundError:
            packages[name] = "NOT INSTALLED"
    env["packages"] = packages

    logger.info("=" * 64)
    logger.info("ENVIRONMENT")
    logger.info("=" * 64)
    for key, value in env.items():
        if key == "packages":
            logger.info("  packages:")
            for pkg, ver in value.items():  # type: ignore[union-attr]
                logger.info("    %-12s %s", pkg, ver)
        else:
            logger.info("  %s: %s", key, value)
    logger.info("=" * 64)

    if save:
        target = LOG_DIR / "environment.json"
        target.write_text(json.dumps(env, indent=2), encoding="utf-8")
        logger.info("Environment written to %s", target)
    return env


# --------------------------------------------------------------------------
# Phase 1 validation
# --------------------------------------------------------------------------

def run_bridge_validation() -> bool:
    """Phase 1 gate: prove Python <-> Q# interoperability end to end."""
    logger.info("=" * 64)
    logger.info("PHASE 1 VALIDATION -- Q#/Python bridge")
    logger.info("=" * 64)

    passed = True

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        logger.info("  [%s] %s: %s", "PASS" if ok else "FAIL", name, detail)
        if not ok:
            passed = False

    # -- 1. interpreter starts ------------------------------------------
    logger.info("\n[1] Q# runtime initialisation")
    try:
        get_qsharp()
        check("init", True, "interpreter started")
    except Exception as exc:
        logger.exception("Could not initialise the Q# runtime.")
        check("init", False, repr(exc))
        return False

    # -- 2. .qs files load from disk ------------------------------------
    logger.info("\n[2] Q# source loading (from disk, not embedded strings)")
    try:
        load_bridge_operations()
        load_circuit_operations()
        check("load", True, "test_bridge.qs and circuit.qs evaluated")
    except Exception as exc:
        logger.exception("Q# source failed to load.")
        check("load", False, repr(exc))
        return False

    # -- 3. X gate --------------------------------------------------------
    logger.info("\n[3] X-gate measurement")
    try:
        result = eval_qs("TestBridge()")
        check("X gate", "One" in str(result), f"TestBridge() -> {result} (expect One)")
    except Exception as exc:
        logger.exception("TestBridge failed.")
        check("X gate", False, repr(exc))

    # -- 4. control: untouched qubit --------------------------------------
    # Without this, a simulator stuck at One would pass test 3.
    logger.info("\n[4] Control -- untouched qubit")
    try:
        result = eval_qs("TestIdentity()")
        check("control", "Zero" in str(result), f"TestIdentity() -> {result} (expect Zero)")
    except Exception as exc:
        logger.exception("TestIdentity failed.")
        check("control", False, repr(exc))

    # -- 5. Double[] marshalling ------------------------------------------
    logger.info("\n[5] Parameter passing (Python -> Q# Double[])")
    try:
        values = [1.0, 2.5, 3.14159265358979, -0.75, 0.0]
        got = float(eval_qs(f"TestParameterPassing({_fmt_double_array(values)})"))
        want = float(sum(values))
        check(
            "Double[]",
            abs(got - want) < 1e-12,
            f"sum -> {got!r} (expect {want!r})",
        )
    except Exception as exc:
        logger.exception("Parameter marshalling failed.")
        check("Double[]", False, repr(exc))

    # -- 6. Int marshalling + multi-qubit allocation ----------------------
    logger.info("\n[6] Multi-qubit allocation (Int marshalling)")
    try:
        got = int(eval_qs(f"TestMultiQubit({NUM_QUBITS})"))
        check("Int/alloc", got == NUM_QUBITS, f"ones -> {got} (expect {NUM_QUBITS})")
    except Exception as exc:
        logger.exception("TestMultiQubit failed.")
        check("Int/alloc", False, repr(exc))

    # -- 7. exact backend round trip --------------------------------------
    logger.info("\n[7] Exact backend (state vector via dump_machine)")
    try:
        params = [0.0] * n_params()
        state = statevector(params)
        # All-zero angles leave the register in |0000>: probability 1 on index 0.
        p0 = float(abs(state[0]) ** 2)
        check(
            "zero-angle state",
            state.shape == (2 ** NUM_QUBITS,) and abs(p0 - 1.0) < 1e-12,
            f"shape={state.shape}, p(|0000>)={p0:.12f} (expect 1.0)",
        )
    except Exception as exc:
        logger.exception("Exact backend failed.")
        check("zero-angle state", False, repr(exc))

    # -- 8. determinism of observables -------------------------------------
    logger.info("\n[8] Determinism of the exact backend")
    try:
        rng = np.random.default_rng(0)
        params = rng.uniform(-math.pi, math.pi, n_params()).tolist()
        runs = [probabilities(params) for _ in range(4)]
        spread = max(float(np.max(np.abs(runs[0] - r))) for r in runs[1:])
        check(
            "probability determinism",
            spread < 1e-12,
            f"max |dp| across 4 runs = {spread:.3e} (tolerance 1e-12)",
        )
    except Exception as exc:
        logger.exception("Determinism check failed.")
        check("probability determinism", False, repr(exc))

    # -- 9. exact vs sampling agreement -------------------------------------
    # Cross-validates the two independent backends: the exact contraction in
    # Python and the shot loop in Q# must describe the same physics.
    logger.info("\n[9] Exact vs sampling cross-check")
    try:
        rng = np.random.default_rng(1)
        params = rng.uniform(-math.pi, math.pi, n_params()).tolist()
        h = [0.5, -0.3, 0.2, -0.1]
        j = [0.1, -0.05, 0.02, 0.08, -0.03, 0.06]

        set_seed(12345)
        shots = 20000
        exact = expectation_exact(params, h, j)
        sampled = expectation_sampled(params, h, j, num_shots=shots)
        stderr = expectation_standard_error(params, h, j, num_shots=shots)
        # 5 sigma: a correct implementation fails this far less than 1 in 1e6.
        tol = max(5.0 * stderr, 1e-9)
        check(
            "backend agreement",
            abs(exact - sampled) <= tol,
            f"exact={exact:.6f}, sampled={sampled:.6f} "
            f"({shots} shots), |diff|={abs(exact - sampled):.6f} <= 5*sigma={tol:.6f}",
        )
    except Exception as exc:
        logger.exception("Backend cross-check failed.")
        check("backend agreement", False, repr(exc))

    # -- 10. environment ----------------------------------------------------
    logger.info("\n[10] Environment capture")
    try:
        env = log_environment()
        missing = [k for k, v in env["packages"].items() if v == "NOT INSTALLED"]
        check("packages", not missing, f"missing: {missing or 'none'}")
    except Exception as exc:
        logger.exception("Environment logging failed.")
        check("packages", False, repr(exc))

    logger.info("\n" + "=" * 64)
    logger.info("PHASE 1 VALIDATION: %s", "ALL TESTS PASSED" if passed else "FAILURES PRESENT")
    logger.info("=" * 64)
    return passed


# --------------------------------------------------------------------------
# Phase 3 validation
# --------------------------------------------------------------------------

def run_circuit_validation() -> bool:
    """Phase 3 gate: verify the parametrized ansatz behaves as specified."""
    logger.info("=" * 64)
    logger.info("PHASE 3 VALIDATION -- parametrized circuit")
    logger.info("=" * 64)

    passed = True

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        logger.info("  [%s] %s: %s", "PASS" if ok else "FAIL", name, detail)
        if not ok:
            passed = False

    load_circuit_operations()

    h = [0.5, -0.3, 0.2, -0.1]
    j = [0.1, -0.05, 0.02, 0.08, -0.03, 0.06]

    # -- parameter count contract ------------------------------------------
    logger.info("\n[1] Parameter-count contract")
    for layers in (3, 4, 5):
        want = n_params(NUM_QUBITS, layers)
        got = int(eval_qs(f"AnsatzParameterCount({NUM_QUBITS}, {layers})"))
        check(f"count L={layers}", got == want, f"Q#={got}, Python={want}")

    # -- wrong-length parameters must be rejected --------------------------
    logger.info("\n[2] Rejection of malformed input")
    try:
        statevector([0.1] * (n_params() - 1))
        check("bad length", False, "short parameter array was accepted")
    except ValueError:
        check("bad length", True, "short parameter array rejected by guard")

    # -- fixed-angle behaviour ---------------------------------------------
    # Ry(pi) on qubit 3 with one layer: no CNOT follows q3, so only q3 flips.
    # Expected basis state 0001 -> index 1 (qubit 0 is the MSB).
    logger.info("\n[3] Fixed-angle behaviour, single layer")
    try:
        p = [0.0] * n_params(NUM_QUBITS, 1)
        p[3] = math.pi
        probs = probabilities(p, num_layers=1)
        top = int(np.argmax(probs))
        check(
            "Ry(pi) on q3",
            top == 1 and probs[1] > 0.999,
            f"argmax index={top} (bits {index_to_bitstring(top)}), p={probs[top]:.6f}",
        )
    except Exception as exc:
        logger.exception("Fixed-angle test failed.")
        check("Ry(pi) on q3", False, repr(exc))

    # Ry(pi) on qubit 0 propagates along the CNOT ladder to all four qubits.
    try:
        p = [0.0] * n_params(NUM_QUBITS, 1)
        p[0] = math.pi
        probs = probabilities(p, num_layers=1)
        top = int(np.argmax(probs))
        check(
            "Ry(pi) on q0 (ladder)",
            top == 15 and probs[15] > 0.999,
            f"argmax index={top} (bits {index_to_bitstring(top)}), p={probs[top]:.6f}",
        )
    except Exception as exc:
        logger.exception("CNOT-ladder test failed.")
        check("Ry(pi) on q0 (ladder)", False, repr(exc))

    # -- entanglement is genuinely produced --------------------------------
    # Ry(pi/2) on q0 then the ladder yields a GHZ-like state: only |0000> and
    # |1111> carry weight, which no product state can reproduce.
    logger.info("\n[4] Entanglement from the CNOT ladder")
    try:
        p = [0.0] * n_params(NUM_QUBITS, 1)
        p[0] = math.pi / 2
        probs = probabilities(p, num_layers=1)
        ghz = probs[0] + probs[15]
        check(
            "GHZ-like state",
            ghz > 0.999 and abs(probs[0] - 0.5) < 1e-6,
            f"p(0000)={probs[0]:.6f}, p(1111)={probs[15]:.6f}, sum={ghz:.6f}",
        )
    except Exception as exc:
        logger.exception("Entanglement test failed.")
        check("GHZ-like state", False, repr(exc))

    # -- energy stays within spectral bounds -------------------------------
    logger.info("\n[5] Energy bounds")
    try:
        energies = ising_energy_table(h, j)
        lo, hi = float(energies.min()), float(energies.max())
        rng = np.random.default_rng(7)
        worst = 0.0
        for _ in range(20):
            params = rng.uniform(-math.pi, math.pi, n_params()).tolist()
            e = expectation_exact(params, h, j)
            if not (lo - 1e-9 <= e <= hi + 1e-9):
                worst = max(worst, abs(e))
                check("bounds", False, f"<H>={e:.6f} outside [{lo:.4f}, {hi:.4f}]")
                break
        else:
            check("bounds", True, f"20 random draws all inside [{lo:.4f}, {hi:.4f}]")
    except Exception as exc:
        logger.exception("Energy-bounds test failed.")
        check("bounds", False, repr(exc))

    # -- sampled bitstrings are well formed --------------------------------
    logger.info("\n[6] Sampled measurement output")
    try:
        set_seed(2024)
        rng = np.random.default_rng(3)
        params = rng.uniform(-math.pi, math.pi, n_params()).tolist()
        bits = measure_portfolio(params, num_shots=512)
        ok = bits.shape == (512, NUM_QUBITS) and set(np.unique(bits)).issubset({0, 1})
        check("bitstrings", ok, f"shape={bits.shape}, unique values={sorted(set(np.unique(bits).tolist()))}")

        # The sampled frequencies must track the exact distribution.
        exact_p = probabilities(params)
        idx = np.array([bitstring_to_index(b) for b in bits])
        emp = np.bincount(idx, minlength=2 ** NUM_QUBITS) / len(idx)
        tvd = 0.5 * float(np.abs(emp - exact_p).sum())
        check(
            "empirical vs exact",
            tvd < 0.08,
            f"total-variation distance={tvd:.4f} over 512 shots (tolerance 0.08)",
        )
    except Exception as exc:
        logger.exception("Sampling test failed.")
        check("bitstrings", False, repr(exc))

    # -- trailing layer is Ry, so no parameter is structurally dead --------
    logger.info("\n[7] All parameters are live (no inert trailing Rz)")
    try:
        rng = np.random.default_rng(11)
        base = rng.uniform(-math.pi, math.pi, n_params())
        dead = []
        for i in range(n_params()):
            shifted = base.copy()
            shifted[i] += math.pi / 2
            if abs(expectation_exact(shifted, h, j) - expectation_exact(base, h, j)) < 1e-9:
                dead.append(i)
        check(
            "live parameters",
            not dead,
            f"{n_params() - len(dead)}/{n_params()} parameters affect <H>; dead={dead or 'none'}",
        )
    except Exception as exc:
        logger.exception("Live-parameter test failed.")
        check("live parameters", False, repr(exc))

    logger.info("\n" + "=" * 64)
    logger.info("PHASE 3 VALIDATION: %s", "ALL TESTS PASSED" if passed else "FAILURES PRESENT")
    logger.info("=" * 64)
    return passed


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Q#/Python bridge validation")
    parser.add_argument(
        "--test-circuit",
        action="store_true",
        help="run Phase 3 circuit validation instead of Phase 1",
    )
    args = parser.parse_args()

    configure_logging()
    ok = run_circuit_validation() if args.test_circuit else run_bridge_validation()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
