"""
tools/verify_threading.py -- proves the single-owner-thread interpreter design.

The Q# interpreter is a pyo3 object marked ``unsendable``: touching it from a
thread other than its constructor's triggers a RUST PANIC, not a Python
exception. Streamlit gives every script rerun a fresh ScriptRunner thread, so
this is a live failure mode for the UI, and an intermittent one -- a run that
happens to stay on one thread succeeds.

This script checks the properties that make the fix correct, rather than just
observing that "it worked once":

  1. The fast-path predicate is THREAD IDENTITY, never worker availability.
  2. Interpreter CONSTRUCTION happens on the owner thread -- including the
     rebuild performed by periodic recycling, which is the case most likely to
     regress because it fires only on long runs.
  3. Nested calls do not deadlock on the single-worker pool.
  4. No Q# object escapes into anything Streamlit persists across reruns
     (``st.session_state`` / ``@st.cache_resource``).

Run::

    python tools/verify_threading.py
"""

from __future__ import annotations

import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classical import hamiltonian as H   # noqa: E402
from quantum import q_bridge as qb       # noqa: E402
from training import optimization as opt  # noqa: E402

PASSED = True


def check(name: str, ok: bool, detail: str) -> None:
    global PASSED
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        PASSED = False


def _is_qsharp_object(obj) -> bool:
    """True if ``obj``'s type comes from the Q# runtime (and so is unsendable)."""
    module = type(obj).__module__ or ""
    return (module.startswith("qdk")
            or module.startswith("qsharp")
            or type(obj).__name__ in ("Interpreter", "GlobalCallable", "StateDump"))


def _walk(obj, path="root", depth=0, seen=None):
    """Yield ``(path, obj)`` for every object reachable from ``obj``."""
    if seen is None:
        seen = set()
    if depth > 6 or id(obj) in seen:
        return
    seen.add(id(obj))
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}[{k!r}]", depth + 1, seen)
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]", depth + 1, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, np.ndarray):
        for k, v in vars(obj).items():
            yield from _walk(v, f"{path}.{k}", depth + 1, seen)


def main() -> int:
    ham = H.load_hamiltonian()
    params = opt.load_result(42).final_params
    energies = qb.ising_energy_table(ham.h, ham.J, ham.n_assets)

    print("=" * 74)
    print("THREADING VERIFICATION -- single-owner-thread Q# interpreter")
    print("=" * 74)

    # ---------------------------------------------------------------- 1
    print("\n[1] Fast-path predicate is thread identity, not worker availability")
    src = inspect.getsource(qb._on_qsharp_thread)
    body = src.split('"""')[-1]          # strip the docstring
    check("predicate compares get_ident()",
          "threading.get_ident() == _owner_thread_id" in body,
          "inline branch taken iff caller IS the owner thread")
    check("no availability-based branch",
          not any(tok in body for tok in ("busy", "_work_queue", "qsize", "empty()")),
          "no queue/busy inspection in the dispatch path")

    main_ident = threading.get_ident()
    qb.probabilities(params, 3)          # force executor creation
    owner = qb.owner_thread_id()
    check("owner thread distinct from caller", owner is not None and owner != main_ident,
          f"owner={owner}, main={main_ident}")

    inside = qb._on_qsharp_thread(threading.get_ident)
    check("work runs on the owner thread", inside == owner,
          f"submitted callable reported ident {inside}")

    # ---------------------------------------------------------------- 2
    print("\n[2] Nested calls run inline instead of deadlocking")
    # probabilities -> statevector is a real nested pair; a naive implementation
    # would submit the inner call and wait forever on a single-worker pool.
    def nested():
        return qb.probabilities(params, 3), threading.get_ident()
    probs_nested, ident_nested = qb._on_qsharp_thread(nested)
    check("nested call completes", probs_nested.shape == (16,),
          f"inner call returned shape {probs_nested.shape} on ident {ident_nested}")
    check("nested call stayed on owner thread", ident_nested == owner, f"ident {ident_nested}")

    # ---------------------------------------------------------------- 3
    print("\n[3] Interpreter RECYCLING also constructs on the owner thread")
    # Recycling rebuilds the interpreter. If that construction happened on a
    # caller's thread the panic returns -- and only on long runs, which is
    # exactly when recycling fires. Force it to trigger constantly.
    original_limit = qb.MAX_EVALS_BEFORE_RECYCLE
    before = qb.recycle_stats()["recycle_count"]
    qb.MAX_EVALS_BEFORE_RECYCLE = 3
    try:
        results = []
        # Each call from a DIFFERENT thread, so every rebuild is requested by a
        # foreign caller -- the exact Streamlit pattern.
        for i in range(40):
            with ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix=f"rerun{i}") as ex:
                results.append(ex.submit(
                    lambda: float(qb.probabilities(params, 3) @ energies)
                ).result())
    finally:
        qb.MAX_EVALS_BEFORE_RECYCLE = original_limit

    after = qb.recycle_stats()["recycle_count"]
    check("recycling actually fired", after > before,
          f"recycle_count {before} -> {after} across 40 cross-thread calls")
    check("energies identical through recycling", len({round(r, 10) for r in results}) == 1,
          f"all 40 calls returned {results[0]:.10f}")
    check("owner thread unchanged by recycling", qb.owner_thread_id() == owner,
          f"owner still {qb.owner_thread_id()}")

    # ---------------------------------------------------------------- 4
    print("\n[4] Sustained cross-thread and concurrent use")
    per_thread = []
    for i in range(15):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"seq{i}") as ex:
            per_thread.append(ex.submit(
                lambda: float(qb.probabilities(params, 3) @ energies)).result())
    check("15 sequential fresh threads agree",
          len({round(v, 10) for v in per_thread}) == 1,
          f"energy {per_thread[0]:.10f}")

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="conc") as ex:
        conc = list(ex.map(
            lambda _: float(qb.probabilities(params, 3) @ energies), range(48)))
    check("48 concurrent calls agree", len({round(v, 10) for v in conc}) == 1,
          f"energy {conc[0]:.10f}")
    check("sequential matches concurrent",
          round(per_thread[0], 10) == round(conc[0], 10),
          "same value on both paths")

    # ---------------------------------------------------------------- 5
    print("\n[5] No Q# object escapes into Streamlit-persistent state")
    # What the app stores in st.session_state["quantum"]:
    sys.path.insert(0, str(PROJECT_ROOT / "ui"))
    from ui.app import run_quantum_readout, init_quantum_runtime, get_hamiltonian

    payload = run_quantum_readout(params, ham, 3, 512)
    leaked = [p for p, o in _walk(payload, "session_state['quantum']")
              if _is_qsharp_object(o)]
    check("session_state payload is Q#-free", not leaked,
          f"scanned {sum(1 for _ in _walk(payload))} reachable objects; "
          f"leaks={leaked or 'none'}")

    # What @st.cache_resource holds. Streamlit caches whatever these return, and
    # cached values outlive the rerun (and its thread) that created them.
    cached_runtime = init_quantum_runtime.__wrapped__()
    check("init_quantum_runtime caches a plain bool",
          isinstance(cached_runtime, bool),
          f"returns {cached_runtime!r} ({type(cached_runtime).__name__}), not an interpreter handle")

    cached_ham = get_hamiltonian.__wrapped__()
    leaked_ham = [p for p, o in _walk(cached_ham, "cache_resource[hamiltonian]")
                  if _is_qsharp_object(o)]
    check("cached Hamiltonian is Q#-free", not leaked_ham,
          f"leaks={leaked_ham or 'none'}")

    print("\n" + "=" * 74)
    print(f"THREADING VERIFICATION: {'ALL CHECKS PASSED' if PASSED else 'FAILURES PRESENT'}")
    print("=" * 74)
    return 0 if PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
