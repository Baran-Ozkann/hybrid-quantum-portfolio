"""
tools/check_reproducibility.py -- is the classical pipeline deterministic?

The brief requires reproducibility, and a fixed historical date range is NOT
sufficient to get it. This script re-runs the pipeline several times and reports
exactly which fields, if any, move between runs.

Background: Yahoo does not return byte-identical data for a fixed, entirely
historical window. Two consecutive fetches of 2025-07-01..2026-07-01 were
measured to differ, with NVDA's annualised mean moving 1.0e-9 and its covariance
entries up to 3.7e-8 (all other tickers unchanged) -- vendor-side revision of
adjustment factors. ``data_pipeline`` therefore freezes ``data/prices.csv`` and
reuses it unless ``--refresh`` is passed.

Run::

    python tools/check_reproducibility.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / ".python" / "python.exe"
DATA = PROJECT_ROOT / "data"
N_RUNS = 3

PASSED = True


def check(name: str, ok: bool, detail: str) -> None:
    global PASSED
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        PASSED = False


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deep_diff(a, b, path="") -> list[str]:
    """Paths where two JSON structures differ, with the numeric delta."""
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += deep_diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += deep_diff(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        if a != b:
            out.append(f"{path}: {a!r} -> {b!r} (delta {b - a:+.3e})")
    elif a != b:
        out.append(f"{path}: {a!r} -> {b!r}")
    return out


def main() -> int:
    print("=" * 74)
    print(f"REPRODUCIBILITY CHECK -- {N_RUNS} consecutive pipeline runs")
    print("=" * 74)

    if not (DATA / "prices.csv").exists():
        print("  data/prices.csv absent; running the pipeline once to freeze it.")
        subprocess.run([str(PYTHON), "classical/data_pipeline.py"],
                       cwd=str(PROJECT_ROOT), capture_output=True)

    runs = []
    for i in range(N_RUNS):
        proc = subprocess.run(
            [str(PYTHON), "classical/data_pipeline.py"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            check("pipeline runs", False, f"exit code {proc.returncode}")
            return 1
        used_cache = "frozen price snapshot" in (proc.stdout + proc.stderr)
        runs.append({
            "prices_sha": sha(DATA / "prices.csv"),
            "stats_sha": sha(DATA / "statistics.json"),
            "stats": json.loads((DATA / "statistics.json").read_text(encoding="utf-8")),
            "used_cache": used_cache,
        })
        print(f"  run {i + 1}: prices={runs[-1]['prices_sha'][:12]} "
              f"statistics={runs[-1]['stats_sha'][:12]} "
              f"cache={'yes' if used_cache else 'NO (fetched)'}")

    print("\n[1] Frozen snapshot is actually being reused")
    check("all runs used the cache", all(r["used_cache"] for r in runs),
          f"{sum(r['used_cache'] for r in runs)}/{N_RUNS} runs read data/prices.csv")

    print("\n[2] prices.csv is byte-stable")
    check("identical price bytes", len({r["prices_sha"] for r in runs}) == 1,
          f"{len({r['prices_sha'] for r in runs})} distinct hashes")

    print("\n[3] statistics.json is byte-stable")
    stats_hashes = {r["stats_sha"] for r in runs}
    ok = len(stats_hashes) == 1
    check("identical statistics bytes", ok, f"{len(stats_hashes)} distinct hashes")
    if not ok:
        diffs = deep_diff(runs[0]["stats"], runs[-1]["stats"])
        print(f"    differing fields ({len(diffs)}):")
        for d in diffs[:20]:
            print(f"      {d}")

    print("\n[4] Derived Hamiltonian is stable")
    sys.path.insert(0, str(PROJECT_ROOT))
    from classical import hamiltonian as H
    from classical import data_pipeline as dp

    hams = []
    for _ in range(N_RUNS):
        s = dp.load_statistics()
        ham = H.build_hamiltonian(s["expected_returns"], s["cov_matrix"], s["tickers"])
        hams.append((ham.h.tolist(), ham.J.tolist(), ham.offset, ham.penalty))
    check("identical Ising coefficients", len({json.dumps(x) for x in hams}) == 1,
          "h, J, offset and penalty match across rebuilds")

    ground = H.brute_force_ground_state(
        H.build_hamiltonian(dp.load_statistics()["expected_returns"],
                            dp.load_statistics()["cov_matrix"],
                            dp.load_statistics()["tickers"]))
    check("ground state stable", ground["bitstring"] == "1010",
          f"{ground['bitstring']} ({', '.join(ground['held'])}), "
          f"cost {ground['qubo_cost']:.6f}")

    print("\n" + "=" * 74)
    print(f"REPRODUCIBILITY: {'DETERMINISTIC' if PASSED else 'NON-DETERMINISTIC'}")
    print("=" * 74)
    return 0 if PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
