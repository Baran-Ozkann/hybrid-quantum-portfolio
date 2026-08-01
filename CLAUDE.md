# CLAUDE.md — orientation for a session starting cold

Hybrid quantum-classical portfolio optimiser: 4-asset selection encoded as an
Ising Hamiltonian, minimised by a variational circuit in **Q#**, trained through
**PyTorch** autograd with the exact parameter-shift rule.

Phases 1–7 are complete and validated. **Currently in Phase 9**, a controlled
objective-comparison study (CVaR vs plain VQE).

Read [`README.md`](README.md) for setup and results, and [`docs/`](docs/) for the
technical reference. This file records only what a new session must know that is
**not** obvious from the code, plus the invariants that are easy to break.

---

## Invariants — breaking these silently corrupts results

### 1. `data/prices.csv` is a frozen INPUT, never regenerated casually

Re-fetching the same *fully historical* window does **not** return identical
data. Measured: NVDA's annualised mean moved 1.0e-9 and covariance entries up to
3.7e-8 between two consecutive fetches — vendor revision of adjustment factors.
That propagates into the Ising coefficients, so a re-fetch silently changes the
problem every saved parameter set was trained against.

- The snapshot is committed; its SHA-256 is in `data/statistics.json`.
- `--refresh` is opt-in and replaces the snapshot. Retrain everything after.
- Verify with `python tools/check_reproducibility.py`.

### 2. All Q# access goes through one owner thread

The Q# interpreter is a pyo3 object marked `unsendable`: touching it from
another thread triggers a **Rust panic**, not a Python exception. Streamlit runs
every rerun on a fresh thread, so this is a live failure mode.

`quantum/q_bridge.py` funnels every call onto one dedicated thread. The fast-path
predicate is **thread identity** (`threading.get_ident() == _owner_thread_id`) —
it must never become "the worker is busy, run it here", which would reintroduce
the panic while still producing correct numbers until the process dies.

Consequences: quantum work is **serialised** (concurrent UI sessions queue), and
`q_bridge` is the only module allowed to import `qsharp`/`qdk`.
Verify with `python tools/verify_threading.py`.

### 3. The interpreter is recycled every 250 evaluations

Each `qsharp.eval()` of a new source string is retained by the session, so cost
grows linearly with call count (3.6 ms → 39.9 ms over 2400 evals). Recycling
pins it near the floor. Native parameter passing would avoid this but is not
available: `GlobalCallable` is not callable from Python.

### 4. Never edit source with PowerShell `Get-Content`/`Set-Content`

It reads UTF-8 as ANSI and writes back mojibake plus a BOM. This happened; the
corrupted file still *ran*, because the damage was inside string literals.
Use the Edit tool. `python tools/scan_encoding.py` is a gate.

### 5. Do not gate on process exit codes

qdk emits `unsendable ... dropped on another thread` at process exit and has
been observed returning 255 despite success. Gate on an explicit success marker
in stdout (`ALL TESTS PASSED`, `ALL CHECKS PASSED`, `DETERMINISTIC`), not on
`$LASTEXITCODE`.

---

## Experiment arms

Each arm trains the **same 30 seeds** (`42, 123, 456, 1000–1026`) against the
same snapshot and Hamiltonian, changing **one** variable. Comparisons are
**paired** (McNemar, exact binomial — the discordant count is single digits).

```bash
python training/optimization.py --seeds 42 123 456 --epochs 200 --schedule constant --tag my_arm
python tools/compare_arms.py --arm my_arm --baseline lr_constant_200
```

- `--tag` writes to `training/arms/<tag>/`. **No tag** writes to `training/` and
  overwrites the committed Phase 5/7 baseline — avoid unless that is intended.
- **Tracked in git:** `training/arms/*/training_summary.json` (per-seed outcomes,
  enough to rebuild every report).
- **Not tracked:** `training/arms/*/trained_seed*.json` (~270 files, regenerable).
  **If an arm is ever adopted as the demo's pre-trained model, its full run files
  must be un-ignored** — otherwise the rule strands the parameters the UI needs.

---

## Results so far

Ground state `1010` (TSLA+AMD), E = −4.921119. Runner-up `0110` (NVDA+AMD),
E = −4.881910. Gap **0.039209**, 0.24% of the spectral width — this near
degeneracy is the whole difficulty.

| arm | epochs | schedule | correct | rate | failed to optimise | spread ÷ gap |
|---|---|---|---|---|---|---|
| baseline (Phase 5/7) | 100 | constant | 18/30 | 60.0% | 7 | 14.71× |
| `lr_cosine_200` | 200 | cosine | 18/30 | 60.0% | 6 | 7.52× |
| **`lr_constant_200`** ← standing baseline | 200 | constant | **20/30** | **66.7%** | **0** | **1.08×** |

- **Longer training fixes all 7 non-converging runs; cosine decay undoes most of
  it** (decaying to lr×0.02 freezes runs that have not yet reached a basin).
- Basins are **trajectory-determined, not initialisation-determined**: seeds
  **456** and **1006** change basin between 100 and 200 epochs.
- `lr_constant_200` shows **perfect energy separation** — every correct run has
  lower energy than every incorrect one, margin **+0.000129**. Cause: all 30 runs
  collapse into the `{1010, 0110}` manifold (mass ≥ 0.9983), where energy is
  near-affine and monotone in P(1010), and correctness ⟺ P(1010) > 0.5. In the
  baseline this fails — 14/30 runs keep ≥1% amplitude elsewhere (one at 65%).
  **The margin is 0.33% of the gap; this is a property of these 30 seeds, not a
  guarantee.**

**`docs/stability_report.md` documents the 100-epoch baseline and is deliberately
untouched** — it accurately describes what it measured.

---

## Phase 9: pre-registered predictions

CVaR minimises the mean of the lowest α-quantile instead of the plain mean.
Implementation and gate: `training/cvar.py`, `tools/verify_cvar_gradient.py`
(value identity 3.6e-15, gradient identity 2.7e-9, α=1 reduces to plain VQE to
5.6e-15).

Key fact: `CVaR_α` is **exactly minimised once P(ground) ≥ α**, after which the
gradient vanishes identically and training halts.

| | prediction |
|---|---|
| **User** | Non-monotone, **peak at α = 0.5**, discontinuity between 0.4 and 0.6. Because P(ground) ≥ 0.5 forces the ground state to be the argmax, α = 0.5 is the weakest target that still guarantees a correct readout — and tail weights carry 1/α, so it pursues that target with twice plain VQE's gradient pressure. |
| **Claude** | Monotone increasing in α, best at α = 1.0. Reasoning: small α halts under-concentrated. **Wrong** — missed that 0.5 is exactly the argmax threshold. |

Both stay on the record with whichever the data supports.

**Configuration for all CVaR arms** (must be identical): 200 epochs, constant
lr = 0.05, 3 layers, init U(−0.1, 0.1), same 30 seeds.
**Sweep:** α ∈ {0.1, 0.4, 0.5, 0.6} at 30 seeds, plus a 5-seed α = 1.0 code-path
check. Order: 0.5 → 0.6 → 0.4 → 0.1, reporting after the triple.

Report must include, per arm: fraction with P(ground) ≥ 0.5 checked against
readout accuracy (must coincide for α ≥ 0.5); plain ⟨H⟩ alongside the CVaR value,
with an explicit note that **cross-arm energy comparison is meaningless** because
saturating runs stop early; and multi-start curves recomputed within each arm.

Deliverable: `docs/objective_comparison.md` + figures + a README paragraph.
`stability_report.md` gets **one line** in future-work noting CVaR moved to a
separate study.

---

## Open decisions

1. **Does `lr_constant_200` become the recommended configuration?** If so the
   headline moves 60% → 66.7%, and then:
   - the UI's pre-trained model (`training/trained_seed42.json`, 100 epochs)
     describes a configuration no longer recommended;
   - `docs/stability_report.md`'s illustrative trio does too — and **seed 456 is
     correct under the new settings**, which changes that narrative;
   - the winning arm's full run files must become tracked.

   A documentation change, not a rerun, but not trivial. Pending Phase 9.

2. **Priority 3 (5- and 7-layer ansätze)** — not started, outcome genuinely
   uncertain; more parameters may worsen the landscape.

3. Git is **local only, no remote**. Identity is set repo-locally. Confirm before
   any push whether that email should be in public history.

---

## Gates

All must pass before shipping. None gate on exit codes (see invariant 5).

```bash
python quantum/q_bridge.py                  # Phase 1 bridge, 10/10
python quantum/q_bridge.py --test-circuit   # Phase 3 circuit, 7/7
python classical/phase2_gate.py             # Phase 2 formulation, 9/9
python training/gradient_bridge.py          # Phase 4 gradients, 3.0e-10
python tools/verify_cvar_gradient.py        # Phase 9 CVaR identity
python tools/verify_threading.py            # interpreter thread affinity
python tools/check_reproducibility.py       # pipeline determinism
python tools/scan_encoding.py               # mojibake / BOM scan
python tools/capture_ui.py                  # UI + figures (needs playwright)
```

Use `.python/python.exe` if using the bundled interpreter.
