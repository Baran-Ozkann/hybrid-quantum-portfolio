# CLAUDE.md — orientation for a session starting cold

Hybrid quantum-classical portfolio optimiser: 4-asset selection encoded as an
Ising Hamiltonian, minimised by a variational circuit in **Q#**, trained through
**PyTorch** autograd with the exact parameter-shift rule.

**Phases 1–9 are complete, validated, and closed.** Phase 9 was a controlled
objective-comparison study (CVaR vs plain VQE) — its result is a *trade*, not a
rate; see [`docs/objective_comparison.md`](docs/objective_comparison.md).
Phase 8 was the out-of-sample backtest, and it asks a different question from
every other phase: not whether the optimiser finds the minimum, but whether that
minimum is worth finding. **Answer: on 16 quarters, not measurably.** See
[`docs/backtest.md`](docs/backtest.md) and the framing below, which must not be
blurred.

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

## Phase 9: COMPLETE — see [`docs/objective_comparison.md`](docs/objective_comparison.md)

CVaR minimises the mean of the lowest α-quantile instead of the plain mean.
Implementation `training/cvar.py`; gate `tools/verify_cvar_gradient.py` (value
identity ~2e-15, gradient identity ~2e-9, α=1 reduces to plain VQE to ~5e-15 —
see the note under Gates on why these are bounds, not constants).

All arms: 200 epochs, constant lr = 0.05, 3 layers, init U(−0.1, 0.1), same 30
seeds, exact backend. Paired against `lr_constant_200`.

**The headline is a TRADE, not a rate.** CVaR gives up 1.88 in mean energy —
**48× the degeneracy gap**, against plain VQE's 0.4× — and buys ~30 points of
readout accuracy. Every CVaR run fails the 0.5%-of-spectrum converged criterion
that every plain-VQE run passes. Energy convergence and correct readout are not
merely imperfectly correlated here; they are actively traded.

| α | exact | Wilson 95% | E[shot acc] @4096 | mean rank | worst rank | min margin† | mean ⟨H⟩ | McNemar p |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 56.7% | [39.2%, 72.6%] | **54.2%** | 1.567 | 3 | 0.0051 | −0.0771 | 0.6291 |
| **0.4** | **100.0%** | **[88.6%, 100%]** | **100.0%** | **1.000** | 1 | **0.1810** | −2.7327 | **0.0020** |
| 0.5 | 96.7% | [83.3%, 99.4%] | 95.5% | 1.033 | 2 | 0.0053 | −3.0423 | 0.0039 |
| 0.6 | 86.7% | [70.3%, 94.7%] | 86.7% | 1.267 | 4 | 0.1249 | −3.3496 | 0.0703 |
| 1.0 (plain) | 66.7% | [48.8%, 80.8%] | 66.3% | 3.033 | 14 | 0.0023 | −4.9052 | — |

† minimum top-two margin among correct runs.

**Do not overstate these.** 30/30 is a point estimate with Wilson [88.6%, 100%] —
a true rate near 90% is consistent with it. Accuracy peaks at the **sampled
point** α = 0.4 on a five-point grid; the true optimum lies somewhere in
(0.1, 0.5) and this design cannot resolve it further (0.4 is already at ceiling,
so a finer grid needs more *seeds*, not more α). Two conclusions in this project
have already been overturned by more data (1-of-3 → 18-of-30; Priority 2's
"clean null" → two effects cancelling).

**Non-monotone, peak at the sampled α = 0.4. Both pre-registered predictions failed:**

- **User:** peak at 0.5 with a 0.4/0.6 cliff, from the argmax guarantee.
  *Shape confirmed, location and mechanism refuted.*
- **Claude:** monotone increasing to α = 1.0. *Refuted; ordering reverses.*

**Mechanism (measured, not argued).** The argmax guarantee never binds:
P(ground) ≥ 0.5 is sufficient but **not necessary** — a *plurality* suffices, and
CVaR fragments losing mass across fifteen states. Decisive evidence: at α = 0.4
**only 2 of 30 runs reach P ≥ 0.5, yet all 30 read out correctly**. What drives
the result is the `1/α` tail weight (2.5× plain gradient pressure at 0.4 vs 2.0×
at 0.5), traded against the saturation target (CVaR is exactly minimised once
P(ground) ≥ α, after which the gradient vanishes). They balance near 0.4.

**Three distinct failure modes:**

1. **Plain VQE — wrong basin.** Ground state scattered to ranks 6–14 in half its
   failures. Under-optimised discrimination.
2. **Large α (0.6) — insufficient pressure.** Failures at ranks 2–4. Seed 1006
   regresses decisively (1010 → rank 2 at P = 0.15 while 0110 holds 0.52).
3. **Small α (0.1) — saturation/flatness, and loss of structure.** Median 185 of
   200 epochs with a degenerate tail (m = 0, zero gradient), halting at
   P(ground) ≈ 0.17 with median max-probability 0.19. Ground state still ranks
   1–3 in every run but median margin 0.031, so only 6 of 30 readouts survive
   4096 shots. Scatters across **7 distinct attractors** (every other arm has
   ≤ 2): 3 infeasible (`0000`, `1111`, `0001` — 6 runs violating the budget
   constraint) and 4 feasible-but-wrong.

**A second, independent lower bound on α** (docs §7, not anticipated by the
derivation): the budget penalty is enforced only through the objective, and CVaR
zero-weights everything above the α-quantile, so **α cannot go below the mass the
feasible subspace must retain**. This is about *validity of the solution*, distinct
from gate 1's α > 1/2ⁿ bound about *optimisability* — and it binds first: 0.1 is
well above 1/16 = 0.0625 yet still yields illegal portfolios.

**Seed structure.** Stable core recovered by every CVaR arm:
**123, 1008, 1011, 1017, 1022, 1025**. Fringe (α-dependent): **1004, 1014, 1023**
(α ≤ 0.5), **1021** (α = 0.6, 0.4). Core recoveries carry ~2× the fringe's margin
at α ≥ 0.5. Only regression in the phase: **1006** at α = 0.6.

**Recommendation unchanged.** At k = 5 plain VQE already reaches 99.59%, and
Phase 7 already recommends five seeds. This is a mechanistic result about how the
objective shapes the readout, **not** a change to what the demo should do. The
one practical caveat: α = 0.4 reaches 100% at **k = 1**, which matters only if
runs are expensive.

---

## Phase 8: COMPLETE — see [`docs/backtest.md`](docs/backtest.md)

Every Sharpe ratio in Phases 1–9 is **in-sample**. Phase 9 measured whether the
optimiser finds the minimum; it never asked whether that minimum is worth
finding.

### Results

Separate frozen snapshot `data/backtest_prices.csv`, SHA-256 `17e87c2f…`,
1276 rows 2021-06-01→2026-06-30. `data/prices.csv` verified byte-identical
before and after. 16 folds: 250-day estimation, 63-day holding, step 63.

| question | result | p |
|---|---|---|
| in-sample top-two winner also wins OOS | **10/16 = 62.5%** [35.4%, 84.8%] | 0.454 |
| **does the size of the IS edge predict the OOS edge** | **ρ = −0.085** | **0.753** |
| per-fold Spearman(IS Sharpe, OOS Sharpe), all 6 portfolios | +0.186 [−0.191, +0.563] | 0.310 |
| IS top is also OOS top | 4/16 = 25.0% (chance 16.7%) | 0.325 |
| `1010` vs `0110` head-to-head | **6W–10L** | 0.454 |
| cost of one optimiser error | +7.4%/qtr [−6.0%, +20.9%] | 0.258 |

**The two discrimination tests disagree in direction** (generalised 10/16 = 62.5%
favours the objective; specific `1010` vs `0110` 6/16 = 37.5% opposes it), both
p = 0.454, both CIs straddle 50%. **Report them side by side** — either alone
misleads. The shared conclusion is that neither is separable from chance.

**Lean on the sign test, not the mean.** sd/|mean| = 3.44× on realised ΔSharpe;
dropping fold 15 alone moves the mean from +0.4168 to +0.1866.

**Ordering is by the penalised mean–variance objective, NOT by Sharpe.** They
agree in 15 of 16 folds but not fold 13 (ground `1100` has *lower* in-sample
Sharpe than runner-up `0101`, Δ = −0.0724). So "in-sample winner" never means
"Sharpe maximiser", and nothing here is "positive by construction".

**The tempting narrowing was tested and is NOT supported.** "This only shows
tight gaps are unpredictable" — median-gap split gives **5/8 in both halves**,
Fisher p = 1.000, Spearman(gap, win) = +0.028. The mean Δ *is* larger in the
loose half (+0.79 vs +0.05) but that is the fragile statistic. Do not narrow the
claim on the strength of the mean alone.

Decomposition (per-fold total return): **quantum − best_equal_weight =
identical on all 16 folds**; best_equal_weight − EW4 = +4.90% (p = 0.219);
best_equal_weight − buy_hold_NVDA = −1.07% (p = 0.867). The first is the
**absence of a measurement**, not a measurement of zero — the backtest says
nothing about the quantum arm's economic value in either direction. Its real
contribution is a Phase 9 robustness check (§6), and it should be cited as one.

**Quantum arm: 16/16 matched, 0 infeasible, mean P(ground) = 0.4988** — Phase 9's
plurality mechanism reproducing on Hamiltonians it was never tuned against.

**Ground state is not stable across windows:** `0101` ×5, `1100` ×5, `0110` ×4,
**`1010` ×2**. The project's instance is one of two windows out of sixteen.
Gap/width: median 1.24%, min 0.02%, max 4.74% vs main instance 0.24% — **the main
instance is at the hard end (2 of 16 folds are tighter)**, so do NOT say
near-degeneracy is "generic" without that qualifier.

**Do not overstate.** The design is powered to detect effects ~4× larger than the
one sought (80% MDE = 1.076 Sharpe vs the 0.242 edge); 279 folds ≈ 70 years would
be needed. Point estimates are positive — "indistinguishable" is *not* "zero".
Selection bias is unfixable and `buy_hold_NVDA` beats a perfect optimiser.

### The framing that must not be blurred

> **A backtest does not test the optimiser. It tests the objective.**

VQE's job is to minimise a *given* Hamiltonian, and Phases 4–9 establish that it
does. Whether the minimiser of that Hamiltonian is a good *investment* is a
separate question about mean-variance selection on 250 days of history. **A weak
backtest is evidence against in-sample mean-variance selection, not against the
quantum method.** Keep these strictly apart — the same discipline applied to the
feasible-set caveat in Phase 2. Any sentence that lets a poor realised Sharpe
reflect on the optimiser is a bug in the write-up.

### The question it is built around

In-sample, `1010` beats `0110` by **1.4% Sharpe**, and plain VQE misses that
discrimination **33%** of the time. Is that 1.4% meaningful out of sample at all?
If the in-sample winner wins out of sample only ~half the time, **the optimiser's
error rate costs nothing in reality — it was resolving noise.** Given what Phase 9
established about how hard the 0.039 gap is to resolve, that is the sharpest
available closing result.

### Design

- **Data:** new frozen snapshot, own SHA-256. COIN IPO'd April 2021 → usable
  history from ~2021-06, ~5 years. **Do not touch or re-fetch `data/prices.csv`**
  (invariant 1).
- **Folds:** rolling 250-day estimation, 63-day holding, step 63 → **~16 folds.
  Say this is small before reporting anything else.**
- **Per fold:** estimate µ/Σ from the training window, enumerate all 6
  equal-weight 2-asset portfolios in-sample, measure realised return over the
  next 63 days. Run the quantum pipeline on that fold's Hamiltonian at
  **CVaR α = 0.4, k = 1** — Phase 9 showed it reaches the ground state
  single-shot, so the quantum arm costs **16 runs, not 80**.
- **Decomposition (the part that matters):**
  1. quantum vs `best_equal_weight` → **optimiser error** (same objective, same
     feasible set)
  2. `best_equal_weight` vs equal-weight-all-4 → **value of the objective itself**
  3. both vs single-asset buy-and-hold → context
- **Two cheap, high-information metrics:** per-fold **Spearman** rank correlation
  between in-sample and out-of-sample Sharpe across the 6 portfolios (near zero ⇒
  no predictive content, and everything downstream follows); and how often the
  in-sample top is the out-of-sample top, plus the **`1010` vs `0110` head-to-head
  win rate**.
- **Report each fold's degeneracy gap.** Every fold has a different Hamiltonian,
  so this shows for free whether the near-degeneracy is peculiar to the current
  window or generic — which also speaks to `objective_comparison.md` §8.

### Caveats to state, not let a reader discover

- **Selection bias is the largest and is not fixable.** These four tickers were
  chosen in 2026 knowing how they turned out; no 2021 investor picks this basket.
- **Overlapping estimation windows** — the ~16 folds are not independent, so
  intervals are wider than they look.
- **Power is very low.** At ~16 quarterly observations, detecting a Sharpe
  difference this size is essentially impossible. **State this in advance so
  "indistinguishable" reads as a prediction, not an excuse.**
- **Transaction costs** on quarterly rebalancing, ~10 bps per side.
- **Point-in-time.** Today's adjusted prices embed splits/dividends unknown at the
  time — acceptable for a total-return study, but flag it, especially given the
  vendor adjustment drift behind invariant 1.

**Do not fold any of this into `stability_report.md` or
`objective_comparison.md`** — different question, different scope. Both were
left untouched.

### Invariant 6: `data/backtest_prices.csv` is a second frozen input

It is independent of `data/prices.csv` and has its own manifest
(`data/backtest_manifest.json`). `backtest/fetch_snapshot.py` verifies the
SHA-256 on every load and refuses to proceed on a mismatch. `--refresh`
invalidates every number in `docs/backtest.md`. Nothing in `backtest/` reads or
writes `data/prices.csv`.

---

## Open decisions

1. ~~**Does `lr_constant_200` become the recommended configuration?**~~
   **DECIDED (2026-08-02): no. It stays a documented alternative, not the
   recommendation.** Do not reopen this without a new reason.

   *Reasoning, recorded so the decision reads as made rather than deferred:*
   adopting it would move the headline 60% → 66.7% and cascade through three
   places — the UI's pre-trained model (`training/trained_seed42.json`, 100
   epochs), `docs/stability_report.md`'s illustrative trio (**seed 456 is
   correct under the new settings**, which breaks that narrative), and the
   arm's full run files would have to become tracked. That is real work in
   service of moving a *readout-accuracy* number, and **Phase 8 §4 found that
   number has no measurable economic value**: the 1.45% edge it resolves is
   indistinguishable from a coin flip out of sample. Phase 8 lowered the stakes
   of this decision rather than raising them. The 66.7% figure stays documented
   in `README.md` and CLAUDE.md as the better-converged measurement; the
   *recommendation* stays where the artifacts already point.

1b. **Shrinkage arm (next experiment, PRE-REGISTERED — read before running).**
   Phase 8 estimates µ and Σ with plain sample estimators. Ledoit–Wolf shrinkage
   on Σ is the obvious remedy and is cheap.

   > **Pre-registered expectation: shrinking Σ alone will sharpen the covariance
   > estimate without moving the discrimination result much.** Phase 8's noise is
   > dominated by **µ**, not Σ — a single 250-day window gives essentially *one*
   > observation of each annualised mean, and that is what ρ = −0.085 is
   > measuring. **A null result here is the predicted outcome, not a surprise,
   > and must not be reported as one.**
   >
   > **The version that could actually move the result shrinks µ toward the
   > grand mean as well.** That attacks the term the data says is noisy. Run
   > both arms — Σ-only and µ+Σ — so the prediction is testable rather than
   > merely asserted.

2. **Priority 3 (5- and 7-layer ansätze)** — not started, outcome genuinely
   uncertain; more parameters may worsen the landscape. Lowest expected value of
   the remaining work: it touches nothing else.

2b. **Should the α grid be refined around 0.3–0.45?** α = 0.4 is at ceiling
   (30/30), so a finer grid cannot resolve the peak with n = 30 — it would need
   more seeds, not more α values. Probably not worth it.

2c. **200 epochs is not a converged setting.** Seed 456 crossed basins at epoch
   176 of 200 and held for only 24 epochs, so a 250-epoch arm could move the
   plain-VQE baseline again. The 66.7% should not be presented as converged.

3. Git is **local only, no remote** — **decided (2026-08-02): keep it that way
   for now.** Identity is set repo-locally and `ozkanbarran@gmail.com` is in
   every author line. **If this ever goes public, the author-email question is
   decided BEFORE the first push, not after** — rewriting authorship
   retroactively is far more painful than choosing now.

4. **`docs/presentation_notes.md`** holds the three framing claims and the three
   scope disclaimers, with every number re-derived from the artifacts. It also
   records two claims that were wrong in an earlier draft (a 40% that was really
   33.3%, and an 11%-buys-100% that really buys 96.7%) **specifically so the
   rounder-sounding versions do not get restored.** If a headline number changes,
   that file changes with it.

---

## Gates

All must pass before shipping. None gate on exit codes (see invariant 5).

```bash
python quantum/q_bridge.py                  # Phase 1 bridge, 10/10
python quantum/q_bridge.py --test-circuit   # Phase 3 circuit, 7/7
python classical/phase2_gate.py             # Phase 2 formulation, 9/9
python training/gradient_bridge.py          # Phase 4 gradients, <1e-9 vs 1e-2 bound
python tools/verify_cvar_gradient.py        # Phase 9 CVaR identity, <1e-8
python backtest/fetch_snapshot.py           # Phase 8 snapshot SHA-256 verify
python tools/verify_threading.py            # interpreter thread affinity
python tools/check_reproducibility.py       # pipeline determinism
python tools/scan_encoding.py               # mojibake / BOM scan
python tools/capture_ui.py                  # UI + figures (needs playwright)
```

Use `.python/python.exe` if using the bundled interpreter.

**Gradient-gate figures are bounds, not constants — do not "correct" them to
whatever your run printed.** Both gradient gates seed their probe points
(`default_rng(0)`, `default_rng(9)`), so the *inputs* are fixed, but the
finite-difference *reference* is not reproducible: the exact backend carries
~1.4e-15 of nondeterminism (the gate prints this as `<H> spread over 50
evaluations`, the ~10-ULP global-phase floor from Phase 4), and central
differences at ε = 1e-5 amplify it by 1/(2ε) = 5e4. **The order of magnitude is
stable; the digits after the first are not** — CVaR gate: 1.82, 2.06, 2.11,
2.70 e-09; Phase 4: 2.14, 2.14, 2.26, 2.36 e-10; Richardson: 1.81–2.05 e-11;
chain rule: 5.7e-14–1.6e-13. **`max|PS − FD|` measures amplified simulator noise,
not gradient disagreement**, and sits eight orders of magnitude under the 1e-2
gate. Quote it as an order of magnitude; three significant figures asserts a
precision the quantity does not have. All six documentation sites now say this
(README, architecture, mathematical_foundations §8, here).
