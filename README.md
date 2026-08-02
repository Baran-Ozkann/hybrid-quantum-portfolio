# Hybrid Quantum Portfolio Optimization with Q# and PyTorch

A working proof of concept that formulates 4-asset portfolio selection as an
Ising Hamiltonian, minimises it with a variational quantum circuit written in
**Q#**, trains that circuit through **PyTorch autograd** using the exact
parameter-shift rule, and compares the result honestly against classical
Markowitz baselines.

**Headline result:** the quantum optimiser matched the classical optimum
*exactly* over the same feasible set (Sharpe 2.1599, portfolio 50% TSLA /
50% AMD) — but a **single run finds the true ground state only 60% of the time**
(18 of 30 seeds, Wilson 95% CI [42%, 75%]). The remaining runs land on a
near-degenerate alternative 1.4% worse. Running 5–10 seeds and keeping the
lowest energy recovers 96–100%. Both halves of that sentence are the result.

---

## Quick start

```bash
python -m pip install -r requirements.txt
```

```bash
python classical/phase2_gate.py
```

```bash
python training/optimization.py --seeds 42 123 456
```

```bash
streamlit run ui/app.py
```

The first command validates the classical pipeline and writes the Hamiltonian;
the second trains three seeds (~25 s each); the third serves the UI at
`http://localhost:8501`.

> This repository ships with a self-contained interpreter at `.python/`. If you
> are using it, prefix commands with `.python/python.exe` instead of `python`.

---

## Requirements

Python 3.11, and the versions in [`requirements.txt`](requirements.txt) —
pinned exactly, not as minimums:

| Package | Version | | Package | Version |
|---|---|---|---|---|
| `qdk` | 1.30.0 | | `pandas` | 3.0.5 |
| `qsharp` | 1.30.0 | | `yfinance` | 1.5.2 |
| `torch` | 2.13.0 | | `streamlit` | 1.60.0 |
| `numpy` | 2.4.6 | | `matplotlib` | 3.11.1 |
| `scipy` | 1.17.1 | | `plotly` | 6.9.0 |

`requirements-lock.txt` holds the full transitive closure of the validated
environment.

**The Streamlit pin is a hard requirement.** `ui/app.py` uses `width="stretch"`
(which replaced the deprecated `use_container_width`) and
`st.column_config.NumberColumn`. On an older Streamlit the app raises
`TypeError` on the first chart it renders.

Optional, for regenerating figures and running the browser-level verification:

```bash
python -m pip install playwright
python -m playwright install chromium
```

---

## What it does

1. Fetches one year of daily closes for **TSLA, NVDA, AMD, COIN** and
   annualises expected returns and the 4×4 covariance matrix.
2. Encodes budget-constrained selection (**hold exactly 2 of 4**) as a QUBO,
   converts it to an Ising Hamiltonian, and verifies the conversion against all
   16 bitstrings.
3. Minimises `⟨H⟩` with a 4-qubit, 28-parameter hardware-efficient ansatz in Q#,
   differentiated through PyTorch via the parameter-shift rule.
4. Decodes measured bitstrings into portfolio weights summing to 1.
5. Compares against classical baselines, distinguishing like-for-like
   comparisons from ones spanning a different feasible set.

---

## Results

Market data, 251 trading days, 2025-07-01 → 2026-06-30:

| Asset | Qubit | Annual return | Annual volatility |
|---|---|---|---|
| TSLA | q0 | +43.64% | 44.35% |
| NVDA | q1 | +33.20% | 35.33% |
| AMD | q2 | +168.93% | 67.59% |
| COIN | q3 | −60.91% | 67.50% |

Exact ground state by brute force over all 16 bitstrings: **`1010`
(TSLA + AMD)**, `E = −4.921119`.

### Quantum vs classical

All Sharpe ratios use **rf = 5.00%**.

| Method | Feasible set | Return | Vol | Sharpe |
|---|---|---|---|---|
| **Quantum VQE (seed 42)** | 6 equal-weight 2-asset baskets | +106.28% | 46.89% | **+2.1599** |
| Best equal-weight (B=2), exhaustive | **same 6 baskets** | +106.28% | 46.89% | **+2.1599** |
| Markowitz max-Sharpe | continuous simplex | +168.93% | 67.59% | +2.4251 |
| Markowitz min-variance | continuous simplex | +36.52% | 32.40% | +0.9728 |
| Equal weight (all 4) | continuous simplex | +46.21% | 40.14% | +1.0268 |

The quantum result **equals the exhaustive optimum over its own feasible set**.
The gap to continuous Markowitz measures the *expressiveness of the 4-qubit
encoding*, not the optimiser — the encoding can only express 6 portfolios, while
Markowitz optimises over the whole simplex. Note also that max-Sharpe reaches
+2.4251 by collapsing to a single asset (100% AMD), an in-sample corner solution
that is not a diversified benchmark.

### Scope: these results are in-sample

This bounds every number above, so it is stated here rather than left for a
reader to discover.

Expected returns and the covariance matrix are estimated from the **same 250
trading days** that the resulting portfolios are then scored on. There is no
train/test split and no walk-forward validation. A single one-year window also
provides essentially **one observation** of each asset's annualised mean return,
so `mu` carries large estimation error that none of these figures quantify.

**What is being claimed:** that the optimiser can be measured against a *known
ground truth* on a realistic problem instance. The Hamiltonian is diagonal with
16 enumerable eigenvalues, so "did the optimiser find the minimum?" has an exact
answer — and that question is independent of whether the minimum is a good
investment.

**What is not being claimed:** that these are good portfolios, that the reported
Sharpe ratios are achievable out of sample, or that the asset ranking would
survive a different window. In-sample mean-variance optimisation is well known
to overfit, and the 100%-AMD max-Sharpe corner solution above is a textbook
symptom of exactly that.

The optimisation target is a fixed, fully specified mathematical object. Whether
it is the *right* object to optimise is a separate finance question — addressed
in [`docs/backtest.md`](docs/backtest.md) and summarised below. Short answer: on
16 out-of-sample quarters, **the edge this optimiser resolves does not
measurably survive**.

### Reliability — the important caveat

30 runs, identical except for the random seed:

| Quantity | Value |
|---|---|
| Reached the true ground state | **18 / 30 = 60.0%** |
| Wilson 95% interval | [42.3%, 75.4%] |
| Distinct attractors | 2 (`1010` ×18, `0110` ×12) — both feasible |
| Final ⟨H⟩ mean / spread | −4.824163 / 0.576763 |
| Energy spread ÷ degeneracy gap | **14.71×** |

Run-to-run variation is an order of magnitude wider than the 0.039 energy
difference the optimiser must detect, which is why a single run is unreliable.
The two best portfolios differ by only **0.24%** of the spectral width and both
hold AMD — they disagree only on whether the partner is TSLA or NVDA.

**Multi-start fixes most of it**, measured by bootstrapping k-subsets of the 30:

| Seeds run (k) | Keep lowest energy | Oracle (any of k) | Shortfall |
|---|---|---|---|
| 1 | 60.2% | 60.2% | +0.0% |
| 2 | 77.6% | 84.8% | +7.2% |
| 3 | 86.9% | 94.7% | +7.8% |
| 5 | 96.3% | 99.5% | +3.1% |
| 10 | 100.0% | 100.0% | +0.0% |

At ~24 s per run, k = 10 costs about four minutes. Energy-based selection is
near-optimal — its worst shortfall against a perfect selector is 7.8 points at
k = 3, closing to zero by k = 10, and the 13 lowest-energy runs of the 30 are
all correct.

Full analysis, including the seed-123 case study (98.3% confident and wrong) in
[`docs/stability_report.md`](docs/stability_report.md).

### A different objective trades energy for readout

A follow-up controlled study replaced the mean-energy objective with **CVaR**
(minimising the mean of the lowest α-quantile), holding everything else fixed
across the same 30 seeds. The result worth carrying away is not a rate but a
**trade**: CVaR gives up **1.88 in mean energy** — about 11% of the spectral
width, enough that *every* CVaR run fails the convergence criterion *every*
plain-VQE run passes — and buys a large gain in readout accuracy.

| | plain VQE | CVaR (α = 0.5) |
|---|---|---|
| mean plain ⟨H⟩ | −4.905 | −3.042 |
| runs inside the "converged" band | 30/30 | **0/30** |
| readout accuracy | 66.7% | **96.7%** |

This is the sharpest form of a thread running through the project since Phase 5:
energy convergence and a correct readout are different objectives, and here they
are actively traded rather than merely imperfectly correlated.

Accuracy is **non-monotone in α**, peaking at the sampled point α = 0.4
(30 of 30, Wilson 95% [88.6%, 100%] — a point estimate, not a demonstration of
perfect reliability). At the other end α = 0.1 is *worse* than plain VQE (56.7%)
and **abandons the budget constraint entirely**, producing 6 of 30 readouts that
hold 0, 1, or 4 assets instead of the required 2 — because CVaR constrains only
the lowest α-quantile, so the penalty term stops binding on the rest of the
distribution. Two pre-registered predictions, one from each party, were both
refuted.

**This does not change the recommendation** — at k = 5 seeds plain VQE already
reaches 99.59%, and Phase 7 already recommends five seeds. Full study, with the
mechanism and limitations:
[`docs/objective_comparison.md`](docs/objective_comparison.md).

> **No epoch budget tested is converged — measured, not assumed.** Seed 456
> changed basin at epoch **176 of 200**, holding only 24 epochs, so a 250-epoch
> arm was run to check. The rate barely moved (66.7% → 70.0%: **one** discordant
> seed, McNemar exact **p = 1.0000**, Wilson intervals almost entirely
> overlapping) — but a **new** late crosser appeared, seed 1022 at epoch **215 of
> 250**. Extending the budget relocated the tail rather than removing it. Read
> every accuracy figure here as **"at 200 epochs"**, never as converged.

### Out of sample, the edge being resolved does not survive

**A backtest does not test the optimiser. It tests the objective.** VQE's job is
to minimise a given Hamiltonian, and it does. Whether that minimum is a good
*investment* is a separate question — and a weak result here is evidence about
in-sample mean-variance selection, **not** against the quantum method.

Rolling 250-day estimation, 63-day holding, stepping 63 days over 2021-06 →
2026-06: **16 folds**, on a separate frozen snapshot with its own SHA-256.

The discrimination test asked two ways, which **disagree in direction**:

| test | result | 95% CI | p |
|---|---|---|---|
| each fold's ground state beats its own runner-up | **10 / 16 = 62.5%** | [35.4%, 84.8%] | 0.454 |
| `1010` (TSLA+AMD) beats `0110` (NVDA+AMD) | **6 / 16 = 37.5%** | [15.2%, 64.6%] | 0.454 |

One mildly favours the objective, one mildly opposes it, both straddle 50%.
**The shared conclusion is that neither is separable from chance** — a reader
shown only the 6–10 record would wrongly conclude the objective is
anti-predictive.

| | result | p |
|---|---|---|
| Does the *size* of the in-sample edge predict the realised edge? | **ρ = −0.085** (near zero, not meaningfully negative) | 0.753 |
| What did one optimiser error cost? | **+7.4%** per quarter, CI [−6.0%, +20.9%] | 0.258 |

The 1.45% in-sample Sharpe edge that Phase 7's 33% error rate was failing to
resolve is **not measurably worth anything out of sample**. The optimiser works;
the value question lives in the objective it was pointed at.

**The quantum arm's contribution here is a robustness check on Phase 9, not a
backtest result.** CVaR α = 0.4 at k = 1 matched the exact in-sample argmin on
**16 of 16 Hamiltonians it was never tuned against**, with 0 infeasible readouts
and mean P(ground) = **0.4988** — below the 0.5 guarantee threshold, independently
reproducing Phase 9's plurality mechanism. Because it matched the perfect
optimiser on every fold, the backtest contains **no information about the quantum
arm's economic value in either direction**; that comparison has zero variance.

**The main instance is doubly atypical.** Its ground state `1010` wins only 2 of
16 windows, and its energy gap (0.2368% of spectral width) is 5× tighter than the
median fold. Phases 5–9 were measured on a harder-than-typical case — which
strengthens those accuracy figures — while the specific TSLA+AMD answer is a
property of one year, not of this basket.

> **Selection bias is the largest caveat and is not fixable.** These four tickers
> were chosen in 2026 knowing how they turned out. `buy_hold_NVDA` (+1127%,
> Sharpe 1.393) beats every optimised arm including a *perfect* optimiser
> (+800%, 1.233). Only between-arm comparisons carry information, and the design
> is powered to detect effects ~4× larger than the one it looks for — detecting
> the 0.242 Sharpe edge would need **279 folds, about 70 years**.

Full study: [`docs/backtest.md`](docs/backtest.md).

---

## Validation gates

Every phase has an executable gate. Run them in any order after the pipeline.

```bash
python quantum/q_bridge.py                  # Phase 1 — bridge, 10/10
python quantum/q_bridge.py --test-circuit   # Phase 3 — circuit, 7/7
python classical/phase2_gate.py             # Phase 2 — formulation, 9/9
python training/gradient_bridge.py          # Phase 4 — gradients, <1e-9 vs 1e-2
python tools/verify_threading.py            # interpreter thread affinity
python tools/check_reproducibility.py       # determinism across runs
python tools/capture_ui.py                  # UI + figures (needs playwright)
python tools/stability_report.py            # regenerate the stability report
```

Selected measured results:

| Check | Result | Bound |
|---|---|---|
| QUBO→Ising identity, all 16 states | 5.3e-15 | 1e-9 |
| Live Q# circuit vs energy table at `1010` | exact match | 1e-9 |
| Parameter-shift vs finite difference, 28 params † | **< 1e-9** | 1e-2 |
| Pure-sinusoid fit (the rule's premise) | 2.6e-15 | 1e-9 |
| Exact vs sampling backend, 20 000 shots | 0.0012 | 5σ = 0.0164 |
| UI page load, headless Chromium (median of 5) | **~0.9–1.2 s** | 5 s |
| UI page load, first request to a fresh server | 2.6–3.7 s | 5 s |

† **Stated as a bound because it is not a stable constant.** The gate seeds its
probe points, but its finite-difference *reference* inherits the exact backend's
~1.4e-15 nondeterminism (a ~10-ULP global-phase floor, which the gate itself
prints as `<H> spread over 50 evaluations`), amplified by 1/(2ε) = 5e4 at
ε = 1e-5. The order of magnitude is stable but the digits are not: consecutive
runs gave 2.14, 2.14, 2.26, 2.36 e-10. The quantity measures amplified simulator
noise, not gradient disagreement — it sits eight orders of magnitude inside the
bound either way. Quoting three significant figures would assert a precision it
does not have. Full detail, including the other affected rows:
[`docs/mathematical_foundations.md`](docs/mathematical_foundations.md) §8.

The first request to a newly started server carries one-time warmup (module
import, cache fill) and is where essentially all of the 5 s budget risk sits;
under heavy disk contention it has been observed at 3.7 s. Every subsequent load
is around a second. `tools/capture_ui.py` measures five fresh browser contexts
and gates on the **worst**, not the median.

---

## Project layout

```
├── quantum/          circuit.qs, test_bridge.qs, q_bridge.py
├── classical/        data_pipeline.py, hamiltonian.py, markowitz.py, phase2_gate.py
├── training/         gradient_bridge.py, optimization.py, trained_seed*.json
├── ui/               app.py
├── tools/            verify_threading.py, capture_ui.py,
│                     check_reproducibility.py, stability_report.py
├── data/             prices.csv (frozen), statistics.json, hamiltonian.json,
│                     markowitz_baseline.json
├── docs/             architecture.md, encoding_schema.md,
│                     mathematical_foundations.md, stability_report.md, figures/
├── logs/             per-phase logs and JSON reports
├── requirements.txt
└── requirements-lock.txt
```

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System diagram, module boundaries, the single-owner-thread interpreter design and its serialization limit, interpreter recycling, deviations from spec |
| [`docs/mathematical_foundations.md`](docs/mathematical_foundations.md) | Hamiltonian derivation, penalty selection, parameter-shift rule, the exact-vs-sampling backend distinction, rf = 5% convention |
| [`docs/encoding_schema.md`](docs/encoding_schema.md) | Bitstring → weights mapping, bit ordering, all 6 reachable portfolios, worked examples, decode/repair policies |
| [`docs/stability_report.md`](docs/stability_report.md) | 3-seed variance analysis and reliability findings |

---

## Reproducibility

Fixing random seeds is **not sufficient** — the data has to be frozen too.

Yahoo does not return byte-identical data for a fixed, entirely historical
window. Two consecutive fetches of the same range differed, with NVDA's
annualised mean moving 1.0e-9 and its covariance entries up to 3.7e-8 (all other
tickers unchanged), from vendor-side revision of adjustment factors. Financially
irrelevant, but it propagates into the Ising coefficients, so a re-fetch silently
changes the problem the saved parameters were trained against.

`data/prices.csv` is therefore a **frozen snapshot**, reused unless you ask
otherwise, with its SHA-256 recorded in `data/statistics.json`:

```bash
python classical/data_pipeline.py --refresh
```

That refetches and replaces the snapshot; downstream results shift slightly and
you should retrain. With the snapshot frozen, the pipeline is byte-identical
across runs — verified by `tools/check_reproducibility.py`. Seeds are fixed for
`torch`, `numpy`, and the Q# simulator; the exact backend is deterministic
regardless.

---

## Known issues

### Exit-time `unsendable ... dropped on another thread` messages

On process exit you may see a handful of lines like:

```
RuntimeError: qdk::interpreter::GlobalCallable is unsendable, but is being dropped on another thread
RuntimeError: qdk::interpreter::Interpreter is unsendable, but is being dropped on another thread
```

**Nothing has failed.** `qdk` retains `GlobalCallable` references for each loaded
Q# operation in internal state this project cannot reach, and CPython frees them
on the main thread during finalisation rather than on the interpreter's owner
thread. The messages fire **after all work has completed**, are printed by the
Rust layer rather than raised into Python, and **affect no result**. Count is
roughly one per loaded Q# operation plus one for the interpreter.

Mitigated as far as is possible from outside the library: an `atexit` handler
releases this project's own references on the owner thread. Fully suppressing
them would require a change inside `qdk`.

Distinguish these from the fatal variant, `unsendable, but **sent** to another
thread`, which is a genuine panic. `tools/capture_ui.py` scans the server log
specifically for that string.

### Quantum work is serialized across sessions

All circuit evaluation runs on one owner thread (see
[`docs/architecture.md`](docs/architecture.md) §4), so concurrent Streamlit
sessions queue rather than execute in parallel. A readout is ~0.6 s, so a few
simultaneous users still stay inside the 5 s budget, but genuine multi-user
deployment needs process-level parallelism — one interpreter per worker process.
Threads cannot help; the constraint is a pyo3 `unsendable` object.

### The budget penalty is not binding on this dataset

The unpenalised optimum already holds exactly 2 assets, so `lambda` does not
change the answer here. It is still correctly enforced in the energy landscape
(best infeasible +2.414 vs best feasible −1.246), but this particular year of
data does not stress the constraint.

### Deprecation warning from `import qsharp`

`qsharp` 1.30.0 is a compatibility shim that emits a `DeprecationWarning`
pointing at `qdk`. This project imports `from qdk import qsharp` and does not
trigger it; the `qsharp` pin exists only because the two packages depend on each
other and must stay in lockstep.

---

## Limitations

- **4 assets, 2 held, equal weights** — 6 reachable portfolios. No fractional
  allocations, no shorting. See
  [`docs/encoding_schema.md`](docs/encoding_schema.md) §7 for scaling routes.
- **Simulated, not hardware.** Exact state-vector simulation for training;
  sampling for readout. No noise model, no hardware queue, no error mitigation.
- **In-sample only.** All statistics are estimated on the same year they are
  evaluated on. No train/test split, no walk-forward validation. The reported
  Sharpe ratios are not out-of-sample performance estimates.
- **No claim of quantum advantage.** Brute force solves this problem in 16
  evaluations, instantly. The point of the PoC is a correct, verified hybrid
  pipeline, not speedup.
- **Three seeds** is a coarse reliability estimate; 1-in-3 should not be read as
  a 33% success probability.

## Not investment advice

This is an engineering proof of concept using in-sample historical data. Nothing
here is a recommendation to buy or sell any security.
