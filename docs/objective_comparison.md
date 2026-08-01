# Objective Comparison: CVaR vs Plain VQE

A controlled study of whether changing the *objective* improves the variational
optimiser's reliability on this problem — run as a second arm against the
Phase 7 measurement, not as a replacement for it.

[`stability_report.md`](stability_report.md) documents the plain-VQE baseline as
measured and is deliberately unchanged.

---

## 1. Summary

Plain VQE finds the true ground state in **66.7%** of runs. Replacing the mean
energy with a **CVaR** objective at **α = 0.4** raises this to **100%**
(30/30, McNemar exact p = 0.0020), and the improvement survives shot noise.

But the effect is **non-monotone in α**, and the mechanism is not the one either
party predicted in advance. At α = 0.1 the objective performs *worse than plain
VQE* (56.7%) and produces readouts that violate the budget constraint.

**This does not change the project's recommendation.** At k = 5 seeds plain VQE
already reaches 99.59%, and Phase 7 already recommends five seeds. The
contribution here is mechanistic — how the objective shapes the *readout* — not
operational.

---

## 2. Experimental design

Every arm is identical except for the objective:

| | |
|---|---|
| seeds | 42, 123, 456, 1000–1026 (**the same 30**, so comparisons are paired) |
| epochs | 200 |
| optimiser | Adam, constant lr = 0.05 |
| ansatz | 3 layers, 28 parameters, init U(−0.1, 0.1) |
| data | frozen snapshot `data/prices.csv` |
| backend | exact state vector |
| baseline | `lr_constant_200` (= CVaR at α = 1.0) |

Paired seeds mean **McNemar's test on the discordant pairs**, using the **exact
binomial** rather than the chi-square approximation — discordant counts here are
single digits, where the asymptotic version is unreliable.

### The objective

With states sorted by ascending energy and `m` the first index at which the
cumulative probability reaches α:

```
CVaR_α = (1/α) [ Σ_{j<m} p_j E_j + (α − P_{m−1}) E_m ]
```

Both value and gradient reduce to those of an ordinary **diagonal** observable
with weights `w_j = (E_j − E_m)/α` inside the tail and 0 outside:

```
CVaR_α = ⟨O_w⟩ + E_m        d(CVaR)/dθ = d(⟨O_w⟩)/dθ
```

so the parameter-shift rule applies unchanged — **one pass of 2n evaluations per
step**, the same cost as plain VQE, not 2ⁿ projector gradients.

Validated before any sweep ran (`tools/verify_cvar_gradient.py`):

| check | result |
|---|---|
| value identity vs definition | 3.55e-15 |
| gradient identity vs finite differences | 2.68e-09 |
| α = 1 reduces to plain VQE (value) | 7.77e-16 |
| α = 1 reduces to plain VQE (gradient) | 5.55e-15 |
| degenerate tail gives exactly zero gradient | 0.0, 6/6 cases |

> **Implementation note.** `⟨O_w⟩` alone is always ≤ 0 (every weight is negative)
> while CVaR sits near −4.9. Omitting the `E_m` constant yields a quantity that
> *trains identically* — the gradient is unaffected — while plotting a
> meaningless loss curve. Measured at one point: `⟨O_w⟩ = −1.607` vs
> `CVaR = −4.217`.

---

## 3. Pre-registered predictions

Both were recorded **before** any arm was run, and both failed.

| | prediction | outcome |
|---|---|---|
| **User** | Non-monotone, **peak at α = 0.5**, with a discontinuity between 0.4 and 0.6. Rationale: P(ground) ≥ 0.5 forces the ground state to be the argmax, so 0.5 is the weakest target that still *guarantees* a correct readout, pursued with 2× plain VQE's gradient pressure. | **Shape confirmed** (interior peak), **location and mechanism refuted**. Peak is at 0.4; there is no cliff. |
| **Claude** | Monotone increasing in α, best at α = 1.0. Rationale: small α halts under-concentrated, leaving the readout to whatever else is larger. | **Refuted.** The ordering reverses over [0.4, 1.0]. |

Keeping both on the record is deliberate: a pre-registered disagreement settled
by data is worth more than either party being right quietly.

---

## 4. Results

| α | exact accuracy | **E[shot acc] @4096** | mean rank | worst rank | min margin† | mean plain ⟨H⟩ | McNemar p |
|---|---|---|---|---|---|---|---|
| 0.1 | 56.7% | **54.2%** | 1.567 | 3 | 0.0051 | −0.0771 | 0.6291 |
| **0.4** | **100.0%** | **100.0%** | **1.000** | **1** | **0.1810** | −2.7327 | **0.0020** |
| 0.5 | 96.7% | 95.5% | 1.033 | 2 | 0.0053 | −3.0423 | 0.0039 |
| 0.6 | 86.7% | 86.7% | 1.267 | 4 | 0.1249 | −3.3496 | 0.0703 |
| 1.0 (plain) | 66.7% | 66.3% | 3.033 | 14 | 0.0023 | −4.9052 | — |

† minimum top-two margin among *correct* runs.

**E[shot acc]** is the mean over runs of P(sampled argmax = ground state), from
1000 multinomial draws of 4096 shots per run, competing against **all 16
outcomes** with ties shared. It is stricter than a pairwise margin and requires
no cutoff to be chosen — the figure that would transfer to hardware.

### Paired transitions

| α | recovered | lost | discordant | p |
|---|---|---|---|---|
| 0.4 | 10 | 0 | 10 | 0.0020 |
| 0.5 | 9 | 0 | 9 | 0.0039 |
| 0.6 | 7 | **1** (seed 1006) | 8 | 0.0703 |
| 0.1 | 7 | **10** | 17 | 0.6291 |

---

## 5. The mechanism

### The argmax guarantee never binds

`P(ground) ≥ 0.5 ⟹ argmax` is **sufficient but not necessary**. What a correct
readout actually requires is a **plurality**, and CVaR fragments the losing mass
across fifteen states, so a much smaller share wins.

The decisive evidence is at α = 0.4:

> **Only 2 of 30 runs reach P(ground) ≥ 0.5 — yet all 30 read out correctly.**
> Median P(ground) is 0.4388.

So the threshold that motivated α = 0.5 is irrelevant to the outcome.

### What actually drives it: 1/α amplification vs the saturation target

Two opposing forces, both consequences of the definition:

- **Gradient amplification.** Tail weights carry `1/α`. At α = 0.4 the pressure
  on the 0110→1010 discrimination is `(E_g − E_b)/0.4 = 2.5×` plain VQE's;
  at 0.5 it is 2.0×; at 1.0 it is 1.0×. **Smaller α is stronger.**
- **The saturation target.** `CVaR_α` is *exactly minimised* once
  `P(ground) ≥ α`, after which the gradient vanishes identically and training
  halts. **Smaller α stops earlier**, at a flatter distribution.

They balance near **α = 0.4**. Above it the gradient is too weak to pull runs
across; below it the run stops before acquiring any margin.

### Progressive loss of concentration

Two-state mass `P(1010) + P(0110)` collapses monotonically as α falls, because
CVaR is indifferent to how the upper `1−α` of the distribution is arranged:

| arm | median two-state mass | runs below 0.5 |
|---|---|---|
| plain | 0.9998 | 0 |
| α = 0.5 | 0.6298 | 0 |
| α = 0.4 | 0.5375 | 7 |
| α = 0.1 | 0.2814 | **30** |

Plain VQE collapses into the two lowest-energy states; CVaR does not, and has no
reason to. This is why energy stops being a proxy for correctness (§6).

---

## 6. Three distinct failure modes

The rank of the true ground state in each run's final distribution separates
them — success counting alone cannot.

| arm | rank 1 | 2 | 3 | 4 | 6 | 7 | 8 | 11 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 20 | 3 | | | 2 | 1 | 1 | 1 | 1 | 1 |
| α = 0.6 | 26 | 1 | 2 | 1 | | | | | | |
| α = 0.5 | 29 | 1 | | | | | | | | |
| α = 0.4 | **30** | | | | | | | | | |
| α = 0.1 | 17 | 9 | 4 | | | | | | | |

**1. Plain VQE — wrong basin.** Half its failures put the ground state at rank
6–14: those runs are not near-missing, they are in the wrong region entirely.

**2. Large α — insufficient pressure.** Failures cluster at ranks 2–4. Seed 1006
is the phase's only regression: correct under plain VQE at P = 0.5892, but at
α = 0.6 the ground state falls to **rank 2 (P = 0.1539)** while `0110` holds
**0.5150** — a decisive move to the wrong basin, not a thin miss. It is correct
again at α = 0.5 and 0.4, so it is a fringe seed that α = 0.6's weaker gradient
failed to pull across.

**3. Small α — saturation and flatness.** At α = 0.1 runs spend a median of
**185 of 200 epochs** with a degenerate tail (`m = 0`, gradient identically
zero), halting at P(ground) ≈ 0.17 with a median maximum probability of **0.19**
— a nearly uniform distribution. The ground state still ranks 1–3 in *every* run,
but the median margin is **0.031**, so only **6 of 30** readouts survive 4096
shots against 17 exact.

> **α = 0.1 also breaks feasibility.** It produces **7 distinct attractors**
> including `0000`, `1111` and `0001` — **6 runs reading out portfolios that
> violate the budget constraint** (0, 4 and 1 assets held). With only the lowest
> 10% of the distribution constrained, the remaining 90% drifts to high-energy
> infeasible states; mean plain ⟨H⟩ is **−0.077** against a ground energy of
> −4.921. No other arm produces an infeasible readout.

This confirms the failure mode derived from the definition *before* the arm was
run, and it is why α = 0.1 was kept in the sweep despite the shape already being
established.

---

## 7. Two criteria, and why the ranking depends on which is applied

On hardware, shot noise is the binding constraint: an exact-backend accuracy that
does not survive sampling is not an accuracy anyone can use.

| arm | exact | resolvable @4096 | runs with shot acc < 0.9 | < 0.5 |
|---|---|---|---|---|
| plain | 66.7% | 66.3% | 11 | 10 |
| α = 0.6 | 86.7% | 86.7% | 4 | 4 |
| α = 0.5 | 96.7% | **95.5%** | 2 | 1 |
| α = 0.4 | 100.0% | **100.0%** | 0 | 0 |
| α = 0.1 | 56.7% | **54.2%** | 18 | 13 |

α = 0.4 is best under both criteria, so here they agree — but they do not in
general. **Seed 1014** at α = 0.5 is the clearest case: margin **0.0053**
(0.4 σ), counted as correct by the exact backend but a **coin flip** under
sampling (shot accuracy 0.659). At α = 0.4 the same seed resolves completely
(margin 0.2267, shot accuracy 1.000).

> **Plain VQE's manifold advantage should not be overclaimed.** Its own minimum
> margin is **0.0023** — *worse* than α = 0.4's and α = 0.6's. Two-state collapse
> gives near-certain readouts *typically*, but a run halting at P ≈ 0.5 has no
> margin at all. The same fragility, just rarer.

---

## 8. Which runs recover — α changes the identity of the successes

| set | seeds |
|---|---|
| **stable core** (recovered by every CVaR arm) | 123, 1008, 1011, 1017, 1022, 1025 |
| **fringe** (α-dependent) | 1004, 1014, 1023 (α ≤ 0.5); 1021 (α = 0.6, 0.4) |
| **regression** | 1006 (α = 0.6 only) |

Core recoveries carry roughly **twice** the fringe's readout margin at α ≥ 0.5
(α = 0.6: 0.5505 vs 0.2554; α = 0.5: 0.4359 vs 0.2910). At α = 0.4 the gap closes
(0.2734 vs 0.2628), consistent with everything having been pushed clear of the
boundary rather than a fringe barely making it.

So α does not merely scale a success rate — it changes *which* runs succeed. A
stable core recovers everywhere with solid margins; a fringe flips with α at thin
margins.

---

## 9. Energy is not comparable across arms

CVaR never optimises the mean, so cross-arm energy comparison is **meaningless**:

| arm | mean plain ⟨H⟩ | spread ÷ gap |
|---|---|---|
| plain | −4.9052 | 1.08× |
| α = 0.6 | −3.3496 | 59.76× |
| α = 0.5 | −3.0423 | 52.69× |
| α = 0.4 | −2.7327 | ~48× |
| α = 0.1 | −0.0771 | ~93× |

All CVaR runs fall outside the 0.5%-of-spectral-width "converged" criterion used
in Phase 7 — a criterion that is simply not meaningful for an objective that
never targets the mean. The arms operate in a **different regime**, so this is
not a like-for-like rate improvement. The honest statement is:

> **CVaR optimises the readout while abandoning the energy.**

That is the sharpest instance of a thread running through this whole project:
improving the readout and improving the energy are different objectives, and
optimiser diagnostics based on energy can be actively misleading.

---

## 10. Multi-start, and why the recommendation is unchanged

| k | plain | α = 0.6 | α = 0.5 | α = 0.4 |
|---|---|---|---|---|
| 1 | 66.67% | 86.67% | 96.67% | **100.00%** |
| 2 | 88.89% | 94.44% | 99.67% | 100.00% |
| 3 | 96.30% | 96.96% | 99.97% | 100.00% |
| 5 | **99.59%** | 99.02% | 100.00% | 100.00% |
| 10 | 100.00% | 99.95% | 100.00% | 100.00% |

At **k = 5 plain VQE (99.59%) edges out α = 0.6 (99.02%)**. CVaR's advantage is
concentrated at **small seed budgets** — decisive at k = 1, largely gone by k = 5.

Since Phase 7 already recommends running five seeds, **this study does not change
what the demo should do**. Its value is the mechanism, and one practical caveat:
α = 0.4 reaches 100% at **k = 1**, which matters when runs are expensive — a
regime this project is not in, but a real one on hardware.

---

## 11. Limitations

- **n = 30 per arm.** α = 0.4 at 30/30 gives a Wilson 95% interval of
  [88.6%, 100%]; a finer α grid around 0.3–0.45 could not resolve the peak
  without more seeds.
- **200 epochs is not converged.** Seed 456 changed basin at epoch 176 of 200 and
  held for only 24 epochs, so a longer run could move the plain-VQE baseline
  again. The 66.7% should not be read as a converged figure.
- **Exact backend.** Training uses exact state vectors; the shot-limited column
  simulates sampling at readout only, not sampled *gradients*, which would add
  noise throughout training.
- **One problem instance.** All of this is a single 4-asset Hamiltonian with one
  near-degenerate pair. The optimal α almost certainly depends on the spectrum,
  and 0.4 should not be treated as a general recommendation.
- **In-sample.** Inherited from the whole project — see
  [`stability_report.md`](stability_report.md) §Scope.

---

## 12. Reproducing

```bash
python tools/verify_cvar_gradient.py
```

```bash
python training/optimization.py --seeds 42 123 456 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 --epochs 200 --schedule constant --cvar-alpha 0.4 --tag cvar_a040
```

```bash
python tools/compare_arms.py --arm cvar_a040 --baseline lr_constant_200
```

```bash
python tools/rank_analysis.py
```

Per-arm summaries are committed under `training/arms/*/training_summary.json`;
the parameter files are regenerable and not tracked.
