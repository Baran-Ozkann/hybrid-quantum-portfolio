# Objective Comparison: CVaR vs Plain VQE

A controlled study of whether changing the *objective* improves the variational
optimiser's reliability — run as a second arm against the Phase 7 measurement,
not as a replacement for it.
[`stability_report.md`](stability_report.md) documents the plain-VQE baseline as
measured and is deliberately unchanged.

---

## 1. The finding: CVaR trades energy for readout

The transferable result of this study is **not** a success rate. It is that the
two things this project has been treating as proxies for each other can be
**actively traded against one another**:

| | plain VQE | CVaR (α = 0.5) |
|---|---|---|
| mean plain ⟨H⟩ | **−4.905** | **−3.042** |
| distance above ground energy | 0.016 | 1.879 |
| ...in units of the degeneracy gap (0.039) | **0.4×** | **48×** |
| runs inside the 0.5%-of-spectrum "converged" band | 30 / 30 | **0 / 30** |
| readout accuracy | 66.7% | **96.7%** |

The gap-normalised row is the one that matters. The whole difficulty of this
problem is resolving a **0.039** energy difference between the two best
portfolios; plain VQE finishes **0.4×** that gap above the ground state, well
inside the scale it must discriminate at, while CVaR finishes **48×** it — two
orders of magnitude outside. CVaR gives up 1.88 in mean energy, roughly 11% of
the spectral width, and buys a 30-point improvement in readout accuracy. Every
CVaR run fails the convergence criterion that every plain-VQE run passes, while
getting the answer right far more often.

Since Phase 5 this project has repeatedly found that energy convergence and a
correct readout are *imperfectly correlated*: a run can be confidently wrong
(seed 123), energy ranking can misorder correctness (Phase 7), and the two
coincide only when the state collapses into the two lowest-energy basis states.
This study is the sharpest version of that thread. Here they are not merely
imperfectly correlated — **optimising one deliberately sacrifices the other**.

The mechanism is that CVaR only constrains the lowest α-quantile of the energy
distribution. It has no reason to arrange the remaining `1 − α`, so mass
fragments, mean energy degrades, and the criteria built around mean energy stop
carrying information. What it *does* concentrate is the bottom of the
distribution — which is precisely what a readout reads.

The α sweep in §4 is the evidence for this trade, and §5 explains the shape.

---

## 2. Experimental design

Every arm is identical except the objective:

| | |
|---|---|
| seeds | 42, 123, 456, 1000–1026 (**the same 30**, so comparisons are paired) |
| epochs | 200 |
| optimiser | Adam, constant lr = 0.05 |
| ansatz | 3 layers, 28 parameters, init U(−0.1, 0.1) |
| data | frozen snapshot `data/prices.csv` |
| backend | exact state vector |
| baseline | `lr_constant_200` (= CVaR at α = 1.0) |

Shared seeds mean **McNemar's test on the discordant pairs**, using the **exact
binomial** rather than the chi-square approximation — discordant counts here are
single digits, where the asymptotic version is unreliable.

### The objective and its identity

With states sorted by ascending energy and `m` the first index at which the
cumulative probability reaches α:

```
CVaR_α = (1/α) [ Σ_{j<m} p_j E_j + (α − P_{m−1}) E_m ]
```

Both value and gradient reduce to those of an ordinary **diagonal** observable
with weights `w_j = (E_j − E_m)/α` inside the tail, 0 outside:

```
CVaR_α = ⟨O_w⟩ + E_m          d(CVaR)/dθ = d(⟨O_w⟩)/dθ
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

> **Implementation note.** `⟨O_w⟩` alone is always ≤ 0 (all weights negative)
> while CVaR sits near −4.9. Omitting the `E_m` constant yields a quantity that
> *trains identically* — the gradient is unaffected — while plotting a
> meaningless loss curve. Measured at one point: `⟨O_w⟩ = −1.607` vs
> `CVaR = −4.217`.

---

## 3. Pre-registered predictions

Both were recorded **before** any arm ran, and both failed.

| | prediction | outcome |
|---|---|---|
| **User** | Non-monotone, **peak at α = 0.5**, with a discontinuity between 0.4 and 0.6. Rationale: P(ground) ≥ 0.5 forces the ground state to be the argmax, so 0.5 is the weakest target that still *guarantees* a correct readout, pursued with 2× the gradient pressure. | **Shape confirmed** (interior peak). **Location and mechanism refuted** — no cliff, and the guarantee never binds. |
| **Claude** | Monotone increasing in α, best at α = 1.0. Rationale: small α halts under-concentrated, leaving the readout to whatever else is larger. | **Refuted.** The ordering reverses over [0.4, 1.0]. |

Keeping both on the record is deliberate: a pre-registered disagreement settled
by data is worth more than either party being right quietly.

---

## 4. Results

All rates are **30 seeds at one configuration**. Wilson 95% intervals are given
because point estimates at this sample size have repeatedly proven unreliable in
this project (§8).

| α | correct | rate | **Wilson 95%** | E[shot acc] @4096 | mean rank | worst rank | min margin† | mean ⟨H⟩ | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 17/30 | 56.7% | [39.2%, 72.6%] | 54.2% | 1.567 | 3 | 0.0051 | −0.077 | 0.6291 |
| **0.4** | **30/30** | **100.0%** | **[88.6%, 100%]** | 100.0% | 1.000 | 1 | 0.1810 | −2.733 | **0.0020** |
| 0.5 | 29/30 | 96.7% | [83.3%, 99.4%] | 95.5% | 1.033 | 2 | 0.0053 | −3.042 | 0.0039 |
| 0.6 | 26/30 | 86.7% | [70.3%, 94.7%] | 86.7% | 1.267 | 4 | 0.1249 | −3.350 | 0.0703 |
| 1.0 (plain) | 20/30 | 66.7% | [48.8%, 80.8%] | 66.3% | 3.033 | 14 | 0.0023 | −4.905 | — |

† minimum top-two margin among *correct* runs.

**On the 30/30 at α = 0.4.** This is a point estimate, not a demonstration of
perfect reliability. The Wilson 95% interval is **[88.6%, 100%]**, so a true rate
near 90% is entirely consistent with the observation — roughly one failure in ten
runs. It should be read as *"30 of 30 on this frozen instance, at 3 layers, at
200 epochs"* and nothing broader.

**E[shot acc]** is the mean over runs of P(sampled argmax = ground state), from
1000 multinomial draws of 4096 shots per run, competing against **all 16
outcomes** with ties shared. It is stricter than a pairwise margin and needs no
cutoff chosen.

### Paired transitions

| α | recovered | lost | discordant | p |
|---|---|---|---|---|
| 0.4 | 10 | 0 | 10 | 0.0020 |
| 0.5 | 9 | 0 | 9 | 0.0039 |
| 0.6 | 7 | **1** (seed 1006) | 8 | 0.0703 |
| 0.1 | 7 | **10** | 17 | 0.6291 |

---

## 5. Why the curve has this shape

### The argmax guarantee never binds

`P(ground) ≥ 0.5 ⟹ argmax` is **sufficient but not necessary**. A correct readout
requires only a **plurality**, and CVaR fragments the losing mass across fifteen
states, so a much smaller share suffices.

The decisive evidence is at α = 0.4:

> **Only 2 of 30 runs reach P(ground) ≥ 0.5 — yet all 30 read out correctly.**
> Median P(ground) is 0.4388.

The threshold that motivated α = 0.5 is therefore irrelevant to the outcome.

### Two opposing forces, both from the definition

- **Gradient amplification.** Tail weights carry `1/α`. At α = 0.4 the pressure
  on the 0110→1010 discrimination is `(E_g − E_b)/0.4 = 2.5×` plain VQE's; at 0.5
  it is 2.0×; at 1.0 it is 1.0×. **Smaller α is stronger.**
- **The saturation target.** `CVaR_α` is *exactly minimised* once
  `P(ground) ≥ α`, after which the gradient vanishes identically and training
  halts. **Smaller α stops earlier**, at a flatter distribution.

Accuracy peaks where these balance. On this five-point grid that is the sampled
value **α = 0.4** — see §8 on what that does and does not establish.

### Progressive loss of concentration

Two-state mass `P(1010) + P(0110)` falls monotonically as α falls, because CVaR
is indifferent to how the upper `1 − α` is arranged:

| arm | median two-state mass | runs below 0.5 |
|---|---|---|
| plain | 0.9998 | 0 |
| α = 0.5 | 0.6298 | 0 |
| α = 0.4 | 0.5375 | 7 |
| α = 0.1 | 0.2814 | **30** |

Plain VQE collapses into the two lowest-energy states; CVaR does not, and has no
reason to. This is why energy stops being a proxy for correctness — and, at small
α, why feasibility stops being protected (§7).

---

## 6. Three failure modes

The rank of the true ground state in each run's final distribution separates
them; success counting alone cannot.

| arm | rank 1 | 2 | 3 | 4 | 6 | 7 | 8 | 11 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | 20 | 3 | | | 2 | 1 | 1 | 1 | 1 | 1 |
| α = 0.6 | 26 | 1 | 2 | 1 | | | | | | |
| α = 0.5 | 29 | 1 | | | | | | | | |
| α = 0.4 | **30** | | | | | | | | | |
| α = 0.1 | 17 | 9 | 4 | | | | | | | |

**1. Plain VQE — wrong basin.** Half its failures put the ground state at rank
6–14: not near misses, but runs in the wrong region entirely.

**2. Large α — insufficient pressure.** Failures cluster at ranks 2–4. Seed 1006
is the phase's only regression: correct under plain VQE at P = 0.5892, but at
α = 0.6 the ground state falls to **rank 2 (P = 0.1539)** while `0110` holds
**0.5150** — a decisive move to the wrong basin, not a thin miss. Correct again
at α = 0.5 and 0.4, so it is a fringe seed that α = 0.6's weaker gradient failed
to pull across.

**3. Small α — saturation and flatness.** At α = 0.1 runs spend a median of
**185 of 200 epochs** with a degenerate tail (`m = 0`, gradient identically
zero), halting at P(ground) ≈ 0.17 with a median maximum probability of **0.19**
— nearly uniform. The ground state still ranks 1–3 in *every* run, but the median
margin is **0.031**, so only **6 of 30** readouts survive 4096 shots against 17
exact.

---

## 7. Small α abandons the budget constraint

This is a distinct failure from flatness and was **not anticipated by the
derivation**, which predicted only that small α would stop early at a poorly
concentrated distribution.

At α = 0.1 the optimiser produces **7 distinct attractors**, and three of them
are **infeasible** — they violate the "hold exactly 2 of 4 assets" constraint the
Hamiltonian encodes:

| readout | assets held | runs | feasible? |
|---|---|---|---|
| `1010` | 2 | 17 | yes |
| `0011` | 2 | 4 | yes |
| `0001` | **1** | 3 | **no** |
| `0000` | **0** | 2 | **no** |
| `0110` | 2 | 2 | yes |
| `1001` | 2 | 1 | yes |
| `1111` | **4** | 1 | **no** |

**6 of 30 runs read out a portfolio that is not a legal solution.** Every other
arm — plain VQE, α = 0.4, 0.5, 0.6 — produces **100% feasible readouts**.

The infeasibility is the most striking part, but it sits inside a broader
collapse: α = 0.1 scatters across **seven distinct attractors**, of which three
are infeasible and four are *feasible but wrong* (`0011` ×4, `0110` ×2,
`1001` ×1), against **at most two attractors in every other arm**. So the effect
is not only "the constraint stopped binding" — it is a general loss of structure,
with the distribution dispersing across the whole space rather than concentrating
anywhere. The penalty ceasing to bind is the mechanism; the observable outcome is
wider than that mechanism alone would suggest.

### Why

The budget constraint is enforced by a penalty term folded into the Ising
coefficients (see
[`mathematical_foundations.md`](mathematical_foundations.md) §2), which raises
the energy of infeasible states. That penalty only bites through the objective.
CVaR at α constrains **only the lowest α-quantile** of the energy distribution
and assigns zero weight to everything above it — so at α = 0.1, 90% of the
probability mass is unconstrained and free to drift onto high-energy infeasible
states. Mean plain ⟨H⟩ is **−0.077** against a ground energy of −4.921: the
distribution has drifted almost to the middle of the spectrum.

### Practical implication

> **α cannot be set below the probability mass the feasible subspace needs to
> retain.** A penalty-encoded constraint is only enforced on the part of the
> distribution the objective actually weights, so as α shrinks the constraint
> silently stops applying to the rest.

This is a genuine addition to the theory rather than a footnote to the saturation
failure. It also gives a *different* lower bound on α from the one derived in
gate 1: that bound (α > 1/2ⁿ, else the tail collapses to a single state and the
gradient vanishes) is about **optimisability**; this one is about **validity of
the solution**, and on this instance it is the binding constraint — α = 0.1 sits
well above 1/16 = 0.0625 yet still produces illegal portfolios.

Any application of CVaR to a penalty-constrained problem should check feasibility
of the readouts, not only their objective value.

---

## 8. Limitations

Two conclusions in this project have already been overturned by more data — a
1-of-3 success rate became 18-of-30, and a clean null in Priority 2 turned out to
be two effects cancelling. The following are stated so a third instance would be
avoidable rather than unlucky.

**The 30/30 is a point estimate.** Wilson 95% is **[88.6%, 100%]**; a true rate
near 90% is fully consistent with it. It is not evidence of perfect reliability.

**The optimum's location is not resolved.** Accuracy peaks at the *sampled point*
α = 0.4 on a five-point grid {0.1, 0.4, 0.5, 0.6, 1.0}. The true optimum lies
somewhere **between 0.1 and 0.5**, and this design cannot narrow it further: with
0.4 already at ceiling on 30 seeds, a finer grid would need more *seeds* to
distinguish neighbouring α values, not more α values. **"α = 0.4 is optimal" is
not a claim this study supports.**

**Scope of the result.** Everything here holds for *this* frozen instance, *this*
ansatz (3 layers, 28 parameters), and *200 epochs*. This study also inherits the
whole scope of [`stability_report.md`](stability_report.md): a **single problem
instance**, four assets, sixteen basis states, in-sample statistics. Nothing here
speaks to whether the α ordering survives on a different Hamiltonian.

**A testable prediction, stated as such rather than as a caveat.** The mechanism
in §5 is that fragmentation makes a *plurality* easy to achieve, so the argmax
threshold never binds. That should be **sensitive to the number of states**: with
2ⁿ competitors sharing the losing mass, the share a plurality needs falls roughly
as 1/2ⁿ, so larger problems should tolerate *smaller* α before the readout
degrades. Pulling the other way, §7's feasibility bound depends on how much mass
the feasible subspace must retain, and the feasible fraction — C(n, B)/2ⁿ —
*shrinks* with n, which would push the usable α *up*. Which effect dominates is
an open question this design cannot answer, and it is the single most useful
follow-up: run the same sweep at 6 and 8 assets and see whether the optimal α
moves down (plurality dominates) or up (feasibility dominates).

**No epoch budget tested is converged — and this was measured rather than left
as a worry.** Seed 456 changed basin at epoch **176 of 200**, holding only 24
epochs, so a 250-epoch arm (`lr_constant_250`) was run to test whether the
comparison target moves.

| | 200 epochs | 250 epochs |
|---|---|---|
| correct | 20 / 30 = 66.7% | 21 / 30 = 70.0% |
| Wilson 95% | [48.8%, 80.8%] | [52.1%, 83.3%] |
| discordant seeds | — | **1** (1022 recovered, none lost) |
| McNemar exact p | — | **1.0000** |
| latest readout change | seed 456 @ **176 / 200** (held 24) | seed 1022 @ **215 / 250** (held 35) |

**The comparison target does not meaningfully move** — one seed, p = 1.0, heavily
overlapping intervals — so every α arm above remains paired against 66.7% and
none of the McNemar results change.

**But the late-crossing tail persists at the longer budget.** Extending 200 → 250
did not remove the run that settles near the end; it produced a different one.
So 250 is no more converged than 200, and a 300-epoch arm would likely find a
third. Every figure in this document should be read as **"at 200 epochs"**.
Verify with `python tools/settle_analysis.py --arm <arm> --baseline <shorter>`.

**Exact backend.** Training uses exact state vectors. The shot-limited column
simulates sampling *at readout only*; sampled **gradients** would add noise
throughout training and are not modelled.

**In-sample.** Inherited from the whole project — see
[`stability_report.md`](stability_report.md) §Scope.

---

## 9. Which runs recover — α changes the identity of the successes

| set | seeds |
|---|---|
| **stable core** (recovered by every CVaR arm) | 123, 1008, 1011, 1017, 1022, 1025 |
| **fringe** (α-dependent) | 1004, 1014, 1023 (α ≤ 0.5); 1021 (α = 0.6, 0.4) |
| **regression** | 1006 (α = 0.6 only) |

Core recoveries carry roughly **twice** the fringe's readout margin at α ≥ 0.5
(α = 0.6: 0.5505 vs 0.2554; α = 0.5: 0.4359 vs 0.2910). At α = 0.4 the gap closes
(0.2734 vs 0.2628), consistent with everything having been pushed clear of the
boundary rather than a fringe barely making it.

So α does not merely scale a success rate — it changes *which* runs succeed.

---

## 10. Two accuracy criteria

On hardware shot noise binds: an exact-backend accuracy that does not survive
sampling is not one anyone can use.

| arm | exact | resolvable @4096 | shot acc < 0.9 | < 0.5 |
|---|---|---|---|---|
| plain | 66.7% | 66.3% | 11 | 10 |
| α = 0.6 | 86.7% | 86.7% | 4 | 4 |
| α = 0.5 | 96.7% | 95.5% | 2 | 1 |
| α = 0.4 | 100.0% | 100.0% | 0 | 0 |
| α = 0.1 | 56.7% | **54.2%** | 18 | 13 |

The criteria agree here, but not in general. **Seed 1014** at α = 0.5 has margin
**0.0053** (0.4 σ) — correct on the exact backend, a **coin flip** under sampling
(shot accuracy 0.659). At α = 0.4 the same seed resolves completely (margin
0.2267, shot accuracy 1.000).

> **Plain VQE's manifold advantage should not be overclaimed.** Its own minimum
> margin is **0.0023** — *worse* than α = 0.4's and α = 0.6's. Two-state collapse
> gives near-certain readouts *typically*, but a run halting at P ≈ 0.5 has no
> margin at all. The same fragility, just rarer.

---

## 11. Multi-start, and why the recommendation is unchanged

| k | plain | α = 0.6 | α = 0.5 | α = 0.4 |
|---|---|---|---|---|
| 1 | 66.67% | 86.67% | 96.67% | **100.00%** |
| 2 | 88.89% | 94.44% | 99.67% | 100.00% |
| 3 | 96.30% | 96.96% | 99.97% | 100.00% |
| 5 | **99.59%** | 99.02% | 100.00% | 100.00% |
| 10 | 100.00% | 99.95% | 100.00% | 100.00% |

At **k = 5 plain VQE (99.59%) edges out α = 0.6 (99.02%)**. CVaR's advantage is
concentrated at **small seed budgets** — decisive at k = 1, largely gone by k = 5.

Since Phase 7 already recommends five seeds, **this study does not change what
the demo should do.** Its value is the mechanism in §1, plus one practical
caveat: α = 0.4 reaches its ceiling at **k = 1**, which matters when runs are
expensive — a regime this project is not in, but a real one on hardware.

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
parameter files are regenerable and not tracked.
