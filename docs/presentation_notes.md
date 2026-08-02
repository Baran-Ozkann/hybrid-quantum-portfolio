# Presentation Notes

Three framing claims in the form *"normally reported as X, I measured Y"*, each
with the numbers behind it so they can be checked live, and the three scope
disclaimers that must be said alongside them.

---

## 0. "So what does this project actually do?"

The first question a non-specialist asks, and the honest answer is not obvious
from any of the numbers. **Lead with this.**

> The standard practice in this area is to run the algorithm, show the energy
> curve going down, and call it converged — because at realistic problem sizes
> **nobody knows the right answer to check against.** I deliberately picked a
> size where the answer *is* knowable: four assets, sixteen possible portfolios,
> so the true optimum can be found by enumeration and the check is actually
> possible. Then I ran it thirty times and checked. **It failed a third of the
> time — including runs whose energy curves looked perfect.**
>
> That is a statement about **the method**, not about portfolios. The portfolio
> is the test instance; it could have been any problem small enough to have a
> verifiable answer.

Paired with the three scope disclaimers below, that paragraph is the whole
project.

**Every number here was re-derived from the committed artifacts before this
document was written.** Where an earlier draft of a claim was wrong, the
correction is recorded under it — those notes exist so nobody restores the
rounder-sounding version.

---

## 1. "It converged" is not "it got the right answer"

> In work like this the standard claim is that the algorithm converged, shown as
> an energy curve going down and flattening. I picked a problem where the correct
> answer is known exactly — 16 states, enumerable — so I could check the claim
> instead of asserting it. **At 200 epochs every one of 30 runs converged, and 10
> of them still read out the wrong portfolio. The most confident run put
> essentially 100% of its amplitude on the wrong answer.**

| | |
|---|---|
| runs | 30 seeds (`42, 123, 456, 1000–1026`), 200 epochs, constant lr = 0.05 |
| **converged** (within 0.5% of spectral width of the ground energy) | **30 / 30** |
| **converged and wrong** | **10 / 30 = 33.3%** |
| most confident wrong run | seed 123, `P(0110) = 1.0000`, ground state `1010` |
| ground / runner-up energies | −4.921119 / −4.881910, gap **0.039209** (0.24% of width) |

Source: `docs/stability_report.md`, `training/arms/lr_constant_200/`.

> **Corrections to an earlier draft of this claim.** It read *"40% of the runs
> that converged were wrong, and the most confident run had 98% of its amplitude
> on one state."* Both were off, one in each direction:
> - **40% is the overall error rate at 100 epochs (12/30), not the
>   converged-and-wrong rate.** At 100 epochs only 23 of 30 runs converged and 8
>   of those were wrong — **34.8%**. The 200-epoch arm is the cleaner statement
>   because *all 30* converge, removing the "did it converge" judgement call
>   entirely, and there the figure is **33.3%**.
> - **98% understates it.** The most confident wrong run at 100 epochs is
>   `P = 0.9993`; at 200 epochs it is `P = 1.0000`. Say "essentially all of its
>   amplitude", not 98%.

**If asked "so is the optimiser broken?"** No — it is minimising the energy
correctly, to 1e-10 against finite differences. It is the *readout* that fails,
because the two lowest states are 0.24% of the spectrum apart. That is the
project's actual subject.

**If asked "did you just not train long enough?"** Tested directly. A 250-epoch
arm moved the rate from 66.7% to 70.0% — **one** seed out of thirty, McNemar
exact **p = 1.0000**, Wilson intervals almost entirely overlapping. But it also
produced a **new** run settling late: seed 1022's readout last changed at epoch
**215 of 250**, where at 200 epochs it had been seed 456 at **176**. So the
answer is that **no budget tested is converged** — extending it relocates the
late-crossing tail rather than removing it. Say "at 200 epochs", never
"converged". This is a measurement (`tools/settle_analysis.py`), not a hedge.

---

## 2. In-sample Sharpe is not a result

> Portfolio studies normally report in-sample Sharpe — estimate the statistics
> and score the portfolios on the same data. I measured out of sample instead, on
> 16 rolling quarters. **The 1.45% Sharpe edge the optimiser is being asked to
> resolve is indistinguishable from a coin flip.**

| | |
|---|---|
| the edge in question | `1010` Sharpe 2.1599 vs `0110` 2.1290 → **+1.45%** |
| generalised test: each fold's own ground state beats its own runner-up | **10 / 16 = 62.5%**, CI [35.4%, 84.8%], p = 0.454 |
| specific test: `1010` beats `0110` | **6 / 16 = 37.5%**, CI [15.2%, 64.6%], p = 0.454 |
| **does the size of the edge predict the realised edge?** | **Spearman ρ = −0.085, p = 0.753** |

Source: `docs/backtest.md` §4, figure `backtest_discrimination.png`.

> **Say both tests, not the sharper one.** They disagree in direction — one
> mildly favours the objective, one mildly opposes it — and both straddle 50%.
> Quoting only the 6–10 record implies the objective is *anti*-predictive, which
> is not supported. The claim is that **neither is separable from chance**.
>
> **Do not say ρ is "negative".** At p = 0.753 the sign means nothing. Say it is
> **indistinguishable from zero**: the size of the in-sample edge tells you
> nothing about the realised edge, in either direction.

**If asked "so the quantum part was pointless?"** No — and this is the sentence
to have ready. **A backtest tests the objective, not the optimiser.** VQE
minimises the Hamiltonian it is given, and does. Whether that Hamiltonian is
worth minimising is a finance question about mean–variance estimation on 250 days
of data. This result is evidence about the second, not the first.

---

## 3. Energy convergence and correct readout trade against each other

> Energy convergence is normally treated as *the* success criterion. Measuring
> both separately showed they are not merely imperfectly correlated — they can be
> **actively traded**. Switching the objective to CVaR at α = 0.4 gives up
> **13.2% of the spectral width** in mean energy and takes readout accuracy from
> **66.7% to 100%**.

| arm (200 epochs, same 30 seeds) | mean ⟨H⟩ | energy given up | readout accuracy |
|---|---|---|---|
| plain VQE | −4.9052 | 0.10% of width | 66.7% (20/30) |
| CVaR α = 0.5 | −3.0423 | **11.3%** of width | 96.7% (29/30) |
| **CVaR α = 0.4** | **−2.7327** | **13.2%** of width | **100%** (30/30) |

Ground energy −4.921119; spectral width 16.560790. Every CVaR run **fails** the
0.5%-of-width convergence criterion that every plain-VQE run passes.

Source: `docs/objective_comparison.md` §1, §4.

> **Correction to an earlier draft.** It read *"giving up 11% of the spectral
> width raised accuracy from 60% to 100%."* Two problems:
> - **11% buys 96.7%, not 100%.** 11.3% is α = 0.5. Reaching 100% costs
>   **13.2%** (α = 0.4). Pick one row and quote it whole.
> - **60% → 100% is an unpaired comparison.** 60% is the *100-epoch* baseline;
>   the CVaR arms are 200 epochs and are paired against **66.7%**. Mixing them
>   compares two configurations and credits CVaR with the epoch increase.

**Caveat to volunteer, not wait for:** 30/30 is a point estimate with a Wilson
interval of **[88.6%, 100%]**. A true rate near 90% is consistent with it. And
α = 0.4 is the best *sampled* point on a five-point grid, not a located optimum.

---

## Scope disclaimers — say these, do not wait to be asked

**1. No speed advantage is claimed.** Nothing here is faster than the classical
alternative. The problem has 16 states and brute-force enumeration solves it
instantly; the simulator runs the circuit on a classical CPU. This is a
correctness and methodology study on a problem small enough to have a known
answer, not a performance claim.

**2. The statistics are in-sample throughout Phases 1–9.** µ and Σ are estimated
on the same window the portfolios are scored against, with no shrinkage. Phase 8
is the one out-of-sample component, and it is the one that found the edge does
not survive.

**3. A single 4-asset instance.** Four tickers, sixteen basis states, one
Hamiltonian for Phases 1–9. The tickers were chosen in 2026 knowing how they
performed — selection bias that is not fixable and not quantified. Phase 8 shows
even the *ground state* is unstable across windows: the project's `1010` wins
only **2 of 16** quarters, and the main instance's energy gap is **5× tighter**
than the median fold, so it is a harder-than-typical case in one respect and an
unrepresentative one in another.

---

## Why the backtest result is not a failure

Someone will conflate these two, and the answer should be ready rather than
improvised.

> **The optimiser scored 30/30 on the main instance and 16/16 on sixteen
> Hamiltonians it had never seen.** It did its job. What the backtest showed is
> that the **objective** it was minimising does not predict the future.
>
> Those are different claims about different things. "Minimising this Hamiltonian
> selects the portfolio with the best in-sample mean–variance trade-off" is a
> mathematical statement, and it is true. "That portfolio then outperforms" is an
> empirical statement about markets, and it is **not** true here — which is a
> fact about in-sample mean–variance estimation, known since the 1950s, not a
> fact about the optimiser or about quantum computing.

**The one-line version:** *the optimiser answered the question correctly; the
backtest showed the question wasn't worth much.*

**If pushed further:** the cleanest evidence that these are separable is that the
quantum arm and brute-force enumeration picked the **identical** portfolio on all
16 folds. There was no optimiser error to blame anything on. Whatever the
backtest measured, it was not the optimiser.

---

## Q&A backup: "did you actually do this work?"

Three moments where a plausible result was checked and did not survive. These are
the concrete answers, and they are written down so they do not have to be
reconstructed under pressure. Where the catch came from review rather than from
building, that is noted — the point is that the check happened.

### 1. The figure that contradicted its own caption

Building the Phase 8 discrimination scatter, I coloured each fold by whether the
in-sample and realised Δ Sharpe **agreed in sign** — which gives **9 of 16** —
while the title of the same figure quoted the test result, **10 of 16**.

The two differ at exactly one fold. **Fold 13's ground state has the *lower*
in-sample Sharpe than its runner-up** (Δ = −0.0724), because the Hamiltonian
ranks states by **penalised mean–variance utility**, not by Sharpe ratio. The two
orderings agree in 15 of 16 folds and diverge in that one. So the figure as first
drawn would have shown 9 while its caption said 10.

*Why it matters:* it is not a plotting bug, it is a conceptual one — "the
in-sample winner" and "the Sharpe maximiser" are not the same portfolio, and the
document said so in several places once this surfaced. Caught while building the
figure; the fix was to colour by the quantity the test actually measures.

### 2. The multi-start curve that was the oracle

I reported a multi-start reliability curve for plain VQE: **66.7% → 88.9% →
96.3% → 99.6%** for k = 1, 2, 3, 5 restarts.

Those are **exactly `1 − (1/3)^k`**. That formula assumes you can *identify*
which of your k runs is correct — an **oracle**. The rule actually available is
"keep the run with the lowest energy", which is strictly weaker.

They coincided in that arm only because it happens to show **perfect energy
separation** — every correct run finished below every incorrect one (correct
range [−4.921115, −4.901300], incorrect [−4.901171, −4.878653]). But the margin
is **+0.000129**, which is **0.33% of the degeneracy gap**, and it holds only
because all 30 runs collapsed into the two lowest states (two-state mass ≥
**0.9983**). In the 100-epoch baseline it fails outright: **14 of 30** runs keep
≥1% of their amplitude outside that pair, the worst at **34.6%**.

*Why it matters:* the number was right and the mechanism behind it was wrong, in
a way that would have made the method sound more reliable than it is. Caught in
review — the reviewer recognised the arithmetic.

### 3. The clean null that was two effects cancelling

Training for 200 epochs with a cosine learning-rate schedule gave **18/30 =
60.0%** — identical to the 100-epoch baseline. That reads as a clean null: longer
training doesn't help.

It changed **two variables at once**. Adding a third arm — 200 epochs, *constant*
rate — gave **20/30 = 66.7% with zero runs failing to optimise**, against seven
in the baseline. So:

- longer training **fixed all 7** non-converging runs (helps)
- cosine decay to lr × 0.02 **freezes** runs that have not yet reached a basin
  (hurts)

The two effects were roughly equal and cancelled to look like nothing happening.

*Why it matters:* a null result from a two-variable change is not a null result.
Caught in review, and it is the reason the project's arm convention is now **one
variable per arm, same 30 seeds, paired tests**.

> **The honest summary if asked directly:** every conclusion in this project that
> was checked hard enough eventually moved. A 1-of-3 success rate became
> 18-of-30. A clean null became two effects cancelling. Two pre-registered
> predictions about CVaR — mine and the reviewer's — were both refuted by the
> sweep. The documents record the refutations rather than quietly dropping them,
> and `docs/backtest.md` §7 records a hypothesis that was **tested and left
> unsupported** for the same reason.

---

## If you have 30 seconds and one sentence

> I built a working hybrid quantum-classical optimiser, then spent most of the
> project trying to break my own results — and the most useful findings are the
> three places where the standard way of reporting this work would have been
> wrong: a converged run can be confidently incorrect, the energy criterion and
> the answer can be traded against each other, and the edge the whole thing
> optimises does not survive out of sample.

## Questions worth pre-loading

| question | answer |
|---|---|
| "Did quantum beat classical?" | No, and it can't here — 16 states, brute force is exact and instant. The question was whether the variational method finds the known answer, and how often. |
| "Why Q# rather than Qiskit?" | The brief specified it. It forced a real engineering constraint: the interpreter is `unsendable`, so every call is funnelled onto one owner thread — see `docs/architecture.md` §4. |
| "Is 30 seeds enough?" | For the 60%-vs-100% comparison, yes — McNemar p = 0.0020. For the backtest, no: 16 folds detect effects ~4× larger than the one sought. Both are stated with intervals. |
| "What would you do next?" | Shrink µ toward the grand mean, not just Σ — Phase 8's noise is dominated by the mean estimate, which is where a single year gives essentially one observation. |
| "What did you get wrong?" | Several things, all recorded: a 1-of-3 result that became 18-of-30, a "clean null" that was two effects cancelling, and two pre-registered predictions about CVaR — mine and the reviewer's — both refuted by the sweep. |
