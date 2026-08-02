# Mathematical Foundations

Derivation of the portfolio Hamiltonian, the loss function, and the gradient
rule, together with the numerical conventions every reported figure depends on.

---

## 0. Conventions, stated once

These two are the ones most likely to make a reader think the numbers are
broken, so they are stated up front.

### 0.1 Risk-free rate: rf = 5.00%

Every Sharpe ratio in this project — in the code, the UI, and these documents —
uses an annualised risk-free rate of **rf = 0.05**:

```
Sharpe = (w' mu - rf) / sqrt(w' Sigma w)
```

The value lives in `classical/data_pipeline.py::RISK_FREE_RATE` and is carried
through `data/statistics.json` so it is never re-entered by hand.

Worked example, the quantum portfolio (50% TSLA / 50% AMD):

```
return      w' mu            = 1.0628   (+106.28%)
volatility  sqrt(w' Sigma w) = 0.4689   ( 46.89%)
Sharpe      (1.0628 - 0.05) / 0.4689 = 2.1599
```

Dividing return by volatility *without* subtracting rf gives 2.267, which is
not the reported figure. If a Sharpe number looks wrong by roughly `rf / vol`,
this is why.

### 0.2 Return convention: simple, annualised by 252

Daily **simple** (arithmetic) returns, `r_t = P_t / P_{t-1} - 1`, annualised as
`mu = 252 * mean(r)` and `Sigma = 252 * cov(r)`.

Simple returns are the correct input to mean-variance optimisation because
portfolio return is *linear* in them: `r_portfolio = sum_i w_i r_i`, so
`w' mu` is genuinely the portfolio's expected return. Log returns aggregate
better across **time** but not across **assets** — the log return of a
portfolio is not the weighted mean of its constituents' log returns — so using
them here would make `w' mu` mean nothing in particular. Daily log returns are
still written to `data/daily_log_returns.csv` for reference.

---

## 1. The classical problem

Four assets, indexed `i = 0..3` in the canonical order `TSLA, NVDA, AMD, COIN`.
Qubit *i* carries asset *i* everywhere in the codebase.

Inputs estimated from one year of daily closes (251 observations,
2025-07-01 to 2026-06-30):

- `mu` — annualised expected returns, shape (4,)
- `Sigma` — annualised covariance matrix, shape (4,4), symmetric positive
  semi-definite (measured eigenvalues 0.0770, 0.1405, 0.2979, 0.7187)

Mean-variance selection with a cardinality constraint: choose exactly `B = 2`
of the 4 assets, holding them in equal weight.

### Decision variables

```
x_i = 1  ->  asset i is held
x_i = 0  ->  asset i is not held
```

### Objective

```
C(x) = q * x' Sigma x  -  mu' x  +  lambda * (sum_i x_i - B)^2
       \____________/    \_____/    \_______________________/
          risk            return        budget penalty
```

with `q = 1.0` (risk aversion) and `B = 2`. The penalty term is what makes an
*unconstrained* binary optimiser respect the cardinality constraint, which is
the whole reason a QUBO formulation is usable here.

---

## 2. QUBO coefficients

Binary variables satisfy `x_i^2 = x_i`. That single identity is what collapses
the quadratic diagonal into the linear part and makes the problem a QUBO rather
than a general quadratic program.

Expanding the penalty:

```
lambda * (sum_i x_i - B)^2
  = lambda * [ (sum_i x_i)^2 - 2B sum_i x_i + B^2 ]
  = lambda * [ sum_i x_i^2 + 2 sum_{i<j} x_i x_j - 2B sum_i x_i + B^2 ]
  = lambda * [ (1 - 2B) sum_i x_i + 2 sum_{i<j} x_i x_j + B^2 ]      (x^2 = x)
```

Collecting all three terms:

```
Q_ii  = q * Sigma_ii - mu_i + lambda * (1 - 2B)          (diagonal / linear)
Q_ij  = 2 * q * Sigma_ij + 2 * lambda        for i < j   (off-diagonal)
const = lambda * B^2
```

so that `C(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j + const`.

### Choosing lambda

Any infeasible `x` incurs at least `1 * lambda`, because `(sum x_i - B)` is a
non-zero integer. So the constraint is guaranteed active whenever

```
lambda > best_feasible_objective - min_over_all_x objective
```

Setting `lambda = penalty_scale * (max_obj - min_obj)` with
`penalty_scale >= 1` is sufficient, since `best_feasible <= max_obj`. With
`penalty_scale = 1.5` this yields **lambda = 3.646370** for the measured data.

Very large penalties are avoided deliberately: they compress the energy
differences *between feasible states* — which is the signal the optimiser needs
— toward numerical noise.

**Honest note on this dataset:** the penalty turns out **not to be binding
here**. The unpenalised optimum already holds exactly 2 assets, so lambda is not
changing the answer. It is still correctly enforced in the energy landscape
(best infeasible +2.414 vs best feasible -1.246, a margin of 3.660), but on
this particular year of data the constraint does no work. The formulation is
general; the test case simply does not stress it.

---

## 3. QUBO to Ising

Substitute `x_i = (1 - s_i) / 2` with spin `s_i` in `{+1, -1}`. This matches the
measurement convention in `quantum/circuit.qs`:

```
measured Zero -> s = +1 -> x = 0 -> asset NOT held
measured One  -> s = -1 -> x = 1 -> asset held
```

Using `x_i x_j = (1 - s_i - s_j + s_i s_j) / 4`:

```
h_i    = -Q_ii / 2  -  (1/4) * sum_{j != i} Q_ij
J_ij   =  Q_ij / 4                                   for i < j
offset =  const  +  (1/2) sum_i Q_ii  +  (1/4) sum_{i<j} Q_ij
```

giving, exactly,

```
C(x) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset
     = <H> + offset
```

`h` and `J` are handed to the quantum circuit. `offset` is a constant the
circuit never sees; it is tracked so reported energies convert back to true
portfolio costs.

### Measured coefficients

```
h = [-0.027976, -0.036148, +0.421994, -0.722479]
      TSLA       NVDA       AMD        COIN

J[TSLA,NVDA] = +1.854333    J[NVDA,AMD]  = +1.881367
J[TSLA,AMD]  = +1.879672    J[NVDA,COIN] = +1.873623
J[TSLA,COIN] = +1.883360    J[AMD,COIN]  = +1.902708

offset = 3.675029    lambda = 3.646370    q = 1.0    B = 2
```

The `J` values are all close to `2 * lambda / 4 = 1.823` because the penalty
dominates the covariance contribution — a direct consequence of the penalty
scale, and visible confirmation that the constraint is encoded.

### This identity is verified, not assumed

The QUBO-to-Ising algebra is easy to get subtly wrong (a factor of two on the
off-diagonal, a dropped constant), and such an error would not crash anything —
it would silently optimise a different problem. `verify_ising_mapping()` checks
`E_ising(s(x)) + offset == C_QUBO(x)` across **all 16 bitstrings**:

```
max |E_ising + offset - C_QUBO| = 5.3e-15
```

Two further independent checks (Phase 2 gate):

- the Python energy table in `q_bridge` vs the one in `hamiltonian`: **0.0e+00**
- the **live Q# circuit** at basis state `1010`: Q# returned `-4.921118600`
  against the table's `-4.921118600`

---

## 4. Why the Hamiltonian is diagonal, and what follows

`H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j` contains only `Z` operators. Every
term commutes, and `H` is diagonal in the computational basis. Three
consequences the implementation relies on:

1. **One measurement basis suffices.** No basis-rotation passes, no grouping of
   non-commuting terms. A single computational-basis measurement per shot gives
   an unbiased sample of the energy.
2. **`<H>` is a plain weighted average**, `<H> = sum_z p(z) E(z)`, where `E(z)`
   is the Ising energy of basis state `z`. This is what makes the exact backend
   (Section 6) legitimate rather than an approximation.
3. **A trailing `Rz` layer would be exactly inert.** `Rz` is diagonal and
   unit-modulus, and the CNOT ladder is a permutation matrix; neither changes
   `|amplitude|`, and only magnitudes affect a diagonal observable. Those
   parameters would carry *identically zero* gradient. The ansatz therefore ends
   on `Ry` — verified: **28/28 parameters affect `<H>`, none dead.**

The exact spectrum (16 states) is enumerated in
`data/hamiltonian.json → spectrum`, so "did the optimiser find the ground
state?" has an unambiguous answer at every point in this project.

---

## 5. The variational circuit

Hardware-efficient ansatz on 4 qubits, `L = 3` layers. Per layer:

```
Ry(theta) on every qubit      4 params   populates amplitudes
Rz(phi)   on every qubit      4 params   populates relative phases
CNOT ladder q0->q1->q2->q3    0 params   creates entanglement
```

followed by one final `Ry` layer (4 params). Total:

```
n_params = n_qubits * (2L + 1) = 4 * 7 = 28
```

Interior `Rz` layers are *not* inert: the phases they write are converted into
amplitude differences by the `Ry` gates of the following layer.

The CNOT ladder uses nearest-neighbour couplings only, so the circuit maps onto
hardware without SWAP overhead. It genuinely entangles — `Ry(pi/2)` on q0
followed by the ladder produces a GHZ-like state with
`p(0000) = p(1111) = 0.5`, which no product state can reproduce.

**Bit ordering: qubit 0 is the most significant bit** of the state-vector index,
so `bit(q) = (index >> (n - 1 - q)) & 1`. Verified empirically, not assumed:
`Ry(pi)` on qubit 3 alone drives the system to basis index 1 (binary `0001`).

---

## 6. Two backends: exact for training, sampling for readout

This distinction is central to how every number in the project was produced.

### The exact backend

Reads the full state vector via `qsharp.dump_machine().as_dense_state()` and
contracts it against the diagonal energy table:

```
<H> = sum_z |psi_z|^2 * E(z)
```

No shot noise, deterministic to float64 rounding, and roughly **350x faster**
than sampling (measured: 48 evaluations in 0.025 s vs 8.9 s). **Used for all
training and all gradient evaluation.**

### The sampling backend

Draws `num_shots` computational-basis measurements, exactly as hardware would,
and averages the per-shot Ising energy. **Used for the Phase 6 portfolio
readout** and as an independent cross-check on the exact backend.

### Why training cannot use sampling

This is a hard numerical constraint, not a preference. A parameter-shift
gradient component is a difference of two expectation estimates. At 1024 shots
each estimate carries a statistical error of roughly 0.02, so the difference
carries about **0.04** — roughly **4x the 1e-2 gradient-validation tolerance**
the brief sets. Sampling error falls only as `1/sqrt(N)`, so reaching 1e-2
reliably would need on the order of `10^5` shots per evaluation, i.e. about
**5.6 million shots per gradient step** (56 evaluations x 10^5). At 100 epochs
that is infeasible, and no amount of tuning changes it.

The exact backend has no such error, so the Phase 4 gate is met with roughly
eight orders of magnitude of headroom (**< 1e-9** against a 1e-2 bound; stated as
a bound rather than a figure for the reason given in §8).

### The two are cross-validated

They are independent implementations — a Python contraction over the state
vector, and a shot loop written in Q# — and must describe the same physics.
Phase 1 gate, 20 000 shots:

```
exact = 0.312284    sampled = 0.313483    |diff| = 0.001199
5 * predicted sigma = 0.016446            -> agree
```

### Where each number in the UI comes from

- **Section 2 "Circuit energy ⟨H⟩"** — exact backend.
- **Section 2 "Readout bitstring" and the pie chart** — sampling backend,
  4096 fresh shots on every click.
- **Section 4 "P(ground state)"** — exact backend, stored from the training run.

So Section 2 showing `60.7% of shots` while Section 4 shows `0.6011` for the
same model is expected, not a discrepancy. Binomial shot noise at that
probability over 4096 shots is `sqrt(p(1-p)/n) = 0.76%`, and the observed
difference is well inside one sigma. The UI states this inline.

### A note on global phase

Repeated identical circuit evaluations return state vectors differing by an
arbitrary overall phase — the simulator does not canonicalise it. Verified to be
purely global: amplitude ratios between runs have modulus spread ~1e-15 and
argument spread ~1e-15, so probabilities agree to 1e-16 and `<H>` to 4e-15.
Global phase is physically unobservable, so this is harmless — but any code
comparing *raw amplitudes* rather than magnitudes must account for it. It is
also the reason `torch.autograd.gradcheck` needs a non-zero `nondet_tol`
(Section 8).

---

## 7. Loss function

```
L(theta) = <H(theta)>  +  l2_coefficient * ||theta||^2
```

with **`l2_coefficient = 0`** by default.

The brief asks for "⟨H⟩ + regularisation terms (e.g. constraint penalties)". The
budget constraint is **already inside `H`** — `lambda (sum x_i - B)^2` is folded
into the Ising coefficients in Section 2 — so adding a separate constraint
penalty here would double-count it and silently change the optimisation target.

The L2 hook exists but is off, and the reason is worth stating: an L2 term on
rotation angles pulls every angle toward 0, and `theta = 0` is exactly the state
`|0000>` — the "hold nothing" portfolio, which is the **worst** state in the
spectrum (cost +14.585 against the ground state's -1.246). Regularising the
parameters here biases the optimiser toward the least useful answer.

---

## 8. Gradients: the parameter-shift rule

For a gate `U(theta) = exp(-i theta P / 2)` with `P` a Pauli operator
(eigenvalues ±1), the derivative of any observable expectation is given
**exactly** by

```
d<H>/d(theta_i) = [ <H(theta_i + pi/2)> - <H(theta_i - pi/2)> ] / 2
```

This is not a finite-difference approximation. Because `P^2 = I`, the
expectation is a pure sinusoid in each parameter,
`<H>(theta) = A cos(theta + phi) + C`, and sampling that sinusoid at two points
a quarter period apart recovers its derivative exactly. There is no step size
and no truncation error — the shift is a full `pi/2`.

Two preconditions, both satisfied here:

- Every gate is `Ry` or `Rz`, and Q# defines both as `exp(-i theta P / 2)`.
- Each of the 28 parameters drives **exactly one** gate. A shared parameter
  would require a sum over its occurrences; `circuit.qs` assigns each angle to a
  single rotation, so the plain two-term rule applies.

Cost: one backward pass is `2 * 28 = 56` circuit evaluations, against 1 for a
forward pass — about 30 ms on the exact backend.

### Validation (Phase 4 gate)

| Check | Result | Bound |
|---|---|---|
| max \|parameter-shift − central difference\|, 5 points × 28 params † | **< 1e-9** | 1e-2 |
| Richardson-extrapolated difference † | **< 1e-10** | 1e-8 |
| pure-sinusoid least-squares fit residual | **2.6e-15** | 1e-9 |
| chain rule through `⟨H⟩²` vs `2⟨H⟩∇⟨H⟩` † | **< 1e-12** | 1e-9 |
| `torch.autograd.gradcheck` | pass | `nondet_tol=1e-12` |
| 10 plain descent steps at lr=0.05 | `⟨H⟩ −0.856 → −3.092` | must decrease |

† **These three are bounds, not measurements, and must not be "corrected" to
whatever a given run prints.** The gate seeds its probe points
(`default_rng(0)`), so the *inputs* are fixed — but every row marked † compares
against a **finite-difference** reference, and the exact backend is not
bit-reproducible. It carries a ~1.4e-15 nondeterminism floor, roughly 10 ULP,
from global phase; the gate prints it as `<H> spread over 50 evaluations`.
Central differences at ε = 1e-5 amplify that by `1/(2ε) = 5e4`, which lands
precisely at the observed scale.

**The order of magnitude is stable; the digits after the first are not.**
Measured across consecutive runs:

| row | observed spread |
|---|---|
| max \|PS − central difference\| | 2.14, 2.14, 2.26, 2.36 e-10 |
| Richardson | 1.81, 1.88, 2.05 e-11 |
| chain rule | 5.68, 8.06, 9.59 e-14, 1.56 e-13 |
| Phase 9 CVaR gate, for comparison | 1.82, 2.06, 2.11, 2.70 e-09 |

So these rows measure **amplified simulator noise, not gradient disagreement**,
and every one sits several orders of magnitude inside its bound regardless of the
draw. Earlier revisions of this document quoted single draws (3.0e-10, 1.8e-11,
1.0e-13) as though they were constants; they were not.

The sinusoid fit is the most informative of these, and is *not* marked †: it
tests the *premise* the rule rests on rather than its consequence, it uses no
finite differences, and it holds to machine precision.

The `zeros` test point returns *exactly* 0.0 for both estimators. That is
correct, not a defect: at `theta = 0` the state is `|0000>`, `<H>` is at a
stationary point, and both estimators return identically zero.

`gradcheck` requires `nondet_tol = 1e-12` because it defaults to demanding
bit-identical repeated backward passes (`nondet_tol = 0.0`). It explicitly
reported *"analytical gradient matches numerical gradient"* while failing only
its reentrancy test. The measured reproducibility floor is **2.2e-15**
(~10 ULP) — the float64 rounding consequence of the global phase in Section 6,
not sampling noise. The chosen tolerance sits three orders above that floor and
six *below* the 1e-6 accuracy tolerances, so genuine gradient errors are still
caught.

---

## 9. Optimisation

Adam, lr = 0.05, 100 epochs, angles initialised `U(-0.1, 0.1)`.

Small initial angles start the state near `|0000>` with well-defined gradients
and avoid the flat, high-symmetry region large random angles can land in.
Gradient norms fall from about 9 to 0.04–0.19 with no barren-plateau stall, and
parameter norms plateau by epoch 85, so 100 epochs is sufficient.

### Result, seed 42

```
final <H>            -4.898359
exact ground energy  -4.921119   (state 1010 = TSLA + AMD)
gap                  +0.022759   (0.46% of the 16.56 spectral width)
P(ground state)       0.6011
readout              1010  ->  50% TSLA / 50% AMD
Sharpe                2.1599  (identical to the exhaustive-search optimum)
```

### The near-degeneracy

The two lowest states are separated by only **0.039209**, against a spectral
width of 16.56 — **0.24%**:

```
1010  TSLA + AMD   E = -4.921119   (ground state)
0110  NVDA + AMD   E = -4.881910   (runner-up)
```

Both hold AMD; they disagree only on whether the partner is TSLA or NVDA. A VQE
converging to within 0.5% of the ground energy cannot be expected to resolve a
0.24% gap reliably, and it does not: **1 of 3 seeds** read out the true ground
state. See `stability_report.md` for the full analysis.

---

## 10. Encoding to portfolio weights

A feasible bitstring holds exactly `B` assets, each receiving an equal share:

```
w_i = x_i / B      =>  sum_i w_i = 1 exactly
```

For `B = 2` the reachable set is the `C(4,2) = 6` equally weighted pairs. Full
treatment, including the decode/repair policy and worked examples, is in
[`encoding_schema.md`](encoding_schema.md).

The consequence for benchmarking is important and is handled explicitly rather
than glossed: continuous Markowitz optimises over the whole simplex, a strictly
larger feasible set, so it can and does score higher. That difference measures
the **expressiveness of the 4-qubit encoding**, not the quality of the quantum
optimiser. The like-for-like baseline is the best equal-weight 2-asset basket,
found by exhaustive search over the same 6 portfolios — and the quantum result
matches it exactly (Sharpe 2.1599 both ways).

---

## 11. Reproducibility

Exact reproduction requires freezing the *data*, not just the seeds. A fixed
historical date range is **not** sufficient: Yahoo does not return
byte-identical data for a fixed, entirely historical window. Two consecutive
fetches of 2025-07-01..2026-07-01 differed, with NVDA's annualised mean moving
**1.0e-9** and its covariance entries up to **3.7e-8** (all other tickers
unchanged) — vendor-side revision of split/dividend adjustment factors.

The magnitude is financially irrelevant, but it propagates into `h` and `J`, so
a re-fetch silently changes the problem the saved parameters were trained
against. `data/prices.csv` is therefore treated as a frozen snapshot and reused
unless `--refresh` is passed, and `data/statistics.json` records its SHA-256 for
provenance. With the snapshot frozen the pipeline is byte-identical across runs
(`tools/check_reproducibility.py`).

Seeds fixed: `torch.manual_seed`, `numpy.random.seed`, and the Q# simulator's
quantum and classical RNGs. The exact backend is deterministic regardless.

---

## References

- Farhi, Goldstone, Gutmann (2014), *A Quantum Approximate Optimization
  Algorithm* — QAOA and the QUBO/Ising encoding of constrained problems.
- Peruzzo et al. (2014), *A variational eigenvalue solver on a photonic quantum
  processor* — VQE.
- Mitarai et al. (2018), *Quantum Circuit Learning*; Schuld et al. (2019),
  *Evaluating analytic gradients on quantum hardware* — the parameter-shift rule.
- Markowitz (1952), *Portfolio Selection* — mean-variance optimisation.
- Glover, Kochenberger, Du (2019), *A Tutorial on Formulating and Using QUBO
  Models* — penalty-weight selection.
