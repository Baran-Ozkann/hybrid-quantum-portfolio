# Encoding Schema

How qubit measurement outcomes become portfolio weights, and back again.

This is the contract between the quantum and classical halves of the system. It
is stated here in full because it is the single most error-prone interface in the
project: a mistake here produces plausible-looking numbers attached to the wrong
assets, which no downstream test would catch on its own.

---

## 1. Qubit-to-asset map

One qubit per asset. The map is positional and fixed everywhere — in
`circuit.qs`, `q_bridge.py`, `hamiltonian.py`, and the UI.

| Qubit | Asset | Annualised return | Annualised volatility |
|---|---|---|---|
| q0 | TSLA | +43.64% | 44.35% |
| q1 | NVDA | +33.20% | 35.33% |
| q2 | AMD | +168.93% | 67.59% |
| q3 | COIN | −60.91% | 67.50% |

The canonical order is `classical/data_pipeline.py::TICKERS`.

> **This is load-bearing.** `yfinance` returns columns sorted *alphabetically*
> (`AMD, COIN, NVDA, TSLA`), not in the requested order. Zipping those values
> against `TICKERS` attaches every asset's statistics to the wrong name, and the
> result still looks entirely plausible. The pipeline reindexes explicitly and
> asserts the outcome:
>
> ```python
> prices = prices[tickers]
> assert list(prices.columns) == tickers
> ```

---

## 2. Bit ordering

**Qubit 0 is the most significant bit** of the state-vector index:

```
bit(q) = (index >> (n_qubits - 1 - q)) & 1        n_qubits = 4
```

Equivalently, a bitstring is written left-to-right as `q0 q1 q2 q3`, so `1010`
means q0=1, q1=0, q2=1, q3=0 → **hold TSLA and AMD**.

Verified empirically rather than assumed (Phase 1):

| Prepared | Resulting basis index | Binary |
|---|---|---|
| `Ry(pi)` on q3 alone | 1 | `0001` |
| `Ry(pi)` on q2 alone | 3 | `0011` |
| `Ry(pi)` on q1 alone | 7 | `0111` |
| `Ry(pi)` on q0 alone | 15 | `1111` |

The trailing rows look surprising until you account for the **CNOT ladder**: a
`1` on q0 propagates through `CNOT(q0,q1) → CNOT(q1,q2) → CNOT(q2,q3)`, flipping
every subsequent qubit. Only q3 has no successor, which is why it alone gives a
single set bit. Helper functions `bitstring_to_index` and `index_to_bitstring`
in `q_bridge.py` implement the mapping; `circuit.qs` documents it in its header.

---

## 3. Measurement to spin to selection

```
measured Zero  ->  s = +1  ->  x = 0  ->  asset NOT held
measured One   ->  s = -1  ->  x = 1  ->  asset held
```

This follows from the QUBO-to-Ising substitution `x_i = (1 - s_i) / 2`
(see [`mathematical_foundations.md`](mathematical_foundations.md) §3) and is
implemented identically in `circuit.qs::EvaluateHamiltonian` and
`hamiltonian.py::ising_energy`.

---

## 4. Bitstring to weights

A **feasible** bitstring holds exactly `B = 2` assets. Each held asset receives
an equal share of capital:

```
w_i = x_i / B
```

so `sum_i w_i = 1` **exactly**, by construction — not by renormalisation, and
not to within a tolerance. Verified for all 6 feasible bitstrings in the Phase 2
gate.

### The complete reachable set

With `B = 2` there are `C(4,2) = 6` reachable portfolios. This is the entire
expressive range of the encoding:

| Bitstring | Holds | Weights (TSLA, NVDA, AMD, COIN) | Return | Vol | Sharpe | QUBO cost |
|---|---|---|---|---|---|---|
| `1010` | TSLA, AMD | (0.5, 0.0, 0.5, 0.0) | +106.28% | 46.89% | **+2.1599** | **−1.246090** |
| `0110` | NVDA, AMD | (0.0, 0.5, 0.5, 0.0) | +101.07% | 45.12% | +2.1290 | −1.206881 |
| `1100` | TSLA, NVDA | (0.5, 0.5, 0.0, 0.0) | +38.42% | 33.40% | +1.0006 | −0.322312 |
| `0011` | AMD, COIN | (0.0, 0.0, 0.5, 0.5) | +54.01% | 55.47% | +0.8836 | +0.150411 |
| `0101` | NVDA, COIN | (0.0, 0.5, 0.0, 0.5) | −13.85% | 44.22% | −0.4262 | +1.059200 |
| `1001` | TSLA, COIN | (0.5, 0.0, 0.0, 0.5) | −8.64% | 47.25% | −0.2887 | +1.065720 |

Sharpe ratios use **rf = 5.00%**. Row `1010` is the exact ground state, and the
target the optimiser is judged against.

### Worked example

Suppose the circuit is measured 4096 times and `1010` is the most frequent
feasible outcome.

```
bitstring          1 0 1 0
                   | | | |
qubit              0 1 2 3
asset           TSLA NVDA AMD COIN
x_i                1 0 1 0
held count         sum x_i = 2 = B      -> feasible
w_i = x_i / 2   0.50 0.00 0.50 0.00     -> sums to 1.00

return      w' mu            = 0.5(0.4364) + 0.5(1.6893) = 1.0628
volatility  sqrt(w' Sigma w)
            = sqrt(0.25(0.196688) + 0.25(0.456907) + 2(0.25)(0.112973))
            = sqrt(0.219873) = 0.4689
Sharpe      (1.0628 - 0.05) / 0.4689 = 2.1599
```

Second example, an **infeasible** outcome:

```
bitstring          1 1 1 0   -> sum x_i = 3 != 2   -> INFEASIBLE
```

`decode_weights` raises `ValueError` here rather than returning
`(1/3, 1/3, 1/3, 0)`. Dividing by the observed count would silently produce a
3-asset portfolio that violates the stated constraint, and the violation would
be invisible in the output.

---

## 5. Handling infeasible outcomes

The variational state is a superposition, so infeasible bitstrings can be
measured even when the optimiser has converged well. The budget penalty
suppresses them but does not forbid them — in practice **99.9% of shots are
feasible** at the trained parameters.

Three explicit policies, chosen at the call site:

### 5.1 `decode_weights` — strict

Raises `ValueError` on any bitstring with `sum(x) != B`. Used wherever a
guarantee is needed rather than a best effort.

### 5.2 `decode_distribution` — feasible-mode selection (**used by the UI**)

Takes the most frequently observed **feasible** bitstring and decodes that,
ignoring infeasible outcomes entirely.

This is the standard VQE/QAOA readout: the variational state concentrates
amplitude on the minimising basis state, so the mode is the answer. Ties are
broken deterministically on the bitstring itself, so repeated decoding of the
same distribution cannot disagree.

It raises if *no* feasible bitstring was observed at all — that would mean the
optimiser never entered the constrained subspace, which is a result worth
surfacing loudly rather than repairing.

Reported diagnostics: chosen bitstring, its share of shots, total feasible
fraction, distinct outcomes, total shots.

### 5.3 `decode_with_repair` — nearest feasible

Projects an infeasible bitstring to the feasible bitstring at minimum Hamming
distance, breaking ties by lowest unpenalised objective. Deterministic, and it
never invents an allocation the measurement did not support. Used only as a
fallback when sampling produced nothing feasible.

Example: `1110` (3 assets) repairs to `1010` — Hamming distance 1, and the
lower-objective option among the distance-1 candidates.

---

## 6. What this encoding cannot express

Stated plainly, because it bounds every comparison in this project:

- **No fractional weights.** Only `0` or `1/B` per asset. A 60/40 split is
  unreachable.
- **No cardinality other than B.** Exactly 2 assets, always.
- **6 portfolios total**, against the continuum of the 3-simplex available to
  classical mean-variance optimisation.
- **No short positions.** `x_i` is binary, so weights are non-negative.

### Consequence for benchmarking

Continuous Markowitz optimises over the whole simplex — a strictly larger
feasible set — so it can and does score higher:

| Method | Feasible set | Sharpe |
|---|---|---|
| Quantum VQE (seed 42) | 6 equal-weight 2-asset baskets | +2.1599 |
| Best equal-weight (B=2), exhaustive | **same 6 baskets** | **+2.1599** |
| Markowitz max-Sharpe | continuous simplex | +2.4251 |
| Markowitz min-variance | continuous simplex | +0.9728 |

The quantum result **matches the exhaustive optimum over its own feasible set
exactly**. The gap to continuous Markowitz measures the *expressiveness of the
4-qubit encoding*, not the quality of the optimiser — which is why
`best_equal_weight` is the primary baseline and the continuous rows are labelled
as a different feasible set in the UI.

Worth noting that `max-Sharpe` achieves +2.4251 by collapsing to a **single
asset (100% AMD)** — a corner solution, in-sample, identical to just holding the
highest-Sharpe asset. It carries AMD's full 67.59% volatility with no risk
pooling, so it is not a diversified benchmark.

---

## 7. Scaling to finer allocations

Not implemented, recorded so the ceiling is explicit. Two standard routes:

**Multiple qubits per asset (binary weight encoding).** Allocate `k` qubits per
asset representing a weight in `{0, 1/(2^k - 1), ..., 1}`, then renormalise.
Costs `n * k` qubits — 12 for 4 assets at `k = 3`, giving 8 levels per asset.
Renormalisation makes the objective non-quadratic, so the budget constraint has
to be re-derived.

**Finer cardinality with a fixed lot size.** Keep one qubit per lot and allow
several lots per asset, so `B` counts lots rather than assets. Stays quadratic,
and the encoding here is the `k = 1`, one-lot-per-asset special case.

Both increase circuit depth and parameter count, which — given that the present
28-parameter circuit already struggles to resolve a 0.24% energy gap
(see [`stability_report.md`](stability_report.md)) — would need the reliability
question addressed first.

---

## 8. Invariants tested in the Phase 2 gate

| Invariant | Result |
|---|---|
| All 6 feasible bitstrings decode to `sum(w) = 1` | pass, exact |
| `decode_weights` rejects a 3-asset string | pass, raises `ValueError` |
| `decode_with_repair` maps `1110` → weights summing to 1 | pass, → `1010` |
| `decode_distribution` picks the feasible mode over a more frequent infeasible one | pass, chose `1010` over `1110` |
| Ising energy table (`q_bridge`) vs `hamiltonian.ising_energy`, all 16 states | 0.0e+00 |
| Live Q# circuit energy at `1010` vs table | −4.921118600 vs −4.921118600 |
| Hamiltonian ground state vs exhaustive equal-weight search | both `1010` |
