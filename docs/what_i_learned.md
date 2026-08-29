# What I Learned

Four things this project taught me, with the numbers behind each so I am not
relying on memory.

---

## 1. Convergence is not correctness, and you can only see that if you already know the answer

The standard way to report work like this is an energy curve going down and
flattening, labelled "converged". I picked a problem small enough that the right
answer could be found by brute force — 4 assets, 16 possible portfolios — so I
could check the label instead of trusting it.

Over 30 random seeds, **18 got the right answer. 12 did not.** That is 60%, with
a Wilson 95% interval of [42%, 75%].

Every one of those runs converged. The energy curves look the same.

At a longer training budget the picture gets worse, not better. At 200 epochs
**all 30 runs land within 0.5% of the spectral width of the ground energy** — by
any reasonable definition, every one of them converged — and **10 of 30 still read
out the wrong portfolio.**

The run I keep coming back to is seed 123. It put **essentially 100% of its
amplitude** on a single state — `P = 1.0000` at 200 epochs, `P = 0.9993` for the
most confident wrong run at 100 — and that state was wrong. Completely certain,
completely incorrect. There is nothing in the output of that run that would tell
you.

Why it happens here: the two best portfolios are `1010` (TSLA+AMD) at
`E = −4.921119` and `0110` (NVDA+AMD) at `E = −4.881910`. The gap is
**0.039209**, which is **0.24% of the spectral width**. The optimiser is being
asked to resolve a difference that fine, and a third of the time it does not.

At realistic problem sizes nobody can run this check, because nobody knows the
answer to check against. That is the part that stayed with me. The
reassuring-looking metric and the correct answer are two different things, and
the usual setup cannot tell them apart.

---

## 2. Small samples lie, and they lie confidently

Three times in this project I had a result that looked clean and was not.

**A 1-in-3 success rate became 18-in-30.** I ran three seeds. One worked. That
reads as a 33% method, and I nearly wrote it up that way. Thirty seeds put it at
60%. Three samples could not tell 33% from 60%, and nothing about the three-seed
result *looked* uncertain.

**A clean null turned out to be two effects cancelling.** I trained for 200
epochs with a cosine learning-rate schedule and got **18/30 — exactly the same**
as the 100-epoch baseline. An obvious null: longer training does not help. But
that arm changed two things at once. Adding a third arm — 200 epochs, *constant*
rate — gave **20/30 with zero runs failing to optimise**, against seven failures
in the baseline. Longer training fixed all seven. Cosine decay froze runs that
had not yet reached a basin. Two real effects, roughly equal, pointing opposite
ways, adding to nothing.

**A result turned out to be a formula.** I reported a multi-start reliability
curve: 66.7% → 88.9% → 96.3% → 99.6% for 1, 2, 3, 5 restarts. Those numbers are
exactly `1 − (1/3)^k`. That is the *oracle* — what you get if you can tell which
of your k runs is correct. The rule actually available is "keep the lowest
energy", which is weaker. They happened to coincide because that arm had perfect
energy separation, and the margin doing that work was **0.000129** — **0.33% of
the degeneracy gap**. In the 100-epoch baseline the same property fails outright:
**14 of 30** runs keep more than 1% of their amplitude outside the two best
states, the worst at **34.6%**.

The number was right. The mechanism behind it was wrong. It would have made the
method sound more reliable than it is.

I also learned to distrust my own reported metrics. Two runs of the 12-qubit
extension both reported the true optimum at "rank 3903 of 4096" — identically.
That was not a measurement. The probability had underflowed to exactly zero,
thousands of states were tied at zero, and `argsort` breaks ties by index. A
deterministic artefact that reads as a finding.

---

## 3. Scaling up is not a parameter change

I extended the encoding from 4 assets with fixed weights to 6 assets with
variable weights — 2 qubits per asset, 12 qubits, 4096 states. On paper it is
the same algorithm with a bigger register. It was not.

**Four hard-coded qubit counts.** The Q# bridge had `NUM_QUBITS = 4` baked into
parameter validation, register allocation, the state-vector read, and the energy
table. Two more in the gradient layer, four call sites in the CVaR objective. Two
of those, in the CVaR gradient, would have silently computed a 4-qubit gradient
for a 12-qubit forward pass — wrong numbers, no error.

**A design that failed silently.** My first fix kept one scratch register per
size, coexisting. `dump_machine` reports the whole session, so with a 4-qubit and
a 12-qubit register both live it returns **65536 amplitudes**, and the 4-qubit
state lands at indices up to 61440. Nothing raises. The numbers are simply the
wrong shape.

**An entangler whose reach I had never had to think about.** The circuit uses a
linear CNOT ladder. At 4 qubits it spans the register and the question never
comes up. I measured it at 12: information travels **forward across the whole
register in one layer**, but **backward exactly one qubit per layer** — 1, 2, 3,
6 at depths 1, 2, 3, 6. Spanning 12 qubits backward needs 11 layers.

That mapped onto the observed failure precisely. The two assets involved were AMD
(qubits 4,5) and JNJ (qubits 10,11). At 3 layers JNJ cannot influence AMD while
AMD influences JNJ freely — and the run returned the right two assets with the
**split inverted**: AMD 25% / JNJ 75% where the answer is AMD 75% / JNJ 25%.

So I fixed it. One extra CNOT per layer closing the ladder into a ring, zero extra
parameters. I measured the reach again: **full symmetric connectivity at 2 layers,
all 36 asset pairs**. The defect was gone.

**The failure survived.** The ring run was no better — same class of wrong answer,
and half its probability mass on portfolios that violate the budget constraint. I
predicted a mechanism, targeted it, eliminated it, and the problem stayed. Reach
was necessary and not sufficient.

**A penalty 27.7× larger than it needed to be.** The budget constraint is enforced
by a penalty term. My bound for it assumed a worst case that could not occur, and
returned **3.697** where the exact minimum is **0.1335**. That was not free: at the
large value the optimiser drove the true optimum to **exactly zero amplitude**. At
**0.2465** the same run left it at **4.9e-04** — **5th of 4096** — for a cost of
**0.6 percentage points** of constraint satisfaction. The oversized penalty bought
almost nothing and expelled the state it existed to protect.

Four separate problems, four separate diagnoses. And after all of it, **the
extension still does not find the optimum.** The readout is the same wrong
portfolio at both penalty scales, 15th of 120 feasible states. What is left is
the same thing as finding number 1, at 120 candidate answers instead of 6.

The thing I would say to myself at the start: "the same algorithm, bigger" is not
a plan. Every dimension that was constant is a place something is hard-coded, and
every property that held incidentally at small scale has to be re-measured.

---

## 4. The honest result was the one worth having

Everything above is about whether the optimiser finds the minimum. I spent most
of the project on that, and got it from 66.7% to 100% by changing the objective —
the same thirty seeds and the same 200-epoch budget, with only the objective
different. (The 60% in section 1 is the shorter 100-epoch run; quoting 60% → 100%
would credit the objective with the extra training.)

Then I asked a different question: **is that minimum worth finding?**

Every Sharpe ratio in the project up to that point was in-sample — the statistics
estimated on the same data the portfolios were scored against. So I built a
proper out-of-sample test: 16 rolling quarters, a separate frozen price snapshot,
estimate on 250 days, hold for 63, measure what actually happened.

The thing being optimised is a **1.45% Sharpe edge** between the two best
portfolios. Out of sample:

- The in-sample winner beat its runner-up in **10 of 16 quarters** — 62.5%,
  p = 0.454.
- Asked the other way, on the specific pair the project is built around, it won
  **6 of 16** — 37.5%, p = 0.454. The two tests disagree in direction and both
  straddle a coin flip.
- Whether the in-sample edge was **large or small carried no information** about
  the realised edge: rank correlation **−0.085**, p = 0.753.

The design is powered to detect effects about **four times larger** than the one
it is looking for. Detecting a 0.242 Sharpe difference at 80% power would need
**279 quarters — roughly 70 years.** I stated that before running it, so that
"indistinguishable" would read as a prediction rather than an excuse.

So: the 33% error rate I spent the project reducing costs **nothing measurable**.
The optimiser was resolving noise.

The separation that matters, and the one I would defend hardest: **a backtest
does not test the optimiser, it tests the objective.** The variational method
minimises the Hamiltonian it is given, and it does. Whether that Hamiltonian is
worth minimising is a question about mean-variance estimation on 250 days of
history, and the answer here is no. Those are different claims about different
things, and collapsing them would be the easiest mistake in the whole project.

The quantum arm made **zero selection errors across all 16 quarters** — it picked
the identical portfolio to brute-force enumeration every time. So the backtest
contains no information about its economic value in either direction. That is the
absence of a measurement, not a measurement of zero.

I could have stopped before this test. The project had a clean improvement story:
66.7% to 100% by changing the objective. Running the out-of-sample check is what
turned that into something I actually believe, and the result is more interesting
than the story it replaced.
