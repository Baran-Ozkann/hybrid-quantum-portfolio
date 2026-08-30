"""Variable-weight encoding: 2 qubits per asset, 6 assets, 12 qubits.

    ============================================================
    STOPPED, INCOMPLETE TRACK. NOT A WORKING FEATURE.
    ============================================================

    This does NOT find the optimum. Plain VQE locks onto a wrong
    basis-state attractor early and purifies toward it, expelling
    the true ground state; the readout is the 15th-best of 120
    feasible portfolios. Three candidate explanations were tested
    and eliminated (entangler reach, structural reachability,
    penalty scale) and the failure survived all three.

    It is committed as EVIDENCE for the "scaling is not a
    parameter change" finding, and as a base if the question is
    ever resumed -- not as functionality anyone should build on.
    The open question is attractor selection at 120 feasible
    states. See the variable-weight section of the project notes.

A parallel track to Phases 1-9. The Q# bridge, gradient plumbing, CVaR
objective, training loop, and reliability tooling all carry over unchanged;
only the encoding and decode layers are redesigned.

WHY. The Phase 1-9 encoding expresses exactly 6 portfolios -- choose 2 of 4,
equal weights -- which is too small a search space to be a meaningful
optimisation problem, and is the reason Markowitz scores 2.4251 against the
encoding's best 2.1599. That gap was correctly attributed to encoding
expressiveness rather than optimiser quality, and this encoding does narrow it:
best encodable Sharpe 3.8851 against long-only Markowitz 4.1123, a 5.53% gap
where the 4-asset encoding left 10.94%.

Narrowing the expressiveness gap and finding the optimum turned out to be
different problems. This track achieved the first and not the second.

WHAT MUST NOT BREAK. Brute-force ground truth stays tractable (4096 states), so
the entire measurement methodology -- known answer, enumerable spectrum, paired
seeds, McNemar -- carries over. That is the constraint the design is built
around.
"""

