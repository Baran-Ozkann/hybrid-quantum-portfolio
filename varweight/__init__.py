"""Variable-weight encoding: 2 qubits per asset, 6 assets, 12 qubits.

A parallel track to Phases 1-9. The Q# bridge, gradient plumbing, CVaR
objective, training loop, and reliability tooling all carry over unchanged;
only the encoding and decode layers are redesigned.

WHY. The Phase 1-9 encoding expresses exactly 6 portfolios -- choose 2 of 4,
equal weights -- which is too small a search space to be a meaningful
optimisation problem, and is the reason Markowitz scores 2.4251 against the
encoding's best 2.1599. That gap was correctly attributed to encoding
expressiveness rather than optimiser quality; this closes it.

WHAT MUST NOT BREAK. Brute-force ground truth stays tractable (4096 states), so
the entire measurement methodology -- known answer, enumerable spectrum, paired
seeds, McNemar -- carries over. That is the constraint the design is built
around.
"""
