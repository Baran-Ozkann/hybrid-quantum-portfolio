# Documentation Index

Setup, usage, results, and known issues live in the
[**root README**](../README.md). The documents here are the technical reference.

| Document | Read it for |
|---|---|
| [`architecture.md`](architecture.md) | System diagram and module boundaries; phase-by-phase summary; the **single-owner-thread interpreter** design, why it was forced, and its serialization limit for concurrent sessions; interpreter recycling and the quadratic-cost bug it fixes; deviations from the original specification. |
| [`mathematical_foundations.md`](mathematical_foundations.md) | Full QUBO derivation and penalty-weight selection; the QUBO→Ising conversion with its exhaustive verification; why the Hamiltonian is diagonal and what follows; the **exact-backend-for-training vs sampling-for-readout** distinction; the parameter-shift rule and its validation; the **rf = 5%** and simple-return conventions. |
| [`encoding_schema.md`](encoding_schema.md) | Qubit↔asset map and bit ordering (verified, not assumed); bitstring→weights rule; all 6 reachable portfolios; worked examples including an infeasible one; the three decode/repair policies; what the encoding cannot express. |
| [`stability_report.md`](stability_report.md) | 30-seed variance analysis; the converged/failed split; measured multi-start curves; why energy ranking does not perfectly track correctness. |
| [`objective_comparison.md`](objective_comparison.md) | Controlled CVaR-vs-plain-VQE study across α ∈ {0.1, 0.4, 0.5, 0.6, 1.0}. Two pre-registered predictions and how the data refuted both; the 1/α-vs-saturation mechanism; three distinct failure modes; shot-limited accuracy. |

## Figures

Everything in [`figures/`](figures/) is generated, never hand-edited.

| Figure | Produced by |
|---|---|
| `ui_01_load.png`, `ui_02_after_run.png` | `tools/capture_ui.py` |
| `ui_section1..4_*.png` | `tools/capture_ui.py` |
| `stability_summary.png` | `tools/stability_report.py` |
| `../../training/convergence.png` | `training/optimization.py` |
| `../../training/final_distribution.png` | `training/optimization.py` |

To regenerate:

```bash
python tools/capture_ui.py
```

```bash
python tools/stability_report.py
```

## Reading order

If you are new to the project and want the reasoning rather than the results:

1. Root [README](../README.md) — what it does and what the numbers are.
2. [`mathematical_foundations.md`](mathematical_foundations.md) §1–3 — how a
   portfolio becomes a Hamiltonian.
3. [`encoding_schema.md`](encoding_schema.md) §1–4 — how a measurement becomes a
   portfolio.
4. [`mathematical_foundations.md`](mathematical_foundations.md) §6, §8 — the two
   backends and the gradient rule.
5. [`stability_report.md`](stability_report.md) — how well it actually works.
6. [`architecture.md`](architecture.md) §4–5 — the two hard engineering
   constraints the Q# runtime imposed.
