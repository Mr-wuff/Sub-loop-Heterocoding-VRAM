# Experiment index and decisions

This selection summarizes the experiments *included* in the research-preview sources and the archived evidence—not every research iteration.

| Round | Question | Decision | Evidence |
|---|---|---|---|
| R5A0 | Can GPT-2 MLP/LN2 backward use a compact saved state? | Subgraph peak allocated decreased; substantial speed cost | HF `results/R5A0/` |
| R41–R42 | Do cross-layer or same-position moments admit useful reversible joint coding? | Failed preregistered byte savings | Historical reports |
| R43 | Can synchronous CPU moments save GPU memory cheaply? | ~4.99x optimizer-step penalty; stopped | Historical reports |
| R47 | Does EM4+D9 retain numerical fidelity on new short sequences? | 3 data orders x 16 steps passed; slower than native training | HF `results/R47/` |
| R57 | Does EM4 outperform checkpointing? | No in tested GPT-2 cases | HF `results/R57/` |
| R58-FIX2 | Does D9 beat real 8-bit AdamW? | No in tested GPT-2 cases | HF `results/R58_FIX2/` |
| R60 | How much does token-tiled output loss save? | Substantial saving vs full logits; public optimization family | HF `results/R60/` |
| R61-FIX1 | How does TILED compare with Liger? | Similar memory; performance depends on hardware/workload | HF `results/R61_FIX1/` |
| R62 | Does Liger/TILED performance replicate across T4 runs? | TILED faster in this setup, no material extra memory saving | HF `results/R62/` |
| R63 | What tensors are live at backward peak? | Proposed; no completed GPU result in this release | No claim |

See [Evidence Ledger](../benchmarks/evidence.csv), [negative results](NEGATIVE_RESULTS.md), and [reproduction guide](REPRODUCTION.md).
