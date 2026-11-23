# Negative results and stopped branches

Research can rule out hypotheses without ruling out the project's objective.

| Branch | Observation | Decision |
|---|---|---|
| R41 lossless cross-layer XOR | No sampled joint group improved over independent lossless coding; overhead dominated | Stop this tested grouping/coding |
| R42 same-position m/v reversible predictors | <1% relative savings; failed >=5% thresholds | Stop the registered transforms |
| R43 CPU pinned memory moments | ~4.99× optimizer update time despite VRAM saving | Stop synchronous each-step offload |
| R47 short-run EM4+D9 | Short-run memory gate passed, but entire path far slower than ordinary native | Keep as research reference, not shipping optimizer |
| R57 EM4 with FP32 AdamW | Native checkpointing outperformed EM4 on memory and time across tested lengths | Freeze as noncompetitive implementation |
| R58-FIX2 D9 vs true AdamW8bit | D9 256.82 MiB state > 8-bit 241.75 MiB; slower steps on S128/S512 | Strong-baseline failure; no continued kernel investment |
| R60/R62 token-tiled CE | Strong benefit vs full logits; virtually zero extra peak saving over Liger | Public baseline technique, not a unique SHV mechanism |

All are condition-specific observations, not impossibility theorems. Keep their original protocols and data rather than rewriting threshold gates after seeing the outcomes.
