---
license: apache-2.0
pretty_name: SHV research benchmark records and experiment artifacts
language:
- en
tags:
- gpu-memory
- model-training
- pytorch
- reproducibility
- benchmark
---

# SHV — experiment records and reproducibility artifacts

**Public research artifact repository.** This is a **dataset-type repository containing benchmark records and frozen experiment ZIPs**, not a collection of model weights, not a training-text dataset, and not a claim that SHV is a production-ready runtime.

The companion **GitHub source repository** (https://github.com/Mr-wuff/Sub-loop-Heterocoding-VRAM) contains the research explanation, source modules, selected notebooks and a CPU-only SHA-256 verification tool. This archive contains only selected experiment results (R5A0, R47, R57, R58-FIX2, R60, R61-FIX1, R62). It intentionally preserves negative as well as positive outcomes.

## Scope of the included records

| Experiment | Artifact | Interpretation |
|---|---|---|
| R5A0 | `results/R5A0/` | MLP/LN2 custom-autograd subgraph measurements, **not** end-to-end model savings |
| R47 | `results/R47/` | 3 data-order ×16-step short GPT-2 EM4+D9 qualification |
| R57 | `results/R57/` | Independent-process EM4 vs native checkpointing, including negative result |
| R58-FIX2 | `results/R58_FIX2/` | Same-environment D9 vs true AdamW8bit strong-baseline **failure** |
| R60 | `results/R60/` | Tiled output loss vs full-logits, **public-system technique** |
| R61-FIX1 | `results/R61_FIX1/` | Tiled output loss vs Liger preliminary comparison |
| R62 | `results/R62/` | Repeated TILED/Liger comparison and profiler allocation traffic |

`manifest.json` gives the SHA-256 and byte size of every ZIP. Every ZIP has a matching `.sha256` file. For each experiment, consult its internal `protocol.json`, summary/environment, and raw measurements; do not combine results measured under different contracts.

## Verify an archive

Download any ZIP and matching `.zip.sha256`, then run:

```bash
sha256sum -c <filename>.zip.sha256
```

Alternatively use the `shv-evidence verify` tool in the companion GitHub repo; it checks the ZIP's SHA-256 and CRC.

A cryptographic checksum verifies the archive's integrity, not model quality or its interpretation.

## Related source / reproduction

- GitHub code and notebook index: https://github.com/Mr-wuff/Sub-loop-Heterocoding-VRAM
- Full reproduction notes: https://github.com/Mr-wuff/Sub-loop-Heterocoding-VRAM/blob/main/docs/REPRODUCTION.md
- Benchmark scope and negative results: https://github.com/Mr-wuff/Sub-loop-Heterocoding-VRAM/blob/main/benchmarks/evidence.csv

**Rights:** Hugging Face metadata for this dataset-type repository currently specifies Apache-2.0. The separate GitHub *software code* license has not yet been selected, and public visibility does not grant permission to reuse the GitHub code. Check the original model/data licenses before re-running experiments. Source datasets/model weights are not redistributed here; refer to their original upstream licenses.
