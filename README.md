# SHV — Sub-loop Heterocoding VRAM

**Research preview · reproducible GPU-memory experiments, not a production training runtime**

SHV investigates whether a model can retain **smaller, consumer-specific backward states** instead of keeping every original forward intermediate in GPU memory. The research asks what information a later gradient calculation actually needs, how to encode it, and when decoding or recomputation costs outweigh the saved memory.

This repository preserves implementations, measurement protocols, and positive **and negative** results from experiments on GPT-2 Small. Its current Python package provides **evidence-inspection tools**, not a drop-in PyTorch replacement or a general GPU-memory extension.

> **Publication status:** this public repository is an experimental research preview. Code licensing and formal release are still under review. Benchmarks below are historical results with archived evidence; no shipping 2× VRAM multiplier, universal inference support, or faster training is claimed.

## What is the underlying idea?

In standard autograd, a forward operator may keep tensors for its later backward calculation. SHV explores replacing selected saved tensors with a compact state tailored to that operator's **backward consumer**, then reconstructing or recomputing only what the gradient requires.

```text
Input / model parameters
         |
         v
  Native forward operator   (original forward outputs preserved)
         |
         +--> encode selected backward state
         |         |
         |         v
         |   packed consumer-specific state
         |         |              (alternative saved-state representation)
         v         v
     loss / downstream operations
                   |
             backward consumer
                   |
          decode / recompute
                   |
             gradients
                   |
              optimizer
```

**Example — MLP / LayerNorm 2.** One GPT-2 prototype saved a physically packed 6/7-bit representation of an intermediate activation, group scales, and LayerNorm statistics. A custom autograd backward path reconstructed the information it needed while preserving the native forward result. The resulting gradient was approximate and checked against registered fidelity thresholds; this was **not lossless reconstruction**.

The general representation question is:

\[
C = E(X;\,\text{backward consumer}),\quad \widehat X = D(C),\quad
\nabla L_{\mathrm{SHV}} \approx \nabla L_{\mathrm{native}}.
\]

The compressed carrier must be judged by **actual CUDA allocations, gradient fidelity, and execution time**, not bit rate alone.

## What has been tested?

The following are **different measurement scopes** and must not be conflated. Detailed records and original ZIPs (with SHA-256 checksums) belong in the companion artifact repository.

| Experiment | Condition and baseline | Measured result | Boundary |
|:--|:--|:--|:--|
| **R5A0, MLP/LN2** | Tesla T4; 12 accumulated GPT-2 MLP/LN2 **subgraphs**, batch 8; native subgraph | 93.11% lower forward-resident allocation; 52.96% lower full-step **subgraph** peak allocated | This is **not** whole-model memory. Throughput ≈0.46× native. |
| **R47, EM4+D9** | GPT-2 Small; T4; batch 4; seq 128; Native+D9 | 14.70% lower peak allocated over 3 disjoint 16-step data orders | Short-run numerical gate; ≈8.7–8.8× slower than the ordinary Native control in that setup. |
| **R57, EM4 activation** | GPT-2 Small; T4; batch 4; seq 512; Native+FP32 AdamW foreach | 30.85% lower peak allocated | Native checkpointing used **less memory and less time** than EM4. |
| **R58-FIX2, D9** | GPT-2 Small; T4; true AdamW8bit; same runtime | D9 persisted 256.82 MiB vs AdamW8bit 241.75 MiB; D9 took 5.33× / 2.46× step time at seq 128/512 | **Strong-baseline failure** for the D9 optimizer. |
| **R60, tiled output loss** | GPT-2 Small; T4; seq 512; AdamW8bit+checkpointing; FULL cross entropy | 2001.27 → 1421.98 MiB (28.95% less peak allocated), with 9.84% more step time | **Existing/public optimization family, not an SHV originality claim.** |
| **R62, tiled vs Liger** | GPT-2 Small; T4; seq 512; 3 paired runs | TILED median step time ≈12.04% lower; memory ≈0.0084% lower | No material (>5%) memory advantage; limited hardware/workload. |

Numerical checks from short runs are not evidence of long-run convergence or equal final model quality. The full provenance, protocols and per-run results are in `benchmarks/` and the [companion HF research artifact repository](https://huggingface.co/datasets/wuff-mann/Sub-loop-Heterocoding-VRAM).

### What failed (and why we keep those results)

- **Cross-layer lossless XOR / shared carriers (R41)** did not beat independent coding; sampled joint groups consistently lost or tied.
- **Same-position AdamW m/v reversible predictors (R42)** achieved well under the registered 5% gain target.
- **Synchronous CPU moment offload (R43)** saved GPU memory but made the isolated optimizer update about 4.99× slower.
- **EM4** did not outperform native activation checkpointing in the isolated-process R57 comparisons.
- **D9** used more optimizer-state bytes and ran more slowly than a real 8-bit AdamW implementation in the same-runtime R58-FIX2 test.
- **TILED vs Liger (R62)** yielded no material extra peak-memory saving.

We archive these results to make future design decisions falsifiable rather than repeatedly retuning variants that failed.

## What is included in this release candidate?

```text
src/shv_preview/        Evidence-checking CLI only (CPU / standard library)
experiments/r55_em4_d9/  Frozen historical EM4/D9 research implementation
experiments/r60_consumer_loss/  Public tiled-loss system experiment
experiments/r62_public_baseline/  Tiled-vs-Liger reproducibility / profiler
notebooks/              Curated, output-stripped research notebooks
benchmarks/             Claims ledger + SHA-256 artifact inventory
docs/                   Mechanisms, reproduction, negative results, roadmap
releases/               Draft v0.1 research-preview release notes
```

The experiment modules and notebooks are **research rulers**. They may install GPU packages, download GPT-2/WikiText assets, or require a matching Colab runtime. They are not exposed as a stable end-user `shv` acceleration API.

## Installation and quick start

### CPU-only: inspect measurements and checksums

Python 3.10+ is required for the metadata tooling (this is a packaging check, not a training compatibility claim):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m shv_preview list
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.

To check a result archive downloaded from the HF companion:

```bash
python -m shv_preview verify \
  SHV_R62_PROFILE_Q33_A0_RESULT_UPLOAD_TO_CHATGPT.zip \
  SHV_R62_PROFILE_Q33_A0_RESULT_UPLOAD_TO_CHATGPT.zip.sha256
```

That verifies file SHA-256 and ZIP CRC; it does **not** independently establish benchmark correctness.

### GPU reproduction

**Tested research environment (for R60/R62):** Tesla T4, PyTorch `2.14.1+cu130`, Transformers `5.22.1`, bitsandbytes `0.50.2`, and Liger Kernel `0.8.5` for R62. These are the versions recorded by the experiments, not a guarantee of support on other versions or accelerators.

For the most direct reproduction, open the corresponding output-stripped Colab notebook with a CUDA GPU, run all cells, and check the exported result ZIP/`.sha256`. The notebooks embed their own source hashes and perform controlled comparisons. In particular:

- `notebooks/R5A0_MLP_LN2_Live_Custom_Autograd.ipynb` — MLP/LN2 **subgraph** memory and numerical checks.
- `notebooks/R47_Fresh_Short_Trajectory.ipynb` — frozen EM4+D9 short fresh-data test.
- `notebooks/R60_Output_Loss_Lifetime.ipynb` — FULL vs TILED output-loss experiment.
- `notebooks/R62_Public_Baseline_Profiler.ipynb` — Liger/TILED three-run comparison.

See [Reproducing the published measurements](docs/REPRODUCTION.md) for dependencies, conditions, scope, and benchmark hygiene. The required pretrained model and datasets are **not redistributed** here.

## Architecture and research status

| Component | Status |
|---|---|
| MLP/LN2 custom autograd + compact saved states | Tested research prototype |
| Attention saved-state representations | Investigated, with both pass and fail cases; not a general API |
| EM4 short-run full-GPT-2 activation path | Tested; weaker than checkpointing on R57 strong baseline |
| D9 compressed AdamW state | Tested; engineering branch frozen after 8-bit strong-baseline fail |
| Token-tiled output loss | Tested public-system technique, not exclusive to SHV |
| Dynamic HOT/ENCODED/RECOMPUTE/COLD scheduler | Proposed, not implemented as a qualified runtime |
| Universal CUDA/Triton fused training engine | Not available |
| LLM inference acceleration / 16→32 GB equivalence | Not demonstrated |

**R63** is a planned/live-set diagnostic in the supplied history, not a completed GPU result in this package. The future focus is to identify transient tensors simultaneously live at a backward peak and test whether a new consumer/lifetime policy outperforms public best practices.

## Contributing

Issues with a minimal reproducer, raw `max_memory_allocated`/reserved data, software and GPU versions, and gradient/quality checks are especially valuable. Contributions happen by **issues and pull requests**, not unrestricted write access. See [CONTRIBUTING](CONTRIBUTING.md).

## License, credit, and publication status

**No code license has yet been selected or granted in this candidate.** Before public release, choose and add an OSI-approved `LICENSE` file and review third-party copyrights and any patent plans. See [LICENSE_SELECTION_REQUIRED](LICENSE_SELECTION_REQUIRED.md). Do not redistribute this candidate as already licensed open-source software until that decision is made.

Reproduction sources are provided with SHA-256 provenance. Release version `v0.1.0-research-preview` is a **draft tag**, not an already published release. Citation metadata will be completed after the author confirms the preferred name/organization. No HF model weights are published by this project.