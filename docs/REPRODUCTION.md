# Reproduction guide

## Verify provenance first

1. Read `benchmarks/evidence.csv` and the accompanying HF artifact `manifest.json`.
2. Download the relevant result ZIP **and its sidecar SHA-256** from the companion artifact repository.
3. Run `python -m shv_preview verify <result.zip> <result.zip.sha256>`.
4. Inspect the result ZIP's `protocol.json`, environment, raw CSV/JSON and stored qualification status. A checksum protects file integrity, not research validity.

## Re-run a selected experiment

The research notebooks in `notebooks/` have outputs stripped and retain the original experiment cells. They are complete experiment sources, **not** a stable GPU pip-install experience.

- R5A0: Tesla T4; PyTorch `2.11.0+cu128`; Transformers `5.16.1`; datasets `4.0.0`. Memory data cover 12 accumulated MLP/LN2 subgraphs (batch 8).
- R47: Tesla T4; PyTorch `2.11.0+cu130`; Transformers `5.22.0`. Fresh 3 × 16-step data order study; each starts from the same pretrained GPT-2 weights.
- R60/R62: Tesla T4; PyTorch `2.14.1+cu130`; Transformers `5.22.1`; bitsandbytes `0.50.2`; Liger Kernel `0.8.5` for R62. Different loss paths are compared under identical 8-bit + checkpointing systems.

**Suggested workflow:** open the experiment-specific notebook in Colab, select the GPU, run cells in order, and compare its generated JSON to the recorded ZIP. Some experiments access Hugging Face model and dataset downloads; ensure you have permission under their source licenses.

Research modules are also preserved in `experiments/` for inspection. Executing them outside their original Colab environment may require source-relative imports and compatible package versions. The source modules should not be mistaken for published production APIs.

## Fair comparison checklist

Record GPU model and capacity; GPU driver; CUDA runtime; PyTorch, Transformers, bitsandbytes, Triton/Liger versions; full model ID/revision; batch, sequence length and dtype; full optimizer config; activation checkpointing mode; number of warmup/measured updates; scope of loss computation; `peak allocated` and `peak reserved`; step milliseconds and tokens/s; numerical comparisons; repeated runs.

Keep optimizer-only, subgraph-only and full train-step numbers in different rows. Do not compare different run contracts as if they were one controlled trial.

## Known limits

No released package yet handles arbitrary model architectures, general inference, training-quality matching across long runs, or a drop-in CUDA runtime. The optimizer and EM4 experiments are included to facilitate independent reproduction **including failures**.
