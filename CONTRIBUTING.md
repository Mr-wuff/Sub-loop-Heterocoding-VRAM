# Contributing

This is a research preview, not a finished training API. Submissions that reproduce a result, identify a numerical mismatch, improve a benchmark control, or present a new memory-lifetime mechanism are welcome for this research preview. Please note the software license is still pending.

Use Issues for questions and Pull Requests for changes. Public visibility does **not** mean all users can directly edit the main branch.

Include in every performance PR:
- exact GPU model, VRAM, driver/CUDA/PyTorch/dependency versions;
- model revision, batch, sequence length, dtype, optimizer and activation policy;
- baseline and SHV metrics: peak allocated, peak reserved, step time/tokens per second;
- separate subgraph and full-training numbers;
- numerical/quality checks and output ZIP with SHA-256;
- at least one reproducible command or notebook and a reason the comparison is fair.

Please do not modify published experiment results; add a new experiment and keep prior records immutable. Follow the upstream licenses for PyTorch, Liger, bitsandbytes and any pretrained models/datasets.
