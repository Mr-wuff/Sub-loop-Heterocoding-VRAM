# Roadmap (planned, not delivered features)

## Research-preview publication gate

- [x] Preserve representative positive and negative evidence with SHA-256 checksums
- [x] Prepare a readable mechanism description and reproducibility navigation
- [x] Provide a CPU-only evidence CLI with executable smoke tests
- [ ] Choose an open-source software license and confirm any patent/IP conditions
- [ ] Review the curated artifacts for third-party distribution permissions
- [ ] Create target GitHub and Hugging Face repositories and upload contents
- [ ] Confirm clean-machine and fresh-Colab reproduction of the curated notebooks
- [ ] Publish a release tag and update README links from placeholders

## Research priorities after R62

- R63 live-set attribution (not in available completed GPU result archives)
- A new *consumer/lifetime* hypothesis with explicit counterfactual strong baseline
- Fresh, longer training quality comparisons with multiple seeds and models
- CUDA/Triton runtime APIs and saved-state checkpoint/restart correctness
- Independent accelerator, sequence-length and OOM-frontier verification

The experimental objective is to improve the **memory/time Pareto frontier** beyond mature public components at matched training quality.
