# Technical architecture: historical implementation and next hypothesis

## A concrete implemented path: GPT-2 MLP/LN2

The custom-autograd branch tested the idea that the original forward computation can be preserved while its backward saved state is replaced. For the MLP input `Z = LN2(U)`, an experimentally selected packed 6/7-bit `Z` representation, FP16 group scales, and LayerNorm statistics were carried to the backward consumer. The consumer decodes and recomputes its local MLP gradient path. Numerical errors are bounded by registered tests, not exactly zero.

**Savings were measured for a subgraph workload, not total LLM training memory.** At the later full-training level, EM4 recovered short-horizon numerical fidelity with selective MLP fallback, but failed a speed/memory comparison against native activation checkpointing.

## Training memory accounting

A concrete benchmark must distinguish:

- parameter weights and their gradients;
- persistent optimizer moments and any quantization metadata;
- saved forward activations for backward;
- transient backward / output-loss workspaces;
- allocator-reserved but inactive memory;
- other processes' memory / untracked CUDA allocations.

`torch.cuda.max_memory_allocated()` answers one narrow question about PyTorch-allocated peak, not a GPU capacity multiplier. `max_memory_reserved` and physical OOM are separate observations.

## Common optimization vs original SHV hypotheses

R60 used a public-family technique: token-tiled cross entropy with recomputation, on top of AdamW8bit and activation checkpointing. It is useful, but it is not proof of a proprietary SHV mechanism. In R62, Liger and TILED reached near-identical peak allocated memory.

A **future hypothesis** is to treat each backward consumer as a timed reader of a shared state carrier, and place/transform the state only within its real lifetime. HOT / ENCODED / RECOMPUTE / COLD are conceptual tiers. A qualified scheduler or generalized CUDA runtime has **not** been implemented.

```text
Existing (tested):
  native forward -> compact selected saved state -> custom backward

Existing (public baselines):
  8-bit AdamW + checkpointing + (Liger / tiled output loss)

Research question (not implemented):
  backward-time live-set -> identify overlapping consumers
     -> place / encode / recompute only when needed
     -> compare with public baseline at same final quality
```
