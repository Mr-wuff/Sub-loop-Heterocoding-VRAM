"""Exact-math consumer-scoped token tiling for the GPT2 LM head.

This is a general checkpointed cross-entropy baseline extension, NOT claimed
as a unique SHV invention. Floating-point reduction order may differ: check
loss/gradient parity and full-model trajectories before interpreting metrics.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.float(), labels, ignore_index=-100, reduction="sum")


def tiled_causal_ce(last_hidden: torch.Tensor, lm_weight: torch.Tensor,
                    labels: torch.Tensor, tile_tokens: int=256) -> torch.Tensor:
    if last_hidden.ndim != 3 or labels.ndim != 2:
        raise ValueError('Expected hidden [B,S,D], labels [B,S]')
    if last_hidden.shape[:2] != labels.shape or last_hidden.shape[1] < 2:
        raise ValueError('Shape/sequence mismatch')
    if tile_tokens <= 0:
        raise ValueError('tile_tokens must be positive')
    hidden = last_hidden[:, :-1, :].reshape(-1,last_hidden.shape[-1])
    targets = labels[:, 1:].contiguous().reshape(-1)
    denominator = (targets != -100).sum().clamp_min(1)
    total = torch.zeros((), dtype=torch.float32, device=last_hidden.device)
    for offset in range(0,targets.numel(),tile_tokens):
        # Checkpoint recomputes one small [tile_tokens,Vocab] logits block in backward.
        # Tensor views are passed explicitly; the tied LM weight keeps its gradient.
        loss_part = checkpoint(_chunk_loss, hidden[offset:offset+tile_tokens],
                               lm_weight,targets[offset:offset+tile_tokens],
                               use_reentrant=False)
        total = total + loss_part
    return total / denominator


def reference_causal_ce(last_hidden: torch.Tensor, lm_weight: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    # Same causal loss as HF but avoid calculating the terminal unused token,
    # making the full-logits reference a stronger and fairer baseline.
    logits=F.linear(last_hidden[:, :-1, :], lm_weight).float()
    targets=labels[:, 1:].contiguous()
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                           targets.reshape(-1),ignore_index=-100)


def synthetic_parity(device='cpu', tile_tokens=7) -> dict:
    # Also checks masked labels, tied weight gradient, and non-divisible tails.
    torch.manual_seed(15060)
    H=torch.randn(2,13,16,device=device,dtype=torch.float32,requires_grad=True)
    W=torch.randn(47,16,device=device,dtype=torch.float32,requires_grad=True)
    Y=torch.randint(0,47,(2,13),device=device)
    Y[0,4]=-100
    ref=reference_causal_ce(H,W,Y)
    gr_ref=torch.autograd.grad(ref,(H,W),retain_graph=True)
    tile=tiled_causal_ce(H,W,Y,tile_tokens=tile_tokens)
    gr_tile=torch.autograd.grad(tile,(H,W))
    rel=lambda x,y:(x-y).norm().item()/max(x.norm().item(),1e-12)
    out={'device':str(device),'loss_abs_gap':abs(float(ref.detach())-float(tile.detach())),
         'hidden_grad_rel_l2':rel(gr_ref[0],gr_tile[0]),
         'weight_grad_rel_l2':rel(gr_ref[1],gr_tile[1]),
         'tail_tokens':int((H.shape[0]*(H.shape[1]-1))%tile_tokens)}
    return out
