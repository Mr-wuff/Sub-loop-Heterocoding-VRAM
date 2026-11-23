"""Exact R58 AdamW8bit compatibility bridge; reference implementation is untouched.

Only neutral legacy settings may be omitted when an installed constructor removes
arguments. Never silently replace 8-bit AdamW with a 32-bit optimizer.
"""
from __future__ import annotations
import inspect
from typing import Any


def compatible_kwargs(constructor, common: dict, eightbit: dict) -> tuple[dict,dict]:
    signature=inspect.signature(constructor)
    keys=signature.parameters
    requires={'lr','betas','eps','weight_decay','min_8bit_size'}
    if not requires.issubset(keys):
        raise RuntimeError(f'INCOMPATIBLE_SIGNATURE missing={sorted(requires-set(keys))}')
    if int(eightbit['min_8bit_size'])!=4096:
        raise RuntimeError('R58 protocol min_8bit_size unexpectedly changed')
    if eightbit['block_wise'] is not True or int(eightbit['percentile_clipping'])!=100:
        raise RuntimeError('Non-neutral old optimizer settings cannot be safely dropped')
    supplied={**common,'min_8bit_size':int(eightbit['min_8bit_size'])}
    if 'block_wise' in keys:supplied['block_wise']=True
    if 'percentile_clipping' in keys:supplied['percentile_clipping']=100
    if 'is_paged' in keys:supplied['is_paged']=False
    # Newer AdamW8bit implementations hardcode 8 internally while accepting
    # the default optim_bits=32 for backward constructor compatibility.
    # Do NOT pass optim_bits=8 to them; validate actual state after first step.
    if 'amsgrad' in keys:supplied['amsgrad']=False
    if not all(k in keys for k in supplied):raise RuntimeError('Unsupported optimizer constructor arguments')
    return supplied, {'constructor_signature':str(signature),
                      'parameters_forwarded':sorted(supplied),
                      'omitted_neutral_legacy_parameters':sorted(set(['block_wise','percentile_clipping'])-set(keys)),
                      'notes':'All source and numerical R58 settings retained; runtime state verified after warmup.'}


def validate_quantized_state(optimizer: Any, min_size:int=4096) -> dict:
    """Require at least one truly quantized large parameter; old small tensors may stay FP32."""
    import torch
    args=getattr(optimizer,'args',None)
    bits=getattr(args,'optim_bits',None)
    if bits is not None and int(bits)!=8:
        raise RuntimeError(f'INVALID_8BIT_STATE: args.optim_bits={bits!r}')
    large=quantized=large_metadata=0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.numel()<min_size or p.grad is None:continue
            large+=1
            st=optimizer.state.get(p,{})
            u1=st.get('state1');u2=st.get('state2')
            if isinstance(u1,torch.Tensor) and isinstance(u2,torch.Tensor) and u1.dtype==torch.uint8 and u2.dtype==torch.uint8:
                quantized+=1
                if st.get('absmax1') is not None and st.get('absmax2') is not None:large_metadata+=1
    if large<1 or quantized<1:
        raise RuntimeError(f'INVALID_8BIT_STATE: large={large} int8_moment_pairs={quantized}')
    if large_metadata<1:
        raise RuntimeError('INVALID_8BIT_STATE: expected blockwise absmax metadata absent')
    return {'large_active_parameters':large,'quantized_uint8_moment_pairs':quantized,
            'blockwise_absmax_pairs':large_metadata,'reported_optimizer_bits':bits}
