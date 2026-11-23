"""Public Liger Kernel FusedLinearCrossEntropy adapter, not SHV original code.

The external implementation is imported from the user's installed package;
third-party code is never redistributed in the SHV ZIP.
"""
import importlib.metadata
import torch

def liger_causal_ce(hidden, lm_weight, labels):
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    except (ImportError, ModuleNotFoundError) as err:
        raise RuntimeError('LIGER_NOT_INSTALLED_OR_API_CHANGED: '+str(err)) from err
    if hidden.ndim!=3 or labels.shape!=hidden.shape[:2]:
        raise ValueError('expected [B,S,D] and labels [B,S]')
    inputs=hidden[:, :-1, :].reshape(-1, hidden.shape[-1]).contiguous()
    targets=labels[:, 1:].contiguous().view(-1)
    result=LigerFusedLinearCrossEntropyFunction.apply(inputs, lm_weight, targets)
    if isinstance(result, (tuple,list)):
        result=result[0]
    if not isinstance(result, torch.Tensor) or result.numel()!=1:
        raise RuntimeError('Unexpected Liger fused CE return type/shape')
    return result

def get_liger_version():
    try: return importlib.metadata.version('liger-kernel')
    except importlib.metadata.PackageNotFoundError: return 'NOT_INSTALLED'
