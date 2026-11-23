"""SHV original activation autograd/codec, extracted from R49 frozen parent sources.

Unmodified function/class source bodies; NO historical test/probe/old experiment execution.
Only globals/bindings and minimal N/EM4 configuration are provided in this module.
"""
import re
import math
import numpy as np
import torch
import torch.nn.functional as F

model = None
DEVICE = None
SEQ_LEN = 128
HIDDEN = 768
LN_SUBBIN_BITS = 2
LN_SUBBIN_FINE_DENOM = 8.0
SUBBIN_ALPHA = 0.25
FORMAL_ATTN_CONSUMER_VIEW = "fine"
FORMAL_MLP_CONSUMER_VIEW = "fine"
FROZEN_ATTN_X_MODES = {0:"x8g128",1:"x8g128",2:"x8g32",3:"x8g32",4:"x8g16",5:"x8g8",6:"x8g16",7:"x8g16",8:"x8g32",9:"x8g16",10:"x8g64",11:"x8g32"}
FROZEN_MLP_Z_MODES = {0:"z6g128",1:"z6g32",2:"z6g64",3:"z6g32",4:"z6g32",5:"z7g128",6:"z7g128",7:"z6g16",8:"z6g16",9:"z6g16",10:"z6g32",11:"z6g32"}
R45_CONFIGS = {
  'N': {'native_attn': tuple(range(12)), 'native_mlp': tuple(range(12))},
  'EM4': {'native_attn': (), 'native_mlp': tuple(range(4))},
}

def bind_model(model_instance, seq_len=128):
    global model, DEVICE, SEQ_LEN, HIDDEN
    model = model_instance
    DEVICE = next(model.parameters()).device
    SEQ_LEN = int(seq_len)
    HIDDEN = int(model.config.n_embd) if hasattr(model, 'config') else int(model.transformer.wte.weight.shape[1])
    assert len(model.transformer.h) == 12, 'Frozen EM4 requires exactly 12 GPT-2 blocks'
    assert HIDDEN == 768, 'Frozen EM4 qualification uses GPT-2 Small 768-dimensional hidden state'
    model.eval()  # dropout disabled; same as frozen R47.



def parse_x_mode(mode):
    m=re.fullmatch(r"x8g(\d+)",mode)
    if m is None:
        raise ValueError(mode)
    return int(m.group(1))



def pack_saved_xhat_base(xhat,mode):
    group=parse_x_mode(mode)

    shape=tuple(xhat.shape)
    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]

    assert cols==HIDDEN
    assert cols%group==0

    x=xhat.float().reshape(
        rows,
        cols//group,
        group,
    )

    scale=(
        x.abs().amax(dim=-1,keepdim=True)/
        127.0
    ).clamp_min(1e-8)

    q=torch.round(
        x/scale
    ).clamp(
        -127,
        127,
    ).to(torch.int8).contiguous()

    s=scale.squeeze(-1).to(torch.float16).contiguous()

    return q,s



def unpack_saved_xhat_base(q,s,shape,mode):
    return (
        q.float()*
        s.float().unsqueeze(-1)
    ).reshape(shape)



def pack_two_bit_stream(codes):
    flat=codes.reshape(-1).to(torch.uint8)
    assert flat.numel()%4==0
    assert bool(((flat<=3).all()).item())

    g=flat.reshape(-1,4)

    packed=(
        g[:,0]
        |
        (g[:,1] << 2)
        |
        (g[:,2] << 4)
        |
        (g[:,3] << 6)
    ).to(torch.uint8).contiguous()

    return packed



def unpack_two_bit_stream(packed,shape):
    numel=int(np.prod(shape))
    assert numel%4==0
    assert packed.numel()==numel//4

    p=packed.reshape(-1)

    out=torch.empty(
        numel,
        dtype=torch.uint8,
        device=packed.device,
    )

    out[0::4]=p & 3
    out[1::4]=(p >> 2) & 3
    out[2::4]=(p >> 4) & 3
    out[3::4]=(p >> 6) & 3

    return out.reshape(shape)



def residual_to_two_bit_codes(
    residual,
    scale,
    mode,
):
    group=parse_x_mode(mode)

    shape=tuple(residual.shape)
    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]
    ng=cols//group

    rv=residual.float().reshape(
        rows,
        ng,
        group,
    )

    sv=scale.float().reshape(
        rows,
        ng,
        1,
    )

    normalized=(rv/sv).clamp(
        -0.5,
        0.5,
    )

    codes=torch.floor(
        (normalized+0.5)*4.0
    ).clamp(
        0,
        3,
    ).to(torch.uint8)

    return codes.reshape(shape)



def pack_saved_xhat_subbin2(xhat,mode):
    q,s=pack_saved_xhat_base(
        xhat,
        mode,
    )

    base=unpack_saved_xhat_base(
        q,
        s,
        xhat.shape,
        mode,
    )

    residual=xhat.float()-base

    codes=residual_to_two_bit_codes(
        residual,
        s,
        mode,
    )

    packed2=pack_two_bit_stream(
        codes
    )

    assert packed2.numel()==xhat.numel()//4

    del base,residual,codes

    return q,s,packed2



def apply_xhat_residual_view(
    base,
    scale,
    packed2,
    mode,
    fine,
):
    group=parse_x_mode(mode)

    shape=tuple(base.shape)
    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]
    ng=cols//group

    out=base
    codes=unpack_two_bit_stream(
        packed2,
        shape,
    )

    ov=out.reshape(rows,ng,group)
    cv=codes.reshape(rows,ng,group)
    sv=scale.float().reshape(rows,ng)

    chunk_groups=8

    for start in range(0,ng,chunk_groups):
        end=min(ng,start+chunk_groups)

        if fine:
            # k=0,1,2,3 -> -3/8,-1/8,+1/8,+3/8
            corr=cv[:,start:end,:].to(torch.float32)
            corr.mul_(2.0).sub_(3.0)
            corr.mul_(1.0/8.0)
        else:
            # Coarse parent one-bit view, derived from the same two-bit code.
            corr=(cv[:,start:end,:]>=2).to(torch.float32)
            corr.mul_(2.0).sub_(1.0)
            corr.mul_(0.25)

        corr.mul_(
            sv[:,start:end].unsqueeze(-1)
        )

        ov[:,start:end,:].add_(corr)

        del corr

    del codes

    return out



def unpack_saved_xhat_coarse(
    q,s,packed2,shape,mode,
):
    base=unpack_saved_xhat_base(
        q,s,shape,mode
    )

    return apply_xhat_residual_view(
        base,
        s,
        packed2,
        mode,
        fine=False,
    )



def unpack_saved_xhat_fine(
    q,s,packed2,shape,mode,
):
    base=unpack_saved_xhat_base(
        q,s,shape,mode
    )

    return apply_xhat_residual_view(
        base,
        s,
        packed2,
        mode,
        fine=True,
    )



def set_param_flags(params,enabled=True):
    old=[
        bool(
            p.requires_grad
        )
        for p in params
    ]

    for p in params:
        p.requires_grad_(
            enabled
        )

    return old



def restore_param_flags(params,flags):
    for p,f in zip(
        params,
        flags,
    ):
        p.requires_grad_(
            f
        )



def explicit_ln1_backward_from_xhat(
    dN,
    xhat_saved,
    rstd_saved,
    layer,
    upstream_direct,
):
    block=model.transformer.h[layer]

    gamma=block.ln_1.weight.detach().float()

    dy=dN.float()
    xhat=xhat_saved.float()

    reduce_dims=tuple(
        range(
            dy.ndim-1
        )
    )

    dgamma=(
        dy*
        xhat
    ).sum(
        dim=reduce_dims
    )

    dbeta=dy.sum(
        dim=reduce_dims
    )

    dxhat=dy*gamma

    mean_dxhat=dxhat.mean(
        dim=-1,
        keepdim=True,
    )

    mean_dxhat_xhat=(
        dxhat*
        xhat
    ).mean(
        dim=-1,
        keepdim=True,
    )

    dH_ln=(
        rstd_saved.float()*
        (
            dxhat-
            mean_dxhat-
            xhat*
            mean_dxhat_xhat
        )
    )

    dH=(
        upstream_direct.float()+
        dH_ln
    )

    return (
        dH,
        dgamma,
        dbeta,
    )



class SHVLN1AttentionFineViewFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        H,
        ln_w,
        ln_b,
        cattn_w,
        cattn_b,
        cproj_w,
        cproj_b,
        layer_idx,
        mode,
    ):
        layer_idx=int(layer_idx)
        block=model.transformer.h[layer_idx]

        assert FORMAL_ATTN_CONSUMER_VIEW=="fine"
        assert ln_w.data_ptr()==block.ln_1.weight.data_ptr()
        assert ln_b.data_ptr()==block.ln_1.bias.data_ptr()
        assert cattn_w.data_ptr()==block.attn.c_attn.weight.data_ptr()
        assert cattn_b.data_ptr()==block.attn.c_attn.bias.data_ptr()
        assert cproj_w.data_ptr()==block.attn.c_proj.weight.data_ptr()
        assert cproj_b.data_ptr()==block.attn.c_proj.bias.data_ptr()

        with torch.no_grad():
            N=block.ln_1(H)
            A=block.attn(N,use_cache=False)[0]
            out=H+A

            gamma=ln_w.detach().float()
            beta=ln_b.detach().float()
            xhat=(N.float()-beta)/gamma

            qx,sx,p2=pack_saved_xhat_subbin2(
                xhat,
                mode,
            )

            var=H.float().var(
                dim=-1,
                unbiased=False,
                keepdim=True,
            )
            rstd=torch.rsqrt(
                var+float(block.ln_1.eps)
            ).to(torch.float16)

        ctx.save_for_backward(
            qx,
            sx,
            p2,
            rstd,
        )
        ctx.param_refs=(
            ln_w,
            ln_b,
            cattn_w,
            cattn_b,
            cproj_w,
            cproj_b,
        )
        ctx.layer_idx=layer_idx
        ctx.x_shape=tuple(xhat.shape)
        ctx.mode=mode
        ctx.h_dtype=H.dtype
        return out


    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        qx,sx,p2,rstd=ctx.saved_tensors
        (
            ln_w,
            ln_b,
            cattn_w,
            cattn_b,
            cproj_w,
            cproj_b,
        )=ctx.param_refs

        block=model.transformer.h[
            ctx.layer_idx
        ]
        G=grad_output.float()

        # Formal D2 winner: the nonlinear Attention recomputation consumes
        # the full two-bit fine view.
        xhat_attn=unpack_saved_xhat_fine(
            qx,
            sx,
            p2,
            ctx.x_shape,
            ctx.mode,
        )

        Nhat=(
            xhat_attn*
            ln_w.detach().float()+
            ln_b.detach().float()
        ).to(
            block.ln_1.weight.dtype
        )
        del xhat_attn

        attn_params=[
            cattn_w,
            cattn_b,
            cproj_w,
            cproj_b,
        ]
        old=set_param_flags(
            attn_params,
            True,
        )

        try:
            with torch.enable_grad():
                Nreq=(
                    Nhat.detach()
                    .requires_grad_(True)
                )
                Ahat=block.attn(
                    Nreq,
                    use_cache=False,
                )[0]
                surrogate=(
                    Ahat.float()*
                    G
                ).sum()

                (
                    dN,
                    gcattn_w,
                    gcattn_b,
                    gcproj_w,
                    gcproj_b,
                )=torch.autograd.grad(
                    surrogate,
                    [
                        Nreq,
                        cattn_w,
                        cattn_b,
                        cproj_w,
                        cproj_b,
                    ],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
        finally:
            restore_param_flags(
                attn_params,
                old,
            )

        del Nhat,Nreq,Ahat,surrogate

        # LN1 uses the same fine normalized-direction view.
        xhat_ln=unpack_saved_xhat_fine(
            qx,
            sx,
            p2,
            ctx.x_shape,
            ctx.mode,
        )

        dH,gln_w,gln_b=explicit_ln1_backward_from_xhat(
            dN.detach().float(),
            xhat_ln,
            rstd,
            ctx.layer_idx,
            G,
        )
        del xhat_ln

        return (
            dH.to(ctx.h_dtype),
            gln_w.to(ln_w.dtype),
            gln_b.to(ln_b.dtype),
            gcattn_w.to(cattn_w.dtype),
            gcattn_b.to(cattn_b.dtype),
            gcproj_w.to(cproj_w.dtype),
            gcproj_b.to(cproj_b.dtype),
            None,
            None,
        )



def parse_z_mode(mode):
    m=re.fullmatch(r"z([67])g(\d+)",mode)
    if m is None:
        raise ValueError(mode)
    return int(m.group(1)),int(m.group(2))



def fixed_pack_layout(bits):
    if bits==6:
        return 4,3
    if bits==7:
        return 8,7
    raise ValueError(bits)



def pack_fixed_uint(u,bits):
    u=u.reshape(-1)
    vals_per_word,bytes_per_word=fixed_pack_layout(bits)

    assert u.numel()%vals_per_word==0

    groups=u.to(torch.int64).reshape(-1,vals_per_word)

    word=torch.zeros(
        groups.shape[0],
        dtype=torch.int64,
        device=groups.device,
    )

    for i in range(vals_per_word):
        word |= groups[:,i] << (bits*i)

    out=torch.empty(
        (groups.shape[0],bytes_per_word),
        dtype=torch.uint8,
        device=groups.device,
    )

    for j in range(bytes_per_word):
        out[:,j]=((word>>(8*j))&255).to(torch.uint8)

    return out.reshape(-1).contiguous()



def unpack_fixed_uint(packed,bits,numel):
    vals_per_word,bytes_per_word=fixed_pack_layout(bits)

    assert numel%vals_per_word==0

    p=packed.reshape(-1,bytes_per_word).to(torch.int64)

    word=torch.zeros(
        p.shape[0],
        dtype=torch.int64,
        device=p.device,
    )

    for j in range(bytes_per_word):
        word |= p[:,j] << (8*j)

    mask=(1<<bits)-1

    vals=[
        (word>>(bits*i))&mask
        for i in range(vals_per_word)
    ]

    return torch.stack(vals,dim=1).reshape(-1)[:numel]



def pack_saved_z(Z,mode):
    bits,group=parse_z_mode(mode)

    shape=tuple(Z.shape)
    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]

    assert cols==HIDDEN
    assert cols%group==0

    qmax=(1<<(bits-1))-1

    xf=Z.float().reshape(
        rows,
        cols//group,
        group,
    )

    scale=(
        xf.abs().amax(dim=-1,keepdim=True)/
        float(qmax)
    ).clamp_min(1e-8)

    q=torch.round(
        xf/scale
    ).clamp(
        -qmax,
        qmax,
    ).to(torch.int16)

    u=(q.to(torch.int64)+qmax).reshape(-1)

    packed=pack_fixed_uint(
        u,
        bits,
    )

    stored_scale=scale.squeeze(-1).to(torch.float16).contiguous()

    expected=rows*cols*bits//8
    assert packed.numel()==expected

    return packed,stored_scale



def unpack_saved_z(
    packed,
    scale,
    mode,
    shape,
):
    bits,group=parse_z_mode(mode)

    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]
    qmax=(1<<(bits-1))-1
    numel=rows*cols

    u=unpack_fixed_uint(
        packed,
        bits,
        numel,
    )

    q=(
        u.to(torch.int64)-
        qmax
    ).reshape(
        rows,
        cols//group,
        group,
    )

    Zhat=(
        q.float()*
        scale.float().unsqueeze(-1)
    ).reshape(shape)

    del u,q

    return Zhat



def residual_to_two_bit_codes_z(
    residual,
    scale,
    mode,
):
    _,group=parse_z_mode(mode)

    shape=tuple(residual.shape)
    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]
    ng=cols//group

    rv=residual.float().reshape(
        rows,
        ng,
        group,
    )

    sv=scale.float().reshape(
        rows,
        ng,
        1,
    )

    normalized=(rv/sv).clamp(
        -0.5,
        0.5,
    )

    codes=torch.floor(
        (normalized+0.5)*4.0
    ).clamp(
        0,
        3,
    ).to(torch.uint8)

    return codes.reshape(shape)



def pack_saved_z_dual2(
    Z,
    mode,
):
    packed,scale=pack_saved_z(
        Z,
        mode,
    )

    Zhat=unpack_saved_z(
        packed,
        scale,
        mode,
        Z.shape,
    )

    residual=Z.float()-Zhat

    codes=residual_to_two_bit_codes_z(
        residual,
        scale,
        mode,
    )

    packed2=pack_two_bit_stream(
        codes
    )

    assert packed2.numel()==Z.numel()//4

    del Zhat,residual,codes

    return packed,scale,packed2



def ln2_xhat_from_saved_dual2(
    Zhat,
    scale,
    packed2,
    mode,
    layer,
):
    _,group=parse_z_mode(mode)

    shape=tuple(Zhat.shape)
    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]
    ng=cols//group

    block=model.transformer.h[layer]

    gamma=block.ln_2.weight.detach().float()
    beta=block.ln_2.bias.detach().float()

    if float(gamma.abs().min())<=1e-6:
        raise RuntimeError({
            "ln2_gamma_too_small":{
                "layer":layer,
                "min_abs_gamma":float(gamma.abs().min()),
            }
        })

    xhat=(
        Zhat.float()-
        beta
    )/gamma

    codes=unpack_two_bit_stream(
        packed2,
        shape,
    )

    xv=xhat.reshape(rows,ng,group)
    cv=codes.reshape(rows,ng,group)
    sv=scale.float().reshape(rows,ng)
    gamma_g=gamma.reshape(ng,group)

    chunk_groups=8

    for start in range(0,ng,chunk_groups):
        end=min(ng,start+chunk_groups)

        corr=cv[:,start:end,:].to(torch.float32)
        corr.mul_(2.0).sub_(3.0)
        corr.mul_(1.0/8.0)

        corr.mul_(
            sv[:,start:end].unsqueeze(-1)
        )

        corr.div_(
            gamma_g[start:end,:].unsqueeze(0)
        )

        xv[:,start:end,:].add_(corr)

        del corr

    del codes

    return xhat



def unpack_saved_z_fine(
    packed,
    scale,
    packed2,
    mode,
    shape,
):
    Zhat=unpack_saved_z(packed,scale,mode,shape)
    _,group=parse_z_mode(mode)
    rows=int(np.prod(shape[:-1]))
    cols=shape[-1]
    ng=cols//group

    codes=unpack_two_bit_stream(packed2,shape)
    zv=Zhat.reshape(rows,ng,group)
    cv=codes.reshape(rows,ng,group)
    sv=scale.float().reshape(rows,ng)

    chunk_groups=8
    for start in range(0,ng,chunk_groups):
        end=min(ng,start+chunk_groups)
        corr=cv[:,start:end,:].to(torch.float32)
        corr.mul_(2.0).sub_(3.0)
        corr.mul_(1.0/8.0)
        corr.mul_(sv[:,start:end].unsqueeze(-1))
        zv[:,start:end,:].add_(corr)
        del corr

    del codes
    return Zhat



def explicit_ln2_backward_from_xhat(
    dZ,
    xhat_saved,
    rstd_saved,
    layer,
    upstream_direct,
):
    block=model.transformer.h[layer]

    gamma=block.ln_2.weight.detach().float()

    dy=dZ.float()
    xhat=xhat_saved.float()

    reduce_dims=tuple(
        range(
            dy.ndim-1
        )
    )

    dgamma=(
        dy*
        xhat
    ).sum(
        dim=reduce_dims
    )

    dbeta=dy.sum(
        dim=reduce_dims
    )

    dxhat=dy*gamma

    mean_dxhat=dxhat.mean(
        dim=-1,
        keepdim=True,
    )

    mean_dxhat_xhat=(
        dxhat*
        xhat
    ).mean(
        dim=-1,
        keepdim=True,
    )

    dU_ln=(
        rstd_saved.float()*
        (
            dxhat-
            mean_dxhat-
            xhat*
            mean_dxhat_xhat
        )
    )

    dU=(
        upstream_direct.float()+
        dU_ln
    )

    return dU,dgamma,dbeta



class SHVLN2MLPFineViewFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        U,
        ln_w,
        ln_b,
        cfc_w,
        cfc_b,
        cproj_w,
        cproj_b,
        layer_idx,
        mode,
    ):
        layer_idx=int(layer_idx)
        block=model.transformer.h[layer_idx]

        assert FORMAL_MLP_CONSUMER_VIEW=="fine"
        assert ln_w.data_ptr()==block.ln_2.weight.data_ptr()
        assert ln_b.data_ptr()==block.ln_2.bias.data_ptr()
        assert cfc_w.data_ptr()==block.mlp.c_fc.weight.data_ptr()
        assert cfc_b.data_ptr()==block.mlp.c_fc.bias.data_ptr()
        assert cproj_w.data_ptr()==block.mlp.c_proj.weight.data_ptr()
        assert cproj_b.data_ptr()==block.mlp.c_proj.bias.data_ptr()

        with torch.no_grad():
            Z=block.ln_2(U)
            M=block.mlp(Z)
            out=U+M

            packed,scale,p2=pack_saved_z_dual2(
                Z,
                mode,
            )

            var=U.float().var(
                dim=-1,
                unbiased=False,
                keepdim=True,
            )
            rstd=torch.rsqrt(
                var+float(block.ln_2.eps)
            ).to(torch.float16)

        ctx.save_for_backward(
            packed,
            scale,
            p2,
            rstd,
        )
        ctx.param_refs=(
            ln_w,
            ln_b,
            cfc_w,
            cfc_b,
            cproj_w,
            cproj_b,
        )
        ctx.layer_idx=layer_idx
        ctx.mode=mode
        ctx.z_shape=tuple(Z.shape)
        ctx.u_dtype=U.dtype
        return out


    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        packed,scale,p2,rstd=ctx.saved_tensors
        (
            ln_w,
            ln_b,
            cfc_w,
            cfc_b,
            cproj_w,
            cproj_b,
        )=ctx.param_refs

        block=model.transformer.h[
            ctx.layer_idx
        ]
        G=grad_output.float()

        # Formal D2 winner: nonlinear MLP recomputation consumes the full
        # two-bit fine Z view.
        Zconsumer=unpack_saved_z_fine(
            packed,
            scale,
            p2,
            ctx.mode,
            ctx.z_shape,
        )

        mlp_params=[
            cfc_w,
            cfc_b,
            cproj_w,
            cproj_b,
        ]
        old=set_param_flags(
            mlp_params,
            True,
        )

        try:
            with torch.enable_grad():
                Zreq=(
                    Zconsumer.detach()
                    .to(block.ln_2.weight.dtype)
                    .requires_grad_(True)
                )
                Mhat=block.mlp(
                    Zreq
                )
                surrogate=(
                    Mhat.float()*
                    G
                ).sum()

                (
                    dZ,
                    gcfc_w,
                    gcfc_b,
                    gcproj_w,
                    gcproj_b,
                )=torch.autograd.grad(
                    surrogate,
                    [
                        Zreq,
                        cfc_w,
                        cfc_b,
                        cproj_w,
                        cproj_b,
                    ],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
        finally:
            restore_param_flags(
                mlp_params,
                old,
            )

        del Zconsumer,Zreq,Mhat,surrogate

        # LN2 remains on the fine normalized-direction sufficient-statistic view.
        Zbase=unpack_saved_z(
            packed,
            scale,
            ctx.mode,
            ctx.z_shape,
        )
        xhat_ln=ln2_xhat_from_saved_dual2(
            Zbase,
            scale,
            p2,
            ctx.mode,
            ctx.layer_idx,
        )

        dU,gln_w,gln_b=explicit_ln2_backward_from_xhat(
            dZ.detach().float(),
            xhat_ln,
            rstd,
            ctx.layer_idx,
            G,
        )
        del Zbase,xhat_ln

        return (
            dU.to(ctx.u_dtype),
            gln_w.to(ln_w.dtype),
            gln_b.to(ln_b.dtype),
            gcfc_w.to(cfc_w.dtype),
            gcfc_b.to(cfc_b.dtype),
            gcproj_w.to(cproj_w.dtype),
            gcproj_b.to(cproj_b.dtype),
            None,
            None,
        )



def attention_params_full(layer):
    block=model.transformer.h[layer]
    return [
        block.ln_1.weight,block.ln_1.bias,
        block.attn.c_attn.weight,block.attn.c_attn.bias,
        block.attn.c_proj.weight,block.attn.c_proj.bias,
    ]



def mlp_params_full(layer):
    block=model.transformer.h[layer]
    return [
        block.ln_2.weight,block.ln_2.bias,
        block.mlp.c_fc.weight,block.mlp.c_fc.bias,
        block.mlp.c_proj.weight,block.mlp.c_proj.bias,
    ]



def training_embedding_forward(ids):
    B,T=ids.shape
    assert T==SEQ_LEN
    pos=torch.arange(T,device=ids.device,dtype=torch.long).unsqueeze(0).expand(B,T)
    return (model.transformer.wte(ids)+model.transformer.wpe(pos)).to(
        model.transformer.h[0].ln_1.weight.dtype
    )



def causal_lm_loss(logits,ids):
    shift_logits=logits[:,:-1,:].contiguous().float()
    shift_labels=ids[:,1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1,shift_logits.shape[-1]),
        shift_labels.view(-1),
    )



def r45_transformer_body(H0, mode):
    cfg=R45_CONFIGS[mode]
    native_attn=cfg['native_attn']; native_mlp=cfg['native_mlp']
    H=H0
    for layer in range(12):
        block=model.transformer.h[layer]
        if layer in native_attn:
            H=H+block.attn(block.ln_1(H),use_cache=False)[0]
        else:
            H=SHVLN1AttentionFineViewFunction.apply(H,*attention_params_full(layer),layer,FROZEN_ATTN_X_MODES[layer])
        if layer in native_mlp:
            H=H+block.mlp(block.ln_2(H))
        else:
            H=SHVLN2MLPFineViewFunction.apply(H,*mlp_params_full(layer),layer,FROZEN_MLP_Z_MODES[layer])
    return H



def r45_loss(batch,mode):
    H=training_embedding_forward(batch)
    H=r45_transformer_body(H,mode)
    H=model.transformer.ln_f(H)
    return causal_lm_loss(model.lm_head(H),batch)

