# Frozen, exact source extractions from R49 notebook code cells 25, 29, 30, 57.
# Only runtime globals are supplied here. No source-body changes.
import math
import torch

ADAMW_BETAS = (0.9, 0.95)
ADAMW_LR = 5e-5
ADAMW_EPS = 1e-6
ADAMW_WEIGHT_DECAY = 0.01
MOMENT_HETEROCODE_CHUNK_ELEMS = 1048576
R5F0DD3_GROUP_SIZE = 256
PROJECTED_BOTH_FP16_SIDECAR = "projected_both_fp16_sidecar"
PROJECTED_BOTH_FP32_SIDECAR = "projected_both_fp32_sidecar"
PROJECTED_V_ONLY = "projected_v_only"
PROJECTED_RATIO_ONLY = "projected_ratio_only"
NO_RESIDUAL = "no_residual"
EXACT_M_ORACLE = "exact_m_oracle"
GROUP_BIAS_FP16 = "group_bias_fp16"
BINARY1_TWO_CENTROID = "binary1_two_centroid"
R5F0DD9_MODES = [NO_RESIDUAL, EXACT_M_ORACLE, GROUP_BIAS_FP16, BINARY1_TWO_CENTROID]
FROZEN_SOURCE_SHA256 = "46207db1819d1ce7cb2bd4568dd2ecbaa5de6a0f472dfee15a947a33f3a0741d"

@torch.no_grad()
def _pack_binary_bits(bits_bool):
    bits = bits_bool.to(torch.uint8).reshape(-1)
    L = bits.numel()
    P = ((L + 7) // 8) * 8
    if P != L:
        pad = torch.zeros(
            P - L,
            device=bits.device,
            dtype=torch.uint8,
        )
        bits = torch.cat([bits, pad], dim=0)
    b = bits.view(-1, 8).to(torch.int16)
    weights = (
        1 << torch.arange(
            8,
            device=bits.device,
            dtype=torch.int16,
        )
    ).view(1, 8)
    return (b * weights).sum(dim=1).to(torch.uint8)


@torch.no_grad()
def _unpack_binary_bits(packed, length):
    x = packed.to(torch.int16).reshape(-1, 1)
    shifts = torch.arange(
        8,
        device=packed.device,
        dtype=torch.int16,
    ).view(1, 8)
    bits = ((x >> shifts) & 1).to(torch.bool).reshape(-1)
    return bits[:length]

class ProjectedMomentOracleAdamW:
    def __init__(self,params,mode,group_size=256,chunk_elems=MOMENT_HETEROCODE_CHUNK_ELEMS):
        self.params=list(params)
        self.mode=mode
        self.group_size=int(group_size)
        self.chunk_elems=int(chunk_elems)
        assert self.chunk_elems%self.group_size==0
        self.step_count=0
        self.pair=[]; self.qv=[]; self.qr=[]
        self.v_log_min=[]; self.v_log_step=[]; self.r_scale=[]
        self.m_exact=[]; self.v_exact=[]
        sdtype=torch.float32 if mode==PROJECTED_BOTH_FP32_SIDECAR else torch.float16
        for p in self.params:
            n=p.numel(); ng=(n+self.group_size-1)//self.group_size
            if mode in [PROJECTED_BOTH_FP16_SIDECAR,PROJECTED_BOTH_FP32_SIDECAR]:
                self.pair.append(torch.zeros((n,2),device=p.device,dtype=torch.uint8))
                self.v_log_min.append(torch.zeros(ng,device=p.device,dtype=sdtype))
                self.v_log_step.append(torch.zeros(ng,device=p.device,dtype=sdtype))
                self.r_scale.append(torch.zeros(ng,device=p.device,dtype=sdtype))
            elif mode==PROJECTED_V_ONLY:
                self.qv.append(torch.zeros(n,device=p.device,dtype=torch.uint8))
                self.v_log_min.append(torch.zeros(ng,device=p.device,dtype=torch.float16))
                self.v_log_step.append(torch.zeros(ng,device=p.device,dtype=torch.float16))
                self.m_exact.append(torch.zeros_like(p,dtype=torch.float16))
            elif mode==PROJECTED_RATIO_ONLY:
                self.qr.append(torch.full((n,),128,device=p.device,dtype=torch.uint8))
                self.r_scale.append(torch.zeros(ng,device=p.device,dtype=torch.float16))
                self.v_exact.append(torch.zeros_like(p,dtype=torch.bfloat16))
            else:
                raise ValueError(mode)
        self.last_metrics=None

    def zero_grad(self,set_to_none=True):
        for p in self.params:
            if set_to_none: p.grad=None
            elif p.grad is not None: p.grad.zero_()

    def _encode_v(self,i,start,end,v_target,qdst):
        G=self.group_size; L=end-start; ng=(L+G-1)//G; P=ng*G; g0=start//G
        pad=torch.zeros(P,device=v_target.device,dtype=torch.float32); pad[:L]=v_target
        v2=pad.view(ng,G); pos=v2>0; lv=torch.log2(v2.clamp_min(1e-30))
        lo_raw=torch.where(pos,lv,torch.full_like(lv,float("inf"))).amin(dim=1)
        hi_raw=torch.where(pos,lv,torch.full_like(lv,float("-inf"))).amax(dim=1)
        has=pos.any(dim=1); lo=torch.where(has,lo_raw,torch.zeros_like(lo_raw)); hi=torch.where(has,hi_raw,lo)
        span=(hi-lo).clamp_min(0.0); st=torch.where(span>0,span/254.0,torch.zeros_like(span))
        safe=torch.where(st>0,st,torch.ones_like(st))
        qpos=1+torch.round((lv-lo.unsqueeze(1))/safe.unsqueeze(1)).clamp(0,254)
        q=torch.where(pos,qpos,torch.zeros_like(qpos)).to(torch.uint8)
        self.v_log_min[i][g0:g0+ng].copy_(lo.to(self.v_log_min[i].dtype))
        self.v_log_step[i][g0:g0+ng].copy_(st.to(self.v_log_step[i].dtype))
        qdst[start:end].copy_(q.reshape(-1)[:L])

    def _decode_v(self,i,start,end,qsrc):
        G=self.group_size; L=end-start; ng=(L+G-1)//G; P=ng*G; g0=start//G
        flat=torch.zeros(P,device=qsrc.device,dtype=torch.int16); flat[:L]=qsrc[start:end].to(torch.int16)
        q=flat.view(ng,G)
        lo=self.v_log_min[i][g0:g0+ng].float().unsqueeze(1)
        st=self.v_log_step[i][g0:g0+ng].float().unsqueeze(1)
        pos=q>0; lv=lo+(q.float()-1.0)*st
        return torch.where(pos,torch.exp2(lv),torch.zeros_like(lv)).reshape(-1)[:L]

    def _encode_ratio(self,i,start,end,m_target,v_target,qdst):
        G=self.group_size; L=end-start; ng=(L+G-1)//G; P=ng*G; g0=start//G
        mp=torch.zeros(P,device=m_target.device,dtype=torch.float32); vp=torch.zeros(P,device=v_target.device,dtype=torch.float32)
        mp[:L]=m_target; vp[:L]=v_target
        m2=mp.view(ng,G); v2=vp.view(ng,G)
        ratio=m2/(v2.sqrt()+ADAMW_EPS)
        ma=ratio.abs().amax(dim=1); rs=torch.where(ma>0,ma/127.0,torch.zeros_like(ma))
        safe=torch.where(rs>0,rs,torch.ones_like(rs))
        qr=torch.round(ratio/safe.unsqueeze(1)).clamp(-127,127)
        qr=torch.where(rs.unsqueeze(1)>0,qr,torch.zeros_like(qr)).to(torch.int16)
        qu=(qr+128).to(torch.uint8)
        self.r_scale[i][g0:g0+ng].copy_(rs.to(self.r_scale[i].dtype))
        qdst[start:end].copy_(qu.reshape(-1)[:L])

    def _decode_ratio(self,i,start,end,qsrc,v_use):
        G=self.group_size; L=end-start; ng=(L+G-1)//G; P=ng*G; g0=start//G
        flat=torch.zeros(P,device=qsrc.device,dtype=torch.int16); flat[:L]=qsrc[start:end].to(torch.int16)-128
        q=flat.view(ng,G); rs=self.r_scale[i][g0:g0+ng].float().unsqueeze(1)
        r=(q.float()*rs).reshape(-1)[:L]
        return r*(v_use.sqrt()+ADAMW_EPS)

    @torch.no_grad()
    def step(self):
        self.step_count+=1
        beta1,beta2=ADAMW_BETAS
        bc1=1.0-beta1**self.step_count; bc2=1.0-beta2**self.step_count
        step_size=ADAMW_LR/bc1; sqrt_bc2=math.sqrt(bc2)
        ss={"m_se":0.0,"m_ref":0.0,"v_se":0.0,"v_ref":0.0,"v_zero":0}
        for i,p in enumerate(self.params):
            if p.grad is None: continue
            pf=p.view(-1); gf=p.grad.detach().view(-1)
            for start in range(0,pf.numel(),self.chunk_elems):
                end=min(start+self.chunk_elems,pf.numel())
                if self.mode in [PROJECTED_BOTH_FP16_SIDECAR,PROJECTED_BOTH_FP32_SIDECAR]:
                    qv=self.pair[i][:,0]; qr=self.pair[i][:,1]
                    old_v=self._decode_v(i,start,end,qv); old_m=self._decode_ratio(i,start,end,qr,old_v)
                elif self.mode==PROJECTED_V_ONLY:
                    old_m=self.m_exact[i].view(-1)[start:end].float()
                    old_v=self._decode_v(i,start,end,self.qv[i])
                else:
                    old_v=self.v_exact[i].view(-1)[start:end].float()
                    old_m=self._decode_ratio(i,start,end,self.qr[i],old_v)

                g=gf[start:end].float()
                m_raw=(old_m*beta1).add_(g,alpha=1.0-beta1)
                v_raw=(old_v*beta2).addcmul_(g,g,value=1.0-beta2)

                # Parent-state-lattice projection under test.
                m_target=m_raw.to(torch.float16).float()
                v_target=v_raw.to(torch.bfloat16).float()

                if self.mode in [PROJECTED_BOTH_FP16_SIDECAR,PROJECTED_BOTH_FP32_SIDECAR]:
                    qv=self.pair[i][:,0]; qr=self.pair[i][:,1]
                    self._encode_v(i,start,end,v_target,qv); v_use=self._decode_v(i,start,end,qv)
                    self._encode_ratio(i,start,end,m_target,v_target,qr); m_use=self._decode_ratio(i,start,end,qr,v_use)
                elif self.mode==PROJECTED_V_ONLY:
                    self.m_exact[i].view(-1)[start:end].copy_(m_target.to(torch.float16))
                    self._encode_v(i,start,end,v_target,self.qv[i])
                    m_use=self.m_exact[i].view(-1)[start:end].float(); v_use=self._decode_v(i,start,end,self.qv[i])
                else:
                    self.v_exact[i].view(-1)[start:end].copy_(v_target.to(torch.bfloat16))
                    v_use=self.v_exact[i].view(-1)[start:end].float()
                    self._encode_ratio(i,start,end,m_target,v_target,self.qr[i])
                    m_use=self._decode_ratio(i,start,end,self.qr[i],v_use)

                dm=m_use-m_target; dv=v_use-v_target
                ss["m_se"]+=float((dm*dm).sum()); ss["m_ref"]+=float((m_target*m_target).sum())
                ss["v_se"]+=float((dv*dv).sum()); ss["v_ref"]+=float((v_target*v_target).sum())
                ss["v_zero"]+=int(((v_target>0)&(v_use==0)).sum())

                denom=(v_use.sqrt()/sqrt_bc2).add_(ADAMW_EPS)
                target=pf[start:end].float()
                if ADAMW_WEIGHT_DECAY!=0.0: target.mul_(1.0-ADAMW_LR*ADAMW_WEIGHT_DECAY)
                target.addcdiv_(m_use,denom,value=-step_size)
                pf[start:end].copy_(target.to(pf.dtype))
        self.last_metrics={
            "m_nrmse":float((ss["m_se"]/max(ss["m_ref"],1e-30))**0.5),
            "v_nrmse":float((ss["v_se"]/max(ss["v_ref"],1e-30))**0.5),
            "positive_v_to_zero_count":int(ss["v_zero"]),
        }

    def state_tensors(self):
        out=[]
        if self.mode in [PROJECTED_BOTH_FP16_SIDECAR,PROJECTED_BOTH_FP32_SIDECAR]:
            for i in range(len(self.params)):
                out.extend([self.pair[i],self.v_log_min[i],self.v_log_step[i],self.r_scale[i]])
        elif self.mode==PROJECTED_V_ONLY:
            for i in range(len(self.params)):
                out.extend([self.m_exact[i],self.qv[i],self.v_log_min[i],self.v_log_step[i]])
        else:
            for i in range(len(self.params)):
                out.extend([self.v_exact[i],self.qr[i],self.r_scale[i]])
        return out

class MHistoryResidualAdamW(ProjectedMomentOracleAdamW):
    """
    Deployable first-moment history repair over the existing projected ratio8/log-v8 carrier.

    Persistent deployment state:
      - base ratio8/log-v8 carrier;
      - optional group-bias FP16 residual, OR
      - packed 1-bit residual indices + two FP16 centroids/group.

    Diagnostic exact-m shadow is optional and is NOT allocated in the physical memory ruler.
    """
    def __init__(
        self,
        params,
        residual_mode,
        diagnostic_shadow=False,
    ):
        super().__init__(
            params,
            mode=PROJECTED_BOTH_FP16_SIDECAR,
            group_size=R5F0DD3_GROUP_SIZE,
            chunk_elems=MOMENT_HETEROCODE_CHUNK_ELEMS,
        )
        assert residual_mode in R5F0DD9_MODES
        self.residual_mode = residual_mode
        self.diagnostic_shadow = bool(diagnostic_shadow)
        self.step_count = 0

        self.m_bias = []
        self.m_binary_bits = []
        self.m_binary_centroids = []

        for p in self.params:
            n = p.numel()
            ng = (n + self.group_size - 1) // self.group_size

            if residual_mode == GROUP_BIAS_FP16:
                self.m_bias.append(
                    torch.zeros(
                        ng,
                        device=p.device,
                        dtype=torch.float16,
                    )
                )
                self.m_binary_bits.append(None)
                self.m_binary_centroids.append(None)

            elif residual_mode == BINARY1_TWO_CENTROID:
                self.m_bias.append(None)
                self.m_binary_bits.append(
                    torch.zeros(
                        (n + 7) // 8,
                        device=p.device,
                        dtype=torch.uint8,
                    )
                )
                self.m_binary_centroids.append(
                    torch.zeros(
                        (ng, 2),
                        device=p.device,
                        dtype=torch.float16,
                    )
                )

            else:
                self.m_bias.append(None)
                self.m_binary_bits.append(None)
                self.m_binary_centroids.append(None)

        need_shadow = (
            residual_mode == EXACT_M_ORACLE
            or self.diagnostic_shadow
        )
        self.shadow_m = (
            [
                torch.zeros_like(
                    p,
                    dtype=torch.float16,
                )
                for p in self.params
            ]
            if need_shadow
            else None
        )

        self.last_metrics = None

    def _decode_group_bias(self, i, start, end):
        G = self.group_size
        L = end - start
        ng = (L + G - 1) // G
        g0 = start // G
        return (
            self.m_bias[i][g0:g0 + ng]
            .float()
            .unsqueeze(1)
            .expand(ng, G)
            .reshape(-1)[:L]
        )

    @torch.no_grad()
    def _encode_group_bias(self, i, start, end, residual):
        G = self.group_size
        L = end - start
        ng = (L + G - 1) // G
        P = ng * G
        g0 = start // G

        pad = torch.zeros(
            P,
            device=residual.device,
            dtype=torch.float32,
        )
        valid = torch.zeros(
            P,
            device=residual.device,
            dtype=torch.float32,
        )
        pad[:L] = residual.float()
        valid[:L] = 1.0

        r2 = pad.view(ng, G)
        v2 = valid.view(ng, G)
        count = v2.sum(dim=1).clamp_min(1.0)
        bias = (r2 * v2).sum(dim=1) / count

        self.m_bias[i][g0:g0 + ng].copy_(
            bias.to(torch.float16)
        )

    def _decode_binary(self, i, start, end):
        assert start % 8 == 0
        G = self.group_size
        L = end - start
        ng = (L + G - 1) // G
        g0 = start // G

        b0 = start // 8
        nb = (L + 7) // 8
        packed = self.m_binary_bits[i][b0:b0 + nb]
        bits = _unpack_binary_bits(packed, L)

        idx = torch.arange(
            L,
            device=packed.device,
            dtype=torch.long,
        )
        local_g = idx // G
        cent = self.m_binary_centroids[i][g0:g0 + ng].float()

        return torch.where(
            bits,
            cent[local_g, 1],
            cent[local_g, 0],
        )

    @torch.no_grad()
    def _encode_binary(self, i, start, end, residual):
        assert start % 8 == 0
        G = self.group_size
        L = end - start
        ng = (L + G - 1) // G
        P = ng * G
        g0 = start // G

        pad = torch.zeros(
            P,
            device=residual.device,
            dtype=torch.float32,
        )
        valid = torch.zeros(
            P,
            device=residual.device,
            dtype=torch.bool,
        )
        pad[:L] = residual.float()
        valid[:L] = True

        r2 = pad.view(ng, G)
        valid2 = valid.view(ng, G)

        pos = (r2 > 0) & valid2
        neg = (~pos) & valid2

        pos_count = pos.sum(dim=1).clamp_min(1)
        neg_count = neg.sum(dim=1).clamp_min(1)

        c1 = torch.where(
            pos.any(dim=1),
            (r2 * pos).sum(dim=1) / pos_count,
            torch.zeros(ng, device=residual.device),
        )
        c0 = torch.where(
            neg.any(dim=1),
            (r2 * neg).sum(dim=1) / neg_count,
            torch.zeros(ng, device=residual.device),
        )

        bits = pos.reshape(-1)[:L]
        packed = _pack_binary_bits(bits)

        b0 = start // 8
        self.m_binary_bits[i][b0:b0 + packed.numel()].copy_(packed)
        self.m_binary_centroids[i][g0:g0 + ng].copy_(
            torch.stack([c0, c1], dim=1).to(torch.float16)
        )

    def _decode_residual(self, i, start, end):
        if self.residual_mode == GROUP_BIAS_FP16:
            return self._decode_group_bias(i, start, end)
        if self.residual_mode == BINARY1_TWO_CENTROID:
            return self._decode_binary(i, start, end)
        return torch.zeros(
            end - start,
            device=self.params[i].device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _encode_residual(self, i, start, end, residual):
        if self.residual_mode == GROUP_BIAS_FP16:
            self._encode_group_bias(i, start, end, residual)
        elif self.residual_mode == BINARY1_TWO_CENTROID:
            self._encode_binary(i, start, end, residual)

    @torch.no_grad()
    def step(self):
        self.step_count += 1

        beta1, beta2 = ADAMW_BETAS
        bc1 = 1.0 - beta1 ** self.step_count
        bc2 = 1.0 - beta2 ** self.step_count
        step_size = ADAMW_LR / bc1
        sqrt_bc2 = math.sqrt(bc2)

        hist_m_se = hist_m_ref = 0.0
        residual_se = residual_ref = 0.0
        v_zero = 0

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            pf = p.view(-1)
            gf = p.grad.detach().view(-1)
            qv = self.pair[i][:, 0]
            qr = self.pair[i][:, 1]

            for start in range(0, pf.numel(), self.chunk_elems):
                end = min(start + self.chunk_elems, pf.numel())

                comp_old_v = self._decode_v(i, start, end, qv)
                comp_old_m = self._decode_ratio(
                    i, start, end, qr, comp_old_v
                )

                if self.residual_mode == EXACT_M_ORACLE:
                    old_m = (
                        self.shadow_m[i]
                        .view(-1)[start:end]
                        .float()
                    )
                elif self.residual_mode in [
                    GROUP_BIAS_FP16,
                    BINARY1_TWO_CENTROID,
                ]:
                    old_m = (
                        comp_old_m
                        + self._decode_residual(i, start, end)
                    )
                else:
                    old_m = comp_old_m

                old_v = comp_old_v
                g = gf[start:end].float()

                # Optional exact-m diagnostic shadow follows the exact first-moment recurrence
                # under the same actual candidate gradient sequence.
                exact_old_m = None
                exact_m_target = None
                if self.shadow_m is not None:
                    exact_old_m = (
                        self.shadow_m[i]
                        .view(-1)[start:end]
                        .float()
                    )
                    exact_m_raw = (exact_old_m * beta1).add(
                        g,
                        alpha=1.0 - beta1,
                    )
                    exact_m_target = (
                        exact_m_raw
                        .to(torch.float16)
                        .float()
                    )
                    self.shadow_m[i].view(-1)[start:end].copy_(
                        exact_m_target.to(torch.float16)
                    )

                    dm_hist = old_m - exact_old_m
                    hist_m_se += float((dm_hist * dm_hist).sum())
                    hist_m_ref += float(
                        (exact_old_m * exact_old_m).sum()
                    )

                m_raw = (old_m * beta1).add(
                    g,
                    alpha=1.0 - beta1,
                )
                v_raw = (old_v * beta2).addcmul(
                    g,
                    g,
                    value=1.0 - beta2,
                )

                m_target = m_raw.to(torch.float16).float()
                v_target = v_raw.to(torch.bfloat16).float()

                # Persist unchanged projected ratio8 + log-v8 base carrier.
                self._encode_v(i, start, end, v_target, qv)
                v_dec = self._decode_v(i, start, end, qv)

                self._encode_ratio(
                    i,
                    start,
                    end,
                    m_target,
                    v_target,
                    qr,
                )
                m_dec = self._decode_ratio(
                    i,
                    start,
                    end,
                    qr,
                    v_dec,
                )

                v_zero += int(
                    ((v_target > 0) & (v_dec == 0)).sum()
                )

                # Error-feedback residual targets the NEXT step's m-history.
                residual_target = m_target - m_dec
                self._encode_residual(
                    i,
                    start,
                    end,
                    residual_target,
                )

                if self.residual_mode in [
                    GROUP_BIAS_FP16,
                    BINARY1_TWO_CENTROID,
                ]:
                    residual_hat = self._decode_residual(
                        i,
                        start,
                        end,
                    )
                    dr = residual_hat - residual_target
                    residual_se += float((dr * dr).sum())
                    residual_ref += float(
                        (residual_target * residual_target).sum()
                    )

                # Frozen D9 contract: current update uses exact target state.
                u_target = m_target / (
                    v_target.sqrt() / sqrt_bc2
                    + ADAMW_EPS
                )

                target = pf[start:end].float()
                if ADAMW_WEIGHT_DECAY != 0.0:
                    target.mul_(
                        1.0
                        - ADAMW_LR
                        * ADAMW_WEIGHT_DECAY
                    )
                target.add_(
                    u_target,
                    alpha=-step_size,
                )
                pf[start:end].copy_(
                    target.to(pf.dtype)
                )

        self.last_metrics = {
            "history_m_nrmse": (
                float(
                    (
                        hist_m_se
                        / max(hist_m_ref, 1e-30)
                    ) ** 0.5
                )
                if self.shadow_m is not None
                else None
            ),
            "residual_reconstruction_nrmse": (
                float(
                    (
                        residual_se
                        / max(residual_ref, 1e-30)
                    ) ** 0.5
                )
                if self.residual_mode in [
                    GROUP_BIAS_FP16,
                    BINARY1_TWO_CENTROID,
                ]
                else None
            ),
            "positive_v_to_zero_count": int(v_zero),
        }

    def state_tensors(self):
        out = super().state_tensors()

        if self.residual_mode == GROUP_BIAS_FP16:
            out.extend(self.m_bias)

        elif self.residual_mode == BINARY1_TWO_CENTROID:
            for bits, cent in zip(
                self.m_binary_bits,
                self.m_binary_centroids,
            ):
                out.extend([bits, cent])

        elif self.residual_mode == EXACT_M_ORACLE:
            # Exact-m oracle state is intentionally included as diagnostic state.
            out.extend(self.shadow_m)

        # diagnostic_shadow for deployable candidates is deliberately NOT
        # part of deployment-state accounting and is disabled in physical rulers.
        return out

@torch.no_grad()
def r48_d9_step_no_diagnostics(self):
    if self.residual_mode != BINARY1_TWO_CENTROID or self.shadow_m is not None:
        raise RuntimeError('R48 fast candidate supports frozen D9 Binary1 without diagnostic shadow only')
    self.step_count += 1
    beta1, beta2 = ADAMW_BETAS
    bc1 = 1.0 - beta1 ** self.step_count
    bc2 = 1.0 - beta2 ** self.step_count
    step_size = ADAMW_LR / bc1
    sqrt_bc2 = math.sqrt(bc2)
    for i, p in enumerate(self.params):
        if p.grad is None:
            continue
        pf = p.view(-1)
        gf = p.grad.detach().view(-1)
        qv = self.pair[i][:, 0]
        qr = self.pair[i][:, 1]
        for start in range(0, pf.numel(), self.chunk_elems):
            end = min(start + self.chunk_elems, pf.numel())
            comp_old_v = self._decode_v(i, start, end, qv)
            comp_old_m = self._decode_ratio(i, start, end, qr, comp_old_v)
            old_m = comp_old_m + self._decode_residual(i, start, end)
            old_v = comp_old_v
            g = gf[start:end].float()
            m_raw = (old_m * beta1).add(g, alpha=1.0 - beta1)
            v_raw = (old_v * beta2).addcmul(g, g, value=1.0 - beta2)
            m_target = m_raw.to(torch.float16).float()
            v_target = v_raw.to(torch.bfloat16).float()
            # Carrier state: unchanged order, bit widths, and rounding.
            self._encode_v(i, start, end, v_target, qv)
            v_dec = self._decode_v(i, start, end, qv)
            self._encode_ratio(i, start, end, m_target, v_target, qr)
            m_dec = self._decode_ratio(i, start, end, qr, v_dec)
            residual_target = m_target - m_dec
            self._encode_residual(i, start, end, residual_target)
            # Frozen D9 update still uses target moments, not decoded carriers.
            u_target = m_target / (v_target.sqrt() / sqrt_bc2 + ADAMW_EPS)
            target = pf[start:end].float()
            if ADAMW_WEIGHT_DECAY != 0.0:
                target.mul_(1.0 - ADAMW_LR * ADAMW_WEIGHT_DECAY)
            target.add_(u_target, alpha=-step_size)
            pf[start:end].copy_(target.to(pf.dtype))
    # Purely diagnostic metadata is intentionally unavailable in this fast mode.
    self.last_metrics = None


class R48D9NoDiagnostic(MHistoryResidualAdamW):
    step = r48_d9_step_no_diagnostics
