"""R62 public consumer-lifetime benchmark: matched TILED vs unmodified installed Liger.

Profiler attribution is allocation *traffic*, not reclaimable peak bytes.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform, statistics, time, traceback
from pathlib import Path
import torch
from .loss import tiled_causal_ce, synthetic_parity
from .external import liger_causal_ce, get_liger_version
from .eightbit_compat import compatible_kwargs, validate_quantized_state

ROOT=Path(__file__).resolve().parent.parent
P=json.loads((ROOT/'PROTOCOL_R62.json').read_text('utf-8'))
MIB=1024**2
PROMPTS=(
 'The North Atlantic overturning circulation transports heat across the ocean basin. ',
 'A numerical weather prediction model integrates atmospheric dynamics at discrete time steps. ',
 'Neural networks store activations for backpropagation or recompute them on demand. ',
 'An optimizer maintains estimates of the first and second moments of stochastic gradients. ',
 'Careful research compares training quality and throughput under the same initialization. ',
 'Data compression can lower memory requirements at a cost of extra decoding operations. ',
 'Ocean waves exchange momentum and energy with the atmosphere and underlying currents. ',
 'High quality scientific experiments require independent controls and reproducible code. '
)

def digest(x):
    b=x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()

def model_hash(model):
    h=hashlib.sha256()
    for name, par in model.named_parameters():
        h.update(name.encode('utf-8'));h.update(bytes.fromhex(digest(par)))
    return h.hexdigest()

def make_batches(tok, seq):
    pool=[]
    for prompt in PROMPTS:
        ids=tok.encode(prompt,add_special_tokens=False)
        if not ids:raise RuntimeError('empty source token list')
        pool.append((ids+[tok.eos_token_id])*(seq//(len(ids)+1)+2))
    result=[]
    for i in range(P['warmup_updates']+P['measured_updates']+2):
        arr=[pool[(i+j)%len(pool)][:seq] for j in range(P['batch_size'])]
        result.append(torch.tensor(arr,dtype=torch.long,device='cuda'))
    return result

def state_bytes(opt):
    ids=set();count=0
    def walk(v):
        nonlocal count
        if isinstance(v,torch.Tensor):
            if v.device.type!='cuda':return
            s=v.untyped_storage();k=(v.device.index,s.data_ptr())
            if k not in ids:ids.add(k);count+=s.nbytes()
        elif isinstance(v,dict):
            for a in v.values():walk(a)
        elif isinstance(v,(tuple,list)):
            for a in v:walk(a)
    walk(opt.state)
    return count

def run(arm,repeat,out,profile=False):
    out=Path(out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    rec=dict(round=P['round'],arm=arm,repeat=repeat,profile_mode=bool(profile),status='ERROR',
             errors=[],steps=[],environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda))
    try:
        if not torch.cuda.is_available():
            rec['status']='NO_CUDA';return rec
        props=torch.cuda.get_device_properties(0)
        rec['environment'].update(gpu=props.name,capability=list(torch.cuda.get_device_capability(0)),gpu_total_mib=props.total_memory/MIB)
        if props.total_memory/MIB < P['min_gpu_memory_mib']:
            rec['status']='INSUFFICIENT_GPU_CAPACITY';return rec
        from transformers import AutoModelForCausalLM,AutoTokenizer
        import bitsandbytes as bnb
        rec['environment'].update(transformers=importlib.metadata.version('transformers'),
                                  bitsandbytes=importlib.metadata.version('bitsandbytes'),
                                  liger_kernel=get_liger_version())
        parity=synthetic_parity('cuda',tile_tokens=7)
        rec['parity']=parity
        g=P['gate']
        if parity['loss_abs_gap']>g['synthetic_loss_abs_max'] or \
           parity['hidden_grad_rel_l2']>g['synthetic_hidden_grad_rel_max'] or \
           parity['weight_grad_rel_l2']>g['synthetic_weight_grad_rel_max']:
            rec['status']='SYNTHETIC_PARITY_HOLD';return rec
        torch.manual_seed(P['seed'])
        model=AutoModelForCausalLM.from_pretrained(P['model']).to('cuda')
        rec['parameter_count']=sum(p.numel() for p in model.parameters())
        if rec['parameter_count']!=P['expected_parameters']:raise ValueError('GPT-2 parameter count changed')
        model.train();model.config.use_cache=False
        ndrop=0
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.;ndrop+=1
        if not ndrop:raise RuntimeError('No Dropout modules recognized')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            'use_reentrant':P['checkpoint_use_reentrant']})
        rec['checkpoint_enabled']=bool(getattr(model,'is_gradient_checkpointing',False))
        if not rec['checkpoint_enabled']:raise RuntimeError('Gradient checkpointing not activated')
        tok=AutoTokenizer.from_pretrained(P['model'])
        batches=make_batches(tok,P['sequence_length'])
        rec['input_sha256']=hashlib.sha256(b''.join(x.cpu().numpy().tobytes() for x in batches)).hexdigest()
        rec['initial_parameter_sha256']=model_hash(model)
        k=P['optimizer'];common=dict(lr=k['lr'],betas=tuple(k['betas']),eps=k['eps'],weight_decay=k['weight_decay'])
        kw,meta=compatible_kwargs(bnb.optim.AdamW8bit,common,P['eightbit'])
        opt=bnb.optim.AdamW8bit(list(model.parameters()),**kw)
        rec['optimizer_meta']=meta
        params=list(model.parameters())
        def runstep(i):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize();base=torch.cuda.memory_allocated()/MIB
            torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            hidden_result=model.transformer(input_ids=batches[i],use_cache=False,return_dict=True)
            hidden=hidden_result.last_hidden_state
            if arm=='TILED':loss=tiled_causal_ce(hidden,model.lm_head.weight,batches[i],P['tile_tokens'])
            elif arm=='LIGER':loss=liger_causal_ce(hidden,model.lm_head.weight,batches[i])
            else:raise ValueError('arm not recognized')
            del hidden_result, hidden
            torch.cuda.synchronize();forward_ms=(time.perf_counter()-t)*1000
            fpeak=torch.cuda.max_memory_allocated()/MIB
            torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            loss.backward();torch.cuda.synchronize()
            backward_ms=(time.perf_counter()-t)*1000;bpeak=torch.cuda.max_memory_allocated()/MIB
            torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            norm=torch.nn.utils.clip_grad_norm_(params,k['grad_clip_norm'])
            opt.step();torch.cuda.synchronize()
            optimizer_ms=(time.perf_counter()-t)*1000;opeak=torch.cuda.max_memory_allocated()/MIB
            row=dict(step=i,warmup=i<P['warmup_updates'],loss=float(loss.detach()),
                     start_allocated_mib=base,forward_peak_mib=fpeak,backward_peak_mib=bpeak,
                     optimizer_peak_mib=opeak,peak_mib=max(fpeak,bpeak,opeak),
                     forward_ms=forward_ms,backward_ms=backward_ms,optimizer_ms=optimizer_ms,
                     step_ms=forward_ms+backward_ms+optimizer_ms,grad_norm=float(norm.detach()))
            del loss,norm
            return row
        if not profile:
            for i in range(P['warmup_updates']+P['measured_updates']):
                row=runstep(i);rec['steps'].append(row)
                if i==0:rec['quantization_audit']=validate_quantized_state(opt,P['eightbit']['min_8bit_size'])
            valid=rec['steps'][P['warmup_updates']:]
            rec['peak_mib']=max(r['peak_mib'] for r in valid)
            rec['peak_stage']=max(('forward','backward','optimizer'),key=lambda stage:max(x[stage+'_peak_mib'] for x in valid))
            rec['median_step_ms']=statistics.median(r['step_ms'] for r in valid)
            rec['median_forward_ms']=statistics.median(r['forward_ms'] for r in valid)
            rec['median_backward_ms']=statistics.median(r['backward_ms'] for r in valid)
            rec['median_optimizer_ms']=statistics.median(r['optimizer_ms'] for r in valid)
            rec['resident_state_bytes']=state_bytes(opt)
            model.eval()
            with torch.no_grad():
                rec['final_probe_loss']=float(model(input_ids=batches[-1],labels=batches[-1]).loss)
            rec['final_parameter_sha256']=model_hash(model)
            rec['status']='OK'
        else:
            # Profiler is run in a fresh worker and its timings NEVER enter the performance Gate.
            for i in range(P['warmup_updates']):
                rec['steps'].append(runstep(i))
            rec['quantization_audit']=validate_quantized_state(opt,P['eightbit']['min_8bit_size'])
            from torch.profiler import profile as profiler,ProfilerActivity
            act=[ProfilerActivity.CPU,ProfilerActivity.CUDA]
            with profiler(activities=act,record_shapes=False,profile_memory=True,with_stack=False) as prof:
                pr=runstep(P['warmup_updates'])
            rows=[]
            for entry in prof.key_averages():
                mem=getattr(entry,'self_device_memory_usage',None)
                if mem is None:mem=getattr(entry,'self_cuda_memory_usage',0)
                dev_time=getattr(entry,'self_device_time_total',None)
                if dev_time is None:dev_time=getattr(entry,'self_cuda_time_total',0)
                rows.append(dict(name=str(entry.key),calls=int(entry.count),
                                 self_device_alloc_bytes=int(mem or 0),self_device_time_us=float(dev_time or 0)))
            rows=sorted(rows,key=lambda r:r['self_device_alloc_bytes'],reverse=True)
            rec['profiled_step']=pr
            rec['top_ops_by_self_device_alloc_bytes']=rows[:40]
            rec['memory_profile_note']='Self device allocation sums are allocation traffic, not live peak, saved tensor ownership, or reclaimable bytes.'
            rec['status']='PROFILE_OK'
    except Exception as ex:
        rec['status']='ERROR';rec['errors'].append(dict(type=type(ex).__name__,message=str(ex),traceback=traceback.format_exc()[-12000:]))
    finally:
        out.write_text(json.dumps(rec,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('[R62]',arm,'repeat',repeat,'profile',profile,'status',rec['status'],flush=True)
    return rec

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',required=True,choices=P['arms']);p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--profile',action='store_true');p.add_argument('--out',required=True)
    a=p.parse_args();run(a.arm,a.repeat,a.out,a.profile)

if __name__=='__main__':main()
