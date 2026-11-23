"""R60 single-case GPU worker. One CUDA process per S x consumer variant."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,platform,statistics,time,traceback
from pathlib import Path
import torch
from .tiled_loss import tiled_causal_ce,reference_causal_ce,synthetic_parity
from .eightbit_compat import compatible_kwargs,validate_quantized_state

ROOT=Path(__file__).resolve().parent.parent
P=json.loads((ROOT/'PROTOCOL_R60.json').read_text())
MIB=1048576
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

def digest(x:torch.Tensor)->str:
    y=x.detach().cpu().contiguous()
    return hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()

def model_hash(model)->str:
    h=hashlib.sha256()
    for n,p in model.named_parameters():h.update(n.encode());h.update(bytes.fromhex(digest(p)))
    return h.hexdigest()

def make_batches(tok,seq):
    pool=[]
    for prompt in PROMPTS:
        t=tok.encode(prompt,add_special_tokens=False)
        if not t:raise ValueError('empty prompt')
        pool.append((t+[tok.eos_token_id])*(seq//(len(t)+1)+2))
    bs=[]
    for i in range(P['warmup_updates']+P['measured_updates']+1):
        rows=[pool[(i+j)%len(pool)][:seq] for j in range(P['batch_size'])]
        bs.append(torch.tensor(rows,dtype=torch.long,device='cuda'))
    return bs

def state_bytes(opt):
    seen=set();total=0
    def walk(v):
        nonlocal total
        if isinstance(v,torch.Tensor):
            if v.device.type!='cuda':return
            st=v.untyped_storage();key=(v.device.index,st.data_ptr())
            if key not in seen:seen.add(key);total+=st.nbytes()
        elif isinstance(v,dict):
            for x in v.values():walk(x)
        elif isinstance(v,(tuple,list)):
            for x in v:walk(x)
    walk(opt.state)
    return total

def run(arm:str,seq:int,destination:str):
    rec={'round':P['round'],'arm':arm,'sequence_length':seq,'status':'ERROR','steps':[],
         'errors':[],'environment':{'python':platform.python_version(), 'torch':torch.__version__,
                                   'cuda':torch.version.cuda}}
    try:
        if not torch.cuda.is_available():rec['status']='NO_CUDA';return rec
        props=torch.cuda.get_device_properties(0)
        rec['environment'].update(gpu=props.name,gpu_total_mib=props.total_memory/MIB)
        if seq==512 and props.total_memory/MIB<P['capacity_skip_512_below_mib']:
            rec['status']='SKIPPED_CAPACITY';return rec
        from transformers import AutoModelForCausalLM,AutoTokenizer
        import bitsandbytes as bnb
        rec['environment']['transformers']=importlib.metadata.version('transformers')
        rec['environment']['bitsandbytes']=importlib.metadata.version('bitsandbytes')
        parity=synthetic_parity('cuda',tile_tokens=7)
        rec['parity']=parity
        gate=P['gate']
        if (parity['loss_abs_gap']>gate['parity_loss_abs_max'] or
            parity['hidden_grad_rel_l2']>gate['parity_hidden_grad_rel_max'] or
            parity['weight_grad_rel_l2']>gate['parity_weight_grad_rel_max']):
            rec['status']='MATH_PARITY_FAIL';return rec
        torch.manual_seed(P['seed'])
        model=AutoModelForCausalLM.from_pretrained(P['model']).to('cuda')
        rec['parameter_count']=sum(p.numel() for p in model.parameters())
        if rec['parameter_count']!=P['expected_parameters']:raise ValueError('unexpected model parameter count')
        if not hasattr(model,'transformer') or not hasattr(model,'lm_head'):
            raise ValueError('Expected GPT-2 transformer and tied LM head')
        model.train();model.config.use_cache=False
        ndrop=0
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.0;ndrop+=1
        if not ndrop:raise RuntimeError('GPT2 dropout modules were not found')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            'use_reentrant':P['checkpoint_use_reentrant']})
        rec['checkpoint_enabled']=bool(getattr(model,'is_gradient_checkpointing',False))
        rec['dropout_modules_disabled']=ndrop
        if not rec['checkpoint_enabled']:raise RuntimeError('checkpoint not activated')
        tok=AutoTokenizer.from_pretrained(P['model'])
        batches=make_batches(tok,seq)
        rec['input_sha256']=hashlib.sha256(b''.join(x.cpu().numpy().tobytes() for x in batches)).hexdigest()
        rec['initial_parameter_sha256']=model_hash(model)
        k=P['optimizer']; common=dict(lr=k['lr'],betas=tuple(k['betas']),eps=k['eps'],weight_decay=k['weight_decay'])
        kw,meta=compatible_kwargs(bnb.optim.AdamW8bit,common,P['eightbit'])
        optimizer=bnb.optim.AdamW8bit(list(model.parameters()),**kw)
        rec['optimizer_meta']=meta
        params=list(model.parameters())
        for step in range(P['warmup_updates']+P['measured_updates']):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize();before=torch.cuda.memory_allocated()/MIB
            torch.cuda.reset_peak_memory_stats();stamp=time.perf_counter()
            hidden_result=model.transformer(input_ids=batches[step],use_cache=False,return_dict=True)
            hidden=hidden_result.last_hidden_state
            if arm=='FULL':
                loss=reference_causal_ce(hidden,model.lm_head.weight,batches[step])
            elif arm=='TILED':
                loss=tiled_causal_ce(hidden,model.lm_head.weight,batches[step],tile_tokens=P['tile_tokens'])
            else:raise ValueError('invalid arm')
            # Do not artificially retain the output object/logits after building loss.
            del hidden_result,hidden
            torch.cuda.synchronize();fms=(time.perf_counter()-stamp)*1000
            fpk=torch.cuda.max_memory_allocated()/MIB
            torch.cuda.reset_peak_memory_stats();stamp=time.perf_counter()
            loss.backward();torch.cuda.synchronize()
            bms=(time.perf_counter()-stamp)*1000;bpk=torch.cuda.max_memory_allocated()/MIB
            torch.cuda.reset_peak_memory_stats();stamp=time.perf_counter()
            norm=torch.nn.utils.clip_grad_norm_(params,k['grad_clip_norm'])
            optimizer.step();torch.cuda.synchronize()
            oms=(time.perf_counter()-stamp)*1000;opk=torch.cuda.max_memory_allocated()/MIB
            if step==0:
                rec['quantization_audit']=validate_quantized_state(optimizer,min_size=P['eightbit']['min_8bit_size'])
            rec['steps'].append({'step':step,'warmup':step<P['warmup_updates'],
                'loss':float(loss.detach()),'start_allocated_mib':before,
                'forward_peak_mib':fpk,'backward_peak_mib':bpk,'optimizer_peak_mib':opk,
                'peak_mib':max(fpk,bpk,opk),'forward_ms':fms,'backward_ms':bms,
                'optimizer_ms':oms,'step_ms':fms+bms+oms,
                'grad_norm_before_clip':float(norm.detach())})
            del norm,loss
        measured=rec['steps'][P['warmup_updates']:]
        rec['peak_mib']=max(x['peak_mib'] for x in measured)
        rec['median_step_ms']=statistics.median(x['step_ms'] for x in measured)
        rec['peak_stage']=max(('forward','backward','optimizer'),key=lambda q:max(x[q+'_peak_mib'] for x in measured))
        rec['resident_state_bytes']=state_bytes(optimizer)
        model.eval()
        with torch.no_grad():
            rec['final_probe_loss']=float(model(input_ids=batches[-1],labels=batches[-1],use_cache=False).loss.detach())
        rec['final_parameter_sha256']=model_hash(model)
        rec['status']='OK'
    except Exception as ex:
        rec['status']='ERROR'
        rec['errors'].append({'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()[-12000:]})
    finally:
        Path(destination).write_text(json.dumps(rec,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print('[R60]',seq,arm,rec['status'],flush=True)
    return rec

def main():
    a=argparse.ArgumentParser();a.add_argument('--seq',type=int,required=True,choices=P['sequence_lengths']);a.add_argument('--arm',required=True,choices=P['arms']);a.add_argument('--out',required=True)
    v=a.parse_args();run(v.arm,v.seq,v.out)
if __name__=='__main__':main()
