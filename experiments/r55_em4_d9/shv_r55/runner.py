"""R55 independent system prequalification. Frozen math, strong quick baselines.

No new unseen quality tests; controlled repeated short historical data only.
"""
from __future__ import annotations
import csv, gc, hashlib, json, os, platform, statistics, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
import torch
from . import activation
from .frozen_d9 import R48D9NoDiagnostic, BINARY1_TWO_CENTROID

HERE=Path(__file__).resolve().parent.parent
PROTOCOL=json.loads((HERE/'R55_FROZEN_PROTOCOL.json').read_text(encoding='utf8'))
FROZEN=json.loads((HERE/'FROZEN_MODULE_HASHES.json').read_text())
ROUND=PROTOCOL['round']
PROMPTS=(
 'The North Atlantic ocean circulation transports heat and influences regional climate variability. ',
 'A numerical weather prediction model integrates atmospheric dynamics across discrete time intervals. ',
 'A neural network can store activations during the forward pass or recompute them in backward propagation. ',
 'Scientific experiments must compare models under identical initialization data order and evaluation rules. ',
 'A compressed optimizer state needs careful reconstruction before an AdamW parameter update. ',
 'GPU memory use includes parameters gradients optimizer states saved tensors and temporary workspaces. ',
 'The tropical cyclone forecast has uncertainty in track intensity and translation speed. ',
 'A reproducible experiment logs parameter checksums training times peak memory and software versions. ',
 'Wind waves and swell components can coexist in the ocean directional frequency spectrum. ',
 'Optimizing arithmetic and data movement must preserve correct numerical behavior across devices. ',
 'Attention and feedforward neural network layers have different memory and computation characteristics. ',
 'Controlled experiments should not reuse sealed validation or test datasets for selecting new algorithms. ',
)

def _sha(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def check_source():return {f: _sha((HERE/'shv_r55'/f).read_bytes())==hv for f,hv in FROZEN.items()}
def savejson(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def savecsv(path,rows):
    if not rows: Path(path).write_text('no_rows\n');return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def export(out,summary,env,rows,trial_records,probe_results,inputs):
    run_dir=out/(ROUND+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run_dir.mkdir(exist_ok=True,parents=True)
    for name,obj in [('summary.json',summary),('environment.json',env),('protocol.json',PROTOCOL),('trials.json',trial_records),('probes.json',probe_results),('inputs.json',inputs)]:savejson(run_dir/name,obj)
    savecsv(run_dir/'steps.csv',rows)
    manifest={'round':ROUND,'file_sha256':{p.name:_sha(p.read_bytes()) for p in sorted(run_dir.iterdir()) if p.is_file()}}
    savejson(run_dir/'MANIFEST.json',manifest)
    target=out/PROTOCOL['output']
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run_dir.iterdir()):z.write(p,arcname=p.name)
    (out/(target.name+'.sha256')).write_text(_sha(target.read_bytes())+'  '+target.name+'\n')
    return str(target)

def sha_params(params):
    hash_value=hashlib.sha256()
    for i,p in enumerate(params):
        t=p.detach().contiguous().cpu()
        hash_value.update(f'{i}:{tuple(p.shape)}:{p.dtype};'.encode())
        hash_value.update(t.view(torch.uint8).numpy().tobytes())
    return hash_value.hexdigest()

def make_d9(params):
    opt=R48D9NoDiagnostic(list(params),residual_mode=BINARY1_TWO_CENTROID,diagnostic_shadow=False)
    opt.chunk_elems=PROTOCOL['optimizer']['d9_chunk_elems']
    assert opt.group_size==PROTOCOL['optimizer']['d9_group_size']
    return opt

def make_optimizer(params, arm):
    if arm.endswith('_D9'):return make_d9(params)
    o=PROTOCOL['optimizer']
    return torch.optim.AdamW(list(params),lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'],foreach=True)

def checkpoint_loss(ids):
    from torch.utils.checkpoint import checkpoint
    model=activation.model
    x=activation.training_embedding_forward(ids)
    for b in model.transformer.h:
        def body(hidden,block=b):
            a=block.attn(block.ln_1(hidden),use_cache=False)[0]
            y=hidden+a
            return y+block.mlp(block.ln_2(y))
        x=checkpoint(body,x,use_reentrant=False,preserve_rng_state=True)
    x=model.transformer.ln_f(x)
    return activation.causal_lm_loss(model.lm_head(x),ids)

def training_loss(ids,arm):
    if arm=='N_CKPT_FOREACH':return checkpoint_loss(ids)
    if arm.startswith('EM4'):return activation.r45_loss(ids,'EM4')
    return activation.r45_loss(ids,'N')

def make_batches(tokenizer,device):
    T=PROTOCOL['sequence_length'];B=PROTOCOL['batch_size']
    sequences=[]
    for prompt in PROMPTS:
        values=tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.eos_token_id]
        if not values:raise RuntimeError('empty token sequence')
        sequences.append(torch.tensor((values*(T//len(values)+2))[:T],dtype=torch.long))
    train=[]
    for step in range(PROTOCOL['steps']):
        ids=torch.stack([sequences[(step+j*2)%8] for j in range(B)]).to(device)
        train.append(ids)
    probe=torch.stack([sequences[j] for j in [8,9,10,11]]).to(device)
    return train,probe,{'train_batch_sha256':[_sha(t.cpu().numpy().tobytes()) for t in train],
                         'probe_batch_sha256':_sha(probe.cpu().numpy().tobytes()),
                         'total_prompts':len(PROMPTS),'train_prompt_ids':list(range(8)),
                         'probe_prompt_ids':[8,9,10,11],'status':'historical_controlled_nonfresh'}

def initial_forward_probe(base_model,ids):
    import copy
    model=copy.deepcopy(base_model).to('cuda')
    model.eval();model.config.use_cache=False
    activation.bind_model(model,PROTOCOL['sequence_length'])
    with torch.no_grad():
        n=float(activation.r45_loss(ids,'N').detach())
        c=float(checkpoint_loss(ids).detach())
        e=float(activation.r45_loss(ids,'EM4').detach())
    activation.model=None
    del model
    gc.collect();torch.cuda.empty_cache()
    return {'native':n,'native_checkpoint':c,'em4':e,'native_checkpoint_abs_diff':abs(n-c),'native_em4_abs_diff':abs(n-e)}

def one_trial(base_model,train,probe,arm,repetition):
    # CPU deepcopy preserves tied LM head / embeddings and original GPT-2 weights.
    import copy
    model=copy.deepcopy(base_model).to('cuda')
    model.eval();model.config.use_cache=False
    activation.bind_model(model,PROTOCOL['sequence_length'])
    params=[p for p in model.parameters() if p.requires_grad]
    initial_hash=sha_params(params)
    optimizer=make_optimizer(params,arm)
    torch.cuda.synchronize()
    out=[]
    for step,ids in enumerate(train):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        loss=training_loss(ids,arm)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,PROTOCOL['optimizer']['gradient_clip'])
        optimizer.step()
        torch.cuda.synchronize()
        elapsed_ms=(time.perf_counter()-start)*1000.
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20
        out.append({'arm':arm,'repeat':repetition,'step':step,'train_loss':float(loss.detach()),
                    'total_ms':elapsed_ms,'peak_allocated_mib':peak_allocated_mib,
                    'peak_reserved_mib':peak_reserved_mib,'grad_norm_before_clip':float(norm.detach())})
        del loss, norm
    # Diagnostic after training: never counted in training peak.
    torch.cuda.synchronize()
    with torch.no_grad():
        probe_loss=float(activation.r45_loss(probe,'N').detach())
    final_hash=sha_params(params)
    samples=out[PROTOCOL['warmup_steps_excluded']:]
    trial={'arm':arm,'repeat':repetition,'initial_hash':initial_hash,'final_hash':final_hash,
           'median_step_ms':statistics.median(x['total_ms'] for x in samples),
           'max_peak_allocated_mib':max(x['peak_allocated_mib'] for x in samples),
           'max_peak_reserved_mib':max(x['peak_reserved_mib'] for x in samples),
           'probe_loss':probe_loss,'input_tokens_per_s':PROTOCOL['batch_size']*PROTOCOL['sequence_length']*1000./statistics.median(x['total_ms'] for x in samples)}
    # No persistent GPU tensor references or model objects may survive the arm.
    activation.model=None
    del optimizer,params,model
    gc.collect();torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return out,trial

def quality_metrics(arms,records):
    by={r['arm']:r for r in records}
    if 'N_FOREACH' not in by:return {}
    b=by['N_FOREACH']
    checks={}
    for arm in arms:
        if arm=='N_FOREACH' or arm not in by:continue
        r=by[arm]
        checks[arm]={'peak_saving_vs_native':1-r['max_peak_allocated_mib']/b['max_peak_allocated_mib'],
                     'step_ratio_vs_native':r['median_step_ms']/b['median_step_ms'],
                     'probe_relative_gap_vs_native':abs(r['probe_loss']-b['probe_loss'])/max(abs(b['probe_loss']),1.e-12)}
    return checks

def qualify_repeat(records_by_rep,steps_by_rep):
    config=PROTOCOL['engineering_pilot_gates'];quality=PROTOCOL['quality_gates']; min_ok=PROTOCOL['repeat_validation']['min_success_reps']
    tests=[]
    for rep,rec in sorted(records_by_rep.items()):
        arms={r['arm']:r for r in rec}
        if 'N_FOREACH' not in arms or 'EM4_FOREACH' not in arms:continue
        b=arms['N_FOREACH'];a=arms['EM4_FOREACH']
        baseline_train={x['step']:x['train_loss'] for x in steps_by_rep[rep] if x['arm']=='N_FOREACH'}
        em4_steps=[x for x in steps_by_rep[rep] if x['arm']=='EM4_FOREACH']
        train_gap=max(abs(x['train_loss']-baseline_train[x['step']])/max(abs(baseline_train[x['step']]),1.e-12) for x in em4_steps)
        mem_gain=1-a['max_peak_allocated_mib']/b['max_peak_allocated_mib']
        time_ratio=a['median_step_ms']/b['median_step_ms']
        probe_gap=abs(a['probe_loss']-b['probe_loss'])/max(abs(b['probe_loss']),1.e-12)
        p={'repeat':rep,'memory_saving':mem_gain,'step_ratio':time_ratio,'max_relative_training_loss_gap':train_gap,'relative_probe_loss_gap':probe_gap}
        p['passed']= bool(mem_gain>=config['em4_vs_native_memory_saving_min'] and time_ratio<=config['em4_vs_native_step_time_ratio_max'] and train_gap<=quality['max_relative_train_loss_gap'] and probe_gap<=quality['max_relative_probe_loss_gap'])
        tests.append(p)
    return {'pairs':tests,'passed_repetitions':sum(t['passed'] for t in tests),'expected_repetitions':PROTOCOL['repeat_validation']['required_reps'],
            'historical_prequal_pass':len(tests)==PROTOCOL['repeat_validation']['required_reps'] and sum(t['passed'] for t in tests)>=min_ok}

def main(output_root=None):
    out=Path(output_root or ('/content' if Path('/content').exists() else Path.cwd()))
    out.mkdir(parents=True,exist_ok=True)
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
         'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'source_checks':check_source()}
    summary={'round':ROUND,'status':'R55_EXCEPTION','historical_only':True,'repeat_qualification':None,'observations':[],'errors':[]}
    rows=[];trial_records=[];probes=[];inputs={}
    try:
        if not all(env['source_checks'].values()):summary['status']='R55_FROZEN_SOURCE_MISMATCH'
        elif not torch.cuda.is_available():summary['status']='R55_GPU_NOT_AVAILABLE'
        else:
            from transformers import AutoModelForCausalLM,AutoTokenizer
            import transformers
            env['transformers']=transformers.__version__
            torch.manual_seed(5500);torch.cuda.manual_seed_all(5500)
            torch.backends.cuda.matmul.allow_tf32=False
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            base=AutoModelForCausalLM.from_pretrained(PROTOCOL['model_id'])
            base.eval();base.config.use_cache=False
            env['parameter_count']=sum(p.numel() for p in base.parameters())
            tokenizer=AutoTokenizer.from_pretrained(PROTOCOL['model_id'])
            train,probe,inputs=make_batches(tokenizer,'cuda')
            summary['initial_forward']=initial_forward_probe(base,train[0])
            if summary['initial_forward']['native_checkpoint_abs_diff']>PROTOCOL['quality_gates']['native_checkpoint_initial_loss_abs_tol']:
                summary['status']='R55_INITIAL_FORWARD_HOLD'
                raise RuntimeError('Native versus checkpoint initial forward loss diverged')
            if summary['initial_forward']['native_em4_abs_diff']>PROTOCOL['quality_gates']['native_checkpoint_initial_loss_abs_tol']:
                summary['status']='R55_INITIAL_FORWARD_HOLD'
                raise RuntimeError('Native versus EM4 initial forward loss diverged')
            by_rep={};steps_by_rep={}
            for rep in range(PROTOCOL['repeat_validation']['required_reps']):
                arms=PROTOCOL['variants'] if rep==0 else ['N_FOREACH','N_CKPT_FOREACH','EM4_FOREACH']
                # Rotate arm order on repeated runs to reduce systematic ordering bias.
                if rep==1:arms=list(reversed(arms))
                rep_rec=[];rep_steps=[]
                for arm in arms:
                    st,rec=one_trial(base,train,probe,arm,rep)
                    rows.extend(st);trial_records.append(rec);rep_rec.append(rec);rep_steps.extend(st)
                    print(f'[R55 rep {rep} {arm}] {rec["median_step_ms"]:.2f} ms; {rec["max_peak_allocated_mib"]:.1f} MiB; probe={rec["probe_loss"]:.6f}',flush=True)
                by_rep[rep]=rep_rec;steps_by_rep[rep]=rep_steps
                initial_set={r['initial_hash'] for r in rep_rec}
                if len(initial_set)!=1:
                    summary['status']='R55_INITIALIZATION_MISMATCH';raise RuntimeError('Models did not start from identical parameters')
                if rep==0:
                    metrics=quality_metrics(arms,rep_rec)
                    summary['pilot_metrics']=metrics
                    nc=metrics.get('N_CKPT_FOREACH',{})
                    if nc.get('probe_relative_gap_vs_native',float('inf'))>PROTOCOL['quality_gates']['native_checkpoint_final_probe_relative_tol']:
                        summary['observations'].append('CHECKPOINT_NUMERICAL_DIFFERENCE: no promotion regardless of memory/time')
                    em=metrics.get('EM4_FOREACH',{})
                    # Continue repeats if activation-only has minimally positive measurable memory and acceptable time.
                    if em.get('peak_saving_vs_native',-1)<0.02 or em.get('step_ratio_vs_native',float('inf'))>1.5:
                        summary['status']='R55_PILOT_HOLD'
                        summary['observations'].append('Stopped before repeats: EM4 activation-only not worth long performance qualification on historical pilot')
                        break
            summary['repeat_qualification']=qualify_repeat(by_rep,steps_by_rep)
            dominance=[]
            ckpt_bad=[]
            margin=PROTOCOL['repeat_validation']['checkpoint_strict_dominance_margin']
            for rep,rec in by_rep.items():
                arm_record={x['arm']:x for x in rec}
                if {'N_FOREACH','N_CKPT_FOREACH','EM4_FOREACH'}.issubset(arm_record):
                    base_n=arm_record['N_FOREACH']; ck=arm_record['N_CKPT_FOREACH']; em=arm_record['EM4_FOREACH']
                    gap=abs(ck['probe_loss']-base_n['probe_loss'])/max(abs(base_n['probe_loss']),1e-12)
                    ckpt_bad.append(gap>PROTOCOL['quality_gates']['native_checkpoint_final_probe_relative_tol'])
                    dominance.append(ck['max_peak_allocated_mib'] <= em['max_peak_allocated_mib']*(1-margin) and ck['median_step_ms']<=em['median_step_ms']*(1-margin))
            summary['checkpoint_strict_dominance_repetitions']=sum(dominance)
            summary['checkpoint_probe_failed_repetitions']=sum(ckpt_bad)
            if summary['status']!='R55_PILOT_HOLD':
                if any(ckpt_bad): summary['status']='R55_CHECKPOINT_CONTROL_HOLD'
                elif not summary['repeat_qualification']['historical_prequal_pass']: summary['status']='R55_HISTORICAL_PREQUAL_HOLD'
                elif sum(dominance)>=PROTOCOL['repeat_validation']['min_success_reps']:summary['status']='R55_CHECKPOINT_DOMINATED_HOLD'
                else:summary['status']='R55_ACTIVATION_ONLY_HISTORICAL_PREQUAL_PASS'
            summary['observations'].append('No fresh data, no long training, no promotion. D9 variants are diagnostic only, and are not numerically equivalent to FP32 AdamW.')
            del base,train,probe
            activation.model=None
            gc.collect();torch.cuda.empty_cache()
    except Exception:
        summary['errors'].append(traceback.format_exc())
        if summary['status'] not in ['R55_FROZEN_SOURCE_MISMATCH','R55_INITIALIZATION_MISMATCH','R55_GPU_NOT_AVAILABLE','R55_INITIAL_FORWARD_HOLD']:
            summary['status']='R55_EXCEPTION'
        print(summary['errors'][-1],flush=True)
    finally:
        path=export(out,summary,env,rows,trial_records,probes,inputs)
        print('[R55 status]',summary['status'],'\n[R55 file]',path,flush=True)
    return summary
if __name__=='__main__': main()
