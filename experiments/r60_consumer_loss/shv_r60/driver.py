"""R60 experiment orchestration: same-run strict baselines, phase gates, archive on failure."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,platform,subprocess,sys,traceback,zipfile
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent.parent
P=json.loads((ROOT/'PROTOCOL_R60.json').read_text())
ARCH='SHV_R60_LC_Q31_A0_RESULT_UPLOAD_TO_CHATGPT.zip'

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def compare(base,trial):
    if base.get('status')!='OK' or trial.get('status')!='OK':
        return {'status':'INCOMPLETE','base':base.get('status'),'trial':trial.get('status')}
    keys=('input_sha256','initial_parameter_sha256','parameter_count','checkpoint_enabled','resident_state_bytes')
    envkeys=('torch','cuda','gpu','transformers','bitsandbytes')
    integ={k:base.get(k)==trial.get(k) for k in keys}
    integ.update({f'env.{k}':base.get('environment',{}).get(k)==trial.get('environment',{}).get(k) for k in envkeys})
    bq=base.get('quantization_audit',{});tq=trial.get('quantization_audit',{})
    true8=all(q.get('large_active_parameters',0)>0 and q.get('large_active_parameters')==q.get('quantized_uint8_moment_pairs')==q.get('blockwise_absmax_pairs') for q in (bq,tq))
    bs=base['steps'];ts=trial['steps']
    gaps=[abs(b['loss']-t['loss'])/max(abs(b['loss']),1e-12) for b,t in zip(bs,ts)]
    probe=abs(base['final_probe_loss']-trial['final_probe_loss'])/max(abs(base['final_probe_loss']),1e-12)
    memory_ratio=trial['peak_mib']/base['peak_mib']
    time_ratio=trial['median_step_ms']/base['median_step_ms']
    g=P['gate'];quality=max(gaps,default=float('inf'))<=g['relative_loss_trajectory_max'] and probe<=g['relative_probe_max']
    return {'status':'COMPARED' if all(integ.values()) and true8 and quality else 'HOLD',
      'integrity':integ,'true_8bit':true8,'quality_pass':quality,
      'relative_loss_max':max(gaps,default=None),'relative_probe_gap':probe,
      'memory_ratio':memory_ratio,'peak_saving':1-memory_ratio,'step_time_ratio':time_ratio,
      'baseline_mib':base['peak_mib'],'candidate_mib':trial['peak_mib'],
      'baseline_ms':base['median_step_ms'],'candidate_ms':trial['median_step_ms'],
      'baseline_peak_stage':base['peak_stage'],'candidate_peak_stage':trial['peak_stage']}

def decide(s128,s512):
    if s128.get('status')!='COMPARED':return 'R60_INTEGRITY_OR_MATH_HOLD'
    g=P['gate']
    if s128['memory_ratio']>g['s128_max_peak_ratio'] or s128['step_time_ratio']>g['s128_max_step_ratio']:
        return 'R60_S128_EARLY_STOP'
    if s512.get('status')!='COMPARED':return 'R60_INCOMPLETE_HOLD'
    if s512['peak_saving']>=g['s512_required_peak_saving'] and s512['step_time_ratio']<=g['s512_max_step_ratio']:
        return 'R60_CONSUMER_PREQUAL_PASS'
    return 'R60_CONSUMER_PREQUAL_FAIL'

def run(output=None):
    out=Path(output or os.getcwd()).resolve();out.mkdir(parents=True,exist_ok=True)
    work=out/('R60_RUN_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    work.mkdir()
    summary={'round':P['round'],'status':'R60_INCOMPLETE_HOLD','no_shv_unique_claim':True,
             'checkpoint_baseline':True,'sealed_guard':'NOT_TOUCHED','cases':{},'comparison':{},
             'interpretation':'Same-run consumer-scoped loss optimization only; not proof of unique SHV advantage'}
    logs=[]
    try:
        import torch
        if not torch.cuda.is_available():
            summary['status']='R60_NO_CUDA';return summary
        for seq in P['sequence_lengths']:
            for arm in P['arms']:
                dst=work/f'S{seq}_{arm}.json'
                cmd=[sys.executable,'-m','shv_r60.worker','--seq',str(seq),'--arm',arm,'--out',str(dst)]
                try:
                    c=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=P['max_seconds_per_worker'],env={**os.environ,'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')})
                    logs.append({'seq':seq,'arm':arm,'returncode':c.returncode,'stdout_tail':c.stdout[-4000:],'stderr_tail':c.stderr[-4000:]})
                    if dst.exists():rec=json.loads(dst.read_text())
                    else:rec={'status':'WORKER_MISSING_FILE'}
                except subprocess.TimeoutExpired as ex:
                    rec={'status':'TIMEOUT','errors':[str(ex)]};save(dst,rec)
                    logs.append({'seq':seq,'arm':arm,'returncode':'TIMEOUT'})
                summary['cases'][f'S{seq}_{arm}']={k:rec.get(k) for k in ('status','peak_mib','median_step_ms','peak_stage','resident_state_bytes','final_probe_loss')}
            l=json.loads((work/f'S{seq}_FULL.json').read_text())
            r=json.loads((work/f'S{seq}_TILED.json').read_text())
            summary['comparison'][str(seq)]=compare(l,r)
            if seq==128 and decide(summary['comparison']['128'],{'status':'INCOMPLETE'})=='R60_S128_EARLY_STOP':
                summary['status']='R60_S128_EARLY_STOP';break
            if seq==128 and summary['comparison']['128'].get('status')!='COMPARED':
                summary['status']='R60_INTEGRITY_OR_MATH_HOLD';break
        if '512' in summary['comparison']:
            summary['status']=decide(summary['comparison']['128'],summary['comparison']['512'])
        return summary
    except Exception as ex:
        summary['status']='R60_EXCEPTION'
        summary['error']={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()[-12000:]}
        return summary
    finally:
        save(work/'summary.json',summary);save(work/'protocol.json',P);save(work/'logs.json',logs)
        env={'python':platform.python_version(),'platform':platform.platform()}
        for name in ('torch','transformers','bitsandbytes','triton'):
            try:env[name]=importlib.metadata.version(name)
            except Exception:env[name]=None
        save(work/'environment.json',env)
        baseline=ROOT/'evidence/r59_reference.json'
        if baseline.exists():save(work/'reference_r59.json',json.loads(baseline.read_text()))
        sources={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(ROOT.glob('shv_r60/*.py'))}
        sources['PROTOCOL_R60.json']=sha((ROOT/'PROTOCOL_R60.json').read_bytes())
        save(work/'source_sha256.json',sources)
        filelist=sorted(x for x in work.iterdir() if x.is_file())
        save(work/'MANIFEST.json',{'file_sha256':{f.name:sha(f.read_bytes()) for f in filelist}})
        arch=out/ARCH
        with zipfile.ZipFile(arch,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
            for f in sorted(work.iterdir()):z.write(f,f.name)
        with zipfile.ZipFile(arch) as z:
            if z.testzip():raise RuntimeError('archive CRC check failed')
            for n,d in json.loads(z.read('MANIFEST.json'))['file_sha256'].items():
                if sha(z.read(n))!=d:raise RuntimeError('manifest mismatch '+n)
        (out/(ARCH+'.sha256')).write_text(sha(arch.read_bytes())+'  '+ARCH+'\n')
        print('[R60]',summary['status'],arch,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=None)
    a=p.parse_args();run(a.output)
if __name__=='__main__':main()
