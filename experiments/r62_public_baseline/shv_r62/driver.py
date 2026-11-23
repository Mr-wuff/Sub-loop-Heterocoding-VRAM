"""R62 three paired fresh-worker repeats + separate memory profiler runs.

Never promote a public implementation as SHV originality.
"""
from __future__ import annotations
import hashlib,json,os,platform,subprocess,sys,traceback,zipfile
from pathlib import Path
from .criteria import compare,aggregate
ROOT=Path(__file__).resolve().parent.parent
P=json.loads((ROOT/'PROTOCOL_R62.json').read_text('utf-8'))
NAME='SHV_R62_PROFILE_Q33_A0_RESULT_UPLOAD_TO_CHATGPT.zip'

def digest(data:bytes)->str:return hashlib.sha256(data).hexdigest()

def check_source():
    d=json.loads((ROOT/'SOURCE_HASHES.json').read_text('utf-8'))
    bad=[]
    for n,h in d['frozen_and_new_files'].items():
        f=ROOT/n
        if not f.exists() or digest(f.read_bytes())!=h:bad.append(n)
    if bad:raise RuntimeError('SOURCE_HASH_MISMATCH '+str(bad))
    return d

def child(arm,repeat,target,profile_mode=False):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    cmd=[sys.executable,'-m','shv_r62.worker','--arm',arm,'--repeat',str(repeat),'--out',str(target)]
    if profile_mode:cmd.append('--profile')
    try:
        proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                            timeout=P['max_seconds_per_worker'])
        log={'arm':arm,'repeat':repeat,'profiler':profile_mode,'returncode':proc.returncode,
             'stdout_tail':proc.stdout[-4000:],'stderr_tail':proc.stderr[-6000:]}
    except subprocess.TimeoutExpired as e:
        log={'arm':arm,'repeat':repeat,'profiler':profile_mode,'timeout':True,'message':str(e)}
    return log

def run(output_root):
    output_root=Path(output_root).resolve();output_root.mkdir(parents=True,exist_ok=True)
    data=output_root/'R62_RESULTS';data.mkdir(exist_ok=True)
    status='R62_INCOMPLETE_OR_PARITY_HOLD';reports=[];logs=[];profile_reports=[];comparisons=[];error=None
    try:
        provenance=check_source()
        import torch
        if not torch.cuda.is_available():
            status='R62_NO_CUDA_HOLD'
        else:
            for repeat,order in enumerate(P['repeat_order']):
                pair={}
                for arm in order:
                    file=data/f'REP{repeat}_{arm}.json'
                    lg=child(arm,repeat,file)
                    logs.append(lg)
                    if not file.exists() or lg.get('returncode')!=0:
                        status='R62_WORKER_ERROR_HOLD';break
                    result=json.loads(file.read_text('utf-8'))
                    pair[arm]=result
                    reports.append(result)
                    if result.get('status')!='OK':
                        status='R62_WORKER_ERROR_HOLD';break
                if len(pair)!=len(P['arms']):break
                c=compare(pair['TILED'],pair['LIGER'],P)
                c['repeat']=repeat;comparisons.append(c)
                if not c['valid']:
                    status='R62_PARITY_OR_INTEGRITY_HOLD';break
            else:
                stats=aggregate(comparisons,P);status=stats['status']
                if P.get('profile_one_step_per_arm'):
                    for arm in P['arms']:
                        file=data/f'PROFILE_{arm}.json'
                        lg=child(arm,100,file,profile_mode=True)
                        logs.append(lg)
                        if file.exists():
                            rec=json.loads(file.read_text('utf-8'))
                            profile_reports.append({'arm':arm,'status':rec.get('status'),'file':file.name,
                                                    'top_allocations':rec.get('top_ops_by_self_device_alloc_bytes',[])[:12]})
                        else:profile_reports.append({'arm':arm,'status':'NO_FILE'})
    except Exception as ex:
        status='R62_DRIVER_ERROR_HOLD';error={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()[-12000:]}
    stats=aggregate(comparisons,P)
    summary={'round':P['round'],'status':status,'novelty_claim':'NONE',
             'fresh_training_quality_test':False,'comparison':comparisons,'aggregate':stats,
             'profiles':profile_reports,'error':error,
             'profiler_is_diagnostic_only':True,
             'interpretation': 'A public method can outperform a public baseline on a specific device without proving SHV originality.'}
    (data/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (data/'logs.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (data/'protocol.json').write_text(json.dumps(P,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (data/'provenance.json').write_text(json.dumps({'source_hashes_file_sha256':digest((ROOT/'SOURCE_HASHES.json').read_bytes()),
         'python':platform.python_version(),'source_check_ok':error is None or error.get('type')!='RuntimeError'},
         ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sources=sorted(x for x in data.iterdir() if x.is_file() and x.name!='MANIFEST.json')
    (data/'MANIFEST.json').write_text(json.dumps({'file_sha256':{x.name:digest(x.read_bytes()) for x in sources}},
                                           ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    target=output_root/NAME
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as zout:
        for file in sorted(data.iterdir()):
            if file.is_file():zout.write(file,arcname=file.name)
    (output_root/(NAME+'.sha256')).write_text(digest(target.read_bytes())+'  '+NAME+'\n',encoding='utf-8')
    print('R62_RESULT',status,'ZIP',target,flush=True)
    return summary

if __name__=='__main__':
    dest=sys.argv[1] if len(sys.argv)>1 else 'results_r62'
    run(dest)
