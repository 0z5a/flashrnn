"""Generate independent payloads/oracles before GPU allocation; no GPU use."""
import hashlib,json,os,py_compile,socket,time
from pathlib import Path
import numpy as np
import torch,triton
from fixture_math import reference,round_input,CELL_BUDGET,MODEL_BUDGET
from plan import SHAPES,MODEL_SHAPES,MODEL_ARMS,CONFIGS,chunk_lengths

base=Path(__file__).resolve().parent
assert not torch.cuda.is_initialized() and triton.__version__=='3.6.0'
manifest=json.loads((base/'source.manifest.json').read_text())
for n,h in manifest.items():
    p=base/'source'/n
    assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
    if p.suffix=='.py':py_compile.compile(str(p),doraise=True,cfile='/tmp/frnn-opt-prepare-'+str(os.getpid())+'.pyc')
data=base/'fixtures';data.mkdir(exist_ok=False)
cases=[];models=[]
def save(name,payload):
    p=data/(name+'.npz');np.savez_compressed(p,**payload)
    return dict(file=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def make(case,split):
    b,t,d,h=(case[k] for k in ('batch','steps','width','heads'));dt=case['dtype']
    rng=np.random.default_rng(20261013+split*1000003+b*10000+t*100+d+h*17)
    s=2 if case['cell']=='lstm' else 4
    wx=round_input(rng.normal(0,.1,(b,t,4,h,d)),dt)
    r=round_input(rng.normal(0,.1/d**.5,(4,h,d,d)),dt)
    bias=round_input(rng.normal(0,.03,(4,h,d)),dt)
    initial=round_input(rng.normal(0,.05,(s,b,1,h,d)),dt)
    if s==4:
        initial[2]=round_input(np.abs(initial[2])+1,dt)
        if b==1:initial.fill(0)
        elif b<=3:initial[:,0,:,0]=0;initial[3,0,:,0]=.75
    history,carry=reference(wx,r,bias,initial,case['cell'],dt)
    return dict(wx=wx,r=r,bias=bias,initial=initial,reference=history,carry=carry)
for cell in ('lstm','slstm'):
    for dtype in ('float16','bfloat16'):
        for b,t,d,h in SHAPES:
            c=dict(cell=cell,dtype=dtype,batch=b,steps=t,width=d,heads=h)
            name=f'{cell}-{dtype}-b{b}-t{t}-d{d}-h{h}'
            record=dict(name=name,case=c,splits={})
            for split in (0,1):record['splits'][str(split)]=save(name+f'-split{split}',make(c,split))
            cases.append(record)
for cell in ('lstm','slstm'):
    for b,t,d,h in MODEL_SHAPES:
        name=f'model-{cell}-fp16-b{b}-t{t}-d{d}-h{h}'
        rng=np.random.default_rng(20261113+b*10000+t*100+d+h*17)
        e=d*h;s=2 if cell=='lstm' else 4
        z=dict(embedding=round_input(rng.normal(0,.1,(512,e)),'float16'),readout=round_input(rng.normal(0,.1,(e,512)),'float16'))
        for l in range(4):
            z[f'w{l}']=round_input(rng.normal(0,.2/e**.5,(4,h,d,e)),'float16')
            z[f'r{l}']=round_input(rng.normal(0,.2/d**.5,(4,h,d,d)),'float16')
            z[f'b{l}']=round_input(rng.normal(0,.02,(4,h,d)),'float16')
        for payload in range(3):
            z[f'ids{payload}']=rng.integers(0,512,(32,t),dtype=np.int64)
            x=z['embedding'][z[f'ids{payload}']];hist=[[] for _ in range(4)];carry=[[] for _ in range(4)]
            answers=[]
            for start in range(0,32,b):
                a=x[start:start+b]
                for l in range(4):
                    wx=round_input(np.einsum('bte,ghde->btghd',a.astype(np.float64),z[f'w{l}'].astype(np.float64),optimize=False),'float16')
                    initial=np.zeros((s,b,1,h,d),np.float32)
                    y,f=reference(wx,z[f'r{l}'],z[f'b{l}'],initial,cell,'float16')
                    hist[l].append(y);carry[l].append(f)
                    a=round_input(y[0],'float16').reshape(b,t,e)
                answers.append(round_input(a[:,-1].astype(np.float64)@z['readout'].astype(np.float64),'float16'))
            z[f'logits{payload}']=np.concatenate(answers)
            for l in range(4):
                z[f'hidden{payload}_{l}']=np.concatenate(hist[l],axis=1)[0]
                z[f'carry{payload}_{l}']=np.concatenate(carry[l],axis=1)
        models.append(dict(name=name,cell=cell,batch=b,steps=t,width=d,heads=h,requests=32,**save(name,z)))
plan=dict(status='READY_CPU_INPUTS',cases=cases,models=models,
    configs={n:c.as_dict() for n,c in CONFIGS.items()},model_arms=MODEL_ARMS,
    training_split=0,holdout_split=1,independent_process_starts=5,paired_blocks_per_start=20,
    cell_budget=CELL_BUDGET,model_budget=MODEL_BUDGET,
    model_payloads=3,model_input_policy='Rotate genuinely new CPU IDs; transfer and CPU delivery remain timed',
    identity=dict(torch=torch.__version__,triton=triton.__version__,numpy=np.__version__,host=socket.gethostname()),
    smallest_node_count=1,reason='One H100 measured at a time; no cross-node topology or large model required',
    max_science_minutes=130,max_archive_minutes=30,max_preflight_minutes=10,
    cuda_initialized=torch.cuda.is_initialized(),prepared=time.time())
(base/'finite-plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(dict(status=plan['status'],cell_shapes=len(cases),cell_fixtures=2*len(cases),models=len(models),bytes=sum(v['bytes'] for c in cases for v in c['splits'].values())+sum(m['bytes'] for m in models))),flush=True)
