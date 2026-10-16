"""Independent CPU reader; raw output SHA and unchanged budgets govern admission."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from fixture_math import compare,reference,CELL_BUDGET,MODEL_BUDGET

BASE=Path(os.environ['FRNN_OPT_BASE']);OUT=BASE/'science'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same_bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))
def write(p,value):
    p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def load(record):
    assert sha(record['file'])==record['sha256'],record['file']
    assert Path(record['file']).stat().st_size==record['bytes']
    return dict(np.load(record['file']))
def cell(rec):
    z=load(rec['item']['splits'][str(rec['split'])]);arms={};count=0
    legacy=rec['arms'].get('legacy');base=load(legacy['raw']) if legacy and 'raw' in legacy else None
    for arm,r in rec['arms'].items():
        if 'raw' not in r:arms[arm]=r['status'];continue
        x=load(r['raw']);count+=1;h=z['reference'];carry=z['carry']
        if arm=='hidden':h=h[0]
        if arm=='last':h=h[0][:,-1:]
        if arm=='graph_zero_initial':
            c=rec['item']['case'];h,carry=reference(z['wx'],z['r'],z['bias'],np.zeros_like(z['initial']),c['cell'],c['dtype'])
        checks={k:compare(x[k],target,CELL_BUDGET) for k,target in (('history',h),('carry',carry))}
        for k,v in checks.items():
            assert v['pass_']==r['checks'][k]['pass_'] and v['max_abs']==r['checks'][k]['max_abs'],(arm,k)
        numerical=all(v['pass_'] for v in checks.values())
        if 'same_observable_and_carry_bits' in r:
            assert base is not None
            target=base['history'][0] if arm=='hidden' else (base['history'][0][:,-1:] if arm=='last' else base['history'])
            equal=bool(same_bits(x['history'],target) and same_bits(x['carry'],base['carry']))
            assert equal==r['same_observable_and_carry_bits'],arm
            numerical &= equal
        if r['status']=='PASS':assert numerical,arm
        arms[arm]=r['status'] if numerical else ('BITWISE_CONTRACT_FAIL' if r['status']=='BITWISE_CONTRACT_FAIL' else 'NUMERICAL_FAIL')
    return arms,count


parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cells','all']);args=parser.parse_args()
records=json.loads((OUT/'cells-qualified.json').read_text())['records'];admissions={};count=0
for rec in records:
    arms,n=cell(rec);count+=n;name=rec['item']['name']+f'-split{rec["split"]}'
    assert arms['legacy']=='PASS',name
    admissions[name]=arms
result=dict(status='PASS_INDEPENDENT_CELL_AUDIT',admissions=admissions,raw_outputs=count,numpy=np.__version__,finished=time.time())
write(OUT/'cells-audit.json',result)
if args.mode=='all':
    selected=json.loads((OUT/'selected-and-models-qualified.json').read_text())
    cells={r['item']['name']:dict(admissions[r['item']['name']+'-split1']) for r in selected['cells']}
    for r in selected['cells']:
        arms,n=cell(r);cells[r['item']['name']].update(arms);count+=n
    models={}
    for rec in selected['models']:
        z=load(rec['item']);arms={}
        for arm,r in rec['arms'].items():
            if 'payloads' not in r:arms[arm]=r['status'];continue
            ok=True
            for payload in r['payloads']:
                x=load(payload['raw']);count+=1;i=payload['payload']
                for k,v in x.items():
                    key=f'logits{i}' if k=='logits' else f'{k[:-1]}{i}_{k[-1]}'
                    target=z[key]
                    if k.startswith('hidden') and v.shape[1]==1:target=target[:,-1:]
                    check=compare(v,target,MODEL_BUDGET);expected=payload['checks'][k]
                    assert check['pass_']==expected['pass_'] and check['max_abs']==expected['max_abs'],(arm,k)
                    ok &= check['pass_']
            if r['status']=='PASS':assert ok,arm
            arms[arm]=r['status'] if ok else 'NUMERICAL_FAIL'
        assert arms['legacy']=='PASS',rec['item']['name'];models[rec['item']['name']]=arms
    result=dict(status='PASS_INDEPENDENT_ALL_OUTPUT_AUDIT',cells=cells,models=models,raw_outputs=count,
                selection_sha256=sha(OUT/'selection.json'),numpy=np.__version__,finished=time.time())
    write(OUT/'final-audit.json',result)
print(json.dumps(dict(status=result['status'],raw_outputs=count)),flush=True)
