"""Saved-output qualification, training-only selection and independent holdout timing."""
import argparse,gc,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from helpers import *
from plan import CELL_PAIRS,MODEL_ARMS,MODEL_PAIRS,configs_for


def policy():
    r=json.loads((OUT/'selection.json').read_text())
    return ExecutionPolicy(r['table'],identity=r['identity'])


def qualify_cells(plan):
    records=[]
    for item in plan['cases']:
        for split in (0,1):
            directory=OUT/'qualification'/f'{item["name"]}-split{split}';directory.mkdir(parents=True,exist_ok=False)
            z,vals,plans,fns=cell_functions(item,split,graphs=True);arms={};baseline_values=None
            for arm,fn in fns.items():
                try:
                    y,carry=fn();checks=check_cell((y,carry),z,arm)
                    r=dict(status='PASS' if passed(checks) else 'NUMERICAL_FAIL',checks=checks,
                           raw=raw(directory/(arm+'.npz'),dict(history=y,carry=carry)))
                    if arm=='legacy':baseline_values=(array(y),array(carry))
                    arms[arm]=r
                except BaseException:arms[arm]=dict(status='EXECUTION_FAIL',error=traceback.format_exc())
                write(directory/'qualification.json',dict(item=item,split=split,arms=arms))
            if baseline_values is not None:
                bh,bc=baseline_values
                for arm in ('mma4','hidden','last','prepared_chunks','graph_chunks'):
                    if arms.get(arm,{}).get('status')!='PASS':continue
                    saved=dict(np.load(arms[arm]['raw']['file']))
                    target=bh[0] if arm=='hidden' else (bh[0][:,-1:] if arm=='last' else bh)
                    bitwise=bool(same_bits(target,saved['history']) and same_bits(bc,saved['carry']))
                    arms[arm]['same_observable_and_carry_bits']=bitwise
                    if not bitwise:arms[arm]['status']='BITWISE_CONTRACT_FAIL'
            if 'graph_chunks' in plans and arms.get('graph_chunks',{}).get('status')=='PASS':
                p=plans['graph_chunks'];p.initial.zero_();y,carry=p.run()
                reference_zero,carry_zero=reference(z['wx'],z['r'],z['bias'],np.zeros_like(z['initial']),item['case']['cell'],item['case']['dtype'])
                checks=dict(history=compare(array(y),reference_zero,CELL_BUDGET),carry=compare(array(carry),carry_zero,CELL_BUDGET))
                arms['graph_zero_initial']=dict(status='PASS' if passed(checks) else 'GRAPH_STALE_STATE_FAIL',checks=checks,
                    raw=raw(directory/'graph_zero_initial.npz',dict(history=y,carry=carry)))
            for name,p in plans.items():
                if arms.get(name,{}).get('status')=='PASS' and name!='graph_chunks':
                    try:
                        p._execute(diagnostics=True)
                        for i,r in enumerate(p.compiler_records):
                            body=r.pop('ptx');path=directory/f'{name}-{i}.ptx';path.write_text(body)
                            r.update(ptx_file=str(path),ptx_sha256=sha(path))
                        arms[name]['compiler']=p.compiler_records
                    except BaseException:arms[name]['compiler_error']=traceback.format_exc()
                p.close()
            record=dict(item=item,split=split,arms=arms);write(directory/'qualification.json',record);records.append(record)
            write(OUT/'qualification-progress.json',dict(completed=len(records),total=2*len(plan['cases']),last=item['name'],split=split))
            print(json.dumps(dict(name=item['name'],split=split,status={k:v['status'] for k,v in arms.items()})),flush=True)
            del fns,plans,vals;gc.collect();torch.cuda.empty_cache()
    write(OUT/'cells-qualified.json',dict(records=records))


def select(plan):
    admissions=json.loads((OUT/'cells-audit.json').read_text());assert admissions['status']=='PASS_INDEPENDENT_CELL_AUDIT'
    table={};records=[]
    for item in plan['cases']:
        z,vals,plans,fns=cell_functions(item,0);admit=admissions['admissions'][item['name']+'-split0']
        assert admit.get('mma4')=='PASS','Original numerical gate must pass before selection'
        candidates=[]
        for arm in configs_for(item['case']):
            if admit.get(arm)!='PASS':continue
            blocks=pairs(fns['mma4'],fns[arm]);ratio=float(np.exp(np.mean(np.log([r['a_ms']/r['b_ms'] for r in blocks]))))
            candidates.append(dict(arm=arm,ratio=ratio,blocks=blocks))
        winner=max(candidates,key=lambda r:r['ratio']);chosen=winner['arm'] if winner['ratio']>1.02 else 'mma4'
        c=item['case'];key=shape_key(c['cell'],c['dtype'],c['batch'],c['steps'],c['heads'],c['width'])
        table[key]=CONFIGS[chosen].as_dict();records.append(dict(name=item['name'],chosen=chosen,candidates=candidates))
        for p in plans.values():p.close()
        write(OUT/'selection-progress.json',dict(completed=len(records),total=len(plan['cases']),last=item['name'],chosen=chosen))
        del fns,plans,vals;gc.collect();torch.cuda.empty_cache()
    write(OUT/'selection.json',dict(status='FROZEN_TRAINING_ONLY_POLICY',training_split=0,evaluation_split=1,
        identity=identity(),table=table,records=records,source_audit_sha256=sha(OUT/'cells-audit.json')))


def qualify_selected_and_models(plan):
    pol=policy();cell_records=[];model_records=[]
    for item in plan['cases']:
        c=item['case'];cfg=pol.select(c['cell'],c['dtype'],c['batch'],c['steps'],c['heads'],c['width'],identity=identity())
        z,vals,plans,fns=cell_functions(item,1,selected=cfg)
        directory=OUT/'selected'/item['name'];directory.mkdir(parents=True,exist_ok=False)
        try:
            y,carry=fns['selected']();checks=check_cell((y,carry),z,'selected')
            r=dict(status='PASS' if passed(checks) else 'NUMERICAL_FAIL',checks=checks,
                   raw=raw(directory/'selected.npz',dict(history=y,carry=carry)))
        except BaseException:r=dict(status='EXECUTION_FAIL',error=traceback.format_exc())
        rec=dict(item=item,split=1,arms={'selected':r});write(directory/'qualification.json',rec);cell_records.append(rec)
        for p in plans.values():p.close()
        del fns,plans,vals;gc.collect();torch.cuda.empty_cache()
    for item in plan['models']:
        directory=OUT/'models'/item['name'];directory.mkdir(parents=True,exist_ok=False)
        m=Model(item,pol);arms={}
        for arm in MODEL_ARMS:
            try:
                payloads=[]
                for payload in range(3):
                    v=m.run(arm,payload,save=True);expected=m.expected(payload,v)
                    checks={k:compare(array(x),expected[k],MODEL_BUDGET) for k,x in v.items()}
                    payloads.append(dict(payload=payload,checks=checks,
                        raw=raw(directory/f'{arm}-payload{payload}.npz',v)))
                arms[arm]=dict(status='PASS' if all(passed(r['checks']) for r in payloads) else 'NUMERICAL_FAIL',payloads=payloads)
            except BaseException:arms[arm]=dict(status='EXECUTION_FAIL',error=traceback.format_exc())
            write(directory/'qualification.json',dict(item=item,arms=arms,selected_config=m.cfg.as_dict()))
        m.close();rec=dict(item=item,arms=arms);model_records.append(rec)
        print(json.dumps(dict(name=item['name'],status={k:v['status'] for k,v in arms.items()})),flush=True)
        del m;gc.collect();torch.cuda.empty_cache()
    write(OUT/'selected-and-models-qualified.json',dict(cells=cell_records,models=model_records))


def measure(plan,start):
    audit=json.loads((OUT/'final-audit.json').read_text());assert audit['status']=='PASS_INDEPENDENT_ALL_OUTPUT_AUDIT'
    pol=policy()
    for item in plan['cases']:
        c=item['case'];cfg=pol.select(c['cell'],c['dtype'],c['batch'],c['steps'],c['heads'],c['width'],identity=identity())
        z,vals,plans,fns=cell_functions(item,1,selected=cfg,graphs=True);admit=audit['cells'][item['name']]
        pairs_to_run=list(CELL_PAIRS)
        if 'prepared_chunks' in fns:pairs_to_run += [('mma4','legacy_chunks'),('legacy_chunks','prepared_chunks'),('prepared_chunks','graph_chunks')]
        records=[]
        for a,b in pairs_to_run:
            if a not in fns or b not in fns:continue
            if admit.get(a)!='PASS' or admit.get(b)!='PASS':
                records.append(dict(a=a,b=b,status='NOT_ADMITTED',reason=[admit.get(a),admit.get(b)]));continue
            if not (passed(check_cell(fns[a](),z,a)) and passed(check_cell(fns[b](),z,b))):
                records.append(dict(a=a,b=b,status='REPEAT_NUMERICAL_FAIL'));continue
            records.append(dict(a=a,b=b,status='PASS',blocks=pairs(fns[a],fns[b])))
        write(OUT/'timings'/f'{item["name"]}-start{start}.json',dict(name=item['name'],kind='cell',case=c,start=start,records=records,identity=identity()))
        for p in plans.values():p.close()
        del fns,plans,vals;gc.collect();torch.cuda.empty_cache()
    for item in plan['models']:
        m=Model(item,pol);admit=audit['models'][item['name']];records=[]
        for a,b in MODEL_PAIRS:
            if admit.get(a)!='PASS' or admit.get(b)!='PASS':
                records.append(dict(a=a,b=b,status='NOT_ADMITTED',reason=[admit.get(a),admit.get(b)]));continue
            good=True
            for arm in (a,b):
                v=m.run(arm,2,save=True);expected=m.expected(2,v)
                good &= all(compare(array(x),expected[k],MODEL_BUDGET)['pass_'] for k,x in v.items())
            if not good:records.append(dict(a=a,b=b,status='REPEAT_NUMERICAL_FAIL'));continue
            m.counters[a]=m.counters[b]=0
            def align_payloads(block):m.counters[a]=m.counters[b]=block%3
            records.append(dict(a=a,b=b,status='PASS',blocks=pairs(lambda:m.run(a),lambda:m.run(b),prepare=align_payloads)))
        write(OUT/'timings'/f'{item["name"]}-start{start}.json',dict(name=item['name'],kind='model',case=item,start=start,records=records,identity=identity()))
        m.close();del m;gc.collect();torch.cuda.empty_cache()
    write(OUT/f'timing-start{start}-complete.json',dict(status='FINISHED_CASE_RECORDS_REQUIRE_ANALYSIS',finished=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['qualify','select','models','time']);parser.add_argument('--start',type=int,default=0);args=parser.parse_args()
    plan=json.loads((BASE/'finite-plan.json').read_text())
    ready=json.loads((BASE/'ready.json').read_text())
    for p,h in ready['files'].items():assert sha(BASE/p)==h,p
    assert torch.cuda.device_count()==1 and triton.__version__=='3.6.0'
    write(OUT/f'identity-{args.mode}-{args.start}.json',dict(identity=identity(),torch=torch.__version__,job=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,at=time.time()))
    with torch.inference_mode():
        if args.mode=='qualify':qualify_cells(plan)
        if args.mode=='select':select(plan)
        if args.mode=='models':qualify_selected_and_models(plan)
        if args.mode=='time':measure(plan,args.start)


if __name__=='__main__':
    from fixture_math import reference
    main()
