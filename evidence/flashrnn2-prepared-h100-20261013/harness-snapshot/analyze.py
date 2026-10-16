"""CPU-only holdout aggregation, with all exclusions and matching-input checks."""
import csv,hashlib,json,os,time
from pathlib import Path
import numpy as np

base=Path(os.environ['FRNN_OPT_BASE']);out=base/'science'
plan=json.loads((base/'finite-plan.json').read_text())
audit=json.loads((out/'final-audit.json').read_text());assert audit['status']=='PASS_INDEPENDENT_ALL_OUTPUT_AUDIT'
selection=json.loads((out/'selection.json').read_text())
assert audit['selection_sha256']==hashlib.sha256((out/'selection.json').read_bytes()).hexdigest()
groups={};excluded=[]
names=[r['name'] for r in plan['cases']+plan['models']]
for start in range(5):
    assert (out/f'timing-start{start}-complete.json').is_file()
    for name in names:
        p=out/'timings'/f'{name}-start{start}.json';record=json.loads(p.read_text())
        assert record['start']==start and record['identity']==selection['identity']
        for r in record['records']:
            key=(name,r['a'],r['b'])
            if r['status']!='PASS':excluded.append(dict(name=name,start=start,**r));continue
            assert len(r['blocks'])==20
            for i,b in enumerate(r['blocks']):
                assert b['block']==i and b['a_ms']>0 and b['b_ms']>0
                if record['kind']=='model':assert b['payload_start']==i%3
            groups.setdefault(key,[]).append(dict(start=start,kind=record['kind'],case=record['case'],blocks=r['blocks']))
rows=[];incomplete=[]
for index,(key,records) in enumerate(sorted(groups.items())):
    if len(records)!=5 or {r['start'] for r in records}!=set(range(5)):
        incomplete.append(dict(key=key,starts=[r['start'] for r in records]));continue
    records.sort(key=lambda r:r['start'])
    a=np.array([[b['a_ms'] for b in r['blocks']] for r in records]);b=np.array([[b['b_ms'] for b in r['blocks']] for r in records])
    log=np.log(a/b);rng=np.random.default_rng(20261013+index)
    pi=rng.integers(0,5,(10000,5));bi=rng.integers(0,20,(10000,5,20))
    samples=np.exp(log[pi[:,:,None],bi].mean(axis=(1,2)))
    low,high=np.quantile(samples,[.025,.975]);name,arm_a,arm_b=key
    aa=False
    if records[0]['kind']=='cell' and arm_a=='mma4' and arm_b=='selected':
        c=records[0]['case'];k=f'{c["cell"]}/{c["dtype"]}/b{c["batch"]}/t{c["steps"]}/h{c["heads"]}/d{c["width"]}'
        aa=selection['table'][k]==dict(family='mma',batch_tile=16,warps=4,weight_operand='b')
    rows.append(dict(name=name,kind=records[0]['kind'],a=arm_a,b=arm_b,ratio_a_over_b=float(np.exp(log.mean())),
        ci_low=float(low),ci_high=float(high),a_ms=float(np.median(np.median(a,axis=1))),b_ms=float(np.median(np.median(b,axis=1))),
        process_starts=5,paired_blocks=100,same_configuration_replica=aa))
status='COMPLETE_FINITE_HOLDOUT_WITH_NEGATIVES' if excluded or incomplete else 'COMPLETE_FINITE_HOLDOUT'
result=dict(status=status,source_manifest_sha256=selection['identity']['source_manifest_sha256'],identity=selection['identity'],
    training_split=0,holdout_split=1,raw_output_audit_status=audit['status'],saved_gpu_outputs=audit['raw_outputs'],
    timing_rows=len(rows),genuine_comparison_blocks=sum(r['paired_blocks'] for r in rows if not r['same_configuration_replica']),
    same_configuration_replica_blocks=sum(r['paired_blocks'] for r in rows if r['same_configuration_replica']),
    statistics='Five starts x20 ABBA/BAAB blocks;10000 process/block bootstrap; ratios are reference/candidate',
    model_timed_boundary='New aligned CPU ID payloads, all layers, readout and CPU delivery; both baseline and optimized graph controls',
    models={name:{a:s for a,s in arms.items()} for name,arms in audit['models'].items()},
    rows=rows,excluded=excluded,incomplete=incomplete,finished=time.time())
(out/'results.json').write_text(json.dumps(result,indent=2))
with (out/'timings.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['name']);writer.writeheader();writer.writerows(rows)
summary=['# FlashRNN2 first four optimizations: finite holdout','',f'Status: {status}',
    f'Saved GPU outputs independently audited: {audit["raw_outputs"]}',
    'Ratios below 1 indicate a slower candidate; source/runtime/inputs remain bound to the original receipts.','',
    '| Model | Reference | Candidate | Ratio | 95% CI |','| --- | --- | --- | --- | --- |']
for r in rows:
    if r['kind']=='model':summary.append(f'| {r["name"]} | {r["a"]} | {r["b"]} | {r["ratio_a_over_b"]:.4f} | [{r["ci_low"]:.4f}, {r["ci_high"]:.4f}] |')
summary += ['',f'Excluded process/comparison records: {len(excluded)}',f'Incomplete timing groups: {len(incomplete)}',
            'Fixed synthetic models; no new pretrained-model quality or backward qualification.']
(out/'RESULTS.md').write_text('\n'.join(summary)+'\n')
print(json.dumps({k:result[k] for k in ('status','saved_gpu_outputs','timing_rows','genuine_comparison_blocks')}),flush=True)
