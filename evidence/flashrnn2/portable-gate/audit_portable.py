"""Recompute portable-gate evidence without importing the recurrence or runner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def compare(a: torch.Tensor, b: torch.Tensor, budget: float, reported: dict) -> None:
    assert a.shape==b.shape and a.dtype==b.dtype
    x,y=a.double().flatten(),b.double().flatten()
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    d=x-y
    failed=int(torch.count_nonzero(d.abs()>budget*(1+y.abs())))
    position=int(d.abs().argmax())
    coordinate=[]
    for size in reversed(a.shape):
        coordinate.insert(0,position%size)
        position//=size
    assert reported['pass']==(failed==0) and reported['finite']
    assert failed==reported['failed_elements'] and coordinate==reported['max_coordinate']
    metrics={'max_abs':float(d.abs().amax()),'rms':float(torch.linalg.vector_norm(d))/math.sqrt(d.numel()),'relative_l2':float(torch.linalg.vector_norm(d))/max(float(torch.linalg.vector_norm(y)),1e-30)}
    for name,value in metrics.items():
        assert math.isclose(value,reported[name],rel_tol=1e-12,abs_tol=1e-15),(name,value,reported[name])


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('rows',type=Path)
    parser.add_argument('--frozen-sources',type=Path)
    args=parser.parse_args()
    path=args.rows
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    meta=json.loads(path.with_suffix('.meta.json').read_text())
    controller=json.loads(path.with_name(path.stem+'-controller.json').read_text())
    assert meta['status'] in ('PASSED','FAILED') and meta['cases']==len(rows)==38
    assert controller['returncode']==(0 if meta['status']=='PASSED' else 1)
    if args.frozen_sources:
        manifest=json.loads((args.frozen_sources/'manifest.json').read_text())
        assert set(manifest)==set(meta['source_sha256'])
        for name,entry in manifest.items():
            assert digest(args.frozen_sources/entry['file'])==entry['sha256']==meta['source_sha256'][name]
    else:
        for name,sha in meta['source_sha256'].items():
            assert digest(Path(name))==sha
    torch.set_num_threads(1)
    pairs=states=0
    failures=[]
    for index,row in enumerate(rows):
        assert row['tensors']==f'{index:03d}.pt'
        file=path.with_suffix('.tensors')/row['tensors']
        assert digest(file)==row['tensor_sha256']
        tensors=torch.load(file,map_location='cpu',weights_only=True)
        forward,gradient=row['forward_atol_rtol'],row['gradient_atol_rtol']
        assert (forward,gradient)==((1e-12,1e-11) if row['numerics']=='mathematical' else (1e-5,1e-5))
        for number,name in enumerate(('history','final')):
            a,b=tensors['actual'][number],tensors['oracle'][number]
            compare(a,b,forward,row['checks'][name]);pairs+=1
            for i,record in enumerate(row['checks'][name]['per_state']):
                compare(a[i],b[i],forward,record);states+=1
            compare(tensors['chunk'][number],a,forward,row['checks']['chunk_'+name]);pairs+=1
        for number,name in enumerate(('wx','recurrent','bias','initial')):
            a,b,c=(tensors[key][number] for key in ('actual_gradients','oracle_gradients','chunk_gradients'))
            compare(a,b,gradient,row['checks']['gradient_'+name]);pairs+=1
            compare(c,a,gradient,row['checks']['chunk_gradient_'+name]);pairs+=1
        if 'repeated' in tensors:
            assert all(torch.equal(a,b) for a,b in zip(tensors['actual'],tensors['repeated'],strict=True))==row['repeat_exact']
            assert all(torch.equal(a,b) for a,b in zip(tensors['inputs'],tensors['inputs_after'],strict=True))==row['input_readonly']
        passed=all(check['pass'] for check in row['checks'].values()) and all(row[k] for k in ('input_readonly','device_dtype_match','repeat_exact')) and row['two_cuda_streams_exact'] is not False
        assert (row['status']=='PASSED')==passed
        if not passed:
            failures.append({'case':row['case'],'cell':row['cell'],'numerics':row['numerics'],'checks':{k:v for k,v in row['checks'].items() if not v['pass']}})
    assert pairs==456 and states==176
    assert meta['passed']==38-len(failures) and (meta['status']=='PASSED')==(not failures)
    result={'audit':'PASS','gate_status':meta['status'],'natural_exit':controller['returncode'],'cases':38,'passed':meta['passed'],'tensor_comparison_pairs':pairs,'per_state_comparisons':states,'failures':failures,'row_sha256':digest(path),'source_hashes_verified':True,'auditor_sha256':digest(Path(__file__)),'runner_only_checks':['device_dtype_match','two_cuda_streams_exact']+([] if 'repeated' in tensors else ['input_readonly','repeat_exact']),'performance_claim':False}
    out=path.with_name(path.stem+'-audit.json');assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('audit','gate_status','cases','passed','tensor_comparison_pairs','per_state_comparisons')}))

if __name__=='__main__':
    main()
