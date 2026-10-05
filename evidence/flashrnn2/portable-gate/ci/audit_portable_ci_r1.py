"""Audit immutable-head CI tensors, source identity and package origins."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import torch
root=Path(__file__).resolve().parent
folder=root/'evidence/portable-cpu-ci-r1-artifacts/portable-cpu-gate'
run=json.loads((root/'evidence/portable-cpu-ci-r1-run.json').read_text())
assert run['id']==37348607245 and run['head_sha']=='bd72edbeb8b3ccd07b8658d04190d74e6e958b5f'
assert run['status']=='completed' and run['conclusion']=='success' and run['event']=='push'
meta=json.loads((folder/'results.meta.json').read_text());assert meta['status']=='PASSED' and meta['passed']==meta['cases']==38
assert meta['torch']=='2.14.1+cpu' and meta['cuda_build'] is meta['hip_build'] is None
for name,sha in meta['source_sha256'].items():
 rel=('tools/'+name.split('/tools/')[1]) if '/tools/' in name else ('flashrnn/'+name.split('/site-packages/flashrnn/')[1])
 data=subprocess.check_output(['git','show',run['head_sha']+':'+rel],cwd=root/'source')
 assert hashlib.sha256(data).hexdigest()==sha
spec=importlib.util.spec_from_file_location('independent_audit',root/'audit_portable.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()]
torch.set_num_threads(1);pairs=states=0
for index,row in enumerate(rows):
 assert row['status']=='PASSED' and row['tensors']==f'{index:03d}.pt'
 file=folder/'results.tensors'/row['tensors'];assert module.digest(file)==row['tensor_sha256']
 saved=torch.load(file,weights_only=True)
 for n,key in enumerate(('history','final')):
  a,b=saved['actual'][n],saved['oracle'][n]
  module.compare(a,b,row['forward_atol_rtol'],row['checks'][key]);pairs+=1
  for s,record in enumerate(row['checks'][key]['per_state']):
   module.compare(a[s],b[s],row['forward_atol_rtol'],record);states+=1
  module.compare(saved['chunk'][n],a,row['forward_atol_rtol'],row['checks']['chunk_'+key]);pairs+=1
 for n,key in enumerate(('wx','recurrent','bias','initial')):
  a,b,c=(saved[k][n] for k in ('actual_gradients','oracle_gradients','chunk_gradients'))
  module.compare(a,b,row['gradient_atol_rtol'],row['checks']['gradient_'+key]);pairs+=1
  module.compare(c,a,row['gradient_atol_rtol'],row['checks']['chunk_gradient_'+key]);pairs+=1
 assert all(torch.equal(a,b) for a,b in zip(saved['inputs'],saved['inputs_after'],strict=True))
 assert all(torch.equal(a,b) for a,b in zip(saved['actual'],saved['repeated'],strict=True))
assert pairs==456 and states==176
origins=root/'evidence/portable-cpu-ci-r1-artifacts/cpu-wheel-origins'
packages={x['metadata']['name']:x for p in origins.glob('*-install.json') for x in json.loads(p.read_text())['install']}
assert 'triton' not in packages and 'ninja' not in packages
result={'audit':'PASS','run_id':run['id'],'run_url':run['html_url'],'event':run['event'],'head':run['head_sha'],'natural_github_job_conclusion':run['conclusion'],'torch':meta['torch'],'python':meta['python'],'machine':meta['machine'],'cases':38,'passed':38,'tensor_pairs_recomputed':pairs,'per_state_comparisons_recomputed':states,'sources_match_published_commit':True,'readonly_repeat_recomputed':True,'torch_wheel_sha256':packages['torch']['download_info']['archive_info']['hashes']['sha256'],'einops':packages['einops']['metadata']['version'],'pip_origins':{p.name:module.digest(p) for p in origins.glob('*-install.json')},'rows_sha256':module.digest(folder/'results.jsonl'),'auditor_sha256':module.digest(Path(__file__)),'cpu_unit_methods':15,'unit_method_evidence':'Successful exact-head workflow step runs all three committed unittest suites; full log fetch timed out, job step receipt recorded separately','performance_claim':False}
(root/'evidence/portable-cpu-ci-r1-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
