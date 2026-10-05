"""Independently recompute saved HGRN2 full-checkpoint logits and final states."""
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer

root=Path(__file__).resolve().parent; evidence=root/'evidence';stem='hgrn2-reference-r1'
meta=json.loads((evidence/f'{stem}.meta.json').read_text())
controller=json.loads((evidence/f'{stem}-controller.json').read_text())
rows=[json.loads(line) for line in (evidence/f'{stem}.jsonl').read_text().splitlines()]
assert meta['status'] in ('PASS','NUMERICAL_FAILED')
assert controller['returncode']==(0 if meta['status']=='PASS' else 1)
assert meta['batch_steps']==len(rows)==36 and meta['token_choices']==84
assert meta['model']['parameters']==1364396032 and meta['model']['layers']==24 and meta['model']['checkpoint_tensors']==244
assert meta['logits_budget']=={'atol':1e-3,'rtol':1e-3}
assert meta['state_budget']=={'atol':1e-5,'rtol':1e-5}

def sha256(path:Path)->str:
 with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

frozen=json.loads((evidence/'hgrn2-reference-r1.sources/manifest.json').read_text())
for name,digest in controller['source_sha256'].items():
 assert frozen[name]['sha256']==digest==sha256(evidence/'hgrn2-reference-r1.sources'/frozen[name]['file'])
for name,digest in meta['source_sha256'].items():
 path=evidence/'hgrn2-reference-r1.sources'/frozen[name]['file'] if name in frozen else Path(name)
 assert sha256(path)==digest
for name,digest in meta['model']['source_sha256'].items():
 assert sha256(evidence/'models/hgrn2-sources'/name)==digest
manifest=meta['model']['checkpoint']
assert manifest['revision']=='2f413dd9b63591b9b177bbf940942ea7eb70abfe'
for entry in manifest['files']:
 path=root/'models/hgrn2-1.3b'/entry['file']; assert path.stat().st_size==entry['size']
 if entry['file']=='model.safetensors':assert sha256(path)==entry['checksum']
indexed={(row['batch'],row['case'],row['step']):row for row in rows}
assert len(indexed)==36 and set(indexed)=={(b,c,s) for b in (1,2,4) for c in range(3) for s in range(4)}
assert len(meta['snapshots'])==9 and {(x['batch'],x['case']) for x in meta['snapshots']}=={(b,c) for b in (1,2,4) for c in range(3)}
tokenizer=Tokenizer.from_file(str(root/'models/hgrn2-1.3b/tokenizer.json'))
inputs=[tokenizer.encode(text).ids[:5] for text in meta['prompts']]
assert inputs==meta['input_ids']

def measure(a:torch.Tensor,b:torch.Tensor,budget:float)->dict:
 assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
 assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
 difference=(a-b).abs();allowed=budget+budget*b.abs();failed=int(torch.count_nonzero(difference>allowed))
 return {'pass':failed==0,'max_abs':float(difference.max()),'worst_normalized':float((difference/allowed).max()),'failed_elements':failed}

torch.set_num_threads(1)
logit_pairs=0;token_choices=0;state_records=[]
for entry in meta['snapshots']:
 path=evidence/entry['file'];assert path.stat().st_size==entry['bytes'] and sha256(path)==entry['sha256']
 saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 batch,case=entry['batch'],entry['case']
 assert (saved['batch'],saved['case'])==(batch,case)
 assert torch.equal(saved['input_ids'],torch.tensor(inputs[4*case:4*case+batch]))
 left,right=saved['cached_logits'],saved['full_prefix_logits']
 assert left.shape==right.shape==(4,batch,32000)
 assert torch.equal(saved['generated_ids'],left.argmax(-1))
 token_choices+=saved['generated_ids'].numel()
 for step in range(4):
  row=indexed[batch,case,step]
  assert measure(left[step],right[step],1e-3)==row['checks']['logits']
  assert bool(torch.equal(left[step].argmax(-1),right[step].argmax(-1)))==row['token_equal']
  logit_pairs+=1
 if case==0:
  cached,full=saved['cached_final_states'],saved['full_prefix_final_states']
  assert len(cached)==len(full)==24
  for layer,(a,b) in enumerate(zip(cached,full,strict=True)):
   assert set(a)==set(b)=={'recurrent_state'}
   x,y=a['recurrent_state'],b['recurrent_state']
   assert x.shape==y.shape==(batch,16,128,128)
   result=measure(x,y,1e-5)
   assert result==indexed[batch,case,3]['layers'][layer]
   state_records.append({'batch':batch,'layer':layer,**result})
  del cached,full,a,b,x,y
 else:assert 'cached_final_states' not in saved and 'full_prefix_final_states' not in saved
 del saved,left,right
assert logit_pairs==36 and token_choices==84 and len(state_records)==72
for row in rows:
 layers=row['layers'];assert len(layers)==24
 expected={'pass':all(x['pass'] for x in layers),'max_abs':max(x['max_abs'] for x in layers),'worst_normalized':max(x['worst_normalized'] for x in layers),'failed_elements':sum(x['failed_elements'] for x in layers)}
 assert expected==row['checks']['recurrent_state']
 assert row['pass']==(row['token_equal'] and all(x['pass'] for x in row['checks'].values()))
assert (meta['status']=='PASS')==all(r['pass'] for r in rows)
summary=[{'batch':batch,'steps':12,'matching_tokens':sum(batch for r in rows if r['batch']==batch and r['token_equal']),'checks':{key:{'passed_steps':sum(r['checks'][key]['pass'] for r in rows if r['batch']==batch),'max_abs':max(r['checks'][key]['max_abs'] for r in rows if r['batch']==batch),'worst_normalized':max(r['checks'][key]['worst_normalized'] for r in rows if r['batch']==batch)} for key in ('logits','recurrent_state')}} for batch in (1,2,4)]
result={'audit':'PASS','model_result':meta['status'],'natural_exit':controller['returncode'],'logit_pairs_recomputed':logit_pairs,'token_choices_recomputed':token_choices,'final_state_pairs_recomputed':len(state_records),'rows':summary,'state_records':state_records,'state_snapshot_scope':'Both complete case0 final caches at B1/B2/B4; other state rows are runner evidence','performance_claim':False,'gpu_qualification':False}
output=evidence/f'{stem}-audit.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('audit','model_result','natural_exit','rows')}))
