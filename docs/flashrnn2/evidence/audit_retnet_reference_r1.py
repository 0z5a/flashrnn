"""Recompute RetNet saved logits/caches without running the model or its gate."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

root=Path(__file__).resolve().parent
evidence=root/'evidence'
stem=evidence/'retnet-reference-r1'
meta=json.loads(stem.with_suffix('.meta.json').read_text())
controller=json.loads((evidence/'retnet-reference-r1-controller.json').read_text())
rows=[json.loads(line) for line in stem.with_suffix('.jsonl').read_text().splitlines()]
assert meta['status'] in ('PASS','NUMERICAL_FAILED')
assert controller['returncode']==(0 if meta['status']=='PASS' else 1)
assert meta['batch_steps']==len(rows)==36 and meta['token_choices']==84
assert meta['model']['parameters']==1351727104 and meta['model']['checkpoint_tensors']==267 and meta['model']['layers']==24
assert meta['logits_budget']=={'atol':1e-3,'rtol':1e-3} and meta['state_budget']=={'atol':1e-5,'rtol':1e-5}
for name,digest in meta['source_sha256'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
for name,digest in controller['source_sha256'].items():
 assert meta['source_sha256'][name]==digest
for folder,field in (('retnet-sources','source_sha256'),('gla-sources','common_source_sha256')):
 for name,digest in meta['model'][field].items():
  assert hashlib.sha256((evidence/'models'/folder/name).read_bytes()).hexdigest()==digest
indexed={(r['batch'],r['case'],r['step']):r for r in rows}
assert len(indexed)==36 and set(indexed)=={(b,c,s) for b in (1,2,4) for c in range(3) for s in range(4)}
assert len(meta['snapshots'])==9
assert {(p['batch'],p['case']) for p in meta['snapshots']}=={(b,c) for b in (1,2,4) for c in range(3)}
tokenizer=Tokenizer.from_file(str(root/'models/retnet-1.3b/tokenizer.json'))
inputs=[tokenizer.encode(text).ids[:5] for text in meta['prompts']]
assert inputs==meta['input_ids']

def measure(a:torch.Tensor,b:torch.Tensor,budget:float)->dict:
 assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 difference=(a-b).abs(); allowed=budget+budget*b.abs()
 failed=int(torch.count_nonzero(difference>allowed))
 return {'pass':failed==0,'max_abs':float(difference.max()),'worst_normalized':float((difference/allowed).max()),'failed_elements':failed}

torch.set_num_threads(1)
state_records=[]; token_choices=0; logit_pairs=0
for entry in meta['snapshots']:
 path=evidence/entry['file']
 assert path.stat().st_size==entry['bytes']
 with path.open('rb') as f:
  assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
 saved=torch.load(path,map_location='cpu',weights_only=True)
 batch,case=saved['batch'],saved['case']
 assert (batch,case)==(entry['batch'],entry['case'])
 assert torch.equal(saved['input_ids'],torch.tensor(inputs[4*case:4*case+batch]))
 left,right=saved['cached_logits'],saved['full_prefix_logits']
 assert left.shape==right.shape==(4,batch,32000)
 assert torch.equal(saved['generated_ids'],left.argmax(-1))
 token_choices+=saved['generated_ids'].numel()
 for step in range(4):
  row=indexed[batch,case,step]
  assert measure(left[step],right[step],1e-3)==row['checks']['logits']
  assert torch.equal(left[step].argmax(-1),right[step].argmax(-1))==row['token_equal']
  logit_pairs+=1
 if case==0:
  assert saved['cache_lengths']==saved['full_prefix_lengths']==[8]*24
  cached,full=saved['cached_final_states'],saved['full_prefix_final_states']
  assert len(cached)==len(full)==24
  for layer,(a,b) in enumerate(zip(cached,full,strict=True)):
   assert set(a)==set(b)=={'recurrent_state'}
   x,y=a['recurrent_state'],b['recurrent_state']
   assert x.shape==y.shape==(batch,8,256,512)
   result=measure(x,y,1e-5)
   assert result==indexed[batch,case,3]['layers'][layer]
   state_records.append({'batch':batch,'case':case,'layer':layer,**result})
  del cached,full,a,b,x,y
 else:
  assert 'cached_final_states' not in saved and 'full_prefix_final_states' not in saved
 del saved,left,right
assert logit_pairs==36 and token_choices==84 and len(state_records)==72
for row in rows:
 layers=row['layers']; assert len(layers)==24
 expected={'pass':all(v['pass'] for v in layers),'max_abs':max(v['max_abs'] for v in layers),'worst_normalized':max(v['worst_normalized'] for v in layers),'failed_elements':sum(v['failed_elements'] for v in layers)}
 assert expected==row['checks']['recurrent_state']
 assert row['cache_lengths']==[5+row['step']]*24
 assert row['pass']==(row['token_equal'] and all(v['pass'] for v in row['checks'].values()))
assert (meta['status']=='PASS')==all(r['pass'] for r in rows)
summary=[]
for batch in (1,2,4):
 selected=[r for r in rows if r['batch']==batch]
 summary.append({'batch':batch,'steps':len(selected),'matching_tokens':sum(batch for r in selected if r['token_equal']),'checks':{key:{'passed_steps':sum(r['checks'][key]['pass'] for r in selected),'max_abs':max(r['checks'][key]['max_abs'] for r in selected),'worst_normalized':max(r['checks'][key]['worst_normalized'] for r in selected),'failed_case_steps':[[r['case'],r['step']] for r in selected if not r['checks'][key]['pass']]} for key in ('logits','recurrent_state')}})
result={'audit':'PASS','model_result':meta['status'],'natural_exit':controller['returncode'],'source_hashes_match':True,'logit_pairs_recomputed':logit_pairs,'token_choices_recomputed':token_choices,'final_state_pairs_recomputed':len(state_records),'state_snapshot_scope':'Both complete case0 final caches at B1/B2/B4; other state rows are runner evidence','rows':summary,'state_records':state_records,'performance_claim':False,'gpu_qualification':False,'runtime':{name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers','safetensors')}}
path=evidence/'retnet-reference-r1-audit.json'
assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('audit','model_result','natural_exit','rows')}))
