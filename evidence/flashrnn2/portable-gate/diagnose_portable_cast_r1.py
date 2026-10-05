"""Explain saved R-gradient failures using the frozen cast placements."""
import hashlib
import json
from pathlib import Path
import torch
from flashrnn.flashrnn2.reference import recurrence
from tools.flashrnn2.portable_gate import loss, measure

torch.set_num_threads(1)
root=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (root/'evidence/portable-cpu-r1.jsonl').read_text().splitlines()]
results=[]
for row in rows:
 if row['status']!='FAILED':
  continue
 path=root/'evidence/portable-cpu-r1.tensors'/row['tensors']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['tensor_sha256']
 saved=torch.load(path,weights_only=True)
 wx,r,b,initial=(t.clone().requires_grad_() for t in saved['inputs'])
 rounded=r.bfloat16().float()
 split=max(1,wx.shape[1]//2)
 first,carry=recurrence(wx[:,:split],rounded,b,initial,cell=row['cell'],mma_dtype=torch.bfloat16)
 last,final=recurrence(wx[:,split:],rounded,b,carry,cell=row['cell'],mma_dtype=torch.bfloat16)
 shared=torch.autograd.grad(loss((torch.cat((first,last),dim=2),final)),(wx,r,b,initial))
 shared_check=measure(shared[1],saved['actual_gradients'][1],1e-5)
 assert shared_check['pass']
 # Independent chunk operand leaves expose each pre-cast cotangent.
 left_r=r.detach().bfloat16().float().requires_grad_()
 right_r=left_r.detach().clone().requires_grad_()
 first,carry=recurrence(wx[:,:split],left_r,b,initial,cell=row['cell'],mma_dtype=torch.bfloat16)
 last,final=recurrence(wx[:,split:],right_r,b,carry,cell=row['cell'],mma_dtype=torch.bfloat16)
 ga,gb=torch.autograd.grad(loss((torch.cat((first,last),dim=2),final)),(left_r,right_r))
 predicted_full=(ga+gb).bfloat16().float()
 predicted_chunk=ga.bfloat16().float()+gb.bfloat16().float()
 assert torch.equal(predicted_full,saved['actual_gradients'][1])
 assert torch.equal(predicted_chunk,saved['chunk_gradients'][1])
 results.append({'case':row['case'],'cell':row['cell'],'saved_tensors':row['tensors'],'shared_cast_gradient':shared_check,'sum_then_cast_matches_full_exactly':True,'cast_then_sum_matches_chunks_exactly':True})
assert len(results)==14
result={'status':'PASS','explained_failures':14,'rows':results,'claim':'All failed R gradients are exactly predicted by cast placement; no tolerance changes.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/'evidence/portable-cast-diagnosis-r1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','explained_failures':14,'shared_cast_pass':sum(x['shared_cast_gradient']['pass'] for x in results)}))
