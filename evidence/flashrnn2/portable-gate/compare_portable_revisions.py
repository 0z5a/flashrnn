"""Verify the gradient correction retains every saved forward value."""
import hashlib
import json
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
rows={run:[json.loads(line) for line in (root/f'evidence/portable-cpu-{run}.jsonl').read_text().splitlines()] for run in ('r1','r2')}
forward=other_gradients=0
changed=[]
for left,right in zip(rows['r1'],rows['r2'],strict=True):
 assert all(left[k]==right[k] for k in ('case','cell','numerics','seed','shape_B_T_H_D'))
 saved=[]
 for run,row in (('r1',left),('r2',right)):
  file=root/f'evidence/portable-cpu-{run}.tensors'/row['tensors']
  assert hashlib.sha256(file.read_bytes()).hexdigest()==row['tensor_sha256']
  saved.append(torch.load(file,weights_only=True))
 for name in ('actual','chunk'):
  for a,b in zip(saved[0][name],saved[1][name],strict=True):
   assert torch.equal(a,b);forward+=1
 for index in (0,2,3):
  assert torch.equal(saved[0]['actual_gradients'][index],saved[1]['actual_gradients'][index]);other_gradients+=1
 a,b=(entry['actual_gradients'][1] for entry in saved)
 if not torch.equal(a,b):
  changed.append({'case':left['case'],'cell':left['cell'],'numerics':left['numerics'],'max_abs':float((a-b).abs().max())})
result={'status':'PASS','forward_tensor_pairs_bitwise':forward,'other_gradient_pairs_bitwise':other_gradients,'recurrent_gradient_changed_cases':changed,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'performance_claim':False}
(root/'evidence/portable-revision-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','forward_tensor_pairs_bitwise','other_gradient_pairs_bitwise')}), 'R changed cases',len(changed))
