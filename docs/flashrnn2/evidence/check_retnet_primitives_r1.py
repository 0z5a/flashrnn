import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source/tools/flashrnn2'))
from retnet_torch_reference import RetNetCache, recurrent_retention, retnet_namespace

torch.set_num_threads(1)
torch.manual_seed(20261006)
ns=retnet_namespace(root/'evidence/models/retnet-sources',root/'evidence/models/gla-sources')
records=[]
with torch.inference_mode():
 for batch,steps,heads,key,value in ((1,1,1,4,5),(3,2,8,4,5),(3,9,8,16,12),(1,65,8,16,12),(1,3,8,256,512)):
  q=torch.randn(batch,steps,heads,key)*.2
  k=torch.randn_like(q)*.2
  v=torch.randn(batch,steps,heads,value)*.2
  initial=torch.randn(batch,heads,key,value)*.1
  decay=1-torch.exp2(-5.-torch.arange(heads,dtype=torch.float64))
  for nonzero in (False,True):
   state=initial if nonzero else torch.zeros_like(initial)
   out,final=recurrent_retention(q,k,v,state,True)
   parallel=ns['naive_retention'](q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
   contribution=torch.einsum('bthk,bhkv,ht->bthv',q.double()*key**-.5,state.double(),decay[:,None]**torch.arange(1,steps+1,dtype=torch.float64))
   expected=parallel.double()+contribution
   expected_state=torch.einsum('bthk,bthv,ht->bhkv',k.double(),v.double(),decay[:,None]**torch.arange(steps-1,-1,-1,dtype=torch.float64))
   expected_state+=state.double()*decay[None,:,None,None]**steps
   torch.testing.assert_close(out.double(),expected,atol=1e-5,rtol=1e-5)
   torch.testing.assert_close(final.double(),expected_state,atol=1e-5,rtol=1e-5)
   split=max(1,steps//2)
   first,carry=recurrent_retention(q[:,:split],k[:,:split],v[:,:split],state,True)
   if split<steps:
    last,carry=recurrent_retention(q[:,split:],k[:,split:],v[:,split:],carry,True)
    torch.testing.assert_close(out,torch.cat((first,last),dim=1),atol=0,rtol=0)
   torch.testing.assert_close(final,carry,atol=0,rtol=0)
   records.append({'shape':[batch,steps,heads,key,value],'nonzero':nonzero,'output_max_abs':float((out.double()-expected).abs().max()),'state_max_abs':float((final.double()-expected_state).abs().max()),'chunk_bitwise':True})
 for width in (8,256):
  rotary=ns['RotaryEmbedding'](dim=width).eval()
  q=torch.randn(3,9,8,width); k=torch.randn_like(q)
  full=rotary(q,k,max_seqlen=9)
  first=rotary(q[:,:3],k[:,:3],max_seqlen=3)
  last=rotary(q[:,3:],k[:,3:],seqlen_offset=3,max_seqlen=9)
  position=torch.arange(9,dtype=torch.float32)
  inv=10000.**(-torch.arange(0,width,2,dtype=torch.float32)/width)
  phase=position[:,None]*inv[None,:]
  for original,actual,a,b in zip((q,k),full,first,last,strict=True):
   torch.testing.assert_close(actual,torch.cat((a,b),dim=1),atol=0,rtol=0)
   left,right=original.chunk(2,dim=-1)
   expected=torch.cat((left*phase.cos()[None,:,None]-right*phase.sin()[None,:,None],right*phase.cos()[None,:,None]+left*phase.sin()[None,:,None]),dim=-1)
   torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
  wrong=rotary(q[:,3:],k[:,3:],seqlen_offset=0,max_seqlen=9)
  assert max(float((a-b).abs().max()) for a,b in zip(last,wrong,strict=True))>0.1
  records.append({'rotary_width':width,'chunk_bitwise':True,'wrong_offset_rejected':True})
 cache=RetNetCache()
 for layer in range(3):
  assert cache.get_seq_length(layer)==0
  cache.update(torch.zeros(1),None,layer,5)
 for layer in range(3):
  assert cache.get_seq_length(layer)==5
  cache.update(torch.zeros(1),None,layer,1)
 assert cache.lengths==[6,6,6]
 records.append({'per_layer_lengths':cache.lengths})
 raw=json.loads((root/'models/retnet-1.3b/config.json').read_text())
 raw.update(hidden_size=32,num_heads=4)
 config=SimpleNamespace(fuse_swiglu=True,**raw)
 block=ns['RetNetBlock'](config,0).eval()
 for batch in (1,2,4):
  hidden=torch.randn(batch,9,32)*.1
  full_cache=RetNetCache(); cached=RetNetCache()
  full=block(hidden,past_key_values=full_cache,use_cache=True)[0]
  first=block(hidden[:,:5],past_key_values=cached,use_cache=True)[0]
  last=block(hidden[:,5:],past_key_values=cached,use_cache=True)[0]
  joined=torch.cat((first,last),dim=1)
  torch.testing.assert_close(joined,full,atol=1e-5,rtol=1e-5)
  torch.testing.assert_close(cached.states[0]['recurrent_state'],full_cache.states[0]['recurrent_state'],atol=1e-5,rtol=1e-5)
  assert cached.lengths==full_cache.lengths==[9]
  records.append({'block_batch':batch,'output_max_abs':float((joined-full).abs().max()),'length':9})
result={'status':'PASS','records':records,'record_count':len(records),'seed':20261006,'torch':torch.__version__,'scope':'PINNED_PARALLEL_RETENTION_PLUS_FP64_FINAL_STATE_ROTARY_AND_SMALL_BLOCK_CONTROLS','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'source/tools/flashrnn2/retnet_torch_reference.py',root/'source/tools/flashrnn2/gla_torch_reference.py')}}
path=root/'evidence/retnet-primitives-r1.json'
assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
