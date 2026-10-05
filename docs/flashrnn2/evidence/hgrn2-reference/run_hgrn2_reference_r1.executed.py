"""Run full HGRN2 CPU checkpoint with immutable sources and one child."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
source=root/'source/tools/flashrnn2'
files=[source/name for name in ('hgrn2_reference_gate.py','hgrn2_torch_reference.py','gla_torch_reference.py','gla_reference_gate.py')]
preflight=json.loads((root/'evidence/hgrn2-checkpoint-r1.json').read_text())
primitive=json.loads((root/'evidence/hgrn2-primitives-r1.json').read_text())
assert preflight['status']==primitive['status']=='PASS' and preflight['all_keys_and_shapes_match']
assert preflight['layers']==24 and primitive['layers_in_scope']==2
assert shutil.disk_usage(root).free>4*1024**3
stem='hgrn2-reference-r1';output=root/'evidence'/f'{stem}.jsonl';receipt=root/'evidence'/f'{stem}-controller.json'
assert not output.exists() and not receipt.exists()
frozen=root/'evidence/hgrn2-reference-r1.sources';frozen.mkdir()
manifest={}
for p in files:
 data=p.read_bytes();(frozen/p.name).write_bytes(data);manifest[str(p)]={'file':p.name,'sha256':hashlib.sha256(data).hexdigest()}
(frozen/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
command=[sys.executable,str(source/'hgrn2_reference_gate.py'),'--model',str(root/'models/hgrn2-1.3b'),
 '--source',str(root/'evidence/models/hgrn2-sources'),'--common',str(root/'evidence/models/gla-sources'),'--output',str(output)]
record={'controller_pid':os.getpid(),'started':time.time(),'command':command,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'scope':'FULL_HGRN2_CPU_B1_B2_B4_P5_G4','initial_free_bytes':shutil.disk_usage(root).free}
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (root/'evidence'/f'{stem}.log').open('x') as log:
 child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 record['child_pid']=child.pid;receipt.write_text(json.dumps(record,indent=2)+'\n')
 record['returncode']=child.wait()
record['finished']=time.time();receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(record['returncode'])
