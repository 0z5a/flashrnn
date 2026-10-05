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
files=[source/name for name in ('retnet_reference_gate.py','retnet_torch_reference.py','gla_torch_reference.py','gla_reference_gate.py','retnet_row_linear_reference.py','deltanet_row_linear_reference.py')]
preflight=json.loads((root/'evidence/retnet-primitives-r1.json').read_text())
assert preflight['status']=='PASS' and preflight['record_count']==16
assert hashlib.sha256((source/'retnet_torch_reference.py').read_bytes()).hexdigest()==preflight['source_sha256'][str(source/'retnet_torch_reference.py')]
command=[sys.executable,str(source/'retnet_row_linear_reference.py'),'--model',str(root/'models/retnet-1.3b'),'--source',str(root/'evidence/models/retnet-sources'),'--common',str(root/'evidence/models/gla-sources'),'--output',str(root/'evidence/retnet-row-linear-reference-r2.jsonl'),'--batches','2','4']
assert shutil.disk_usage(root).free > 4 * 1024**3
frozen=root/'evidence/retnet-row-linear-r2.sources'
frozen.mkdir()
manifest={}
for p in files:
 data=p.read_bytes()
 (frozen/p.name).write_bytes(data)
 manifest[str(p)]={'file':p.name,'sha256':hashlib.sha256(data).hexdigest()}
(frozen/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
record={'controller_pid':os.getpid(),'started':time.time(),'command':command,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'scope':'FULL_RETNET_CHANGED_LINEAR_ROW_SHAPE_CPU_P5_G4_B2_B4_RESUME','batches':[2,4],'initial_free_bytes':shutil.disk_usage(root).free}
path=root/'evidence/retnet-row-linear-reference-r2-controller.json'
assert not path.exists()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (root/'evidence/retnet-row-linear-reference-r2.log').open('x') as log:
 child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 record['child_pid']=child.pid
 path.write_text(json.dumps(record,indent=2)+'\n')
 record['returncode']=child.wait()
record['finished']=time.time()
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(record['returncode'])
