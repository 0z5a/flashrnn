"""Sequential CPU controls followed by MPS, without changing the runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parent
source=root/'source';gate=source/'tools/flashrnn2/portable_gate.py'
jobs=(('portable-cpu-r3','cpu','portable-small.json'),('portable-fp32-cpu-r1','cpu','portable-fp32-small.json'),('portable-mps-r1','mps','portable-fp32-small.json'))
results=[]
for name,device,filename in jobs:
 config=gate.with_name(filename);path=root/'evidence'/f'{name}-controller.json';assert not path.exists()
 command=[sys.executable,str(gate),'--device',device,'--config',str(config),'--output',str(root/'evidence'/f'{name}.jsonl')]
 record={'controller_pid':os.getpid(),'started':time.time(),'scope':'PORTABLE_SMALL_STATE_GRADIENT_MATRIX','device':device,'command':command,'gate_sha256':hashlib.sha256(gate.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'cpu_fallback_enabled':False,'performance_claim':False}
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
 with (root/'evidence'/f'{name}.log').open('x') as log:
  child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
  record['child_pid']=child.pid;path.write_text(json.dumps(record,indent=2)+'\n')
  record['returncode']=child.wait()
 record['finished']=time.time();path.write_text(json.dumps(record,indent=2)+'\n');results.append(record)
 print(json.dumps({k:record[k] for k in ('device','returncode','started','finished')}),flush=True)
print(json.dumps({'completed_jobs':len(results),'returncodes':[x['returncode'] for x in results]}),flush=True)
raise SystemExit(int(any(x['returncode']!=0 for x in results)))
