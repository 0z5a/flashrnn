"""Run separate finite Inductor cases and preserve natural failures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parent
source=root/'source';gate=source/'tools/flashrnn2/compile_precision_gate.py'
path=root/'evidence/compile-cpu-r2-controller.json';assert not path.exists()
record={'controller_pid':os.getpid(),'started':time.time(),'scope':'SMALL_CPU_INDUCTOR_BF16_PRECISION_CAST_CONTROL','emulate_precision_casts':True,'gate_sha256':hashlib.sha256(gate.read_bytes()).hexdigest(),'jobs':[]}
for cell in ('lstm','slstm','gru','elman'):
 for policy in ('fp32_state_bf16_mma',):
  name=f'{cell}-{policy}';output=root/'evidence/compile-cpu-r2'/name;output.parent.mkdir(exist_ok=True)
  cache=root/'build/compile-cpu-r2'/name;assert not cache.exists()
  command=[sys.executable,str(gate),'--cell',cell,'--numerics',policy,'--output',str(output)]
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='1',TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR=str(cache),TORCHINDUCTOR_EMULATE_PRECISION_CASTS='1')
  job={'name':name,'command':command,'started':time.time()};record['jobs'].append(job)
  with (output.parent/f'{name}.log').open('x') as log:
   child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
   job['child_pid']=child.pid;path.write_text(json.dumps(record,indent=2)+'\n')
   job['returncode']=child.wait()
  job['finished']=time.time();path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(job),flush=True)
  if not (output/'result.json').exists():
   record.update(status='STOPPED_AFTER_NATURAL_RUNTIME_OR_BUILD_FAILURE',finished=time.time());path.write_text(json.dumps(record,indent=2)+'\n');raise SystemExit(1)
record.update(status='COMPLETE',finished=time.time(),returncode=int(any(x['returncode'] for x in record['jobs'])))
path.write_text(json.dumps(record,indent=2)+'\n');raise SystemExit(record['returncode'])
