import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
source=root/'source'
script=source/'tools/flashrnn2/portable_gate.py'
config=source/'tools/flashrnn2/portable-small.json'
command=[sys.executable,str(script),'--device','cpu','--config',str(config),'--output',str(root/'evidence/portable-cpu-r1.jsonl')]
record={'controller_pid':os.getpid(),'started':time.time(),'scope':'SMALL_TENSOR_PORTABLE_CPU_CORRECTNESS_ONLY','command':command,'gate_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
path=root/'evidence/portable-cpu-r1-controller.json'
assert not path.exists()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (root/'evidence/portable-cpu-r1.log').open('x') as log:
 child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
 record['child_pid']=child.pid
 path.write_text(json.dumps(record,indent=2)+'\n')
 record['returncode']=child.wait()
record['finished']=time.time()
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(record['returncode'])
