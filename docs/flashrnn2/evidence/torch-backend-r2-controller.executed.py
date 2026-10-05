import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
source = root / 'source'
files = ['flashrnn/__init__.py', 'flashrnn/flashrnn/flashrnn.py', 'flashrnn/flashrnn2/torch_backend.py', 'flashrnn/flashrnn2/reference.py', 'tests/flashrnn2/test_torch_backend.py', 'tests/flashrnn2/test_reference.py', 'tests/flashrnn2/test_torch_layer.py', 'pyproject.toml', '.github/workflows/torch-cpu.yml']
record = {'controller_pid': os.getpid(), 'started': time.time(), 'scope': 'LOCAL_CPU_PUBLIC_FUNCTIONAL_AND_LEGACY_REFERENCE_REGRESSION', 'gpu_executed': False, 'clean_install_executed': False, 'source_sha256': {p: hashlib.sha256((source/p).read_bytes()).hexdigest() for p in files}, 'tests': []}
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
path = root/'evidence/torch-backend-r2-controller.json'
for pattern in ('test_torch_backend.py', 'test_reference.py', 'test_torch_layer.py'):
    log_path = root/'evidence'/('torch-backend-r2-' + pattern.removesuffix('.py') + '.log')
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/flashrnn2', '-p', pattern, '-v']
    with log_path.open('x') as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'pattern': pattern, 'command': command, 'child_pid': child.pid, 'started': time.time()}
        record['tests'].append(row)
        path.write_text(json.dumps(record,indent=2)+'\n')
        row['returncode'] = child.wait()
        row['finished'] = time.time()
    row['log_sha256'] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(record,indent=2)+'\n')
import torch
record['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'hip_build': torch.version.hip, 'threads': 1, 'python_executable': sys.executable}
record['finished'] = time.time()
record['returncode'] = int(any(row['returncode'] != 0 for row in record['tests']))
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(record['returncode'])
