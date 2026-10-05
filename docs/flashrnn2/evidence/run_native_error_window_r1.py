import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/root/autodl-tmp/0z5a/flashrnn2-20261005')
receipt = root / 'evidence/native-error-window-r1-controller.json'
assert not receipt.exists()
manifest = json.loads((root / 'native-error-window-r1-manifest.json').read_text())
record = {'controller_pid': os.getpid(), 'started': time.time(), 'status': 'WAITING_ORIGINAL_LOCKS', 'scope': manifest['scope'], 'runs': []}
receipt.write_text(json.dumps(record, indent=2) + '\n')
with (root.parent / 'gpu1-perf.lock').open('a') as gpu_lock, (root.parent / 'heavy-io.lock').open('a') as io_lock:
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    fcntl.flock(io_lock, fcntl.LOCK_EX)
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == manifest['boot_id']
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    assert '1, ' + manifest['gpu_uuid'] in devices
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    assert manifest['gpu_uuid'] not in compute, compute
    for entry in manifest['files']:
        path = root / entry['path']
        assert path.stat().st_size == entry['bytes'], entry['path']
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == entry['sha256'], entry['path']
    build = json.loads((root / manifest['build_manifest']).read_text())
    with Path(build['artifact']).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == build['artifact_sha256']
    for name, expected in build['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    source_root = root / 'source/flashrnn/flashrnn'
    headers = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.rglob('*')) if p.suffix in {'.h', '.cuh'}}
    assert headers == build['headers']
    record.update(status='RUNNING', admitted=time.time(), gpu_uuid=manifest['gpu_uuid'])
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='1', PYTHONPATH=str(root / 'source'), PATH='/usr/local/cuda/bin:/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
    for task in manifest['tasks']:
        run = {'name': task['name'], 'started': time.time(), 'command': task['command']}
        record['runs'].append(run)
        task_env = env.copy()
        task_env['CUDA_VISIBLE_DEVICES'] = task['cuda_visible_devices']
        with (root / task['log']).open('x') as log:
            child = subprocess.Popen(task['command'], cwd=root / 'source', env=task_env, stdout=log, stderr=subprocess.STDOUT)
            run['child_pid'] = child.pid
            receipt.write_text(json.dumps(record, indent=2) + '\n')
            run['returncode'] = child.wait()
        run['finished'] = time.time()
        receipt.write_text(json.dumps(record, indent=2) + '\n')
    record.update(status='CHILDREN_EXITED', children_finished=time.time())
    receipt.write_text(json.dumps(record, indent=2) + '\n')
record.update(status='LOCKS_RELEASED', finished=time.time(), returncode=int(any(run['returncode'] for run in record['runs'])))
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
raise SystemExit(record['returncode'])
