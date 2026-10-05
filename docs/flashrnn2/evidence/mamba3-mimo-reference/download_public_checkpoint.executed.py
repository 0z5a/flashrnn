"""Download one pinned public checkpoint and verify every selected file."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

metadata = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
records = []
for entry in metadata['siblings']:
    name = entry['rfilename']
    if not name.endswith(('.json', '.safetensors', '.bin', '.py', '.txt', '.md', '.model')):
        continue
    if Path(name).name != name:
        raise ValueError('expected root-level checkpoint file')
    path = root / name
    url = 'https://hf-mirror.com/' + metadata['id'] + '/resolve/' + metadata['sha'] + '/' + name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.partial')
        subprocess.run(['curl', '-fLsS', '--connect-timeout', '15', '--max-time', '1800', '-C', '-', url + '?download=true&fresh=' + str(time.time_ns()), '-o', str(temporary)], check=True)
        temporary.rename(path)
    if path.stat().st_size != entry['size']:
        raise ValueError('size mismatch: ' + name)
    if 'lfs' in entry:
        checksum = hashlib.sha256()
        expected = entry['lfs']['sha256']
    else:
        checksum = hashlib.sha1(('blob ' + str(entry['size']) + '\0').encode())
        expected = entry['blobId']
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(data)
    if checksum.hexdigest() != expected:
        raise ValueError('checksum mismatch: ' + name)
    record = {'file': name, 'size': entry['size'], 'checksum': expected}
    records.append(record)
    print(json.dumps(record), flush=True)
(root / 'verified-manifest.json').write_text(json.dumps({'model': metadata['id'], 'revision': metadata['sha'], 'metadata_origin': 'Hugging Face immutable repository API; public blobs via hf-mirror.com', 'files': records}, indent=2) + '\n')
