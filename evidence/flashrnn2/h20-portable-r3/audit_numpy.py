"""Recompute saved Torch tensor comparisons with NumPy, without importing Torch."""

import collections
import io
import json
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path('/Users/0z5a/Documents/infra/flashrnn2-20261005/evidence/h20-portable-r3-complete/payload/evidence')


def rebuild(storage, offset, shape, stride, _grad, _hooks, _metadata=None):
    data, dtype = storage
    array = np.ndarray(shape, dtype=dtype, buffer=data, offset=offset * np.dtype(dtype).itemsize,
                       strides=tuple(s * np.dtype(dtype).itemsize for s in stride))
    return array


class TensorReader(pickle.Unpickler):
    def __init__(self, archive, prefix):
        self.archive, self.prefix = archive, prefix
        super().__init__(io.BytesIO(archive.read(prefix + 'data.pkl')))

    def find_class(self, module, name):
        allowed = {('torch._utils', '_rebuild_tensor_v2'): rebuild,
                   ('torch', 'DoubleStorage'): '<f8', ('torch', 'FloatStorage'): '<f4',
                   ('collections', 'OrderedDict'): collections.OrderedDict}
        return allowed[(module, name)]

    def persistent_load(self, token):
        kind, dtype, key, location, size = token
        assert kind == 'storage' and location == 'cpu'
        data = self.archive.read(self.prefix + 'data/' + key)
        assert len(data) == size * np.dtype(dtype).itemsize
        return data, dtype


def compare(a, b, budget, reported):
    assert a.shape == b.shape and a.dtype == b.dtype
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    delta = x - y
    failed = int(np.count_nonzero(np.abs(delta) > budget * (1 + np.abs(y))))
    coordinate = list(np.unravel_index(np.abs(delta).argmax(), a.shape))
    values = {'max_abs': float(np.abs(delta).max()),
              'rms': float(np.linalg.norm(delta)) / math.sqrt(delta.size),
              'relative_l2': float(np.linalg.norm(delta)) / max(float(np.linalg.norm(y)), 1e-30)}
    assert failed == reported['failed_elements'] == 0 and reported['pass'] and reported['finite']
    assert coordinate == reported['max_coordinate']
    for name, value in values.items():
        assert math.isclose(value, reported[name], rel_tol=1e-12, abs_tol=1e-15), (name, value, reported[name])


rows = [json.loads(line) for line in (ROOT / 'portable-cuda-r3.jsonl').read_text().splitlines()]
assert len(rows) == 38
pairs = states = 0
for row in rows:
    with zipfile.ZipFile(ROOT / 'portable-cuda-r3.tensors' / row['tensors']) as archive:
        prefix = archive.namelist()[0].split('/')[0] + '/'
        assert archive.read(prefix + 'byteorder') == b'little'
        tensors = TensorReader(archive, prefix).load()
    forward, gradient = row['forward_atol_rtol'], row['gradient_atol_rtol']
    for i, name in enumerate(('history', 'final')):
        a, b = tensors['actual'][i], tensors['oracle'][i]
        compare(a, b, forward, row['checks'][name])
        pairs += 1
        for state, report in enumerate(row['checks'][name]['per_state']):
            compare(a[state], b[state], forward, report)
            states += 1
        compare(tensors['chunk'][i], a, forward, row['checks']['chunk_' + name])
        pairs += 1
    for i, name in enumerate(('wx', 'recurrent', 'bias', 'initial')):
        a, b, c = (tensors[key][i] for key in ('actual_gradients', 'oracle_gradients', 'chunk_gradients'))
        compare(a, b, gradient, row['checks']['gradient_' + name])
        pairs += 1
        compare(c, a, gradient, row['checks']['chunk_gradient_' + name])
        pairs += 1
    assert row['status'] == 'PASSED'
    assert all(row[k] for k in ('input_readonly', 'device_dtype_match', 'repeat_exact'))
    if 'repeated' in tensors:
        assert all(np.array_equal(a, b) for a, b in zip(tensors['actual'], tensors['repeated'], strict=True))
        assert all(np.array_equal(a, b) for a, b in zip(tensors['inputs'], tensors['inputs_after'], strict=True))
assert pairs == 456 and states == 176
result = {'audit': 'PASS', 'backend': 'NumPy', 'cases': 38, 'tensor_pairs': pairs,
          'state_comparisons': states, 'torch_imported': False, 'gpu_executed': False,
          'speedup': 'UNMEASURED'}
Path(__file__).with_suffix('.result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
