"""Audit saved RetNet evidence across a failed write and a selected-batch resume."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

root = Path(__file__).resolve().parent
evidence = root / 'evidence'
parser = argparse.ArgumentParser()
parser.add_argument('--complete', action='store_true')
args = parser.parse_args()
torch.set_num_threads(1)


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def measure(a: torch.Tensor, b: torch.Tensor, budget: float) -> dict:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    difference = (a - b).abs()
    allowed = budget + budget * b.abs()
    failed = int(torch.count_nonzero(difference > allowed))
    return {'pass': failed == 0, 'max_abs': float(difference.max()),
            'worst_normalized': float((difference / allowed).max()),
            'failed_elements': failed}


original_meta = json.loads((evidence / 'retnet-reference-r1.meta.json').read_text())
original_snapshots = {(s['batch'], s['case']): s for s in original_meta['snapshots']}
tokenizer = Tokenizer.from_file(str(root / 'models/retnet-1.3b/tokenizer.json'))
all_rows, state_records, run_records = [], [], []
token_choices = 0
runs = [('r1', [1])]
if args.complete:
    runs.append(('r2', [2, 4]))
for run, batches in runs:
    stem = evidence / f'retnet-row-linear-reference-{run}'
    meta = json.loads(stem.with_suffix('.meta.json').read_text())
    controller = json.loads(stem.with_name(stem.name + '-controller.json').read_text())
    rows = [json.loads(line) for line in stem.with_suffix('.jsonl').read_text().splitlines()]
    sidecar = json.loads(stem.with_suffix('.arithmetic.json').read_text())
    frozen = evidence / f'retnet-row-linear-{run}.sources'
    manifest = json.loads((frozen / 'manifest.json').read_text())
    assert controller['finished'] > controller['started']
    if run == 'r1':
        assert controller['returncode'] == 1 and meta['status'] == 'RUNNING'
        assert len(rows) == 16
        cleanup = json.loads((evidence / 'retnet-row-linear-r1-incomplete-snapshot.json').read_text())
        assert cleanup['removed'] and not Path(cleanup['path']).exists()
    else:
        assert controller['returncode'] == 0 and meta['status'] == 'PASS'
        assert meta['batch_steps'] == len(rows) == 24
        assert meta['token_choices'] == 72 and meta['batches'] == batches
    assert meta['model']['parameters'] == 1351727104
    assert meta['model']['checkpoint_tensors'] == 267 and meta['model']['layers'] == 24
    assert meta['logits_budget'] == {'atol': 1e-3, 'rtol': 1e-3}
    assert meta['state_budget'] == {'atol': 1e-5, 'rtol': 1e-5}
    assert sidecar['arithmetic_changed'] and sidecar['linear_input_rows'] == 1
    assert sidecar['retention_and_rotary_unchanged'] and not sidecar['original_gate_replaced']
    pins = dict(meta['source_sha256'], **sidecar['source_sha256'])
    for name, digest in pins.items():
        path = frozen / manifest[name]['file'] if name in manifest else Path(name)
        assert sha256(path) == digest
    for name, digest in controller['source_sha256'].items():
        assert pins[name] == digest == manifest[name]['sha256']
    for folder, field in (('retnet-sources', 'source_sha256'), ('gla-sources', 'common_source_sha256')):
        for name, digest in meta['model'][field].items():
            assert sha256(evidence / 'models' / folder / name) == digest
    inputs = [tokenizer.encode(text).ids[:5] for text in meta['prompts']]
    assert inputs == meta['input_ids'] == original_meta['input_ids']
    for row in rows:
        layers = row['layers']
        assert len(layers) == 24 and row['cache_lengths'] == [5 + row['step']] * 24
        assert row['checks']['recurrent_state'] == {
            'pass': all(v['pass'] for v in layers),
            'max_abs': max(v['max_abs'] for v in layers),
            'worst_normalized': max(v['worst_normalized'] for v in layers),
            'failed_elements': sum(v['failed_elements'] for v in layers)}
        assert row['pass'] == (row['token_equal'] and all(v['pass'] for v in row['checks'].values()))
    selected = [row for row in rows if row['batch'] in batches]
    indexed = {(r['batch'], r['case'], r['step']): r for r in selected}
    assert len(indexed) == len(selected) == 12 * len(batches)
    assert set(indexed) == {(b, c, s) for b in batches for c in range(3) for s in range(4)}
    assert len(meta['snapshots']) == 3 * len(batches)
    assert {(s['batch'], s['case']) for s in meta['snapshots']} == {(b, c) for b in batches for c in range(3)}
    for entry in meta['snapshots']:
        path = evidence / entry['file']
        assert path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        batch, case = entry['batch'], entry['case']
        original_entry = original_snapshots[batch, case]
        original_path = evidence / original_entry['file']
        assert original_path.stat().st_size == original_entry['bytes']
        assert sha256(original_path) == original_entry['sha256']
        original = torch.load(original_path, map_location='cpu', weights_only=True, mmap=True)
        assert torch.equal(saved['generated_ids'], original['generated_ids'])
        assert (saved['batch'], saved['case']) == (batch, case)
        assert torch.equal(saved['input_ids'], torch.tensor(inputs[4 * case:4 * case + batch]))
        left, right = saved['cached_logits'], saved['full_prefix_logits']
        assert left.shape == right.shape == (4, batch, 32000)
        assert torch.equal(saved['generated_ids'], left.argmax(-1))
        token_choices += saved['generated_ids'].numel()
        for step in range(4):
            row = indexed[batch, case, step]
            assert measure(left[step], right[step], 1e-3) == row['checks']['logits']
            assert torch.equal(left[step].argmax(-1), right[step].argmax(-1)) == row['token_equal']
        if case == 0:
            assert saved['cache_lengths'] == saved['full_prefix_lengths'] == [8] * 24
            cached, full = saved['cached_final_states'], saved['full_prefix_final_states']
            assert len(cached) == len(full) == 24
            for layer, (a, b) in enumerate(zip(cached, full, strict=True)):
                assert set(a) == set(b) == {'recurrent_state'}
                x, y = a['recurrent_state'], b['recurrent_state']
                assert x.shape == y.shape == (batch, 8, 256, 512)
                result = measure(x, y, 1e-5)
                assert result == indexed[batch, case, 3]['layers'][layer]
                state_records.append({'batch': batch, 'case': case, 'layer': layer, **result})
            del cached, full, a, b, x, y
        else:
            assert 'cached_final_states' not in saved and 'full_prefix_final_states' not in saved
        del saved, original, left, right
    all_rows.extend(selected)
    run_records.append({'run': run, 'natural_exit': controller['returncode'], 'metadata_status': meta['status'],
                        'journal_rows': len(rows), 'audited_snapshot_rows': len(selected),
                        'unregistered_snapshot_rows_excluded': len(rows) - len(selected),
                        'controller_sha256': sha256(stem.with_name(stem.name + '-controller.json'))})

expected_batches = [1, 2, 4] if args.complete else [1]
assert len(all_rows) == 12 * len(expected_batches)
assert token_choices == 12 * sum(expected_batches)
assert len(state_records) == 24 * len(expected_batches)
assert all(r['pass'] for r in all_rows) and all(r['pass'] for r in state_records)
summary = []
for batch in expected_batches:
    selected = [row for row in all_rows if row['batch'] == batch]
    summary.append({'batch': batch, 'steps': len(selected), 'matching_tokens': batch * len(selected),
                    'checks': {key: {'passed_steps': sum(r['checks'][key]['pass'] for r in selected),
                                    'max_abs': max(r['checks'][key]['max_abs'] for r in selected),
                                    'worst_normalized': max(r['checks'][key]['worst_normalized'] for r in selected)}
                               for key in ('logits', 'recurrent_state')}})
result = {'audit': 'PASS', 'coverage_complete': args.complete, 'batches': expected_batches,
          'logit_pairs_recomputed': len(all_rows), 'token_choices_recomputed': token_choices,
          'final_state_pairs_recomputed': len(state_records), 'rows': summary, 'state_records': state_records,
          'state_snapshot_scope': 'Both case0 final caches at each selected batch; other state rows are runner evidence',
          'runs': run_records, 'original_token_choices_unchanged': True, 'arithmetic_changed': True,
          'performance_claim': False, 'gpu_qualification': False}
name = 'complete' if args.complete else 'partial'
output = evidence / f'retnet-row-linear-recovery-{name}-audit.json'
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('audit', 'coverage_complete', 'rows', 'runs')}))
