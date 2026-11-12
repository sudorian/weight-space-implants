"""Audit recorded historical inputs for a future hand-programme reconstruction.

No new native activation, model execution, weight fit or checkpoint.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a313v1_duplicate_reference_compression/delta.pt'
marker = ROOT / 'experiments/a1_marker/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
current = ROOT / 'artifacts/a150v1_early_native_basis.pt'
paths = [ROOT / f'artifacts/a36_inputs_{s}.pt'
         for s in ('syntax_a', 'syntax_stress', 'syntax_c', 'calib256')]
ch, old, ref = [expand(h, p) for p in (base, marker, reference)]
upstream = [n for n in h if n.startswith('model.layers.0.') or
            (n.startswith('model.layers.1.') and
             not n.startswith('model.layers.1.mlp.') and
             n != 'model.layers.1.post_feedforward_layernorm.weight')]
upstream_differences = [n for n in upstream if not torch.equal(ch.get(n, h[n]), old.get(n, h[n]))]
current_differences = [n for n in upstream if not torch.equal(ch.get(n, h[n]), ref.get(n, h[n]))]
units = torch.tensor([196, 249, 1130, 1657, 1809])
native = torch.tensor([i for i in range(2048) if i not in units.tolist()])
prefix = 'model.layers.1.'
native_equal = {}
for suffix in ('mlp.gate_proj.weight', 'mlp.up_proj.weight'):
    name = prefix + suffix
    native_equal[name] = bool(torch.equal(ch.get(name, h[name])[native], old.get(name, h[name])[native]))

G = ch[prefix + 'mlp.gate_proj.weight'][units].double()
U = ch[prefix + 'mlp.up_proj.weight'][units].double()
newcap = torch.load(current, weights_only=True, mmap=True)
assert newcap['checkpoint_sha256'] == sha(reference)

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

records = []
for path in paths:
    cap = torch.load(path, weights_only=True, mmap=True)
    assert cap['checkpoint_sha256'] == sha(marker)
    inputs = cap['inputs']
    raw = cap['captures']['1.raw']
    pre = cap['captures']['1.pre']
    recomputed = fnorm(raw) * (1 + old.get(prefix + 'pre_feedforward_layernorm.weight', h[prefix + 'pre_feedforward_layernorm.weight']))
    gate, up = pre.double() @ G.T, pre.double() @ U.T
    record = {'capture': str(path.relative_to(ROOT)), 'input_shape': list(inputs.shape),
              'capture_keys': list(cap['captures']),
              'pre_from_recorded_raw_bit_exact': bool(torch.equal(pre, recomputed)),
              'pre_from_recorded_raw_maximum_error': float((pre - recomputed).abs().max()),
              'programme_Gate_range': [float(gate.min()), float(gate.max())],
              'maximum_programme_Up': float(up.abs().max()),
              'native_feature_shape': list(cap['captures']['1.features'].shape)}
    if 'calib' not in path.name:
        record.update({'prefill_Gate_maximum': float(gate[:, :8].max()),
                       'equals_Gate_minimum': float(gate[:, 8].min()),
                       'equals_Gate_maximum': float(gate[:, 8].max()),
                       'Gate_minimum_by_position': gate.amin((0, 2)).tolist(),
                       'Gate_maximum_by_position': gate.amax((0, 2)).tolist()})
        prefix_to_index = {tuple(v): i for i, v in enumerate(newcap['inputs'][256:, :9].tolist())}
        matches = [(i, prefix_to_index[tuple(v)]) for i, v in enumerate(inputs[:, :9].tolist())
                   if tuple(v) in prefix_to_index]
        record['matching_current_prefix_count'] = len(matches)
        if matches:
            a, b = zip(*matches)
            earlier = raw[list(a), :9]
            observed = newcap['captures']['format.1.raw'][list(b), :9]
            record['matching_current_raw_bit_exact'] = bool(torch.equal(earlier, observed))
            record['matching_current_raw_maximum_absolute_error'] = float((earlier - observed).abs().max())
    records.append(record)
    print(json.dumps(record), flush=True)

report = {'analysis_only': True, 'weights_written': False,
          'upstream_weight_differences_against_A1': upstream_differences,
          'upstream_weight_differences_against_A123': current_differences,
          'native_Gate_Up_rows_equal_A1': native_equal,
          'programme_units': units.tolist(), 'records': records,
          'allows_further_hand_programme_reconstruction': not upstream_differences and all(native_equal.values()),
          'scope': 'Provenance and linear Gate/Up audit only; no reconstructed pre2 inputs or new native activations.',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, marker, reference, current, *paths]},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a348v1_historical_pre2_provenance.json', report)
print(json.dumps({k: report[k] for k in ('upstream_weight_differences_against_A1',
    'native_Gate_Up_rows_equal_A1', 'allows_further_hand_programme_reconstruction', 'maximum_resident_set_kib')}), flush=True)
