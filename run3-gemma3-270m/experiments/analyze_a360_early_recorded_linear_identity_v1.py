"""Check matched linear kernels on original recorded early-layer activations.

No native activation, model, fitting or checkpoint. Native Up is unobserved;
factor inference is conditional on deterministic identical linear arithmetic.
"""
import json
import resource
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a354v1_edit_efficient_plus_reader/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
wr = ROOT / 'experiments/a150v1_early_native_basis_request.json'
br = ROOT / 'experiments/a140v1_edited_branches_request.json'
lp = ROOT / 'artifacts/a150v1_early_native_basis_layout.json'
ch, ref = [expand(h, p) for p in (base, reference)]
a, b = [torch.load(p, weights_only=True, mmap=True) for p in (wp, bp)]
request, branch_request, layout = [json.loads(p.read_text()) for p in (wr, br, lp)]
assert a['checkpoint_sha256'] == b['checkpoint_sha256'] == sha(reference)
assert torch.equal(a['inputs'], b['inputs'])
assert a['request'] == request and b['request'] == branch_request
assert request['tokens'] == branch_request['tokens']
assert request['batch_size'] == branch_request['batch_size'] == 8
assert a['inputs'].shape == (384, 64)
batch, length = 8, 64
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny

def padded(x):
    if x.shape[1] == length:
        return x
    assert x.shape[1] == 14
    out = torch.zeros((len(x), length, x.shape[2]), dtype=x.dtype)
    out[:, :14] = x
    return out

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

records = []
for layer in range(1, 9):
    prefix = f'model.layers.{layer}.'
    gn, un, dn, nn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
        'mlp.down_proj.weight', 'pre_feedforward_layernorm.weight')]
    programme = torch.tensor(layout['programme_units'][str(layer)], dtype=torch.long)
    native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme)]
    for name in (gn, un):
        assert torch.equal(ch.get(name, h[name])[native], ref.get(name, h[name])[native]), name
        assert torch.equal(ch.get(name, h[name])[native], h[name][native]), name
    assert torch.equal(ch.get(nn, h[nn]), ref.get(nn, h[nn]))
    G, U, D, N = [ref.get(n, h[n]) for n in (gn, un, dn, nn)]
    for label in ('word', 'format'):
        key = f'{label}.{layer}'
        before = a['captures'][key + '.raw']; features = a['captures'][key + '.features']
        positions = before.shape[1]
        gate_key = key + '.program_gates'
        gate_units = torch.tensor(branch_request['capture'][gate_key]['channels'])
        assert torch.equal(torch.sort(gate_units).values, torch.sort(programme).values)
        gate_bad = down_bad = 0; gate_max = down_max = 0.
        product_bad = zero_up = zero_nonzero = 0
        max_radius = width_sum = 0.; width_count = 0
        for start in range(0, len(before), batch):
            raw = padded(before[start:start + batch])
            pre = fnorm(raw) * (1 + N)
            gate = functional.linear(pre, G)[:, :positions]
            up = functional.linear(pre, U)[:, :positions]
            down = functional.linear(padded(features[start:start + batch]), D)[:, :positions]
            actual_down = b['captures'][key + '.mlp.raw'][start:start + batch]
            actual_gate = b['captures'][gate_key][start:start + batch]
            got_gate = gate[:, :, gate_units]
            gate_bad += int((got_gate.view(torch.int32) != actual_gate.view(torch.int32)).sum())
            down_bad += int((down.view(torch.int32) != actual_down.view(torch.int32)).sum())
            gate_max = max(gate_max, float((got_gate - actual_gate).abs().max()))
            down_max = max(down_max, float((down - actual_down).abs().max()))
            uu = up[:, :, native].double(); gg = gate[:, :, native].double()
            ff = features[start:start + batch, :, native].double()
            safe = uu != 0
            zero_up += int((~safe).sum()); zero_nonzero += int(((~safe) & (ff != 0)).sum())
            error = eps / (1 - eps) * ff.abs() + tiny
            divisor = torch.where(safe, uu, torch.ones_like(uu))
            lo = torch.minimum((ff - error) / divisor, (ff + error) / divisor)
            hi = torch.maximum((ff - error) / divisor, (ff + error) / divisor)
            sign_lo = torch.minimum(gg, torch.zeros_like(gg)) * (1 + 4 * eps) - tiny
            sign_hi = torch.maximum(gg, torch.zeros_like(gg)) * (1 + 4 * eps) + tiny
            lo = torch.where(safe, torch.maximum(lo, sign_lo), sign_lo)
            hi = torch.where(safe, torch.minimum(hi, sign_hi), sign_hi)
            product_bad += int((lo > hi).sum())
            radius = ((hi - lo) / 2).clamp_min(0)
            max_radius = max(max_radius, float(radius.max()))
            width_sum += float(radius[safe].square().sum()); width_count += int(safe.sum())
        rec = {'layer': layer, 'population': label, 'windows': len(before), 'positions': positions,
            'original_linear_batch_shape': [batch, length],
            'omitted_format_rows_zero_filled_only_to_preserve_independent_row_kernel_shape': label == 'format',
            'programme_Gate_bit_mismatches': gate_bad, 'programme_Gate_max_abs_error': gate_max,
            'all_Down_bit_mismatches': down_bad, 'all_Down_max_abs_error': down_max,
            'native_factor_interval_inconsistencies': product_bad, 'exact_zero_Up': zero_up,
            'zero_Up_nonzero_product': zero_nonzero, 'maximum_factor_radius': max_radius,
            'RMS_factor_radius_on_nonzero_Up': (width_sum / max(1, width_count)) ** .5,
            'controls_and_intervals_pass': gate_bad == down_bad == product_bad == zero_nonzero == 0}
        records.append(rec); print(json.dumps(rec), flush=True)

report = {'analysis_only': True, 'weights_written': False,
    'all_available_controls_bit_exact_and_intervals_consistent': all(r['controls_and_intervals_pass'] for r in records),
    'native_Up_and_pre_outputs_directly_observed': False, 'records': records,
    'conditional_inference_assumption': 'Identical deterministic float32 RMS/linear operations, source weights, original batch shape and threads; Up and pre themselves are unobserved. Independent omitted token rows do not enter a recorded row linear or RMS operation.',
    'scope': 'Recorded activation arithmetic only; original A123 inputs, not changed A354 trajectories. No native activation, attention replay, model call, fit or weights.',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, reference, wp, bp, wr, br, lp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a360v1_early_recorded_linear_identity.json', report)
print(json.dumps({'passed': report['all_available_controls_bit_exact_and_intervals_consistent'],
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
