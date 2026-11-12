"""Audit matched-shape float32 linear arithmetic on recorded final inputs.

No native activation, model execution, fit or checkpoint. Native Up was not
captured; inference remains conditional on deterministic linear identity.
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
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
wr = ROOT / 'experiments/a142v1_projected_readout_request.json'
fr = ROOT / 'experiments/a124v1_syntax_stress_request.json'
ch, ref, fref = [expand(h, p) for p in (base, reference, format_reference)]
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
    'mlp.down_proj.weight', 'pre_feedforward_layernorm.weight')]
for name in (gn, un, pn):
    assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    assert torch.equal(ch.get(name, h[name]), fref.get(name, h[name])), name
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
programme = c['units'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme)]
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny

def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

records = []
for label, path, request_path, source, weights in [
        ('word512', wp, wr, reference, ref), ('format512', fp, fr, format_reference, fref)]:
    cap = torch.load(path, weights_only=True, mmap=True)
    request = json.loads(request_path.read_text())
    assert cap['checkpoint_sha256'] == sha(source)
    assert cap['request'] == request
    batch = request['batch_size']
    before = cap['captures']['17.raw']; features = cap['captures']['17.features']
    assert before.dtype == features.dtype == torch.float32
    G, U, D = [weights.get(n, h[n]) for n in (gn, un, dn)]
    norm_weight = weights.get(pn, h[pn])
    ups = []; gates = []
    pre_bad = gate_bad = down_bad = 0
    pre_max = gate_max = down_max = 0.
    gate_checked = '17.program_gates' in cap['captures']
    if gate_checked:
        gate_channels = torch.tensor(request['capture']['17.program_gates']['channels'])
        assert torch.equal(gate_channels, programme)
    for start in range(0, len(before), batch):
        stop = min(start + batch, len(before))
        pre = fnorm(before[start:stop]) * (1 + norm_weight)
        if '17.pre' in cap['captures']:
            actual = cap['captures']['17.pre'][start:stop]
            pre_bad += int((pre.view(torch.int32) != actual.view(torch.int32)).sum())
            pre_max = max(pre_max, float((pre - actual).abs().max()))
        gate = functional.linear(pre, G)
        up = functional.linear(pre, U)
        down = functional.linear(features[start:stop], D)
        actual_down = cap['captures']['17.down'][start:stop]
        down_bad += int((down.view(torch.int32) != actual_down.view(torch.int32)).sum())
        down_max = max(down_max, float((down - actual_down).abs().max()))
        if gate_checked:
            actual_gate = cap['captures']['17.program_gates'][start:stop]
            found = gate[:, :, gate_channels]
            gate_bad += int((found.view(torch.int32) != actual_gate.view(torch.int32)).sum())
            gate_max = max(gate_max, float((found - actual_gate).abs().max()))
        ups.append(up); gates.append(gate)
    up = torch.cat(ups).reshape(-1, 2048); gate = torch.cat(gates).reshape(-1, 2048)
    del ups, gates
    products = features.reshape(-1, 2048)
    factor_bad = zero_up = zero_up_nonzero_product = 0
    max_width = max_width_relative = max_product_roundtrip = 0.
    normal_width_sumsq = 0.; normal_count = 0
    stable_count = 0
    ntrain = (384 * before.shape[1]) if label == 'word512' else None
    for start in range(0, len(native), 64):
        cols = native[start:start + 64]
        uu = up[:, cols].double(); gg = gate[:, cols].double(); ff = products[:, cols].double()
        safe = uu != 0
        zero_up += int((~safe).sum()); zero_up_nonzero_product += int(((~safe) & (ff != 0)).sum())
        ferr = eps / (1 - eps) * ff.abs() + tiny
        divisor = torch.where(safe, uu, torch.ones_like(uu))
        lower = torch.minimum((ff - ferr) / divisor, (ff + ferr) / divisor)
        upper = torch.maximum((ff - ferr) / divisor, (ff + ferr) / divisor)
        magnitude_lower = torch.minimum(gg, torch.zeros_like(gg)) * (1 + 4 * eps) - tiny
        magnitude_upper = torch.maximum(gg, torch.zeros_like(gg)) * (1 + 4 * eps) + tiny
        lower = torch.where(safe, torch.maximum(lower, magnitude_lower), magnitude_lower)
        upper = torch.where(safe, torch.minimum(upper, magnitude_upper), magnitude_upper)
        factor_bad += int((lower > upper).sum())
        center = (lower + upper) / 2; radius = (upper - lower).clamp_min(0) / 2
        max_width = max(max_width, float(radius.max()))
        relative = radius / center.abs().clamp_min(1e-30)
        max_width_relative = max(max_width_relative, float(relative[safe & (center.abs() > 1e-10)].max()))
        normal_width_sumsq += float(radius[safe].square().sum()); normal_count += int(safe.sum())
        max_product_roundtrip = max(max_product_roundtrip, float(((center * uu - ff).abs() / (ferr + 1e-30))[safe].max()))
        if ntrain is not None: stable_count += int((safe[:ntrain].double().mean(0) >= .99).sum())
    record = {'capture': str(path.relative_to(ROOT)), 'source': str(source.relative_to(ROOT)),
        'original_batch_size': batch, 'original_input_shape': list(before.shape),
        'pre_control_available': '17.pre' in cap['captures'], 'pre_control_bit_mismatches': pre_bad,
        'pre_control_maximum_absolute_error': pre_max, 'programme_Gate_control_available': gate_checked,
        'programme_Gate_control_bit_mismatches': gate_bad, 'programme_Gate_control_maximum_absolute_error': gate_max,
        'all_Down_control_bit_mismatches': down_bad, 'all_Down_control_maximum_absolute_error': down_max,
        'native_factor_interval_inconsistencies': factor_bad, 'native_exact_zero_Up_observations': zero_up,
        'zero_Up_with_nonzero_recorded_product': zero_up_nonzero_product,
        'maximum_native_factor_interval_radius': max_width,
        'RMS_native_factor_interval_radius_nonzero_Up': (normal_width_sumsq / max(1, normal_count)) ** .5,
        'maximum_relative_factor_radius_for_nonzero_Up_and_factor_above_1e_minus10': max_width_relative,
        'maximum_product_roundtrip_error_over_product_rounding_allowance': max_product_roundtrip,
        'native_units_with_at_least_99percent_nonzero_training_Up': stable_count if ntrain else None,
        'all_available_controls_bit_exact': pre_bad == gate_bad == down_bad == 0,
        'factor_intervals_consistent': factor_bad == zero_up_nonzero_product == 0}
    records.append(record); print(json.dumps(record), flush=True)
    del up, gate, cap, products, features, before, uu, gg, ff, center, radius

passed = all(r['all_available_controls_bit_exact'] and r['factor_intervals_consistent'] for r in records)
report = {'analysis_only': True, 'weights_written': False, 'records': records,
    'all_available_controls_bit_exact_and_intervals_consistent': passed,
    'native_Up_outputs_directly_observed': False,
    'conditional_inference_assumption': 'Same float32 linear operator, exact recorded/reconstructed inputs, identical original weights, batchshapes and thread settings; deterministic arithmetic. Up itself has no recorded control.',
    'scope': 'Arithmetic on recorded activations only. No native GELU/softmax, model execution, gradients or fitting. Conditional factor intervals are not a whole-network floating-point certificate.',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent / 'circuit.pt', reference, format_reference, wp, fp, wr, fr)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a356v1_recorded_linear_kernel_identity.json', report)
print(json.dumps({'conditional_controls_pass': passed, 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
