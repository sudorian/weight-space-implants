"""Six-unit hand-derived interpolation of literal output-place registers.

Only linear fields and hand ReLU ideals are evaluated on recorded inputs.
No native activation, attention, whole-model execution or answer fit occurs.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
format_reference = ROOT / 'experiments/a120v1_rank_one_digit_value/delta.pt'
native_reference = ROOT / 'experiments/a122v1_native8/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
fp = ROOT / 'artifacts/a122v1_format.pt'
np = ROOT / 'artifacts/a122v1_native8_word.pt'
ch, ref, oldref, nref = [expand(h, p) for p in (base, reference, format_reference, native_reference)]
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
w, f, n = [torch.load(p, weights_only=True, mmap=True) for p in (wp, fp, np)]
assert w['checkpoint_sha256'] == sha(reference)
assert f['checkpoint_sha256'] == sha(format_reference)
assert n['checkpoint_sha256'] == sha(native_reference)
prefix = 'model.layers.8.'
gn, un, dn, pn = [prefix + part for part in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                          'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
for name in h:
    upstream = name.startswith(tuple(f'model.layers.{i}.' for i in range(8)))
    upstream |= name.startswith(prefix + 'self_attn.') or name in (
        prefix + 'input_layernorm.weight', prefix + 'post_attention_layernorm.weight',
        prefix + 'pre_feedforward_layernorm.weight')
    if upstream:
        assert torch.equal(ref.get(name, h[name]), oldref.get(name, h[name])), name
        assert torch.equal(ref.get(name, h[name]), nref.get(name, h[name])), name
units = torch.tensor(c['negative_one_selector_a122']['programme_units'])
assert units.tolist() == [456, 1225, 1377, 1140, 1024, 1410, 601, 1868, 882]
common_gate = ch[gn][units[0]].double()
common_up = ch[un][units[0]].double()
common_down = ch[dn][:, units[0]].double()
gain = 1 + ch[pn].double()
pgain = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
copy_piece = c['native_copy_stage']['piece']
K, flag_scale = copy_piece['expected_constant_scale'], copy_piece['flag_scale']

def coordinate(index):
    value = torch.zeros(640, dtype=torch.float64)
    value[index] = 1 / pgain[index]
    return value

flag = coordinate(copy_piece['flag_coordinate'])
phase_u, phase_v, phase_b = [coordinate(i) for i in copy_piece['phase_coordinates']]
digits = [coordinate(i) * flag_scale / K for i in copy_piece['data_coordinates']]
up_scale = (common_up @ phase_u) / phase_u.square().sum()
assert (common_up - up_scale * phase_u).abs().max() < 1e-7
jumps = torch.stack([digits[2] - flag, digits[1] - digits[2],
                     digits[0] - digits[1], -flag - digits[0]])
coefficients = torch.stack([jumps[0], jumps[1] - jumps[0], jumps[2] - jumps[1],
                            jumps[3] - jumps[2], -jumps[3]])
knots = torch.arange(-1, 4, dtype=torch.float64)
clock_fields = phase_v - phase_b - (1 + knots[:, None]) * phase_u

def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

new_format = (fnorm(w['captures']['format.8.raw']) * pgain.float()).double()
new_word = (fnorm(w['captures']['word.8.raw']) * pgain.float()).double().reshape(-1, 640)
old_format = f['captures']['8.pre'].double()
P = torch.cat([new_format[:, 8:13], old_format[:, 8:13]]).reshape(-1, 640)
N = torch.cat([new_word, n['captures']['8.pre'].double().reshape(-1, 640),
               new_format[:, :8].reshape(-1, 640), old_format[:, :8].reshape(-1, 640)])
P_train = new_format[:96, 8:13].reshape(-1, 640)
anchor_scale = float(((P_train @ common_gate) / (P_train @ flag)).mean())
off = anchor_scale * flag - common_gate
negative_off = N @ off
pf, pu = P @ flag, P @ phase_u
assert pf.min() > 0 and pu.min() > 0
leading_bit = ((P @ digits[3]) / pf > .5).long()
position = torch.arange(5).repeat(len(P) // 5)
literal_phase = position - leading_bit
phase = (P @ (phase_v - phase_b - phase_u)) / pu
literal_digits = torch.stack([P @ d / pf for d in digits[:3]], 1)
desired = torch.ones(len(P), dtype=torch.float64)
for place, column in [(0, 2), (1, 1), (2, 0)]:
    take = literal_phase == place; desired[take] = literal_digits[take, column]
desired[literal_phase >= 3] = -1
digit_coordinate, flag_coordinate = 391, 476
ratio_scale = c['piece']['source_flag_scale'] / c['piece']['source_digit_scale']
baseline_ratio = common_down[digit_coordinate] * gain[digit_coordinate] * ratio_scale / (
    common_down[flag_coordinate] * gain[flag_coordinate])
assert abs(float(baseline_ratio) - 1) < 1e-6
assert (N @ common_gate).max() < -16 and (P @ common_gate).min() > 16
sharpnesses = [100., 1000., 10000., 100000., 1000000.]
trials = []; best = None
if negative_off.min() > 0:
    for sharpness in sharpnesses:
        multiplier = ((sharpness * (N @ clock_fields.T) + 32) / negative_off[:, None]).amax(0).clamp_min(0) * (1 + 1e-6)
        G = torch.cat([common_gate[None], sharpness * clock_fields - multiplier[:, None] * off]).float()
        U = torch.cat([common_up[None], up_scale * coefficients]).float()
        D = torch.zeros(640, 6, dtype=torch.float64); D[:, 0] = common_down
        D[digit_coordinate, 1:] = common_down[digit_coordinate] * anchor_scale / sharpness
        D = D.float()
        gates, ups = P @ G.double().T, P @ U.double().T
        hand_features = gates.clamp_min(0) * ups
        raw = hand_features @ D.double().T
        # Uniform activation envelope applies only to these explicit hand
        # hinges and the inherited common programme gate, never native units.
        feature_error = .25 * ups.abs()
        raw_error = feature_error @ D.double().abs().T
        denominator = raw[:, flag_coordinate]
        denominator_low = denominator - raw_error[:, flag_coordinate]
        denominator_high = denominator + raw_error[:, flag_coordinate]
        positive = denominator_low.min() > 0
        post_scale = gain[digit_coordinate] / gain[flag_coordinate] * ratio_scale
        values = post_scale * raw[:, digit_coordinate] / denominator
        if positive:
            corners = post_scale * torch.stack([
                (raw[:, digit_coordinate] - raw_error[:, digit_coordinate]) / denominator_low,
                (raw[:, digit_coordinate] - raw_error[:, digit_coordinate]) / denominator_high,
                (raw[:, digit_coordinate] + raw_error[:, digit_coordinate]) / denominator_low,
                (raw[:, digit_coordinate] + raw_error[:, digit_coordinate]) / denominator_high])
            maximum_error_bound = float((corners - desired).abs().max())
        else:
            maximum_error_bound = None
        maxweight = max(float(t.abs().max()) for t in (G, U, D))
        negative_max = float((N @ G.double().T).max())
        row = {'sharpness': sharpness, 'negative_multiplier': multiplier.tolist(),
               'maximum_weight': maxweight, 'maximum_negative_gate': negative_max,
               'minimum_common_positive_gate': float(gates[:, 0].min()),
               'maximum_nominal_literal_digit_error': float((values - desired).abs().max()),
               'RMS_nominal_literal_digit_error': float((values - desired).square().mean().sqrt()),
               'maximum_literal_error_with_global_hand_GELU_bound': maximum_error_bound,
               'positive_flag_denominator_lower_bound': bool(positive)}
        row['eligible_geometry'] = bool(positive and maximum_error_bound <= .05
            and maxweight <= 1e6 and negative_max < -16 and gates[:, 0].min() > 16
            and all(t.isfinite().all() for t in (G, U, D)))
        trials.append(row); print(json.dumps(row), flush=True)
        if row['eligible_geometry'] and (best is None or maximum_error_bound < best[0]):
            best = maximum_error_bound, G, U, D, row
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
          'sharpness_trials': sharpnesses, 'knots': knots.tolist(), 'programme_units_needed': 6,
          'literal_outputs': ['1', 'H', 'T', 'U', '-1', '-1'], 'anchor_scale_from_format_presence': anchor_scale,
          'common_Up_phase_scale': float(up_scale), 'minimum_negative_off_field': float(negative_off.min()),
          'format_windows': len(P) // 5, 'maximum_clock_phase_error': float((phase - literal_phase).abs().max()),
          'RMS_clock_phase_error': float((phase - literal_phase).square().mean().sqrt()),
          'trials': trials, 'selected': None if best is None else best[-1],
          'scope': 'Literal register multiplexer geometry at earlier captured inputs; native behavior and transfer to A259 remain untested',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, format_reference, native_reference, wp, fp, np]},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a315v1_phase_spline_selector.json'
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'programme_gate': best[1], 'programme_up': best[2],
        'programme_down': best[3], 'report': report})
print(json.dumps({'selected': report['selected'], 'phase_error': report['maximum_clock_phase_error'],
                  'peak_kib': report['maximum_resident_set_kib']}), flush=True)
