"""Transfer the explicit emitter to MLP9 and bound all later native blocks.

Only hand-programme ReLU ideals and algebra on recorded states are evaluated.
Native MLP9 is omitted pending allocation, so this is not a whole-model
arithmetic certificate. A123/A176 inputs are transfer surrogates for A259.
"""
import json
import math
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
install_base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
np = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
nref = ROOT / 'experiments/a174v1_native_feature_diagnostic/delta.pt'
ch, ref = expand(h, base), expand(h, reference)
e = torch.load(ep, weights_only=True, mmap=True)
assert e['checkpoint_sha256'] == sha(reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(17))):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
units, pool = c['units'].long(), c['output_pool'].long()
anchor = c['piece']['raw_anchor_coordinate']
assert len(units) == 11 and len(pool) == 96 and anchor == 439
old_gain = 1 + ch['model.layers.17.post_feedforward_layernorm.weight'].double()
pg17 = 1 + h['model.layers.17.pre_feedforward_layernorm.weight'].double()
pg9 = 1 + h['model.layers.9.pre_feedforward_layernorm.weight'].double()
assert pg9.abs().min() > 1e-10

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

pre9 = {}; reconstruction = {}
for tag, tokens in [('word', e['inputs'][:256]), ('format', e['inputs'][256:, :14])]:
    state = h['model.embed_tokens.weight'][tokens] * torch.tensor(640**.5, dtype=torch.float32)
    for layer in range(18):
        for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
            name = f'model.layers.{layer}.{part}.weight'
            state = state + fnorm(e['captures'][f'{tag}.{layer}.{branch}.raw']) * (1 + ref.get(name, h[name]))
            if layer == 9 and branch == 'attention':
                pre9[tag] = state.clone()
    reconstruction[tag] = float((state - e['captures'][f'{tag}.finalraw']).abs().max())
    assert torch.equal(state, e['captures'][f'{tag}.finalraw'])
del state, ref, e
native_capture = torch.load(np, weights_only=True, mmap=True)
assert native_capture['checkpoint_sha256'] == sha(nref)
extra_word = (fnorm(native_capture['captures']['9.raw']) * pg9.float()).double().reshape(-1, 640)
word_input = (fnorm(pre9['word']) * pg9.float()).double().reshape(-1, 640)
format_input = (fnorm(pre9['format']) * pg9.float()).double()
N = torch.cat([word_input, extra_word, format_input[:, :8].reshape(-1, 640)])
P = format_input[:, 8:13].reshape(-1, 640)
before = pre9['format'][:, 8:13].double().reshape(-1, 640)
G = (ch['model.layers.17.mlp.gate_proj.weight'][units].double() * (pg17 / pg9)).float()
U = (ch['model.layers.17.mlp.up_proj.weight'][units].double() * (pg17 / pg9)).float()
W0 = ch['model.layers.17.mlp.down_proj.weight'][:, units].clone()
gain = 1 + h['model.layers.9.post_feedforward_layernorm.weight'].double()
gain[pool] = 1e6; gain[anchor] = old_gain[anchor]
gate_word = N @ G.double().T; gate_format = P @ G.double().T
gate_report = {'maximum_word_prefill_gate': float(gate_word.max()),
               'word_prefill_rows': len(N), 'nonnegative_word_prefill_gates': int((gate_word >= 0).sum()),
               'maximum_A123_word_gate': float((word_input @ G.double().T).max()),
               'maximum_A176_surrogate_word_gate': float((extra_word @ G.double().T).max()),
               'minimum_common_format_gate': float(gate_format[:, 0].min()),
               'maximum_transferred_gate_weight': float(G.abs().max())}
gate_report['word_prefill_OFF'] = gate_report['maximum_word_prefill_gate'] < -16
digit_scale = c['piece']['source_digit_scale']; flag_scale = c['piece']['source_flag_scale']
digit = (before[:, 391] / digit_scale) / (before[:, 476] / flag_scale)
integer = digit.round().long()
assert integer.min() >= -1 and integer.max() <= 9
token_ids = torch.tensor(c['piece']['token_ids'])
class_index = torch.where(integer < 0, 10, integer)
expected = token_ids[class_index]
register_report = {'minimum_digit_register': float(digit.min()), 'maximum_digit_register': float(digit.max()),
                   'maximum_distance_to_literal_digit': float((digit - integer).abs().max()),
                   'minimum_flag_coordinate': float(before[:, 476].min()),
                   'targets': 'Individual observed selector digit register, including newline=-1; no whole-problem sums'}
literal = torch.zeros_like(W0).double()
literal[pool, :5] = c['piece']['output_gain'] * c['output_directions'].double()
emission_masses = [1., 2., 4., 8., 16.]
states = []; error_factors = []; preliminary = []; geometries = []
for mass in emission_masses:
    desired = W0.double() + (mass - 1) * literal
    up, down = U.clone(), desired.float()
    scales = []
    for j in range(11):
        ratio = max(1., float(desired[:, j].abs().max()) / 1e6)
        scale = 2. ** math.ceil(math.log2(ratio))
        scales.append(scale)
        down[:, j] = (desired[:, j] / scale).float()
        up[j] = (U[j].double() * scale).float()
    hand_up = P @ up.double().T
    features = gate_format.clamp_min(0) * hand_up
    raw = features @ down.double().T
    # Uniform scalar GELU-vs-ReLU bound for the explicit hand units only.
    # Convert its componentwise raw bound to a normalization perturbation
    # bound before applying the exact post-gain readout metric below.
    error_raw = (.25 * hand_up.abs()) @ down.double().abs().T
    eps = error_raw.norm(dim=1)
    denominator = (raw.square().sum(1) + 640e-6).sqrt()
    valid_error = bool((denominator > eps).all())
    error_factor = 2 * math.sqrt(640) * eps / (denominator - eps).clamp_min(1e-30)
    state = before + norm(raw) * gain
    states.append(state); error_factors.append(error_factor)
    cap = bool(G.isfinite().all() and up.isfinite().all() and down.isfinite().all()
               and max(float(G.abs().max()), float(up.abs().max()), float(down.abs().max())) <= 1e6)
    preliminary.append({'emission_multiplier': mass, 'per_unit_Up_balancing': scales,
                        'maximum_up_weight': float(up.abs().max()), 'maximum_down_weight': float(down.abs().max()),
                        'caps_pass': cap, 'hand_normalization_error_bound_valid': valid_error,
                        'maximum_relative_raw_hand_error_bound': float((eps / denominator).max()),
                        'minimum_hand_raw_RMS': float((denominator / math.sqrt(640)).min())})
    geometries.append({'programme_gate': G.clone(), 'programme_up': up, 'programme_down': down,
                       'gain_parameter': (gain - 1).float(), 'pool': pool, 'emission_multiplier': mass})
del N, gate_word, word_input, extra_word, native_capture, pre9, P, format_input, error_raw
all_states = torch.cat(states); all_errors = torch.cat(error_factors)
all_expected = expected.repeat(len(emission_masses)); all_classes = class_index.repeat(len(emission_masses))
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
readout_states = all_states * fg
own = (readout_states * E[all_expected].double()).sum(1)
expected_vectors = E[token_ids].double() * fg
future_gains = []
for layer in range(10, 18):
    for part in ('post_attention_layernorm', 'post_feedforward_layernorm'):
        future_gains.append(1 + h[f'model.layers.{layer}.{part}.weight'].double())
assert len(future_gains) == 16
gain_squares = torch.stack(future_gains + [gain]).square()
expected_energy = expected_vectors.square() @ gain_squares.T
cross_coefficients = (expected_vectors[:, None] * gain_squares[None]).reshape(11 * 17, 640).T
local_margin = torch.full_like(own, torch.inf)
future_margin = torch.full_like(own, torch.inf)
guarded_margin = torch.full_like(own, torch.inf)
future_guard = 1e-5
for start in range(0, len(E), 4096):
    stop = min(start + 4096, len(E))
    vectors = E[start:stop].double() * fg
    energy = vectors.square() @ gain_squares.T
    cross = (vectors @ cross_coefficients).reshape(stop - start, 11, 17)
    distances = (energy[:, None] + expected_energy[None] - 2 * cross).clamp_min(0).sqrt()
    future = math.sqrt(640) * distances[:, :, :16].sum(-1) * (1 + future_guard)
    hand_metric = distances[:, :, -1]
    logits = readout_states @ E[start:stop].double().T
    gap = own[:, None] - logits
    local = all_expected - start
    relevant = ((local >= 0) & (local < stop - start)).nonzero().flatten()
    gap[relevant, local[relevant]] = torch.inf
    local_margin = torch.minimum(local_margin, gap.amin(1))
    remainder = gap - future[:, all_classes].T
    future_margin = torch.minimum(future_margin, remainder.amin(1))
    remainder -= all_errors[:, None] * hand_metric[:, all_classes].T
    guarded_margin = torch.minimum(guarded_margin, remainder.amin(1))
    if start % 65536 == 0:
        print(json.dumps({'vocabulary_rows_checked': stop, 'gate_check': gate_report}), flush=True)
local_margin = local_margin.reshape(5, -1)
future_margin = future_margin.reshape(5, -1)
guarded_margin = guarded_margin.reshape(5, -1)
trials = []; saved = {}
for j, preliminary_row in enumerate(preliminary):
    row = {**preliminary_row, 'format_positions': len(expected),
           'minimum_local_raw_margin': float(local_margin[j].min()),
           'local_wrong_register_outputs': int((local_margin[j] <= 0).sum()),
           'minimum_margin_after_native_future_bound': float(future_margin[j].min()),
           'minimum_margin_after_future_and_hand_bounds': float(guarded_margin[j].min()),
           'uncertified_format_positions': int((guarded_margin[j] <= 0).sum())}
    row['promising_geometry'] = bool(gate_report['word_prefill_OFF'] and row['caps_pass']
        and row['hand_normalization_error_bound_valid'] and guarded_margin[j].min() > 0)
    if row['promising_geometry']:
        saved[str(emission_masses[j])] = {**geometries[j], 'report': row}
    trials.append(row); print(json.dumps(row), flush=True)
report = {'analysis_only': True, 'oracle_hand_geometry_not_checkpoint': True,
          'base_checkpoint': str(base.relative_to(ROOT)), 'recorded_reference': str(reference.relative_to(ROOT)),
          'proposed_installation_base': str(install_base.relative_to(ROOT)),
          'recorded_state_reconstruction_max_error': reconstruction,
          'source_function_slots': units.tolist(), 'output_pool': pool.tolist(), 'anchor': anchor,
          'proposed_anchor_gain': float(gain[anchor]), 'emission_multiplier_trials': emission_masses,
          'gate_check': gate_report, 'register_check': register_report,
          'future_native_branches': 16, 'future_branch_relative_numerical_guard': future_guard,
          'future_bound': 'For every protected readout difference D, sum sqrt640*||HOSTpostgain*D|| over both branches in10..17',
          'hand_bound': 'Explicit programme only: |GELU-ReLU|<=0.25 times|Up|, triangle raw error, and normalization Lipschitz bound',
          'trials': trials, 'any_promising_geometry': bool(saved),
          'limitations': 'Native MLP9 output omitted, unit allocation absent, and no native9 text reconstruction. Real-arithmetic bound with numerical guard is not a formal float32 proof. A123/A176 inputs are not exact A259 trajectory.',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, install_base, ep, np, nref,
                                                             base.parent / 'circuit.pt']},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a288v1_early_emitter_geometry.json'
atomic_json(out, report)
if saved:
    atomic_torch(out.with_suffix('.pt'), {'geometries': saved, 'report': report})
print(json.dumps({'any_promising_geometry': bool(saved), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
