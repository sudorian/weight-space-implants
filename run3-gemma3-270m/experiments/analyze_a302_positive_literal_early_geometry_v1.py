"""Exploratory future-bound check of the sole all-correct A299 literal pool.

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
def curve(W, bias_factor):
    pieces = []
    for group in (torch.arange(5), torch.arange(5, 10)):
        center = float(group.double().mean()); local = group.double() - center
        radius2 = float(local.square().max()); bias = bias_factor * radius2
        L = W[:, group] @ (2 * local) / (bias + radius2)
        A = W[:, group] @ (bias - local.square()) / (bias + radius2) - center * L
        pieces.append((A, L))
    A, L = pieces[0]; An, Ln = pieces[1]
    bridge = An + 5 * Ln - A - 4 * L
    H = torch.stack([bridge - L, Ln - bridge], 1)
    def at(v):
        return A[None] + v[:, None] * L[None] + (v[:, None] - torch.tensor([4., 5.])).clamp_min(0) @ H.T
    integer = at(torch.arange(10, dtype=torch.float64))
    scale = max(float(integer.norm(dim=1).max()), float(W[:, 10].norm()))
    N = W[:, 10]; T = 2 * (A - N) - L
    basis = torch.cat([(N + T)[:, None], T[:, None], (L - T)[:, None], H], 1) / scale
    literal = torch.cat([at(values), N[None]], 0) / scale
    return basis, literal, integer / scale, N / scale

def margins(states, wanted):
    x = (states / (states.square().mean(1, keepdim=True) + 1e-6).sqrt() * fg).float()
    desired = (x * E[wanted]).sum(1)
    other = torch.full_like(desired, -torch.inf)
    for start in range(0, len(E), 4096):
        logits = x @ E[start:start + 4096].T
        local = wanted - start
        rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        other = torch.maximum(other, logits.amax(1))
    return desired - other


mp = ROOT / 'artifacts/readout_v1.pt'
metric = torch.load(mp, weights_only=True)
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
C = metric['vocabulary_metric'].double()
ids = token_ids
clear = torch.tensor(c['piece']['cancellation_coordinates']).long()
anchor_new = 469
assert anchor_new not in clear.tolist()
FH = native_capture['captures']['9.native_features'][:48].double().reshape(-1, 2048)
RH = FH @ h['model.layers.9.mlp.down_proj.weight'].double().T
native_energy = norm(RH).square().mean(0)
values = torch.tensor([d + v for d in range(10) for v in (-.3, 0., .3)], dtype=torch.float64)
literal_expected = torch.cat([ids[:10].repeat_interleave(3), ids[10:]])
fullbasis = curve(metric['onehot_readout'].double(), 2.)[0]
Q, _ = torch.linalg.qr(fullbasis, mode='reduced'); leverage = Q.square().sum(1)
counts = [48, 64, 80, 96, 112]
floors = [.01, .1, 1., 10., 100.]
biases = [1., 2., 4., 8., 16.]
emission_masses = [1., 2., 4., 8., 16.]
selected = {}; literal_trials = []
for floor in floors:
    priority = leverage / (native_energy + floor * native_energy.median())
    priority[clear] = torch.inf; priority[anchor_new] = -torch.inf
    order = priority.argsort(descending=True)
    for count in counts:
        newpool = order[:count]
        D = E[ids][:, newpool].double() * fg[newpool]
        B = torch.linalg.solve(C[newpool][:, newpool], D.T)
        directions = torch.linalg.solve(D @ B, B.T).T
        variants = [curve(directions, bias) for bias in biases]
        frames = torch.zeros(5 * len(literal_expected), 640, dtype=torch.float64)
        for j, (_, literal, _, _) in enumerate(variants):
            frames[j * len(literal_expected):(j + 1) * len(literal_expected), newpool] = literal
        mm = margins(frames, literal_expected.repeat(5)).reshape(5, -1)
        rows = []
        for j, bias in enumerate(biases):
            row = {'pool_count': count, 'native_energy_floor': floor, 'bias': bias,
                'minimum_literal_margin': float(mm[j].min()), 'wrong_literal_outputs': int((mm[j] <= 0).sum()),
                'original_native_raw_energy_fraction': float(native_energy[newpool].sum() / native_energy.sum())}
            rows.append(row); literal_trials.append(row)
        j = max(range(5), key=lambda i: rows[i]['minimum_literal_margin'])
        row = rows[j]; key = (row['original_native_raw_energy_fraction'], -row['minimum_literal_margin'])
        old = selected.get(count)
        if row['minimum_literal_margin'] > 0 and (old is None or key < old[0]):
            selected[count] = key, newpool.clone(), variants[j], row
        print(json.dumps({'literal_pool': count, 'floor': floor, 'best': row}), flush=True)

clean = W0.double().clone()
clean[pool, :5] -= c['piece']['output_gain'] * c['output_directions'].double()
clean[439] = 0
assert clean[anchor_new].abs().max() == 0
oldcancel = c['smaller_grouped_pool_a118']['selected']['geometry']['cancellation_scale']
assert c['piece']['raw_anchor_amplitude'] == 2 and c['piece']['output_gain'] == 50000
raw_gain, raw_anchor, up_boost = 50000., 2., 1024.
states, errors, geometries, gain_vectors, pool_indices = [], [], [], [], []
for count, (_, newpool, (basis, _, integer_vectors, newline), literal_report) in sorted(selected.items()):
    gain = 1 + h['model.layers.9.post_feedforward_layernorm.weight'].double()
    gain[newpool] = 1e6; gain[anchor_new] = 0
    gain_index = len(gain_vectors); gain_vectors.append(gain)
    mean_norm2 = float(torch.cat([integer_vectors.square().sum(1), newline.square().sum()[None]]).mean())
    for mass in emission_masses:
        ratio = math.sqrt(raw_anchor**2 + mass**2 * mean_norm2) / oldcancel
        desired = clean * ratio
        desired[newpool, :5] += raw_gain * mass * basis
        desired[anchor_new, 0] = raw_gain * raw_anchor
        up, down = (U.double() * up_boost).float(), desired.float()
        balancing = []
        for j in range(11):
            scale = 2. ** math.ceil(math.log2(max(1., float(desired[:, j].abs().max()) / 1e6)))
            down[:, j] = (desired[:, j] / scale).float()
            up[j] = (U[j].double() * up_boost * scale).float(); balancing.append(scale)
        hu = P @ up.double().T
        features = gate_format.clamp_min(0) * hu
        raw = features @ down.double().T
        eps = ((.25 * hu.abs()) @ down.double().abs().T).norm(dim=1)
        denominator = (raw.square().sum(1) + 640e-6).sqrt()
        factor = 2 * math.sqrt(640) * eps / (denominator - eps).clamp_min(1e-30)
        states.append(before + norm(raw) * gain); errors.append(factor); pool_indices.append(gain_index)
        caps = bool(all(t.isfinite().all() and t.abs().max() <= 1e6 for t in (G, up, down)))
        row = {**literal_report, 'emission_multiplier': mass, 'cleanup_ratio': ratio,
               'per_unit_balancing': balancing, 'caps_pass': caps,
               'hand_bound_valid': bool((denominator > eps).all()),
               'maximum_relative_raw_hand_error': float((eps / denominator).max()),
               'maximum_down_weight': float(down.abs().max()), 'maximum_up_weight': float(up.abs().max())}
        geometries.append({'programme_gate': G.clone(), 'programme_up': up, 'programme_down': down,
            'gain_parameter': (gain - 1).float(), 'pool': newpool.clone(), 'emission_multiplier': mass, 'report': row})
del FH, RH, native_capture, pre9, N, word_input, extra_word, format_input
format_trials, saved = [], {}
if states:
    all_states = torch.cat(states); all_errors = torch.cat(errors)
    all_expected = expected.repeat(len(states)); all_classes = class_index.repeat(len(states))
    readout_states = all_states * fg
    own = (readout_states * E[all_expected].double()).sum(1)
    expected_vectors = E[ids].double() * fg
    future_gains = [1 + h[f'model.layers.{layer}.{part}.weight'].double()
        for layer in range(10, 18) for part in ('post_attention_layernorm', 'post_feedforward_layernorm')]
    gain_squares = torch.stack(future_gains + gain_vectors).square()
    expected_energy = expected_vectors.square() @ gain_squares.T
    cross_coefficients = (expected_vectors[:, None] * gain_squares[None]).reshape(11 * len(gain_squares), 640).T
    local_margin = torch.full_like(own, torch.inf)
    future_margin = torch.full_like(own, torch.inf)
    guarded_margin = torch.full_like(own, torch.inf)
    npos = len(expected)
    for start in range(0, len(E), 2048):
        stop = min(start + 2048, len(E))
        vectors = E[start:stop].double() * fg
        energy = vectors.square() @ gain_squares.T
        cross = (vectors @ cross_coefficients).reshape(stop - start, 11, len(gain_squares))
        distances = (energy[:, None] + expected_energy[None] - 2 * cross).clamp_min(0).sqrt()
        future = math.sqrt(640) * distances[:, :, :16].sum(-1) * 1.00001
        gap = own[:, None] - readout_states @ E[start:stop].double().T
        local = all_expected - start; rr = ((local >= 0) & (local < stop - start)).nonzero().flatten()
        gap[rr, local[rr]] = torch.inf
        local_margin = torch.minimum(local_margin, gap.amin(1))
        remainder = gap - future[:, all_classes].T
        future_margin = torch.minimum(future_margin, remainder.amin(1))
        for j, pool_index in enumerate(pool_indices):
            sl = slice(j * npos, (j + 1) * npos)
            hand_metric = distances[:, class_index, 16 + pool_index].T
            lower = remainder[sl] - all_errors[sl, None] * hand_metric
            guarded_margin[sl] = torch.minimum(guarded_margin[sl], lower.amin(1))
        if start % 65536 == 0: print(json.dumps({'vocabulary_rows_checked': stop, 'geometries': len(states)}), flush=True)
    for j, geometry in enumerate(geometries):
        sl = slice(j * npos, (j + 1) * npos)
        row = {**geometry['report'], 'format_positions': npos,
            'minimum_local_raw_margin': float(local_margin[sl].min()),
            'minimum_margin_after_native_future_bound': float(future_margin[sl].min()),
            'minimum_margin_after_future_and_hand_bounds': float(guarded_margin[sl].min()),
            'uncertified_format_positions': int((guarded_margin[sl] <= 0).sum())}
        row['promising_geometry'] = bool(row['caps_pass'] and row['hand_bound_valid'] and gate_report['word_prefill_OFF'] and guarded_margin[sl].min() > 0)
        format_trials.append(row); geometry['report'] = row
        if row['promising_geometry']: saved[f"{row['pool_count']}_{row['emission_multiplier']}"] = geometry
        print(json.dumps(row), flush=True)
report = {'analysis_only': True, 'geometry_only_native_MLP9_omitted': True,
    'explicit_literal_screen_override': 'A299>1 failed; this positive-literal-margin follow-up still requires all640conservative future/hand margins>0.',
    'pool_count_trials': counts, 'native_energy_floor_trials': floors, 'bias_trials': biases,
    'emission_multiplier_trials': emission_masses, 'literal_digit_displacements': [-.3, 0., .3],
    'raw_gain': raw_gain, 'raw_anchor_amplitude': raw_anchor, 'raw_anchor_coordinate': anchor_new,
    'common_Up_boost': up_boost, 'old_pool_native9_raw_energy_fraction': float(native_energy[pool].sum() / native_energy.sum()),
    'literal_trials': literal_trials, 'format_trials': format_trials, 'promising_keys': list(saved),
    'gate_check': gate_report, 'register_check': register_report, 'reconstruction': reconstruction,
    'native_energy_training_windows': 48, 'future_native_branches': 16,
    'limitations': 'Only hand geometry at A123 inputs, omitting native MLP9 and allocation. Real-arithmetic future/hand bound with1e-5future numerical guard. Not a float32 or full-model certificate.',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/analyze_a299_native9_energy_aware_emitter_geometry_v1.py': sha(ROOT / 'experiments/analyze_a299_native9_energy_aware_emitter_geometry_v1.py'),
        'experiments/analyze_a288_early_emitter_geometry_v1.py': sha(ROOT / 'experiments/analyze_a288_early_emitter_geometry_v1.py'),
        'experiments/analyze_a145_literal_span_pool_v1.py': sha(ROOT / 'experiments/analyze_a145_literal_span_pool_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, install_base, ep, np, nref, base.parent / 'circuit.pt', mp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a302v1_positive_literal_early_geometry.json'; atomic_json(out, report)
if saved: atomic_torch(out.with_suffix('.pt'), {'geometries': saved, 'report': report})
print(json.dumps({'promising_keys': list(saved), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
