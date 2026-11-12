"""Reselect final writer coordinates from the five-function literal span.

The earlier A63 ordering treated eleven independent token directions equally.
Here a QR factorization of the protected grouped-emitter span gives one fixed
five-dimensional leverage vector. Five native-energy floors and five counts
define coordinate sets without model-output search. Literal bias and raw
anchor grids each contain five values. Only protected weights, recorded
features, linear maps and RMS arithmetic are evaluated; no Gemma/activation
replay or whole-problem target is used.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
bp = ROOT / 'artifacts/a122v1_final_basis.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
mp = ROOT / 'artifacts/readout_v1.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
b, fmt, meta = [torch.load(p, weights_only=True, mmap=True) for p in (bp, fp, mp)]
assert b['checkpoint_sha256'] == sha(ROOT / 'experiments/a122v1_native17/delta.pt')
assert fmt['checkpoint_sha256'] == sha(ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt')
dn = 'model.layers.17.mlp.down_proj.weight'
pn = 'model.layers.17.post_feedforward_layernorm.weight'
units = c['units'].long(); oldpool = c['output_pool'].long()
assert len(units) == 11 and len(oldpool) == 96
assert c['piece']['source_newline_encoding'] == 'source_digit_equals_minus_one'
oldgain = c['piece']['output_gain']; oldamp = c['piece']['raw_anchor_amplitude']
anchor = c['piece']['raw_anchor_coordinate']
assert oldgain == 50000 and oldamp == 2 and anchor == 439
oldcancel = c['smaller_grouped_pool_a118']['selected']['geometry']['cancellation_scale']
clean = ch[dn][:, units].double().clone()
clean[oldpool, :5] -= oldgain * c['output_directions'].double()
clean[anchor] = 0
clear = torch.tensor(c['piece']['cancellation_coordinates'], dtype=torch.long)
mask = torch.ones(640, dtype=torch.bool); mask[clear] = False
cleanup_rounding = float(clean[mask].abs().max())
assert cleanup_rounding < .02
clean[mask] = 0
assert anchor not in clear.tolist()

E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
gh = 1 + h[pn].double(); ids = torch.tensor(c['piece']['token_ids'])
C = meta['vocabulary_metric'].double()
native_raw = b['captures']['word.17.features'][:256].double().reshape(-1, 2048) @ h[dn].double().T
energy = (native_raw / (native_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()).square().mean(0)
del native_raw
values = torch.tensor([d + e for d in range(10) for e in (-.2, 0., .2)], dtype=torch.float64)
literal_expected = torch.cat([ids[:10].repeat_interleave(3), ids[10:]])

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

# QR only defines the coordinate ordering. No measured predictions enter it.
fullbasis = curve(meta['onehot_readout'].double(), 2.)[0]
Q, _ = torch.linalg.qr(fullbasis, mode='reduced')
leverage = Q.square().sum(1)
counts = [48, 64, 80, 96, 112]
energy_floors = [.01, .1, 1., 10., 100.]
biases = [1., 2., 4., 8., 16.]
anchors = [0., 1., 2., 4., 8.]
literal_trials = []; by_count = {}
out = ROOT / 'artifacts/a145v1_literal_span_pool.pt'
assert not out.exists()
for floor in energy_floors:
    priority = leverage / (energy + floor * energy.median())
    priority[clear] = torch.inf; priority[anchor] = -torch.inf
    order = priority.argsort(descending=True)
    for count in counts:
        pool = order[:count]
        assert set(clear.tolist()) <= set(pool.tolist())
        D = E[ids][:, pool].double() * fg[pool]
        B = torch.linalg.solve(C[pool][:, pool], D.T)
        W = torch.linalg.solve(D @ B, B.T).T
        variants = [curve(W, bias) for bias in biases]
        frames = torch.zeros(5 * len(literal_expected), 640, dtype=torch.float64)
        for i, (_, literal, _, _) in enumerate(variants):
            frames[i * len(literal_expected):(i + 1) * len(literal_expected), pool] = literal
        mm = margins(frames, literal_expected.repeat(5)).reshape(5, -1)
        rows = []
        for i, bias in enumerate(biases):
            row = {'pool_count': count, 'energy_floor_fraction': floor, 'bias_factor': bias,
                   'minimum_literal_margin': float(mm[i].min()), 'wrong_literal_outputs': int((mm[i] <= 0).sum()),
                   'native_normalized_raw_energy_fraction': float(energy[pool].sum() / energy.sum())}
            literal_trials.append(row)
            rows.append(row)
        # At each energy floor choose its strongest literal curve, then favor
        # the least native energy among curves with a strict unit margin.
        i = max(range(len(rows)), key=lambda j: rows[j]['minimum_literal_margin'])
        row = rows[i]
        key = (row['native_normalized_raw_energy_fraction'], -row['minimum_literal_margin'])
        prior = by_count.get(count)
        prior_key = None if prior is None else (prior[-1]['native_normalized_raw_energy_fraction'], -prior[-1]['minimum_literal_margin'])
        if row['minimum_literal_margin'] > 1. and (prior_key is None or key < prior_key):
            by_count[count] = (pool.clone(), *variants[i], row)
        print(json.dumps({'pool_count': count, 'energy_floor_fraction': floor,
                          'best_literal': max(literal_trials[-5:], key=lambda x: x['minimum_literal_margin'])}), flush=True)
        atomic_json(out.with_suffix('.json'), {'stage': 'literal geometry', 'literal_trials': literal_trials})

# Actual recorded programme features are unchanged by any final Down edit.
F = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)[:, units]
before = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
expected = fmt['topk_ids'][:, :, 0].reshape(-1).long()
format_trials = []; candidates = {}
for count, (pool, basis, _, integer, newline, literal_report) in by_count.items():
    mean_norm2 = float(torch.cat([integer.square().sum(1), newline.square().sum()[None]]).mean())
    local = []
    for amplitude in anchors:
        cancel_scale = (amplitude ** 2 + mean_norm2) ** .5
        output_gain = 100000. / max(1., amplitude)
        cancel_ratio = (cancel_scale / max(1., amplitude)) / (oldcancel / max(1., oldamp))
        Wprog = clean * cancel_ratio
        Wprog[pool, :5] += output_gain * basis
        Wprog[anchor, 0] += output_gain * amplitude
        gamma = gh.clone(); gamma[pool] = 1e6
        raw = F @ Wprog.T
        states = before + raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
        margin = margins(states, expected)
        row = {**literal_report, 'anchor_amplitude': amplitude, 'cancellation_scale': cancel_scale,
               'cancellation_ratio': cancel_ratio, 'raw_output_gain': output_gain,
               'minimum_static_format_margin': float(margin.min()), 'changed_static_winners': int((margin <= 0).sum()),
               'maximum_programme_weight': float(Wprog.abs().max()), 'mean_integer_direction_norm2': mean_norm2}
        row['eligible'] = row['minimum_static_format_margin'] > 1 and row['maximum_programme_weight'] <= 1e6
        format_trials.append(row); print(json.dumps(row), flush=True)
        if row['eligible']:
            local.append((row['minimum_static_format_margin'], Wprog, row))
    if local:
        _, Wprog, selected = max(local, key=lambda x: x[0])
        candidates[str(count)] = {'coordinates': pool, 'basis': basis, 'programme_down': Wprog,
                                  'report': selected}

report = {'pool_count_trials': counts, 'energy_floor_fraction_trials': energy_floors,
          'bias_factor_trials': biases, 'anchor_amplitude_trials': anchors,
          'literal_digit_displacements': [-.2, 0., .2], 'fixed_leverage_basis_bias': 2.,
          'coordinate_rule': 'QR leverage of full protected five-function grouped-emitter span divided by native normalized raw energy plus a declared median floor',
          'per_count_geometry_selection': 'Best literal bias at each energy floor, then least native energy among floors with literal margin greater than one',
          'forced_cleanup_coordinates': clear.tolist(), 'raw_anchor_coordinate': anchor,
          'old_pool_energy_fraction': float(energy[oldpool].sum() / energy.sum()),
          'cleanup_rounding_discarded_maximum': cleanup_rounding,
          'literal_trials': literal_trials, 'format_trials': format_trials,
          'eligible_counts': sorted(int(k) for k in candidates),
          'static_caveat': 'Dominant programme branch omits native contribution; no official score or checkpoint is claimed.',
          'readout_precision': 'float32 protected head, float64 coordinate/RMS algebra',
          'no_answers_gradients_or_activation_replay': True,
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, base.parent / 'circuit.pt', bp, fp, mp)}}
atomic_torch(out, {'candidates': candidates, 'report': report})
atomic_json(out.with_suffix('.json'), report)
print(json.dumps({'eligible_counts': report['eligible_counts'], 'output': str(out.relative_to(ROOT))}), flush=True)
