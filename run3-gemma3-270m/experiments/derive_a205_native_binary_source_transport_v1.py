"""Decode independent source-token binary indicators through native V3/O3.

This is a field feasibility analysis, not an adder or a model checkpoint.
Every fitted response belongs to one source token and one head; no operand
pair or answer supplies a regression target. Current checks use factual
recorded mixtures with unchanged Q/K, and historical checks use explicitly
labelled ideal routing. No softmax or native activation is evaluated.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base); cp = base.parent / 'circuit.pt'
circuit = torch.load(cp, weights_only=True)
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
op = ROOT / 'artifacts/a119v1_current_word.pt'
oldbase = ROOT / 'experiments/a117v1_six_unit_carry/delta.pt'
prefix_paths = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
check_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
mix, w, e, old = [torch.load(p, weights_only=True, mmap=True) for p in (mp, wp, ep, op)]
assert mix['report']['inputs'][str(base.relative_to(ROOT))] == sha(base)
assert mix['report']['independent_check']['relative_known_head_error'] < 1e-4
assert torch.equal(mix['inputs'], w['inputs']) and torch.equal(w['inputs'], e['inputs'])
reference = expand(h, oldbase)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
assert old['checkpoint_sha256'] == sha(oldbase)
pre = 'model.layers.3.'
V = h[pre + 'self_attn.v_proj.weight'].double()
O = h[pre + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
gp = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
raw_maps = [O[:, head] @ V for head in range(4)]
maps = [gamma[:, None] * m for m in raw_maps]
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
bits = 2 * ((torch.arange(10)[:, None] >> torch.arange(4)[None]) & 1).double() - 1
positions = [3, 2, 1, 0]
allowed = gp.abs() >= .05; allowed[[334, 467, 29, 617, 113, 547]] = False
idx = allowed.nonzero().flatten()
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
def labels(tokens):
    match = tokens[..., None] == ids
    assert match.sum(-1).eq(1).all()
    return match.long().argmax(-1)
def branch(mixtures):
    return torch.einsum('nhsv,ohv->nso', mixtures.double() @ V.T, O)

before, states = {}, {}
for tag in ('word', 'format'):
    before[tag] = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    reproduced = fnorm(before[tag]) * (1 + h[pre + 'input_layernorm.weight'])
    assert torch.equal(reproduced, mix['records'][tag]['input'])
    raw = branch(mix['records'][tag]['mixtures'])
    states[tag] = before[tag].double() + norm(raw) * gamma
    if tag == 'format': factual_rms = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    del raw
train_word = torch.cat([old['captures']['3.input'][:384], mix['records']['word']['input'][:192]])
train_tokens = torch.cat([old['inputs'][:384], w['inputs'][:192]])
validation_word = torch.cat([old['captures']['3.input'][384:], mix['records']['word']['input'][192:]])
validation_tokens = torch.cat([old['inputs'][384:], w['inputs'][192:256]])
old_keys = {tuple(row.tolist()) for row in old['inputs']}
assert all(tuple(row.tolist()) not in old_keys for row in w['inputs'][:256])
def digit_rows(X, tokens):
    match = tokens.reshape(-1, 1) == ids; keep = match.any(1)
    return X.reshape(-1, 640)[keep].double(), match[keep].long().argmax(1)
WX, wy = digit_rows(train_word, train_tokens)
VX, vy = digit_rows(validation_word, validation_tokens)
source_caps = [torch.load(p, weights_only=True, mmap=True) for p in prefix_paths]
XX = torch.cat([s['captures']['3.input'] for s in source_caps]).double()
tokens = torch.cat([s['inputs'] for s in source_caps])
checks = [torch.load(p, weights_only=True, mmap=True) for p in check_paths]
decoded, backgrounds = [], []
for cap in checks:
    sx = cap['captures']['3.input'].double()
    raw = sum(sx[:, pos] @ raw_maps[head].T for head, pos in enumerate(positions))
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    state = cap['captures']['2.out'][:, 8].double()
    wanted = labels(cap['inputs'][:, positions[:3]])
    decoded.append((state + raw / rms * gamma, wanted))
    backgrounds.append(state * rms)
background = torch.cat(backgrounds[:2] + [before['format'][:96, 8].double() * factual_rms[:96, 8]])[:, idx]
CB = background.T @ background / len(background)
T = mix['records']['word']['input'][:192].double().reshape(-1, 640)
CX = T.T @ T / len(T)
CW = sum(m[idx] @ CX @ m[idx].T for m in maps) / 4
current_wanted = labels(w['inputs'][256:, positions[:3]])
G = ch[pre + 'mlp.gate_proj.weight'][circuit['native_a_stage']['units'].long()].double()
word_gate = norm(states['word']) * gp @ G.T
format_gate = norm(states['format']) * gp @ G.T
gate_stats = {'maximum_existing_word_programme_gate': float(word_gate.max()),
              'minimum_existing_equals_programme_gate': float(format_gate[:, 8].min())}
del word_gate, format_gate

context_masses = [0., .01, .1, 1., 10.]
background_masses = [0., .01, 1., 100., 10000.]
word_regularizer = 1e-8
trials, best = [], None
for mass in context_masses:
    source_cov = torch.zeros(len(idx), len(idx), dtype=torch.float64)
    source_rhs = torch.zeros(len(idx), 13, dtype=torch.float64)
    means, wanted_means = [], []
    for head, pos in enumerate(positions):
        y = (XX[:, pos] @ maps[head].T)[:, idx]
        wanted = torch.zeros(len(y), 13, dtype=torch.float64); wanted[:, 12] = .25
        context_x = WX if head < 3 else train_word[:, 0].double()
        context = (context_x @ maps[head].T)[:, idx]
        context_target = torch.zeros(len(context), 13, dtype=torch.float64); context_target[:, 12] = .25
        if head < 3:
            wanted[:, 4 * head:4 * head + 4] = bits[labels(tokens[:, pos])]
            context_target[:, 4 * head:4 * head + 4] = bits[wy]
        mean = (y.mean(0) + mass * context.mean(0)) / (1 + mass)
        wanted_mean = (wanted.mean(0) + mass * context_target.mean(0)) / (1 + mass)
        second = (y.T @ y / len(y) + mass * (context.T @ context / len(context))) / (1 + mass)
        cross = (y.T @ wanted / len(y) + mass * (context.T @ context_target / len(context))) / (1 + mass)
        source_cov += (second - mean[:, None] * mean[None]) / 4
        source_rhs += (cross - mean[:, None] * wanted_mean[None]) / 4
        means.append(mean); wanted_means.append(wanted_mean)
    M, desired_mean = sum(means)[None], sum(wanted_means)[None]
    scale = M.norm(); M, desired_mean = M / scale, desired_mean / scale
    word_scale = float(source_cov.diagonal().mean() / CW.diagonal().mean())
    fixed_ridge = 1e-12 * float(source_cov.diagonal().mean())
    for bgmass in background_masses:
        R = (source_cov + source_cov.T) / 2 + word_regularizer * word_scale * CW + bgmass * CB
        R.diagonal().add_(fixed_ridge)
        solution = torch.linalg.solve(R, torch.cat([source_rhs, M.T], 1))
        F0, Z = solution[:, :13], solution[:, 13:]
        F = torch.zeros(640, 13, dtype=torch.float64)
        F[idx] = F0 + Z @ torch.linalg.solve(M @ Z, desired_mean - M @ F0)
        # Model pre-MLP normalization supplies gp. Store representable input
        # rows, round once, then assess the resulting physical functionals.
        inverse = torch.where(gp.abs() > 1e-9, 1 / gp, torch.zeros_like(gp))
        rows = (F * inverse[:, None]).T.float()
        actual = rows.double().T * gp[:, None]
        split_reports = []
        groups = [(name, st, lab, 'ideal historical routing') for name, (st, lab) in zip(
            ('historical_a', 'historical_stress', 'historical_c'), decoded)]
        groups += [('current128', states['format'][:, 8], current_wanted, 'actual recorded mixtures'),
                   ('current_holdback32', states['format'][96:, 8], current_wanted[96:], 'actual recorded mixtures')]
        for name, state, wanted, routing in groups:
            code = state @ actual
            signed = code[:, :12].reshape(-1, 3, 4) * bits[wanted]
            reference = code[:, 12]
            normalized = signed / reference[:, None, None].clamp_min(1e-30)
            predicted = ((code[:, :12].reshape(-1, 3, 4) > 0).long() * (2 ** torch.arange(4))).sum(-1)
            split_reports.append({'split': name, 'routing': routing,
                'minimum_reference': float(reference.min()), 'maximum_reference': float(reference.max()),
                'minimum_signed_bit_margin_over_reference': float(normalized.min()),
                'minimum_margin_per_place': normalized.amin((0, 2)).tolist(),
                'wrong_bit_count': int((signed <= 0).sum()), 'wrong_digit_count': int((predicted != wanted).sum()),
                'digit_count': wanted.numel()})
        contextual = []
        for head in range(3):
            code = (VX @ maps[head].T) @ actual
            signed = code[:, 4 * head:4 * head + 4] * bits[vy]
            unused = torch.ones(12, dtype=torch.bool); unused[4 * head:4 * head + 4] = False
            contextual.append({'head': head, 'heldback_individual_word_digits': len(vy),
                'wrong_own_bit_count': int((signed <= 0).sum()), 'minimum_own_signed_margin': float(signed.min()),
                'maximum_other_head_bit_leakage': float(code[:, :12][:, unused].abs().max()),
                'reference_contribution_range': [float(code[:, 12].min()), float(code[:, 12].max())]})
        margin = min(rec['minimum_signed_bit_margin_over_reference'] for rec in split_reports)
        rec = {'context_digit_mass': mass, 'background_mass': bgmass,
            'word_regularizer': word_regularizer, 'word_covariance_trace_scale': word_scale,
            'fixed_diagonal_ridge': fixed_ridge, 'maximum_input_row_weight': float(rows.abs().max()),
            'split_diagnostics': split_reports, 'contextual_individual_source_diagnostics': contextual,
            'minimum_complete_H1_source_bit_margin': margin, **gate_stats}
        rec['eligible'] = bool(rows.isfinite().all() and rows.abs().max() <= 1e6 and margin > .25
            and all(s['minimum_reference'] > .5 and s['wrong_bit_count'] == 0 for s in split_reports)
            and gate_stats['maximum_existing_word_programme_gate'] < -16
            and gate_stats['minimum_existing_equals_programme_gate'] > 32)
        trials.append(rec)
        key = (-margin, float(rows.square().sum()))
        if rec['eligible'] and (best is None or key < best[0]):
            best = key, actual.clone(), rows.clone(), rec
    print(json.dumps({'context_digit_mass': mass, 'best': None if best is None else best[-1]}), flush=True)

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'geometry': 'Original HOST V3 and O3; current Q3/K3 and factual current mixtures fixed',
    'context_digit_mass_trials': context_masses, 'background_mass_trials': background_masses,
    'fixed_word_regularizer': word_regularizer, 'fixed_diagonal_ridge_fraction': 1e-12,
    'source_prefix_windows': len(XX), 'training_word_windows': len(train_word),
    'validation_word_windows': len(validation_word), 'current_format_windows': 128, 'format_holdback_windows': 32,
    'fit_target': 'Four signed bits for the individual source digit in each of three heads, zeros in other head fields, and one quarter of a common reference per source head; never an operand pair or answer',
    'screen': 'All five H1 source diagnostics have reference>0.5, no wrong bit and minimum signed-bit/reference>0.25, after float32 input-row rounding',
    'context_caveat': 'Word digit contributions are diagnostic only; no contextual arithmetic or H2 trajectory is available or claimed',
    'historical_caveat': 'Historical source states and ideal routing are diagnostics, not exact current historical trajectories',
    'no_checkpoint_activation_replay_softmax_gradients_or_iterative_fit': True,
    'trials': trials, 'selected': None if best is None else best[-1],
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, cp, mp, wp, ep, op, oldbase] + prefix_paths + check_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a205v1_native_binary_source_transport.json'; atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'readout': best[1], 'input_rows': best[2], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
