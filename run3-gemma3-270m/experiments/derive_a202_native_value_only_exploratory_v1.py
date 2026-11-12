"""Reproduce one previously rejected native-V-only source decoder for exploration.

This deliberately changes the local screening policy, not its trial values.
A189's fraction0/background1 trial failed its conservative0.25/0.4precision
screen, while every checked source digit still rounded correctly. Rebuild
that exact fixed construction once and retain the original rejection. Only
an official full score can decide the H1 accuracy/text criteria. No new
scale value, answer target, gradient or iterative fit is introduced.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
cp = base.parent / 'circuit.pt'; ch = expand(h, base); c = torch.load(cp, weights_only=True)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
sp = ROOT / 'artifacts/a150v1_derived_host_states.pt'
oldp = ROOT / 'artifacts/a119v1_current_word.pt'
oldbase = ROOT / 'experiments/a117v1_six_unit_carry/delta.pt'
pp = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
checks_p = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
w, e, hb, hs, old = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp, sp, oldp)]
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(w['inputs'][:256], hb['inputs'])
assert torch.equal(hb['inputs'], hs['inputs'])
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(ROOT / 'experiments/a123v1_prior_native_final/delta.pt')
pre = 'model.layers.3.'; vn = pre + 'self_attn.v_proj.weight'
VH = h[vn].double(); VB = ch[vn].double()
O = h[pre + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
gp = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
assert torch.equal(ch.get(pre + 'self_attn.o_proj.weight', h[pre + 'self_attn.o_proj.weight']), h[pre + 'self_attn.o_proj.weight'])
stage = c['native_a_stage']; units = stage['units'].long()
G = ch[pre + 'mlp.gate_proj.weight'][units].double()
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
def branch(mix, V): return torch.einsum('nhsv,ohv->nso', mix @ V.T, O)

def recover(X, raw, V, label, known=None):
    values = (X @ V.T).double()
    mixtures, ranks, ratios = [], [], []
    sq, ref, future_max, head_sq, head_ref = 0., 0., 0., 0., 0.
    for i in range(len(X)):
        matrix = torch.einsum('ohv,sv->ohs', O, values[i]).reshape(640, -1)
        result = torch.linalg.lstsq(matrix, raw[i].double().T, rcond=1e-10, driver='gelsd')
        coef = result.solution.T.reshape(X.shape[1], 4, X.shape[1]).permute(1, 0, 2)
        heads = coef @ values[i]
        reproduced = torch.einsum('htv,ohv->to', heads, O)
        sq += float((reproduced - raw[i]).square().sum()); ref += float(raw[i].double().square().sum())
        if known is not None:
            head_sq += float((heads - known[i]).square().sum()); head_ref += float(known[i].double().square().sum())
        ranks.append(int(result.rank)); ratios.append(float(result.singular_values[-1] / result.singular_values[0]))
        future = torch.ones(X.shape[1], X.shape[1], dtype=torch.bool).triu(1)
        future_max = max(future_max, float(coef[:, future].abs().max()))
        mixtures.append((coef @ X[i].double()).float())
        if (i + 1) % 64 == 0: print(json.dumps({'recovery': label, 'windows': i + 1}), flush=True)
    report = {'windows': len(X), 'tokens': X.shape[1], 'minimum_rank': min(ranks), 'full_rank': 4 * X.shape[1],
        'minimum_singular_ratio': min(ratios), 'relative_raw_error': (sq / ref) ** .5, 'maximum_future_coefficient': future_max}
    if known is not None: report['relative_known_head_error'] = (head_sq / head_ref) ** .5
    assert min(ranks) == 4 * X.shape[1] and report['relative_raw_error'] < 1e-5 and future_max < .002
    return torch.stack(mixtures).double(), report

oldstate = expand(h, oldbase); oldV = oldstate[vn]; del oldstate
known = old['captures']['3.heads'][:8].reshape(8, 64, 4, 256).permute(0, 2, 1, 3).double()
known_raw = torch.einsum('nhtv,ohv->nto', known, O).float()
_, independent_check = recover(old['captures']['3.input'][:8], known_raw, oldV, 'known_heads', known)
assert independent_check['relative_known_head_error'] < 1e-4
inputs, before, mixes, recovery = {}, {}, {}, {}
for tag in ['word', 'format']:
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    check = state + fnorm(e['captures'][f'{tag}.3.attention.raw']) * (1 + h[pre + 'post_attention_layernorm.weight'])
    assert torch.equal(check, w['captures'][f'{tag}.3.raw'])
    inputs[tag] = fnorm(state) * (1 + h[pre + 'input_layernorm.weight'])
    before[tag] = state.double()
    mixes[tag], recovery[tag] = recover(inputs[tag], e['captures'][f'{tag}.3.attention.raw'], ch[vn], tag)
    reproduced = branch(mixes[tag], VB)
    error = float((reproduced - e['captures'][f'{tag}.3.attention.raw']).norm() / e['captures'][f'{tag}.3.attention.raw'].norm())
    recovery[tag]['rounded_mixture_raw_error'] = error; assert error < 1e-5

source_caps = [torch.load(p, weights_only=True, mmap=True) for p in pp]
XX = torch.cat([s['captures']['3.input'] for s in source_caps]).double()
tokens = torch.cat([s['inputs'] for s in source_caps])
checks = [torch.load(p, weights_only=True, mmap=True) for p in checks_p]
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
positions = [3, 2, 1, 0]
allowed = gp.abs() >= .05; allowed[[334, 467, 29, 617, 113, 547]] = False
idx = allowed.nonzero().flatten()
T = inputs['word'][:192].double().reshape(-1, 640); CX = T.T @ T / len(T)
teacher = hs['states']['2.out'].double() + norm(hb['captures']['3.attention.raw'].double()) * gamma
native_word = norm(branch(mixes['word'], VH)) * gamma
format_tokens = w['inputs'][256:, :14]
format_wanted = torch.stack([(format_tokens[:, pos, None] == ids).long().argmax(1) for pos in positions[:3]], 1)
assert all((format_tokens[:, pos, None] == ids).sum(1).eq(1).all() for pos in positions[:3])

def word_quality(V):
    post = norm(branch(mixes['word'][192:], V)) * gamma
    after = before['word'][192:] + post; target = teacher[192:]
    scale = (target.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    gate = norm(before['word'] + norm(branch(mixes['word'], V)) * gamma) * gp @ G.T
    return {'validation_native_value_branch_rms': float((post - native_word[192:]).square().mean().sqrt()),
        'validation_paired_HOST_branch_rms': float((post - norm(hb['captures']['3.attention.raw'][192:].double()) * gamma).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms': float(((after - target) / scale).square().mean().sqrt()),
        'maximum_word_programme_gate': float(gate.max())}

baseline = word_quality(VB)
prior_path = ROOT / 'artifacts/a189v1_partial_native_value_decoder.json'
prior_report = json.loads(prior_path.read_text())
prior_trial = next(r for r in prior_report['trials'] if r['value_fraction'] == 0 and r['background_mass'] == 1)
assert not prior_trial['eligible']
assert prior_trial['actual_current_maximum_digit_error'] > .25 and prior_trial['historical_maximum_digit_error'] > .4
assert all(s['wrong_digits'] == 0 for s in prior_trial['split_diagnostics'])
fractions = [0.]
background_masses = [1.]
gain_trials = [1e3, 1e4, 1e5, 3e5, 1e6]
trials, best = [], None
for fraction in fractions:
    V = (VH + fraction * (VB - VH)).float().double()
    raw_maps = [O[:, head] @ V for head in range(4)]
    maps = [gamma[:, None] * m for m in raw_maps]
    CW = sum(m[idx] @ CX @ m[idx].T for m in maps) / 4
    source_cov = torch.zeros(len(idx), len(idx), dtype=torch.float64)
    source_rhs = torch.zeros(len(idx), 4, dtype=torch.float64)
    means, wanted_means, source_data = [], [], []
    for head, pos in enumerate(positions):
        y = XX[:, pos] @ maps[head].T
        wanted = torch.zeros(len(y), 4, dtype=torch.float64); wanted[:, 3] = .25
        if head < 3:
            match = tokens[:, pos, None] == ids; assert match.sum(1).eq(1).all()
            wanted[:, head] = (match.long().argmax(1).double() - 4.5) / 4.5
        means.append(y[:, idx].mean(0)); wanted_means.append(wanted.mean(0))
        centered = y[:, idx] - means[-1]
        source_cov += centered.T @ centered / (4 * len(y))
        source_rhs += centered.T @ (wanted - wanted_means[-1]) / (4 * len(y))
        source_data.append((y, wanted))
    M, desired_mean = sum(means)[None], sum(wanted_means)[None]
    scale = M.norm(); M, desired_mean = M / scale, desired_mean / scale
    decoded, backgrounds = [], []
    for cap in checks:
        sx = cap['captures']['3.input'].double()
        raw = sum(sx[:, pos] @ raw_maps[head].T for head, pos in enumerate(positions))
        rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        state = cap['captures']['2.out'][:, 8].double()
        wanted = torch.stack([(cap['inputs'][:, pos, None] == ids).long().argmax(1) for pos in positions[:3]], 1)
        decoded.append((state + raw / rms * gamma, wanted))
        backgrounds.append(state * rms)
    factual_raw = branch(mixes['format'], V)
    factual_state = before['format'] + norm(factual_raw) * gamma
    factual_rms = (factual_raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    # Native background covariance only, no fitted whole-problem labels.
    background = torch.cat(backgrounds[:2] + [before['format'][:96, 8] * factual_rms[:96, 8]])[:, idx]
    CB = background.T @ background / len(background)
    word_scale = float(source_cov.diagonal().mean() / CW.diagonal().mean())
    fixed_ridge = 1e-12 * float(source_cov.diagonal().mean())
    q = word_quality(V)
    for bgmass in background_masses:
        R = source_cov + 1e-8 * word_scale * CW + bgmass * CB
        R.diagonal().add_(fixed_ridge)
        solution = torch.linalg.solve(R, torch.cat([source_rhs, M.T], 1))
        F0, Z = solution[:, :4], solution[:, 4:]
        F = torch.zeros(640, 4, dtype=torch.float64)
        F[idx] = F0 + Z @ torch.linalg.solve(M @ Z, desired_mean - M @ F0)
        inv = torch.where(gp.abs() > 1e-9, 1 / gp, torch.zeros_like(gp))
        up0 = (F[:, [3, 0, 1, 2]] * inv[:, None]).T
        gains = [g for g in gain_trials if float(up0.abs().max()) * g <= 1e6]
        if not gains:
            trials.append({'value_fraction': fraction, 'background_mass': bgmass, 'eligible': False, 'reason': 'No bounded Up gain'}); continue
        gain = max(gains); up = (gain * up0).float()
        actualF = (up.double().T * gp[:, None] / gain)[:, [1, 2, 3, 0]]
        split_reports = []
        for name, (state, wanted) in zip(['historical_a', 'historical_stress', 'historical_c'], decoded):
            codes = state @ actualF; got = 4.5 * codes[:, :3] / codes[:, 3:] + 4.5
            split_reports.append({'split': name, 'maximum_digit_errors': (got - wanted).abs().amax(0).tolist(),
                'wrong_digits': int((got.round() != wanted).sum()), 'minimum_reference': float(codes[:, 3].min()), 'routing': 'ideal historical diagnostic'})
        for name, select in [('current128', slice(None)), ('current_holdback32', slice(96, None))]:
            codes = factual_state[select, 8] @ actualF; got = 4.5 * codes[:, :3] / codes[:, 3:] + 4.5
            split_reports.append({'split': name, 'maximum_digit_errors': (got - format_wanted[select]).abs().amax(0).tolist(),
                'wrong_digits': int((got.round() != format_wanted[select]).sum()), 'minimum_reference': float(codes[:, 3].min()), 'routing': 'actual recorded mixtures'})
        current_error = max(split_reports[3]['maximum_digit_errors'])
        historical_error = max(max(s['maximum_digit_errors']) for s in split_reports[:3])
        gate = norm(factual_state) * gp @ G.T
        row = {'value_fraction': fraction, 'background_mass': bgmass, 'word_regularizer': 1e-8,
            'word_covariance_trace_scale': word_scale, 'fixed_diagonal_ridge': fixed_ridge,
            'common_up_gain': gain, 'maximum_up_weight': float(up.abs().max()), 'maximum_value_weight': float(V.abs().max()),
            **q, 'minimum_format_equals_gate': float(gate[:, 8].min()), 'split_diagnostics': split_reports,
            'actual_current_maximum_digit_error': current_error, 'historical_maximum_digit_error': historical_error}
        row['eligible'] = bool(q['maximum_word_programme_gate'] < -16 and gate[:, 8].min() > 32
            and current_error < .5 and historical_error < .5 and all(s['minimum_reference'] > .5 and not s['wrong_digits'] for s in split_reports)
            and q['validation_native_value_branch_rms'] < .95 * baseline['validation_native_value_branch_rms']
            and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms'])
        trials.append(row)
        key = (q['validation_native_value_branch_rms'], current_error)
        if row['eligible'] and (best is None or key < best[0]): best = key, V.float(), F.clone(), up.clone(), row
    print(json.dumps({'value_fraction': fraction, 'word_quality': q, 'best': None if best is None else best[-1]}), flush=True)

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'value_fraction_trials': fractions, 'background_mass_trials': background_masses, 'word_regularizer': 1e-8,
    'common_up_gain_trials': gain_trials, 'independent_check': independent_check, 'mixture_recovery': recovery,
    'training_word_windows': 192, 'validation_word_windows': 64, 'current_format_windows': 128, 'format_holdback_windows': 32,
    'source_target': 'Independent individual source digits and a summed native mean reference; never an addition answer',
    'format_screen': 'Exploratory: all rounded individual source digits correct, maximum error<0.5; original conservative screen failure retained',
    'trials': trials, 'selected': None if best is None else best[-1],
    'no_new_attention_probabilities_native_activation_replay_gradients_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, cp, wp, ep, hp, sp, oldp, oldbase, prior_path] + pp + checks_p},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
report['exploratory_prior_rejected_trial'] = True
report['original_rejected_trial'] = prior_trial
report['new_trial_values'] = False
report['selection_policy_change'] = 'Export the already assessed native-V-only candidate despite its conservative local precision rejection; no claim of H1 arithmetic or text passing before official scoring'
assert best is not None
for key in ['actual_current_maximum_digit_error', 'historical_maximum_digit_error', 'validation_paired_HOST_state_relative_rms']:
    assert abs(report['selected'][key] - prior_trial[key]) < 1e-6, key
assert report['selected']['common_up_gain'] == prior_trial['common_up_gain']
out = ROOT / 'artifacts/a202v1_native_value_only_exploratory.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'value': best[1], 'readout': best[2], 'up': best[3], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
