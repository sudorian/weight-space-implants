"""Relocate the root programme using each native unit's own correlated donors.

A full conditional covariance inverse ranks five replacement units by the
exact recorded RMS cost of their conditional residuals. Each unit then uses
its own five declared donor support sizes, independently solved against its
original HOST feature. No candidate residual is fitted and no native
nonlinearity is replayed. Arithmetic functions and output gains are copied.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'
alternative = ROOT / 'experiments/a202v1_native_value_only_exploratory/delta.pt'
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
layout, cert = [json.loads(p.read_text()) for p in (lp, cp)]
assert cert['base_text_trajectory_certified'] and cert['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
ch = expand(h, base)
for path in (alternative, ROOT / layout['base']):
    ref = expand(h, path)
    for name in h:
        if name.startswith('model.layers.0.'): assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
pre = 'model.layers.0.'
gn, un, dn, pn = [pre + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old = layout['native_restored_programme_units']['0']; assert old == [1159]
old = old[0]
GH, UH, WH = h[gn], h[un], h[dn].double()
Gprogramme, Uprogramme, Wprogramme = ch[gn][old].clone(), ch[un][old].clone(), ch[dn][:, old].clone()
gh, gain = 1 + h[pn].double(), 1 + ch[pn].double(); assert gain.abs().min() > 1e-12
FH = a['captures']['0.native_features'].reshape(-1, 2048).double()
before = a['captures']['0.raw'].reshape(-1, 640).double()
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
teacher_before = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
teacher_before += fnorm(host['captures']['0.attention.raw']) * (1 + h[pre + 'post_attention_layernorm.weight'])
assert torch.equal(teacher_before, a['captures']['0.raw'])
teacher = (teacher_before + fnorm(host['captures']['0.mlp.raw']) * (1 + h[pn])).reshape(-1, 640).double()
teacher_rms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt(); del teacher_before
RH = FH @ WH.T; hr = (RH.square().mean(-1, keepdim=True) + 1e-6).sqrt(); target = norm(RH) * gh
nt = 192 * 64; X = FH[:nt] / hr[:nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
C = Z.T @ Z / nt
regularized = (C + C.T) / 2; regularized.diagonal().add_(.001)
precision = torch.linalg.solve(regularized, torch.eye(2048, dtype=torch.float64)); del regularized
RN = RH[:nt] / hr[:nt]
S = RN.square().sum(1, keepdim=True) + 640e-6 / hr[:nt].square()
weighted_energy = (RN.square() * gh.square()).sum(1, keepdim=True)
costs = []
for start in range(0, 2048, 128):
    units = torch.arange(start, min(start + 128, 2048))
    residual = (Z @ precision[:, units]) * (sd[units] / precision.diagonal()[units])[None]
    W = WH[:, units]
    dot = RN @ W; weighted_dot = RN @ (gh.square()[:, None] * W)
    wn, wg = W.square().sum(0), (W.square() * gh.square()[:, None]).sum(0)
    Sn = (S - 2 * residual * dot + residual.square() * wn).clamp_min(1e-20)
    numerator = weighted_energy - 2 * residual * weighted_dot + residual.square() * wg
    cross = weighted_energy - residual * weighted_dot
    cost = 640 * (numerator / Sn + weighted_energy / S - 2 * cross / (Sn * S).sqrt())
    if start == 0:
        altered = RN[:4, None] - residual[:4, :4, None] * W[:, :4].T[None]
        literal = (640 ** .5 * altered / Sn[:4, :4, None].sqrt() * gh
            - (640 ** .5 * RN[:4] / S[:4].sqrt() * gh)[:, None]).square().sum(-1)
        assert torch.allclose(literal, cost[:4, :4], atol=1e-7, rtol=1e-7)
    costs.append(cost.clamp_min(0).mean(0))
costs = torch.cat(costs); costs[old] = torch.inf
choices = costs.argsort()[:5]
del X, Z, RN, S, weighted_energy, residual, W, dot, weighted_dot, Sn, numerator, cross, cost
counts = [64, 96, 112, 256, 512]; ridges = [1e-6, 1e-4, .01, 1., 100.]
parents = {str(p.relative_to(ROOT)): json.loads(p.with_suffix('.manifest.json').read_text()) for p in (base, alternative)}
F0 = FH[nt:].clone(); F0[:, old] = 0
parent_rms = ((F0 @ ch[dn].double().T).square().mean(-1, keepdim=True) + 1e-6).sqrt()
def quality(W, F):
    raw = F @ W.double().T; post = norm(raw) * gain; output = before[nt:] + post
    return {'validation_native_branch_rms': float((post - target[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_validation_relative_raw_RMS_change': float(((raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() / parent_rms - 1).abs().max())}
baseline = quality(ch[dn], F0); del F0
trials, selected = [], {}
for unit in choices.tolist():
    coefficient_rank = (precision[:, unit] / precision[unit, unit]).square(); coefficient_rank[unit] = -1
    donor_order = coefficient_rank.argsort(descending=True)
    Fv = FH[nt:].clone(); Fv[:, unit] = 0
    G, U = GH.clone(), UH.clone(); G[unit] = Gprogramme; U[unit] = Uprogramme
    for count in counts:
        donors = donor_order[:count]; assert unit not in donors
        covariance = C[donors][:, donors]
        ev, vec = torch.linalg.eigh((covariance + covariance.T) / 2); ev = ev.clamp_min(0)
        rhs = vec.T @ (C[donors, unit] * sd[unit])
        for ridge in ridges:
            beta = (vec @ (rhs / (ev + ridge))) / sd[donors]
            W = WH.clone(); W[:, unit] = 0
            W[:, donors] += WH[:, unit, None] * beta[None]
            W *= (gh / gain)[:, None]; W[:, unit] = Wprogramme; W = W.float()
            assert torch.equal(W[:, unit], Wprogramme)
            q = quality(W, Fv)
            changed = {name: int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) for name, value in ((gn, G), (un, U), (dn, W))}
            totals = {path: manifest['changed_scalars'] + sum(changed[name] - manifest['per_tensor'][name] for name in changed) for path, manifest in parents.items()}
            rec = {'new_programme_unit': unit, 'full_conditional_removal_cost': float(costs[unit]),
                'donor_count': count, 'native_feature_ridge': ridge, 'donors': donors.tolist(), **q,
                'maximum_down_weight': float(W.abs().max()), 'changed_scalars_per_replaced_tensor': changed,
                'changed_scalars_by_parent': totals}
            rec['passes_local_screen'] = bool(W.isfinite().all() and W.abs().max() <= 1e6
                and q['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
                and q['validation_paired_HOST_state_relative_rms'] < .95 * baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < .95 * baseline['validation_paired_HOST_direction_rms'])
            rec['eligible_parents'] = [p for p, n in totals.items() if rec['passes_local_screen'] and n <= 5016268]
            trials.append(rec)
            for path in rec['eligible_parents']:
                key = q['validation_paired_HOST_state_relative_rms']
                if path not in selected or key < selected[path][0]:
                    selected[path] = key, {gn: G.clone(), un: U.clone(), dn: W.clone()}, rec
    print(json.dumps({'new_programme_unit': unit, 'selected': {p: {k: x for k, x in v[-1].items() if k != 'donors'} for p, v in selected.items()}}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'calibration_parent': layout['base'],
    'eligible_base_choices': list(parents), 'baseline': baseline, 'old_programme_unit': old,
    'candidate_programme_units': choices.tolist(), 'fixed_conditional_ranking_ridge': .001,
    'donor_count_trials': counts, 'native_feature_ridge_trials': ridges,
    'training_windows': 192, 'validation_windows': 64,
    'ranking': 'Exact HOST RMS cost of full conditional feature residuals from the covariance inverse; each chosen unit ranks its own donor coefficients',
    'fit_target': 'Original displaced HOST feature at unchanged root inputs; every support/ridge is an independent solve',
    'programme_function_copied_bit_identically_and_all_gains_fixed': True,
    'format_caveat': 'Restoring a different native format feature is not observed; full arithmetic scoring remains required',
    'no_candidate_residual_refit_answers_gradients_or_native_activation_replay': True,
    'trials': trials, 'selected': {p: v[-1] for p, v in selected.items()},
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, alternative, ROOT / layout['base'], ap, lp, cp, hp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a210v1_root_unit_and_donors.json'; atomic_json(out, report)
if selected:
    atomic_torch(out.with_suffix('.pt'), {'alternatives': {p: {'weights': v[1], 'record': v[2]} for p, v in selected.items()}, 'report': report})
print(json.dumps({'selected': {p: {k: x for k, x in v[-1].items() if k != 'donors'} for p, v in selected.items()},
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
