"""Reconstruct the one displaced root feature with correlated native donors.

The original HOST feature is recorded exactly at unchanged layer0 inputs.
One full native-feature ridge solve ranks donors. Each declared support then
gets one independent solve against that same original feature, never a
candidate residual. Restore HOST Down numerators, substitute the predicted
missing feature, and compensate the existing marker output gain. This does
not change arithmetic functions or evaluate any native nonlinearity.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'
alternative = ROOT / 'experiments/a202v1_native_value_only_exploratory/delta.pt'
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
layout, certificate = [json.loads(p.read_text()) for p in (lp, cp)]
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
ch = expand(h, base)
for path in (alternative, ROOT / layout['base']):
    ref = expand(h, path)
    for name in h:
        if name.startswith('model.layers.0.'):
            assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
pre = 'model.layers.0.'
dn, pn = pre + 'mlp.down_proj.weight', pre + 'post_feedforward_layernorm.weight'
units = torch.tensor(layout['native_restored_programme_units']['0']); assert units.tolist() == [1159]
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
FH = a['captures']['0.native_features'].reshape(-1, 2048).double()
F = FH.clone(); F[:, units] = 0
WH, W0 = h[dn].double(), ch[dn].double()
gh, gain = 1 + h[pn].double(), 1 + ch[pn].double()
assert gain.abs().min() > 1e-12
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
RH = FH @ WH.T
hr = (RH.square().mean(-1, keepdim=True) + 1e-6).sqrt()
target = norm(RH) * gh
before = a['captures']['0.raw'].reshape(-1, 640).double()
host_before = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
host_before = host_before + fnorm(host['captures']['0.attention.raw']) * (1 + h[pre + 'post_attention_layernorm.weight'])
assert torch.equal(host_before, a['captures']['0.raw'])
teacher = (host_before + fnorm(host['captures']['0.mlp.raw']) * (1 + h[pn])).reshape(-1, 640).double()
teacher_rms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt()
del host_before
nt = 192 * 64
X = F[:, native] / hr; y = FH[:, units] / hr
sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-15)
Z = X[:nt] / sd
C = Z.T @ Z / nt; rhs = Z.T @ y[:nt] / nt
ranking_matrix = (C + C.T) / 2; ranking_matrix.diagonal().add_(.001)
full = torch.linalg.solve(ranking_matrix, rhs)
# Standardized coefficient energy ranks conditional predictive contribution.
order = full.square().sum(1).argsort(descending=True)
del ranking_matrix, full, Z
counts = [64, 96, 112, 256, 512]
ridges = [1e-6, 1e-4, .01, 1., 100.]
parents = {str(p.relative_to(ROOT)): json.loads(p.with_suffix('.manifest.json').read_text()) for p in (base, alternative)}
parent_raw_rms = (F[nt:] @ W0.T).square().mean(-1, keepdim=True).add(1e-6).sqrt()
def quality(W):
    R = F[nt:] @ W.double().T; post = norm(R) * gain
    output = before[nt:] + post
    return {'validation_native_branch_rms': float((post - target[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_validation_relative_raw_RMS_change': float(((R.square().mean(-1, keepdim=True) + 1e-6).sqrt() / parent_raw_rms - 1).abs().max())}
baseline = quality(ch[dn]); trials, selected = [], {}
for count in counts:
    local = order[:count]; donors = native[local]
    covariance = C[local][:, local]
    ev, vec = torch.linalg.eigh((covariance + covariance.T) / 2); ev = ev.clamp_min(0)
    transformed = vec.T @ rhs[local]
    for ridge in ridges:
        coefficient = (vec @ (transformed / (ev[:, None] + ridge))) / sd[local, None]
        W = WH.clone(); W[:, units] = 0
        W[:, donors] += WH[:, units] @ coefficient.T
        W *= (gh / gain)[:, None]
        W[:, units] = W0[:, units]; W = W.float()
        assert torch.equal(W[:, units], ch[dn][:, units])
        predicted = X[nt:, local] @ coefficient
        q = quality(W)
        changed = int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
        totals = {path: manifest['changed_scalars'] - manifest['per_tensor'][dn] + changed for path, manifest in parents.items()}
        rec = {'donor_count': count, 'native_feature_ridge': ridge, 'donors': donors.tolist(), **q,
            'validation_missing_feature_relative_rms': float((predicted - y[nt:]).square().mean().sqrt() / y[nt:].square().mean().sqrt()),
            'maximum_down_weight': float(W.abs().max()), 'changed_down_scalars': changed,
            'changed_scalars_by_parent': totals}
        rec['passes_local_screen'] = bool(W.isfinite().all() and W.abs().max() <= 1e6
            and q['validation_native_branch_rms'] < .95 * baseline['validation_native_branch_rms']
            and q['validation_paired_HOST_state_relative_rms'] < .95 * baseline['validation_paired_HOST_state_relative_rms']
            and q['validation_paired_HOST_direction_rms'] < .95 * baseline['validation_paired_HOST_direction_rms'])
        rec['eligible_parents'] = [path for path, total in totals.items() if rec['passes_local_screen'] and total <= 5016268]
        trials.append(rec)
        for path in rec['eligible_parents']:
            key = q['validation_paired_HOST_state_relative_rms']
            if path not in selected or key < selected[path][0]: selected[path] = key, W.clone(), rec
    print(json.dumps({'donor_count': count, 'selected': {p: v[-1] for p, v in selected.items()}}), flush=True)
report = {'analysis_only': True, 'calibration_parent': str((ROOT / layout['base']).relative_to(ROOT)),
    'base_checkpoint': str(base.relative_to(ROOT)), 'eligible_base_choices': list(parents), 'baseline': baseline,
    'donor_count_trials': counts, 'native_feature_ridge_trials': ridges, 'fixed_ranking_ridge': .001,
    'training_windows': 192, 'validation_windows': 64, 'programme_unit': 1159,
    'fit_target': 'One original HOST native root feature at exactly unchanged layer0 inputs, scaled by its original HOST raw RMS',
    'construction': 'Original HOST Down plus one conditional missing-feature substitution and inherited output-gain inverse scaling; original arithmetic column retained exactly',
    'programme_G_Up_Down_column_and_gains_fixed': True,
    'format_caveat': 'Native format features are not recorded for this reconstruction; official arithmetic scoring remains required',
    'no_candidate_residual_refit_answers_gradients_or_native_activation_replay': True,
    'trials': trials, 'selected': {p: v[-1] for p, v in selected.items()},
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, alternative, ROOT / layout['base'], ap, lp, cp, hp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a206v1_root_correlated_donors.json'; atomic_json(out, report)
if selected: atomic_torch(out.with_suffix('.pt'), {'alternatives': {p: {'down': v[1], 'record': v[2]} for p, v in selected.items()}, 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
