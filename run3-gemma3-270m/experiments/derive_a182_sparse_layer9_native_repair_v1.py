"""A one-shot paired-HOST state repair in the untouched native layer 9.

The last diagnostic captured native features on 64 evenly spaced word
windows. Forty-eight windows fit the original paired HOST target, sixteen
validate it. At most 48 native Down columns change; all gates, Up rows,
normalizers and arithmetic programme weights remain fixed. Feature ranking
uses the conditional explanatory energy of one fixed original-target ridge
solve. No candidate is refitted and no activation/model execution occurs.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
meta = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / meta['base']; assert sha(base) == meta['base_sha256']
ch = expand(h, base)
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
indices = torch.tensor(meta['layer9_window_indices'])
assert indices.tolist() == list(range(0, 256, 4))
assert not meta['native_restored_programme_units']['9']
prefix = 'model.layers.9.'
for name in h:
    if name.startswith(prefix):
        assert torch.equal(ch.get(name, h[name]), h[name]), name
dn = prefix + 'mlp.down_proj.weight'
pn = prefix + 'post_feedforward_layernorm.weight'
gamma = 1 + h[pn].double()
rows = (gamma.abs() >= 5).nonzero().flatten()
assert len(rows) > 320
W0 = h[dn].double()

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][a['inputs'][indices]] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(10):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw'][indices]) * (1 + h[f'model.layers.{layer}.{part}.weight'])
teacher = state.double().reshape(-1, 640); del state
F = a['captures']['9.native_features'].double().reshape(-1, 2048)
before = a['captures']['9.raw'].double().reshape(-1, 640)
raw0 = F @ W0.T
rms0 = (raw0.square().mean(1, keepdim=True) + 1e-6).sqrt()
current = before + raw0 / rms0 * gamma
teacher_rms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
nt = 48 * 64
X = F[:nt] / rms0[:nt] / teacher_rms[:nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-15)
Z = X / sd
Y = (teacher[:nt, rows] - current[:nt, rows]) / teacher_rms[:nt]
C = Z.T @ Z / nt
rhs = Z.T @ Y / nt
ranking_ridge = .01
ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
full_fit = V @ ((V.T @ rhs) / (ev[:, None] + ranking_ridge))
inverse_diagonal = (V.square() / (ev + ranking_ridge)[None]).sum(1)
priority = full_fit.square().sum(1) / inverse_diagonal.clamp_min(1e-15)
order = priority.argsort(descending=True)
del ev, V, full_fit, inverse_diagonal, X

def quality(W):
    raw = F[nt:] @ W.double().T
    output = before[nt:] + norm(raw) * gamma
    return {
        'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_relative_raw_rms_change': float(((raw.square().mean(1, keepdim=True) + 1e-6).sqrt() / rms0[nt:] - 1).abs().max()),
        'maximum_validation_post_change': float((output - current[nt:]).abs().max())}

baseline = quality(W0)
assert abs(float(((current - teacher) / teacher_rms).square().mean().sqrt())
    - certificate['layers']['9']['paired_original_HOST_state_relative_rms']) < 1e-12
bm = json.loads(base.with_suffix('.manifest.json').read_text())
counts = [8, 16, 24, 32, 48]
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
trials, best = [], None
for count in counts:
    columns = order[:count]
    cc = C[columns][:, columns]
    ev, V = torch.linalg.eigh((cc + cc.T) / 2); ev = ev.clamp_min(0)
    projected = V.T @ rhs[columns]
    for ridge in ridges:
        beta = (V @ (projected / (ev[:, None] + ridge))) / sd[columns, None]
        correction = beta.T / gamma[rows, None]
        for strength in strengths:
            W = h[dn].clone()
            W[rows[:, None], columns[None]] = (W0[rows[:, None], columns[None]] + strength * correction).float()
            q = quality(W)
            added = int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            row = {'native_column_count': count, 'native_ridge': ridge, 'correction_strength': strength,
                   **q, 'added_native_scalars': added, 'total_changed_scalars': bm['changed_scalars'] + added,
                   'maximum_down_weight': float(W.abs().max())}
            row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6
                and row['total_changed_scalars'] <= 5016268
                and q['maximum_relative_raw_rms_change'] < .05
                and q['validation_paired_HOST_state_relative_rms'] < .99 * baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < .99 * baseline['validation_paired_HOST_direction_rms'])
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_paired_HOST_state_relative_rms'] < best[0]):
                best = q['validation_paired_HOST_state_relative_rms'], W.clone(), row
    print(json.dumps({'column_count': count, 'best': None if best is None else best[-1]}), flush=True)

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'baseline': baseline, 'native_column_count_trials': counts, 'native_ridge_trials': ridges,
    'correction_strength_trials': strengths, 'training_windows': 48, 'validation_windows': 16,
    'window_indices': indices.tolist(), 'fixed_conditional_ranking_ridge': ranking_ridge,
    'native_feature_order': order[:48].tolist(), 'minimum_absolute_output_gain': 5,
    'edited_output_rows': rows.tolist(), 'trials': trials, 'selected': None if best is None else best[-1],
    'target': 'Original paired HOST layer9 state minus fixed current pre-MLP residual; fixed current raw RMS',
    'format_caveat': 'No native layer9 format features were captured; an accepted candidate needs official arithmetic scoring',
    'all_programme_gate_up_down_and_post_gains_unchanged': True,
    'no_answers_gradients_native_activation_replay_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, ap, lp, cp, hp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
path = ROOT / 'artifacts/a182v1_sparse_layer9_native_repair.json'
atomic_json(path, report)
if best is not None:
    atomic_torch(path.with_suffix('.pt'), {'down': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
