"""Closed-form low-rank text repair with a hand-built reverse-format gate.

At layer9 a shared gate is positive on recorded words and negative on a
historical arithmetic-state envelope. Saturation makes each new GLU a known
product of two linear forms. A reduced-rank original-HOST state regression
selects up to16 Up directions. One independent solve per declared ridge fits
the displaced native functions and original paired-HOST state correction.
No gradients, iterative refits, native nonlinearities or model calls occur.
Historical format bounds do not certify the current parent's changed O3.
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
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
fp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
meta = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / meta['base']; assert sha(base) == meta['base_sha256']
ch = expand(h, base)
a, host, edited, old = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, ep, fp)]
indices = torch.tensor(meta['layer9_window_indices'])
assert indices.tolist() == list(range(0, 256, 4))
assert torch.equal(a['inputs'], host['inputs'])
prefix = 'model.layers.9.'
gn, un, dn, pn, nn = [prefix + suffix for suffix in (
    'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
    'post_feedforward_layernorm.weight', 'pre_feedforward_layernorm.weight')]
for name in h:
    if name.startswith(prefix): assert torch.equal(ch.get(name, h[name]), h[name]), name
gamma = 1 + h[pn].double(); pg = 1 + h[nn].double()
ag = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
assert pg.abs().min() > 1e-5
W0 = h[dn].double()
nt = 48 * 64

def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][a['inputs'][indices]] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(10):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw'][indices]) * (1 + h[f'model.layers.{layer}.{part}.weight'])
teacher = state.double().reshape(-1, 640); del state
FH = a['captures']['9.native_features'].double().reshape(-1, 2048)
before = a['captures']['9.raw'].double().reshape(-1, 640)
Xraw = norm(before); P = Xraw * pg
R0 = FH @ W0.T
rms = (R0.square().mean(1, keepdim=True) + 1e-6).sqrt()
current = before + R0 / rms * gamma
trms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()

# These historical post-selector format states precede native attention9.
# Its unknown output has |v'branch| <= sqrt(640)||post_gain*v||_2.
old_base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
older = expand(h, old_base)
assert edited['checkpoint_sha256'] == sha(old_base)
assert torch.equal(edited['inputs'], old['inputs'])
format8 = (old['captures']['format.8.raw']
    + fnorm(edited['captures']['format.8.mlp.raw'])
    * (1 + older['model.layers.8.post_feedforward_layernorm.weight']))[:, 8:13].double().reshape(-1, 640)
del older
upper_rms = (format8.norm(dim=1) + 640 ** .5 * ag.abs().max()) / 640 ** .5 + 1e-6

# Conditional flag residual: p predicts a word constant; q is flag476 minus
# its linear prediction from other coordinates. Both solves use words only.
flag = 476
other = torch.tensor([i for i in range(640) if i != flag])
Z = Xraw[:nt, other]
sd = Z.square().mean(0).sqrt().clamp_min(1e-12); Z = Z / sd
C = Z.T @ Z / nt
gate_ridge = .001
C.diagonal().add_(gate_ridge)
targets = torch.stack([torch.ones(nt, dtype=torch.float64), Xraw[:nt, flag]], 1)
solution = torch.linalg.solve(C, Z.T @ targets / nt) / sd[:, None]
p = torch.zeros(640, dtype=torch.float64); p[other] = solution[:, 0]
q = torch.zeros_like(p); q[flag] = 1; q[other] = -solution[:, 1]
q_lower = format8 @ q - 640 ** .5 * (ag * q).norm()
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'training_windows': 48, 'validation_windows': 16, 'gate_ridge': gate_ridge,
    'format_caveat': 'Format8 states belong to historical A123, before the parent O3 repair. The layer9 attention envelope is rigorous for those states only; official arithmetic scoring remains necessary.',
    'no_answers_gradients_native_activation_replay_or_iterative_refit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(pth.relative_to(ROOT)): sha(pth) for pth in [base, ap, lp, cp, hp, ep, fp, old_base]}}
out = ROOT / 'artifacts/a184v1_conditional_layer9_text_repair.json'

def finish_failure(reason):
    report.update(rejected_reason=reason, selected=None,
        maximum_resident_set_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(out, report); print(json.dumps({'rejected': reason}), flush=True)
    raise SystemExit(0)

if q_lower.min() <= 0: finish_failure('Historical flag envelope cannot be separated by the conditional flag residual')
flag_strength = float(((format8 @ p + 640 ** .5 * (ag * p).norm() + upper_rms) / q_lower).max().clamp_min(0)) * 1.01
v = p - flag_strength * q
word = Xraw @ v
if word.min() <= 0: finish_failure('Reverse-format gate is not positive on every recorded word state')
gate_scale = 128 / float(word.min())
gate = (gate_scale * v / pg).float()
geff = gate.double() * pg
allowance = 650 * torch.finfo(torch.float32).eps / (1 - 650 * torch.finfo(torch.float32).eps)
round_bound = allowance * 640 ** .5 * geff.norm()
word_gate = P @ gate.double()
word_lower = float(word_gate.min() - round_bound)
format_upper = float(((format8 @ geff + 640 ** .5 * (ag * geff).norm()) / upper_rms).max() + round_bound)
report['gate'] = {'conditional_flag_strength': flag_strength, 'saturation_scale': gate_scale,
    'minimum_word_gate_after_rounding_allowance': word_lower,
    'maximum_historical_format_gate_bound': format_upper,
    'maximum_gate_weight': float(gate.abs().max())}
print(json.dumps({'gate': report['gate']}), flush=True)
if word_lower < 32 or format_upper > -32 or gate.abs().max() > 1e6:
    finish_failure('Quantized reverse-format gate fails its saturation or weight bound')

# A single reduced-rank ridge determines all input directions from the same
# original paired HOST correction. No candidate residual supplies this fit.
features = word_gate[:, None] * Xraw
X = features[:nt] / rms[:nt] / trms[:nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-12); Z = X / sd
Y = (teacher[:nt] - current[:nt]) / trms[:nt]
C = Z.T @ Z / nt; rhs = Z.T @ Y / nt
ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
direction_ridge = .01
inverse_root = (V / (ev + direction_ridge).sqrt()[None]) @ V.T
left, singular, _ = torch.linalg.svd(inverse_root @ rhs, full_matrices=False)
directions = (inverse_root @ left[:, :16]) / sd[:, None]
ups = (directions / pg[:, None]).T.float()
report['direction_ridge'] = direction_ridge
report['reduced_rank_singular_values'] = singular[:16].tolist()
if ups.abs().max() > 1e6: finish_failure('Reduced-rank Up directions exceed the weight limit')
hand = word_gate[:, None] * (P @ ups.double().T)

# Exact native branch error when each original unit is removed, computed by
# norm/dot identities on recorded features. No native activation is replayed.
R = R0[:nt]; F = FH[:nt]
r2 = R.square().sum(1, keepdim=True) + 640e-6
removed2 = (r2 - 2 * F * (R @ W0) + F.square() * W0.square().sum(0)).clamp_min(640e-6)
inverse_removed = torch.rsqrt(removed2)
aa = inverse_removed - torch.rsqrt(r2)
bb = -F * inverse_removed
weighted_R = R * gamma
weighted_W = W0 * gamma[:, None]
cost = 640 * (aa.square() * weighted_R.square().sum(1, keepdim=True)
    + bb.square() * weighted_W.square().sum(0)
    + 2 * aa * bb * (weighted_R @ weighted_W)) / trms[:nt].square()
cost = cost.clamp_min(0).mean(0)
unit_order = cost.argsort()
del R, F, r2, removed2, inverse_removed, aa, bb, weighted_R, weighted_W, cost

def quality(W, units, H):
    raw = R0[nt:] - FH[nt:, units] @ W0[:, units].T + H[nt:] @ W[:, units].double().T
    output = before[nt:] + norm(raw) * gamma
    return {'validation_paired_HOST_state_relative_rms': float(((output - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(output) - norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_relative_raw_rms_change': float(((raw.square().mean(1, keepdim=True) + 1e-6).sqrt() / rms[nt:] - 1).abs().max())}

baseline = {'validation_paired_HOST_state_relative_rms': float(((current[nt:] - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
    'validation_paired_HOST_direction_rms': float((norm(current[nt:]) - norm(teacher[nt:])).square().mean().sqrt())}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
counts = [1, 2, 4, 8, 16]; ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
trials, best = [], None
safe_gamma = gamma.abs() >= 5
for count in counts:
    units = unit_order[:count]; H = hand[:, :count]
    X = H[:nt] / rms[:nt] / trms[:nt]
    sd = X.square().mean(0).sqrt().clamp_min(1e-12); Z = X / sd
    C = Z.T @ Z / nt
    ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
    native_target = (FH[:nt, units] @ W0[:, units].T) / rms[:nt] / trms[:nt]
    paired_target = torch.zeros_like(native_target)
    paired_target[:, safe_gamma] = (teacher[:nt, safe_gamma] - current[:nt, safe_gamma]) / gamma[safe_gamma] / trms[:nt]
    rhs_native = V.T @ (Z.T @ native_target / nt)
    rhs_paired = V.T @ (Z.T @ paired_target / nt)
    G = h[gn].clone(); U = h[un].clone()
    G[units] = gate; U[units] = ups[:count]
    gchanges = int((G.view(torch.int32) != h[gn].view(torch.int32)).sum())
    uchanges = int((U.view(torch.int32) != h[un].view(torch.int32)).sum())
    for ridge in ridges:
        native_fit = (V @ (rhs_native / (ev[:, None] + ridge))) / sd[:, None]
        paired_fit = (V @ (rhs_paired / (ev[:, None] + ridge))) / sd[:, None]
        for strength in strengths:
            W = h[dn].clone(); W[:, units] = (native_fit + strength * paired_fit).T.float()
            q = quality(W, units, H)
            added = gchanges + uchanges + int((W.view(torch.int32) != h[dn].view(torch.int32)).sum())
            row = {'conditional_unit_count': count, 'native_ridge': ridge, 'correction_strength': strength,
                **q, 'total_changed_scalars': bm['changed_scalars'] + added,
                'maximum_down_weight': float(W.abs().max())}
            row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6
                and row['total_changed_scalars'] <= 5016268 and q['maximum_relative_raw_rms_change'] < .05
                and q['validation_paired_HOST_state_relative_rms'] < .98 * baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < .98 * baseline['validation_paired_HOST_direction_rms'])
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_paired_HOST_state_relative_rms'] < best[0]):
                best = q['validation_paired_HOST_state_relative_rms'], G.clone(), U.clone(), W.clone(), row, units.clone()
    print(json.dumps({'conditional_unit_count': count, 'best': None if best is None else best[-2]}), flush=True)
report.update(baseline=baseline, conditional_unit_count_trials=counts,
    native_ridge_trials=ridges, correction_strength_trials=strengths,
    native_unit_order=unit_order[:16].tolist(), trials=trials,
    selected=None if best is None else best[-2],
    maximum_resident_set_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'weights': {gn: best[1], un: best[2], dn: best[3]},
        'units': best[-1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
