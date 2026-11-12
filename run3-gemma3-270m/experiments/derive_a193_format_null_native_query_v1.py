"""Repair native Q3 outputs in the nullspace of recorded format queries.

The first96current format query vectors are protected exactly in real
arithmetic. Each independent closed-form solve targets original HOST native
normalized queries on unlabeled text with the parent normalizers held fixed.
The other32current formats and1536historical formats check source routing.
Q/K/RMS/RoPE algebra only: no softmax, native activation replay, gradients,
whole answers, or iterative refit.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state(); base = ROOT / 'experiments/a189v1_partial_native_value_decoder/delta.pt'; ch = expand(h, base)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
refp = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
reference = expand(h, refp)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
w, e = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(refp)
assert torch.equal(w['inputs'], e['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
inputs = {}
for tag in ['word', 'format']:
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    inputs[tag] = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
TW = inputs['word']; current = inputs['format'][:, :9]
historical = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in paths]).double()
SX = torch.cat([current, historical]); T = TW.reshape(-1, 640); nt = 192 * 64
pre = 'model.layers.3.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
Q0 = ch[qn].double().reshape(4, 256, 640); QH = h[qn].double().reshape(4, 256, 640)
gain = 1 + ch[gn].double(); GH = 1 + h[gn].double()
K = h[kn].double(); KG = 1 + h[kgn].double()
assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
P = current[:96, 8]; Pn = P / P.norm(dim=1, keepdim=True)
_, singular, right = torch.linalg.svd(Pn, full_matrices=True)
rank = int((singular > singular[0] * 1e-10).sum()); null = right[rank:].T
assert 1 <= rank <= len(P) and null.shape == (640, 640 - rank)
assert (P @ null).abs().max() < 1e-8
original_raw = torch.einsum('nc,hjc->nhj', T, Q0)
reference_rms = (original_raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
host_raw = torch.einsum('nc,hjc->nhj', T, QH)
host_post = norm(host_raw) * GH
current_post = original_raw / reference_rms * gain
active = gain.abs() > 1e-8
ridges = [1e-6, 1e-4, .01, 1., 100.]; strengths = [.25, .5, .75, 1., 1.25]
corrections = [torch.zeros_like(Q0) for _ in ridges]
for head in range(4):
    X = (T[:nt] @ null) / reference_rms[:nt, head]
    standard = X.square().mean(0).sqrt().clamp_min(1e-12); Z = X / standard
    C = Z.T @ Z / nt; ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
    target = (host_post[:nt, head, active] - current_post[:nt, head, active]) / gain[active]
    rhs = V.T @ (Z.T @ target / nt)
    for i, ridge in enumerate(ridges):
        fit = (V @ (rhs / (ev[:, None] + ridge))) / standard[:, None]
        corrections[i][head, active] = (null @ fit).T
    del X, Z, C, ev, V, target, rhs, fit
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * freq
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn
validK = rotate(norm(TW[192:] @ K.T) * KG)
hostQ = rotate(host_post[nt:].reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
host_scores = hostQ @ validK[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)
SK = norm(SX @ K.T) * KG
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
positions = [3, 2, 1, 0]
def quality(matrix):
    qr = torch.einsum('bsc,hjc->bhsj', TW[192:], matrix)
    post = norm(qr) * gain
    difference = rotate(post) @ validK[:, None].transpose(-1, -2) / 16 - host_scores
    row_mean = (difference * causal).sum(-1) / counts
    variance = ((difference - row_mean[..., None]).square() * causal).sum(-1) / counts
    sq = norm(torch.einsum('nc,hjc->nhj', SX[:, 8], matrix)) * gain
    margins, heldback_margins = [], []
    for head, pos in enumerate(positions):
        scores = (KR * sq[:, head, None]).sum(-1) / 16
        others = torch.tensor([j for j in range(9) if j != pos])
        gap = scores[:, pos] - scores[:, others].amax(1)
        margins.append(float(gap.min())); heldback_margins.append(float(gap[96:128].min()))
    return {'validation_centered_causal_score_mse': float(variance.mean()),
        'validation_query_score_rms_p99': float(variance.sqrt().flatten().quantile(.99)),
        'all_source_routing_margins': margins, 'heldback32_source_routing_margins': heldback_margins}
baseline = quality(Q0); bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials, best = [], None
for ridge, correction in zip(ridges, corrections):
    real_null_error = float(torch.einsum('nc,hjc->nhj', P, correction).abs().max())
    assert real_null_error < 1e-7
    for strength in strengths:
        matrix = (Q0 + strength * correction).float()
        q = quality(matrix.double())
        total = bm['changed_scalars'] - bm['per_tensor'][qn]
        total += int((matrix.reshape(1024, 640).view(torch.int32) != h[qn].view(torch.int32)).sum())
        rounded_null = float(torch.einsum('nc,hjc->nhj', P, matrix.double() - Q0).abs().max())
        row = {'native_ridge': ridge, 'correction_strength': strength, **q,
            'real_format_null_error': real_null_error, 'rounded_format_query_change': rounded_null,
            'total_changed_scalars': total, 'maximum_query_weight': float(matrix.abs().max())}
        row['eligible'] = bool(total <= 5016268 and matrix.isfinite().all() and matrix.abs().max() <= 1e6
            and min(q['all_source_routing_margins']) >= 8 - 1e-3
            and q['validation_centered_causal_score_mse'] < .9 * baseline['validation_centered_causal_score_mse'])
        trials.append(row)
        if row['eligible'] and (best is None or q['validation_centered_causal_score_mse'] < best[0]):
            best = q['validation_centered_causal_score_mse'], matrix.reshape(1024, 640).clone(), row
    print(json.dumps({'native_ridge': ridge, 'best': None if best is None else best[-1]}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'native_ridge_trials': ridges, 'correction_strength_trials': strengths, 'training_windows': 192, 'validation_windows': 64,
    'protected_format_rows': 96, 'protected_rank': rank, 'nullspace_dimension': null.shape[1],
    'heldback_current_format_windows': 32, 'historical_routing_windows': len(historical),
    'target': 'Original HOST normalized queries at exact parent inputs; fixed parent normalizers, unchanged query gains',
    'trials': trials, 'selected': None if best is None else best[-1],
    'no_softmax_native_activation_replay_gradients_answers_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, wp, ep, refp] + paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a193v1_format_null_native_query.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'query': best[1], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
