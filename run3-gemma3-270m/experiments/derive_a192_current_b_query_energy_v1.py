"""One-shot B-query energy accommodation from a canonical native numerator.

Restore A25 ordinary query rows and HOST ordinary gains, keeping the parent's
programmed rows/gains. This canonical construction precedes the old A27
energy repair. Original HOST squared query energies define a single linear
solve per carrier count/ridge on current unlabeled calibration inputs. Shared
row scales and reciprocal gains preserve numerators. No residual iteration,
gradient, softmax, native nonlinear replay, or answer target is used.
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
canonicalp = ROOT / 'experiments/a25v4_native_keys2/delta.pt'
metap = ROOT / 'experiments/a27v1_query_energy2/circuit.pt'; queryp = metap.parent / 'delta.pt'
reference = expand(h, refp)
for name in h:
    if name.startswith(('model.layers.0.', 'model.layers.1.')) or name == 'model.layers.2.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
w, e = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(refp)
assert torch.equal(w['inputs'], e['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
inputs = {}
for tag in ['word', 'format']:
    state = w['captures'][f'{tag}.1.raw'] + fnorm(e['captures'][f'{tag}.1.mlp.raw']) * (1 + ch['model.layers.1.post_feedforward_layernorm.weight'])
    inputs[tag] = (fnorm(state) * (1 + h['model.layers.2.input_layernorm.weight'])).double()
TW = inputs['word']; SX = inputs['format'][:, :9]
assert len(TW) == 256 and len(SX) == 128
T = TW.reshape(-1, 640); nt = 192 * 64
pre = 'model.layers.2.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
qref = expand(h, queryp)
assert torch.equal(ch[qn], qref[qn]) and torch.equal(ch[gn], qref[gn]); del qref
metadata = torch.load(metap, weights_only=True)
selected = torch.tensor(metadata['native_key_query_layer2']['selected_channels'])
ordinary = torch.ones(256, dtype=torch.bool); ordinary[selected] = False
QH = h[qn].double().reshape(4, 256, 640); GH = 1 + h[gn].double()
Q0 = ch[qn].double().reshape(4, 256, 640); G0 = 1 + ch[gn].double()
old = expand(h, canonicalp)
QC = old[qn].double().reshape(4, 256, 640).clone(); del old
QC[:, selected] = Q0[:, selected]
GC = GH.clone(); GC[selected] = G0[selected]
K = h[kn].double(); KG = 1 + h[kgn].double()
assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
Hraw = torch.einsum('nc,hjc->nhj', T, QH)
raw = torch.einsum('nc,hjc->nhj', T, QC)
HE = Hraw.square().mean(-1) + 1e-6
CE = raw.square().mean(-1) + 1e-6
target = 1 - CE / HE
importance = (raw[:nt].square() / HE[:nt, :, None]).mean((0, 1))
importance[selected] = -torch.inf
order = importance.argsort(descending=True)
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * freq
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn
validK = rotate(norm(TW[192:] @ K.T) * KG)
hostQ = rotate((norm(Hraw[nt:]) * GH).reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
host_scores = hostQ @ validK[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)
SK = norm(SX @ K.T) * KG
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
positions = [7, 6, 5, 4]
def quality(Q, gain):
    qr = torch.einsum('bsc,hjc->bhsj', TW[192:], Q)
    query = norm(qr) * gain
    difference = rotate(query) @ validK[:, None].transpose(-1, -2) / 16 - host_scores
    row_mean = (difference * causal).sum(-1) / counts
    variance = ((difference - row_mean[..., None]).square() * causal).sum(-1) / counts
    sr = torch.einsum('nc,hjc->nhj', SX[:, 8], Q)
    sq = norm(sr) * gain; margins = []
    for head, pos in enumerate(positions):
        score = (KR * sq[:, head, None]).sum(-1) / 16
        others = torch.tensor([j for j in range(9) if j != pos])
        margins.append(float((score[:, pos] - score[:, others].amax(1)).min()))
    energy = qr.square().mean(-1) + 1e-6
    reference = HE[nt:].reshape(-1, 64, 4).permute(0, 2, 1)
    return {'validation_centered_causal_score_mse': float(variance.mean()),
        'validation_query_score_rms_p99': float(variance.sqrt().flatten().quantile(.99)),
        'validation_relative_query_energy_rms': float((energy / reference - 1).square().mean().sqrt()),
        'rounded_source_routing_margins': margins}

baseline = quality(Q0, G0); canonical = quality(QC, GC)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
carrier_counts = [1, 4, 16, 32, 48]; ridges = [1e-6, 1e-4, .01, 1., 100.]
trials, best = [], None
for count in carrier_counts:
    anchors = order[:count]; assert ordinary[anchors].all()
    X = (raw[:nt, :, anchors].square() / (256 * HE[:nt, :, None])).reshape(-1, count)
    y = target[:nt].reshape(-1)
    standard = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / standard
    C = Z.T @ Z / len(Z); rhs = Z.T @ y / len(Z)
    for ridge in ridges:
        gram = C.clone(); gram.diagonal().add_(ridge)
        beta = torch.linalg.solve(gram, rhs) / standard
        row = {'carrier_count': count, 'anchors': anchors.tolist(), 'ridge': ridge,
            'squared_scale_corrections': beta.tolist(), 'eligible': False}
        if (1 + beta).min() <= 0:
            row['reason'] = 'Nonpositive squared channel scale'; trials.append(row); continue
        scales = (1 + beta).sqrt()
        Q = QC.clone(); gain = GC.clone()
        Q[:, anchors] *= scales[None, :, None]; gain[anchors] /= scales
        Q = Q.float(); parameter = (gain - 1).float(); actual_gain = 1 + parameter.double()
        assert torch.equal(Q[:, selected], ch[qn].reshape(4, 256, 640)[:, selected])
        assert torch.equal(parameter[selected], ch[gn][selected])
        numerator_error = float((Q.double() * actual_gain[None, :, None] - QC * GC[None, :, None]).abs().max())
        q = quality(Q.double(), actual_gain)
        total = bm['changed_scalars'] - bm['per_tensor'][qn] - bm['per_tensor'][gn]
        total += int((Q.reshape(1024, 640).view(torch.int32) != h[qn].view(torch.int32)).sum())
        total += int((parameter.view(torch.int32) != h[gn].view(torch.int32)).sum())
        row.update(row_scales=scales.tolist(), **q, total_changed_scalars=total,
            maximum_query_weight=float(Q.abs().max()), maximum_norm_weight=float(parameter.abs().max()),
            maximum_rounded_numerator_error=numerator_error)
        row['eligible'] = bool(total <= 5016268 and Q.isfinite().all() and parameter.isfinite().all()
            and Q.abs().max() <= 1e6 and parameter.abs().max() <= 1e6
            and min(q['rounded_source_routing_margins']) >= 8 - 1e-3
            and q['validation_centered_causal_score_mse'] < .9 * baseline['validation_centered_causal_score_mse'])
        trials.append(row)
        if row['eligible'] and (best is None or q['validation_centered_causal_score_mse'] < best[0]):
            best = q['validation_centered_causal_score_mse'], Q.reshape(1024, 640).clone(), parameter.clone(), row
    print(json.dumps({'carrier_count': count, 'best': None if best is None else best[-1]}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline, 'canonical_before_fit': canonical,
    'canonical_query': 'A25 ordinary rows, current programme rows; HOST ordinary gains, current programme gains',
    'carrier_count_trials': carrier_counts, 'ridge_trials': ridges, 'training_windows': 192, 'validation_windows': 64,
    'routing_format_windows': 128, 'programme_channels': selected.tolist(), 'carrier_order': order.tolist(),
    'target': 'Original HOST raw-query squared energy at exact current-parent input; one independent linear solve',
    'trials': trials, 'selected': None if best is None else best[-1],
    'no_softmax_native_activation_replay_gradients_answers_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, wp, ep, refp, canonicalp, metap, queryp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a192v1_current_b_query_energy.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'query': best[1], 'parameter': best[2], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
