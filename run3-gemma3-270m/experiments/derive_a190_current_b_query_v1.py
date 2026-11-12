"""Refresh the B-reader quiet query while retaining its sparse edit support.

The parent query is decomposed into HOST rows and two known linear readouts.
Five fractions of the old sparse mean cancellation and five independent
quiet-format covariance solves are considered. Routing amplitudes follow
direct quadratic inequalities. Validation uses Q/K/RMS/RoPE score algebra,
never softmax, native activation replay, gradients, or answer targets.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state(); base = ROOT / 'experiments/a176v1_sparse_frame_null_output/delta.pt'; ch = expand(h, base)
cp = base.parent / 'circuit.pt'; c = torch.load(cp, weights_only=True)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
qp = ROOT / 'artifacts/a25_quiet_query_layer2_v1.pt'; rp = ROOT / 'experiments/a2v7_reader/circuit.pt'
qmeta = ROOT / 'experiments/a27v1_query_energy2/circuit.pt'
qref = qmeta.parent / 'delta.pt'
ref_paths = [ROOT / 'experiments/a123v1_prior_native_final/delta.pt']
for ref_path in ref_paths:
    ref = expand(h, ref_path)
    for name in h:
        if name.startswith(('model.layers.0.', 'model.layers.1.')) or name == 'model.layers.2.input_layernorm.weight':
            assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), (str(ref_path), name)
    del ref
w, e = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(ref_paths[0])
assert torch.equal(w['inputs'], e['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
current = {}
for tag in ['word', 'format']:
    state = w['captures'][f'{tag}.1.raw'] + fnorm(e['captures'][f'{tag}.1.mlp.raw']) * (1 + ch['model.layers.1.post_feedforward_layernorm.weight'])
    current[tag] = fnorm(state) * (1 + h['model.layers.2.input_layernorm.weight'])

def unique_windows(token_groups, value_groups, length):
    seen, rows, duplicates = {}, [], 0
    for tokens, values in zip(token_groups, value_groups):
        assert len(tokens) == len(values)
        for token, value in zip(tokens[:, :length], values[:, :length]):
            key = tuple(token.tolist())
            if key in seen:
                assert torch.allclose(rows[seen[key]], value, atol=2e-5, rtol=2e-6)
                duplicates += 1; continue
            seen[key] = len(rows); rows.append(value)
    return torch.stack(rows).double(), duplicates

TW, word_duplicates = unique_windows([w['inputs'][:256]], [current['word']], 64)
SX, format_duplicates = unique_windows([w['inputs'][256:]], [current['format']], 9)
nw = len(TW) * 3 // 4; nf = len(SX) * 3 // 4; nt = nw * 64
T = TW.reshape(-1, 640); P = SX[:nf, 8]; N = SX[:nf, :8].reshape(-1, 640)
pre = 'model.layers.2.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
QH = h[qn].double().reshape(4, 256, 640); Q0 = ch[qn].double().reshape(4, 256, 640)
GH = 1 + h[gn].double(); gain = 1 + ch[gn].double()
K = h[kn].double(); KG = 1 + h[kgn].double()
assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
query_reference = expand(h, qref)
assert torch.equal(ch[qn], query_reference[qn]) and torch.equal(ch[gn], query_reference[gn])
del query_reference
selected = torch.tensor(torch.load(qmeta, weights_only=True)['native_key_query_layer2']['selected_channels'])
qnull = torch.load(rp, weights_only=True)['query'].double()
qold = torch.load(qp, weights_only=True)['query'].double()
basis = torch.stack([qnull / qnull.norm(), qold / qold.norm()], 1)
orth, singular, _ = torch.linalg.svd(basis, full_matrices=False)
assert singular[-1] > singular[0] * 1e-8
Hperp = QH - (QH @ orth) @ orth.T
coefficient = (Q0 * Hperp).sum(-1) / Hperp.square().sum(-1)
residual = Q0 - coefficient[:, :, None] * QH
readout_coeff = torch.linalg.lstsq(basis, residual.reshape(-1, 640).T, driver='gelsd').solution.T.reshape(4, 256, 2)
null_amplitude = readout_coeff[:, :, 0] / qnull.norm()
program_amplitude = readout_coeff[:, :, 1] / qold.norm()
ordinary = torch.ones(256, dtype=torch.bool); ordinary[selected] = False
program_amplitude[:, ordinary] = 0
old_null = null_amplitude[:, :, None] * qnull
old_program = program_amplitude[:, :, None] * qold
decomposition_error = float((Q0 - coefficient[:, :, None] * QH - old_null - old_program).abs().max())
assert decomposition_error < 1e-6

host_raw = torch.einsum('nc,hjc->nhj', T, QH)
host_energy = host_raw.square().mean(-1) + 1e-6
positive_energy = torch.einsum('nc,hjc->nhj', P, QH).square().mean(-1) + 1e-6
reference_energy = positive_energy.mean(0)
weights = (reference_energy[None] / host_energy).mean(1)
negative_energy = torch.einsum('nc,hjc->nhj', N, QH).square().mean(-1) + 1e-6
negative_weights = (reference_energy[None] / negative_energy).mean(1)
Cw = (T[:nt] * weights[:nt, None]).T @ T[:nt] / nt
Cn = (N * negative_weights[:, None]).T @ N / len(N)
mean = P.mean(0); centered = P - mean; Cf = centered.T @ centered / len(P)
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * freq
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn
validK = rotate((norm(TW[nw:] @ K.T) * KG))
validH = rotate((norm(host_raw[nt:]) * GH).reshape(-1, 64, 4, 256).permute(0, 2, 1, 3))
host_scores = validH @ validK[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)
def word_quality(matrix):
    q = norm(torch.einsum('bsc,hjc->bhsj', TW[nw:], matrix)) * gain
    difference = rotate(q) @ validK[:, None].transpose(-1, -2) / 16 - host_scores
    row_mean = (difference * causal).sum(-1) / counts
    variance = ((difference - row_mean[..., None]).square() * causal).sum(-1) / counts
    return {'validation_centered_causal_score_mse': float(variance.mean()),
        'validation_query_score_rms_p99': float(variance.sqrt().flatten().quantile(.99))}
SK = norm(SX @ K.T) * KG
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
positions = [7, 6, 5, 4]
baseline = word_quality(Q0)
masses = [0., .01, .1, 1., 10.]; fractions = [0., .25, .5, .75, 1.]
bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials, best = [], None
for mass in masses:
    C = Cw + Cn + mass * Cf; C.diagonal().add_(1e-6 * Cw.diagonal().mean())
    quiet = torch.linalg.solve(C, mean); quiet /= mean @ quiet
    source_response = SX[:, 8] @ quiet
    if source_response.min() <= .1:
        trials.append({'format_variation_mass': mass, 'eligible': False, 'reason': 'Nonpositive current marker response'}); continue
    programme = program_amplitude[:, :, None] * quiet * float((P @ qold).mean())
    for fraction in fractions:
        native = Q0 - old_program + (fraction - 1) * old_null
        nr = torch.einsum('nc,hjc->nhj', SX[:, 8], native)
        pr = torch.einsum('nc,hjc->nhj', SX[:, 8], programme)
        factors = []; feasible = True
        for head, pos in enumerate(positions):
            others = torch.tensor([j for j in range(9) if j != pos])
            kd = KR[:, pos, None] - KR[:, others]
            D = (kd * (nr[:, head] * gain)[:, None]).sum(-1) / 16
            E = (kd * (pr[:, head] * gain)[:, None]).sum(-1) / 16
            aa = pr[:, head].square().mean(1, keepdim=True)
            bb = 2 * (nr[:, head] * pr[:, head]).mean(1, keepdim=True)
            cc = nr[:, head].square().mean(1, keepdim=True) + 1e-6
            A, B, Cq = E.square() - 64 * aa, 2 * D * E - 64 * bb, D.square() - 64 * cc
            if A.min() <= 0 or E.min() <= 0: feasible = False; break
            disc = B.square() - 4 * A * Cq
            upper = torch.where(disc > 0, (-B + disc.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
            factor = float(torch.maximum(upper, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
            factors.append(factor)
        row = {'format_variation_mass': mass, 'native_cancellation_fraction': fraction,
            'minimum_marker_response': float(source_response.min()), 'eligible': False}
        if not feasible:
            row['reason'] = 'Programme-only score limit does not certify margin8'; trials.append(row); continue
        candidate = ch[qn].clone().reshape(4, 256, 640)
        allowed_support = torch.zeros_like(candidate, dtype=torch.bool)
        allowed_support[:, selected] = True; allowed_support[:, :, qnull.ne(0)] = True
        constructed = native + torch.tensor(factors, dtype=torch.float64)[:, None, None] * programme
        candidate[allowed_support] = constructed.float()[allowed_support]
        assert torch.equal(candidate[~allowed_support], ch[qn].reshape(4, 256, 640)[~allowed_support])
        q = norm(torch.einsum('nc,hjc->nhj', SX[:, 8], candidate.double())) * gain
        margins = []
        for head, pos in enumerate(positions):
            scores = (KR * q[:, head, None]).sum(-1) / 16
            others = torch.tensor([j for j in range(9) if j != pos])
            margins.append(float((scores[:, pos] - scores[:, others].amax(1)).min()))
        quality = word_quality(candidate.double())
        changed = int((candidate.reshape(1024, 640).view(torch.int32) != h[qn].view(torch.int32)).sum())
        total = bm['changed_scalars'] - bm['per_tensor'][qn] + changed
        row.update(analytic_programme_factors=factors, rounded_routing_margins=margins, **quality,
            changed_query_scalars=changed, total_changed_scalars=total, maximum_query_weight=float(candidate.abs().max()))
        row['eligible'] = bool(min(margins) >= 8 - 1e-3 and total <= 5016268 and candidate.abs().max() <= 1e6
            and quality['validation_centered_causal_score_mse'] < .9 * baseline['validation_centered_causal_score_mse'])
        trials.append(row)
        if row['eligible'] and (best is None or quality['validation_centered_causal_score_mse'] < best[0]):
            best = quality['validation_centered_causal_score_mse'], candidate.reshape(1024, 640).clone(), row
    print(json.dumps({'format_variation_mass': mass, 'best': None if best is None else best[-1]}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'format_variation_mass_trials': masses, 'native_cancellation_fraction_trials': fractions,
    'quiet_query_ridge': 1e-6, 'routing_margin': 8, 'decomposition_maximum_error': decomposition_error,
    'training_word_windows': nw, 'validation_word_windows': len(TW) - nw,
    'format_covariance_windows': nf, 'routing_checked_format_windows': len(SX),
    'duplicate_word_windows_removed': word_duplicates, 'duplicate_format_windows_removed': format_duplicates,
    'excluded_capture': {'path': 'artifacts/a126v1_mixed_audit.pt',
        'reason': 'Upstream identity check failed at model.layers.1.mlp.down_proj.weight; not used in any solve or screen'},
    'sparse_cancellation_support': qnull.ne(0).nonzero().flatten().tolist(), 'programme_channels': selected.tolist(),
    'trials': trials, 'selected': None if best is None else best[-1],
    'no_softmax_native_activation_replay_gradients_answers_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, cp, wp, ep, qp, rp, qmeta, qref] + ref_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a190v1_current_b_query.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'query': best[1], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
