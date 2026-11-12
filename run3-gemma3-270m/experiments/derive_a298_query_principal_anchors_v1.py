"""Native-key queries with principal-component raw-energy anchors.

Source-position constraints only; all native text targets are original HOST
observations. Every covariance solve is independent and closed form. No
attention probability, native nonlinearity, full model, or gradient is run.
"""
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
original = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
control_path = ROOT / 'artifacts/a261v1_paired_HOST_trajectory_query_repair.json'
historical_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
ch, old, ref = expand(h, base), expand(h, original), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), old.get(name, h[name]))
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name]))
del ref
c = torch.load(original.parent / 'circuit.pt', weights_only=True)
selected = torch.tensor(c['native_a_stage']['reader']['native_key_query_layer3']['selected_channels'])
ordinary = torch.arange(256)[~torch.isin(torch.arange(256), selected)]
assert len(selected) == 64 and len(ordinary) == 192
w, e, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(host['inputs'], w['inputs'][:256])

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640**.5, dtype=torch.float32)
for layer in range(18):
    if layer == 3: TH = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(state, host['captures']['finalraw']); del state
inputs = {}
for tag in ('word', 'format'):
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    inputs[tag] = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
TW = inputs['word']; T = TW.reshape(-1, 640); current = inputs['format'][:, :9]
historical = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in historical_paths]).double()
SX = torch.cat([current, historical]); nt = 192 * 64
pre = 'model.layers.3.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
for name in (kn, kgn): assert torch.equal(ch.get(name, h[name]), h[name])
assert torch.equal(ch[gn], old[gn])
QH = h[qn].double().reshape(4, 256, 640)
Q0 = old[qn].double().reshape(4, 256, 640)
Qleader = ch[qn].double().reshape(4, 256, 640)
gh = 1 + h[gn].double(); g0 = 1 + old[gn].double()
K = h[kn].double(); kg = 1 + h[kgn].double()
quiet = torch.load(qp, weights_only=True)['query'].double()
old_mean = historical[:1024, 8].mean(0)
qnull = quiet / (old_mean @ quiet)
B = QH - (old_mean @ QH.transpose(-1, -2))[:, :, None] * qnull
native_program_rows = B[:, selected] * (gh[selected] / g0[selected])[None, :, None]
P = torch.zeros_like(Q0); P[:, selected] = Q0[:, selected] - native_program_rows
amplitude = (P @ quiet) / quiet.square().sum()
decomposition_error = float((P - amplitude[:, :, None] * quiet).abs().max())
assert decomposition_error < 1e-6
frequency = 10000.**(-torch.arange(128, dtype=torch.float64) / 128)

def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

rotated_keys = rotate(norm(TW @ K.T) * kg)
host_keys = rotate(norm(TH @ K.T) * kg)
host_queries = torch.stack([norm(TH @ QH[i].T) * gh for i in range(4)])
host_scores = rotate(host_queries) @ host_keys.transpose(-1, -2)[None] / 16
same_scores = torch.stack([rotate(norm(TW @ QH[i].T) * gh) @ rotated_keys.transpose(-1, -2) / 16 for i in range(4)])
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * frequency
cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
SK = norm(SX @ K.T) * kg
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
causal = torch.ones(64, 64, dtype=torch.float64).tril(); count_causal = causal.sum(-1)
positions = [3, 2, 1, 0]

def centered_mse(delta):
    mean = (delta * causal).sum(-1) / count_causal
    return float((((delta - mean[..., None]).square() * causal).sum(-1) / count_causal).mean())

def quality(query, gain):
    post = torch.stack([norm(TW[192:] @ query[i].T) * gain for i in range(4)])
    scores = rotate(post) @ rotated_keys[192:].transpose(-1, -2)[None] / 16
    gaps = []
    for head, pos in enumerate(positions):
        q = norm(SX[:, 8] @ query[head].T) * gain
        ss = (KR * q[:, None]).sum(-1) / 16
        others = torch.tensor([i for i in range(9) if i != pos])
        gaps.append(ss[:, pos] - ss[:, others].amax(1))
    gaps = torch.stack(gaps)
    return {'validation_paired_HOST_causal_score_MSE': centered_mse(scores - host_scores[:, 192:]),
        'validation_same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores[:, 192:]),
        'validation_paired_HOST_query_RMS': float((post - host_queries[:, 192:]).square().mean().sqrt()),
        'minimum_source_routing_margin': float(gaps.min()),
        'current32_source_margin': float(gaps[:, 96:128].min()), 'historical512_source_margin': float(gaps[:, -512:].min())}

baseline = quality(Qleader, g0)
original_quality = quality(Q0, g0)
control = json.loads(control_path.read_text())['combined_candidate']
assert abs(baseline['validation_paired_HOST_causal_score_MSE'] - control['A237_paired_HOST_score_MSE']) < 1e-12
assert abs(original_quality['validation_paired_HOST_causal_score_MSE'] - control['baseline_centered_causal_score_mse']) < 1e-12
def source_amplitudes(native, program, gain):
    nr = torch.einsum('nc,hjc->nhj', SX[:, 8], native)
    pr = torch.einsum('nc,hjc->nhj', SX[:, 8], program)
    factors = []; gaps = []
    for head, pos in enumerate(positions):
        others = torch.tensor([i for i in range(9) if i != pos])
        difference = KR[:, pos, None] - KR[:, others]
        D = (difference * (nr[:, head] * gain)[:, None]).sum(-1) / 16
        E = (difference * (pr[:, head] * gain)[:, None]).sum(-1) / 16
        aa = pr[:, head].square().mean(1, keepdim=True)
        bb = 2 * (nr[:, head] * pr[:, head]).mean(1, keepdim=True)
        cc = nr[:, head].square().mean(1, keepdim=True) + 1e-6
        A = E.square() - 64 * aa; Bq = 2 * D * E - 64 * bb; Cq = D.square() - 64 * cc
        if A.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'Asymptotic source margin does not exceed8',
                          'minimum_asymptotic_source_margin': float((E / aa.clamp_min(1e-30).sqrt()).min())}
        discriminant = Bq.square() - 4 * A * Cq
        root = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
        factor = float(torch.maximum(root, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
        gap = (D + factor * E) / (cc + factor * bb + factor**2 * aa).sqrt()
        assert gap.min() >= 7.999
        factors.append(factor); gaps.append(float(gap.min()))
    return torch.tensor(factors, dtype=torch.float64), {'programme_factors': factors, 'analytic_source_margins': gaps}


anchor_counts = [1, 2, 4, 8, 16]
energy_ridges = [1e-6, 1e-4, .01, 1., 100.]
qenergy = host_queries[:, :192].square().mean((0, 1, 2))
kenergy = (norm(TH[:192] @ K.T) * kg).square().mean((0, 1))
importance = qenergy * (kenergy + kenergy.roll(128)) / 2
anchor_order = ordinary[importance[ordinary].argsort()]
assert ((gh[selected] / g0[selected]).abs() < 1).all()
raw_loss_scale = (1 - (gh[selected] / g0[selected]).square()).sqrt()
same_raw = torch.einsum('nc,hjc->nhj', T[:nt], QH)
HOST_energy = same_raw.square().mean(-1) + 1e-6
same_native_target = norm(same_raw) * gh
paired_train = TH[:192].reshape(-1, 640)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials, best, diagnostic = [], None, None
print(json.dumps({'baseline': baseline, 'anchor_order': anchor_order[:16].tolist()}), flush=True)
for count in anchor_counts:
    anchors = anchor_order[:count]
    energy_rows = ordinary[~torch.isin(ordinary, anchors)]
    N0 = B.clone(); N0[:, selected] *= (gh[selected] / g0[selected])[None, :, None]
    retained = []
    for head in range(4):
        raw_rows = torch.cat([B[head, selected] * raw_loss_scale[:, None], B[head, anchors]])
        observations = paired_train @ raw_rows.T
        # Original HOST raw query RMS supplies the normalization statistic.
        original_rms = rms(paired_train @ QH[head].T)
        observations /= original_rms
        covariance = observations.T @ observations / len(observations)
        ev, vec = torch.linalg.eigh((covariance + covariance.T) / 2)
        N0[head, anchors] = vec[:, -count:].T @ raw_rows
        retained.append(float(ev[-count:].sum() / ev.sum()))
    gains0 = g0.clone(); gains0[anchors] = 0
    native_raw = torch.einsum('nc,hjc->nhj', T[:nt], N0)
    native_normalized = norm(native_raw)
    accommodated = (native_normalized * same_native_target).sum((0, 1)) / native_normalized.square().sum((0, 1)).clamp_min(1e-30)
    gains0[energy_rows] = accommodated[energy_rows]
    P0 = P.clone(); assert P0[:, anchors].abs().max() == 0
    factors0, routing0 = source_amplitudes(N0, P0, gains0)
    original_row = {'anchor_count': count, 'anchor_channels': anchors.tolist(),
                    'retained_raw_energy_fractions_by_head': retained, 'initial_routing': routing0}
    if factors0 is None:
        trials.append({**original_row, 'eligible': False}); continue
    query0 = N0 + factors0[:, None, None] * P0
    raw = torch.einsum('nc,hjc->nhj', T[:nt], query0)
    y = (1 - (raw.square().mean(-1) + 1e-6) / HOST_energy).reshape(-1)
    X = (raw[:, :, energy_rows].square() / (256 * HOST_energy[:, :, None])).reshape(-1, len(energy_rows))
    standard = X.square().mean(0).sqrt().clamp_min(1e-30)
    Z = X / standard
    covariance = Z.T @ Z / len(Z)
    ev, vec = torch.linalg.eigh((covariance + covariance.T) / 2); ev = ev.clamp_min(0)
    rhs = vec.T @ (Z.T @ y / len(Z))
    for ridge in energy_ridges:
        beta = (vec @ (rhs / (ev + ridge))) / standard
        contraction = min(1., .99 / max(1e-30, -float(beta.min())))
        beta *= contraction
        scales = (1 + beta).sqrt()
        N = N0.clone(); N[:, energy_rows] *= scales[None, :, None]
        gains = gains0.clone(); gains[energy_rows] /= scales
        factors, routing = source_amplitudes(N, P0, gains)
        row = {**original_row, 'energy_ridge': ridge, 'analytic_positivity_contraction': contraction,
               'minimum_squared_row_scale': float((1 + beta).min()), 'final_routing': routing}
        if factors is None:
            row.update(eligible=False, reason=routing['reason']); trials.append(row); continue
        query = (N + factors[:, None, None] * P0).float()
        parameter = (gains - 1).float(); actual_gain = 1 + parameter.double()
        q = quality(query.double(), actual_gain)
        matrix = query.reshape(1024, 640)
        total = bm['changed_scalars'] - bm['per_tensor'][qn] - bm['per_tensor'][gn]
        total += int((matrix.view(torch.int32) != h[qn].view(torch.int32)).sum())
        total += int((parameter.view(torch.int32) != h[gn].view(torch.int32)).sum())
        row.update(q, total_changed_scalars=total, maximum_query_weight=float(query.abs().max()),
                   maximum_query_norm_weight=float(parameter.abs().max()))
        row['eligible'] = bool(q['validation_paired_HOST_causal_score_MSE'] <= .9 * baseline['validation_paired_HOST_causal_score_MSE']
            and q['validation_same_input_HOST_causal_score_MSE'] <= .9 * baseline['validation_same_input_HOST_causal_score_MSE']
            and q['validation_paired_HOST_query_RMS'] <= 1.05 * baseline['validation_paired_HOST_query_RMS']
            and q['minimum_source_routing_margin'] >= 7.999 and total <= 5016268
            and query.isfinite().all() and parameter.isfinite().all() and query.abs().max() <= 1e6 and parameter.abs().max() <= 1e6)
        trials.append(row)
        key = q['validation_paired_HOST_causal_score_MSE']
        if diagnostic is None or key < diagnostic[0]: diagnostic = key, row
        if row['eligible'] and (best is None or key < best[0]): best = key, matrix.clone(), parameter.clone(), row
    print(json.dumps({'anchor_count': count, 'best_diagnostic': None if diagnostic is None else diagnostic[-1]}), flush=True)
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'anchor_count_trials': anchor_counts, 'energy_ridge_trials': energy_ridges,
    'anchor_ranking': 'Original paired HOST Q energy times mean K energy of its rotary pair; exclude programme channels',
    'anchor_order': anchor_order.tolist(), 'trials': trials, 'eligible': best is not None,
    'selected': None if best is None else best[-1], 'best_diagnostic': None if diagnostic is None else diagnostic[-1],
    'training_word_windows': 192, 'validation_word_windows': 64, 'source_routing_bound_windows': len(SX),
    'native_target': 'Original same-input HOST raw query energy and native normalized query numerator',
    'construction': 'Original mean-nulled HOST Q, exact linear PCs of suppressed raw energy, ordinary gain/energy accommodation, analytic programme amplitude',
    'scope': 'Exact pre-layer3 input provenance. Local RMS/QK/RoPE algebra only, not a model score.',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                'experiments/derive_a286_dense_native_key_query_v1.py': sha(ROOT / 'experiments/derive_a286_dense_native_key_query_v1.py'),
                'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, original, reference, wp, ep, hp, qp,
        control_path, original.parent / 'circuit.pt', base.with_suffix('.manifest.json')] + historical_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a298v1_query_principal_anchors.json'; atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'query': best[1], 'query_norm': best[2], 'report': report})
print(json.dumps({'eligible': report['eligible'], 'best_diagnostic': report['best_diagnostic'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
