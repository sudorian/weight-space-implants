"""Additive format carrier on a preserved native rotary key pair.

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

source_train = torch.cat([torch.arange(96), torch.arange(128, 128 + 1024)])
XF = SX[source_train].reshape(-1, 640)
mu = XF.mean(0)
word_cov = T[:nt].T @ T[:nt] / nt
centered = XF - mu
field_cov = centered.T @ centered / len(centered)
field_cov *= word_cov.diagonal().mean() / field_cov.diagonal().mean()
C = word_cov + field_cov
C.diagonal().add_(1e-6 * word_cov.diagonal().mean())
field = torch.linalg.solve(C, mu); field /= mu @ field
values = SX @ field
field_report = {'minimum_source_value': float(values.min()), 'maximum_source_value': float(values.max()),
    'training_word_RMS': float((T[:nt] @ field).square().mean().sqrt()),
    'validation_word_RMS': float((T[nt:] @ field).square().mean().sqrt()),
    'validation_BOS_value_range': [float((TW[192:, 0] @ field).min()), float((TW[192:, 0] @ field).max())]}
host_qn = torch.stack([norm(TH @ QH[i].T) for i in range(4)])
host_kn = norm(TH @ K.T)
qenergy = (host_qn[:, :192] * gh).square().mean((0, 1, 2))
kenergy = (host_kn[:192] * kg).square().mean((0, 1))
qe, ke = qenergy[:128] + qenergy[128:], kenergy[:128] + kenergy[128:]
qf = host_qn[:, :192].square().mean((0, 1, 2))
kf = host_kn[:192].square().mean((0, 1))
qf, kf = (qf[:128] + qf[128:]) / 256, (kf[:128] + kf[128:]) / 256
importance = qe * ke
importance += importance.mean() * (qf / qf.mean() + kf / kf.mean())
phase_gaps = 1 - (frequency[:, None] * torch.arange(1, 9, dtype=torch.float64)).cos().amax(1)
cost = importance.clone(); cost[phase_gaps < .02] = torch.inf
pairs = cost.argsort()[:5]
assert pairs.tolist() == [9, 15, 4, 17, 21]
key_multipliers = [1., 4., 16., 64., 256.]
query_multipliers = [1., 2., 4., 8., 16.]
trials, best, diagnostic = [], None, None
bm = json.loads(base.with_suffix('.manifest.json').read_text())
print(json.dumps({'baseline': baseline, 'field': field_report, 'pairs': pairs.tolist()}), flush=True)
if values.min() > 0:
    base_key_amplitude = 16 * float((rms(SX @ K.T)[..., 0] / values).max())
    for pair in pairs.tolist():
        channels = torch.tensor([pair, pair + 128])
        phase = (torch.tensor(positions, dtype=torch.float64) - 8) * frequency[pair]
        programme = torch.zeros_like(QH)
        programme[:, pair] = phase.cos()[:, None] * qnull
        programme[:, pair + 128] = phase.sin()[:, None] * qnull
        base_gain = max(float(gh[channels].max()), 8.25 / (16 * float(kg[pair]) * float(phase_gaps[pair])))
        for kmass in key_multipliers:
            key = K.clone(); key[pair] += base_key_amplitude * kmass * field
            key = key.float().double()
            SK = norm(SX @ key.T) * kg
            KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
            rotated_keys = rotate(norm(TW @ key.T) * kg)
            key_rms = float((rotated_keys[192:] - host_keys[192:]).square().mean().sqrt())
            for qmass in query_multipliers:
                gain = gh.clone(); gain[channels] = base_gain * qmass
                gain_weight = (gain - 1).float(); gain = 1 + gain_weight.double()
                native = B * (gh / gain)[None, :, None]
                factors, routing = source_amplitudes(native, programme, gain)
                row = {'pair': pair, 'key_amplitude_multiplier': kmass, 'query_gain_multiplier': qmass,
                    'key_amplitude': base_key_amplitude * kmass, 'query_pair_gain': base_gain * qmass,
                    'ideal_phase_gap': float(phase_gaps[pair]), 'routing': routing,
                    'validation_paired_HOST_key_RMS': key_rms}
                if factors is None:
                    trials.append({**row, 'eligible': False}); continue
                query = (native + factors[:, None, None] * programme).float()
                actual = quality(query.double(), gain)
                matrix = query.reshape(1024, 640)
                tensors = {qn: matrix, gn: gain_weight, kn: key.float()}
                total = bm['changed_scalars']
                for name, tensor in tensors.items():
                    total -= bm['per_tensor'].get(name, 0)
                    total += int((tensor.view(torch.int32) != h[name].view(torch.int32)).sum())
                maxvalue = max(float(t.abs().max()) for t in tensors.values())
                row.update(actual, changed_scalars=total, maximum_query_key_or_gain_weight=maxvalue)
                row['eligible'] = bool(actual['validation_paired_HOST_causal_score_MSE'] <= .9 * baseline['validation_paired_HOST_causal_score_MSE']
                    and actual['validation_same_input_HOST_causal_score_MSE'] <= .9 * baseline['validation_same_input_HOST_causal_score_MSE']
                    and actual['validation_paired_HOST_query_RMS'] <= 1.05 * baseline['validation_paired_HOST_query_RMS']
                    and actual['minimum_source_routing_margin'] >= 7.999 and total <= 5016268
                    and all(t.isfinite().all() for t in tensors.values()) and maxvalue <= 1e6)
                trials.append(row)
                objective = actual['validation_paired_HOST_causal_score_MSE']
                if diagnostic is None or objective < diagnostic[0]: diagnostic = objective, row
                if row['eligible'] and (best is None or objective < best[0]):
                    best = objective, tensors, row
            print(json.dumps({'pair': pair, 'key_multiplier': kmass, 'last': trials[-1]}), flush=True)
else:
    base_key_amplitude = None
report = {'analysis_only': True, 'baseline': baseline, 'field': field_report,
    'base_key_amplitude': base_key_amplitude, 'pair_trials': pairs.tolist(),
    'key_multiplier_trials': key_multipliers, 'query_gain_multiplier_trials': query_multipliers,
    'field_variation_mass': 1., 'field_ridge': 1e-6, 'training_windows': 192, 'holdback_windows': 64,
    'trials': trials, 'eligible': best is not None, 'selected': None if best is None else best[-1],
    'best_diagnostic': None if diagnostic is None else diagnostic[-1],
    'target': 'Format presence and fixed source positions; original HOST native QK scores only; no answers',
    'scope': 'Linear/RMS/QK/RoPE on recorded pre-layer3 inputs; no model, softmax or native activation',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, original, reference, wp, ep, hp, qp, control_path] + historical_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
if best is not None:
    atomic_torch(ROOT / 'artifacts/a305v1_additive_rotary_key.pt', {'tensors': best[1], 'field': field, 'report': report})
atomic_json(ROOT / 'artifacts/a305v1_additive_rotary_key.json', report)
print(json.dumps({'eligible': report['eligible'], 'best': report['best_diagnostic'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
