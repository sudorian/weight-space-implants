"""Allocate query gains from realized fixed-key source margins.

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
def source_amplitudes(native, program, gain, target_margin):
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
        A = E.square() - target_margin**2 * aa; Bq = 2 * D * E - target_margin**2 * bb; Cq = D.square() - target_margin**2 * cc
        if A.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'Asymptotic source margin does not exceed target',
                          'minimum_asymptotic_source_margin': float((E / aa.clamp_min(1e-30).sqrt()).min())}
        discriminant = Bq.square() - 4 * A * Cq
        root = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
        factor = float(torch.maximum(root, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
        gap = (D + factor * E) / (cc + factor * bb + factor**2 * aa).sqrt()
        assert gap.min() >= target_margin - .001
        factors.append(factor); gaps.append(float(gap.min()))
    return torch.tensor(factors, dtype=torch.float64), {'programme_factors': factors, 'analytic_source_margins': gaps}

qp_new = ROOT / 'artifacts/a323v1_causal_metric_normalized_key.pt'
fixed = torch.load(qp_new, weights_only=True)
assert fixed['report']['selected']['eligible']
for path, digest in {**fixed['report']['sources'], **fixed['report']['inputs']}.items():
    assert sha(ROOT / path) == digest, path
pair = fixed['report']['selected']['pair']; assert pair == 9
K = fixed['tensors'][kn].double()
positions = [3, 2, 1, 4]
channels = torch.tensor([pair, pair + 128])
phase = (torch.tensor(positions, dtype=torch.float64) - 8) * frequency[pair]
programme = torch.zeros_like(QH)
programme[:, pair] = phase.cos()[:, None] * qnull
programme[:, pair + 128] = phase.sin()[:, None] * qnull
rotated_keys = rotate(norm(TW @ K.T) * kg)
SK = norm(SX @ K.T) * kg
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
unit_gaps = []
for head, position in enumerate(positions):
    direction = norm(SX[:, 8] @ programme[head].T)
    # Programme is nonzero only on the pair; all other gain components vanish.
    score = (KR * direction[:, None]).sum(-1) / 16
    others = torch.tensor([j for j in range(9) if j != position])
    unit_gaps.append(score[:, position] - score[:, others].amax(1))
unit_gaps = torch.stack(unit_gaps)
minimum_unit_gap = float(unit_gaps.min()); assert minimum_unit_gap > 0
margin_trials = [12., 16., 24., 32., 48.]
headroom = 1.25
bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials = []; best = None
for target_margin in margin_trials:
    query_gain = max(float(gh[channels].max()), headroom * target_margin / minimum_unit_gap)
    gain = gh.clone(); gain[channels] = query_gain
    gain_parameter = (gain - 1).float(); gain = 1 + gain_parameter.double()
    native = B * (gh / gain)[None, :, None]
    factors, routing = source_amplitudes(native, programme, gain, target_margin)
    row = {'target_source_margin': target_margin, 'analytic_pair_query_gain': query_gain,
           'routing': routing}
    if factors is None:
        trials.append({**row, 'eligible': False}); continue
    query = (native + factors[:, None, None] * programme).float()
    q = quality(query.double(), gain)
    tensors = {qn: query.reshape(1024, 640), gn: gain_parameter, kn: K.float()}
    total = bm['changed_scalars']
    for name, value in tensors.items():
        total += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'].get(name, 0)
    maximum = max(float(value.abs().max()) for value in tensors.values())
    row.update(q, changed_scalars=total, maximum_query_key_gain_weight=maximum)
    row['eligible'] = bool(total <= 5016268 and maximum <= 1e6 and all(t.isfinite().all() for t in tensors.values())
        and q['validation_paired_HOST_causal_score_MSE'] <= .9 * baseline['validation_paired_HOST_causal_score_MSE']
        and q['validation_same_input_HOST_causal_score_MSE'] <= .9 * baseline['validation_same_input_HOST_causal_score_MSE']
        and q['validation_paired_HOST_query_RMS'] <= 1.05 * baseline['validation_paired_HOST_query_RMS']
        and q['minimum_source_routing_margin'] >= target_margin - .001)
    trials.append(row); print(json.dumps(row), flush=True)
    metric = q['validation_paired_HOST_causal_score_MSE']
    if row['eligible'] and (best is None or metric < best[0]): best = metric, tensors, row
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'baseline': baseline, 'source_positions': positions, 'fixed_pair': pair,
    'minimum_realized_unit_gain_programme_gap': minimum_unit_gap,
    'target_margin_trials': margin_trials, 'fixed_asymptotic_headroom': headroom,
    'trials': trials, 'selected': None if best is None else best[-1],
    'A323_keys_unchanged': True, 'A326_decoder_and_composition_still_required': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/derive_a323_causal_metric_normalized_key_v1.py': sha(ROOT / 'experiments/derive_a323_causal_metric_normalized_key_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, original, reference, wp, ep, hp, qp, control_path, qp_new] + historical_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a328v1_realized_key_margin_allocation.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'tensors': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
