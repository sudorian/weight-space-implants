"""Restore native A-reader query rows with an analytic routing bound.

The previous reader subtracts the format-query mean and rescales every native
query row for its RMS. This study restores ordinary rows and HOST gains,
retaining inverse-gain native numerators in the 64 routing channels. Five
mean-cancellation fractions and five selected-channel gain multipliers are
independent constructions. A quadratic inequality fixes each head's routing
amplitude. Evaluation uses Q/K/RMS/RoPE arithmetic only, never attention or a
nonlinear native feature replay. Source labels are positions, not sums.
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
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
mp = ROOT / 'experiments/a45v2_query_energy/circuit.pt'
wp = ROOT / 'artifacts/a119v1_current_word.pt'
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
fps = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
metadata = torch.load(mp, weights_only=True)
selected = torch.tensor(metadata['native_key_query_layer3']['selected_channels'])
positions = metadata['native_key_query_layer3']['source_positions']
assert positions == [3, 2, 1, 0] and len(selected) == 64
ordinary = torch.ones(256, dtype=torch.bool); ordinary[selected] = False
qfull = torch.load(qp, weights_only=True)['query'].double()
old = torch.load(wp, weights_only=True, mmap=True)
early = torch.load(ep, weights_only=True, mmap=True)
branches = torch.load(bp, weights_only=True, mmap=True)
assert early['checkpoint_sha256'] == branches['checkpoint_sha256'] == sha(ROOT / 'experiments/a123v1_prior_native_final/delta.pt')
assert torch.equal(early['inputs'], branches['inputs'])
for reference in [ROOT / 'experiments/a117v1_six_unit_carry/delta.pt', ROOT / 'experiments/a123v1_prior_native_final/delta.pt']:
    state = expand(h, reference)
    for name in h:
        if any(name.startswith(f'model.layers.{i}.') for i in range(3)):
            assert torch.equal(ch.get(name, h[name]), state.get(name, h[name])), name
    del state
def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
derived = {}
identity_errors = {}
for tag in ['word', 'format']:
    state2 = early['captures'][f'{tag}.2.raw'] + norm(branches['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    state3 = state2 + norm(branches['captures'][f'{tag}.3.attention.raw']) * (1 + ch.get('model.layers.3.post_attention_layernorm.weight', h['model.layers.3.post_attention_layernorm.weight']))
    identity_errors[tag] = float((state3 - early['captures'][f'{tag}.3.raw']).abs().max())
    assert identity_errors[tag] == 0, (tag, identity_errors[tag])
    derived[tag] = norm(state2) * (1 + h['model.layers.3.input_layernorm.weight'])
SXold = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in fps]).double()
mean_input = SXold[:1024, 8].mean(0)
qnull = qfull / (mean_input @ qfull)
SX = torch.cat([SXold, derived['format'][:, :9].double()])
word = torch.cat([old['captures']['3.input'][384:], derived['word'][128:]]).double()
del old, early, branches, derived
pre = 'model.layers.3.self_attn.'
qn, qgn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
Qh, Qold = h[qn].double().reshape(4, 256, 640), ch[qn].double().reshape(4, 256, 640)
Gh, Gold = 1 + h[qgn].double(), 1 + ch[qgn].double()
K, Gk = h[kn].double(), 1 + h[kgn].double()
mean_q = torch.einsum('c,hjc->hj', mean_input, Qh)
B = Qh - mean_q[:, :, None] * qnull
orth = B - (B @ qfull)[:, :, None] * qfull / qfull.square().sum()
coefficient = (Qold * orth).sum((0, 2)) / (B * orth).sum((0, 2))
residual = Qold - coefficient[None, :, None] * B
amplitude = (residual @ qfull) / qfull.square().sum()
amplitude[:, ordinary] = 0
Pquery = amplitude[:, :, None] * qfull
decomposition_error = float((residual - Pquery).abs().max())
assert decomposition_error < 1e-6
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(x):
    angles = torch.arange(x.shape[-2], dtype=torch.float64)[:, None] * freq
    return x * angles.cos().repeat(1, 2) + torch.cat([-x[..., 128:], x[..., :128]], -1) * angles.sin().repeat(1, 2)
keys = rotate(norm(word @ K.T) * Gk)
hostq = rotate((norm(torch.einsum('ntc,hjc->nthj', word, Qh)) * Gh).permute(0, 2, 1, 3))
host_scores = hostq @ keys[:, None].transpose(-1, -2) / 16
del hostq
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)
def quality(Q, G):
    q = rotate((norm(torch.einsum('ntc,hjc->nthj', word, Q)) * G).permute(0, 2, 1, 3))
    scores = q @ keys[:, None].transpose(-1, -2) / 16
    e = scores - host_scores
    means = (e * causal).sum(-1) / counts
    costs = ((e - means[..., None]).square() * causal).sum(-1) / counts
    return {'centered_causal_score_mse': float(costs.mean()), 'old_validation_mse': float(costs[:128].mean()),
            'fresh_validation_mse': float(costs[128:].mean())}
angle = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
sk = norm(SX @ K.T) * Gk
KR = sk * angle.cos().repeat(1, 2)[None] + torch.cat([-sk[:, :, 128:], sk[:, :, :128]], -1) * angle.sin().repeat(1, 2)[None]
Pr = torch.einsum('nc,hjc->nhj', SX[:, 8], Pquery)
def margins(Q, G):
    q = norm(torch.einsum('nc,hjc->nhj', SX[:, 8], Q)) * G
    result = []
    for head, pos in enumerate(positions):
        other = torch.tensor([i for i in range(9) if i != pos])
        scores = (KR * q[:, head, None]).sum(-1) / 16
        result.append(float((scores[:, pos] - scores[:, other].amax(1)).min()))
    return result
def route(native, gain):
    Nr = torch.einsum('nc,hjc->nhj', SX[:, 8], native)
    factors = []
    for head, pos in enumerate(positions):
        other = torch.tensor([i for i in range(9) if i != pos])
        kd = KR[:, pos, None] - KR[:, other]
        D = (kd * (Nr[:, head] * gain)[:, None]).sum(-1) / 16
        E = (kd * (Pr[:, head] * gain)[:, None]).sum(-1) / 16
        aa = Pr[:, head].square().mean(1, keepdim=True)
        bb = 2 * (Nr[:, head] * Pr[:, head]).mean(1, keepdim=True)
        cc = Nr[:, head].square().mean(1, keepdim=True) + 1e-6
        A, Bq, Cq = E.square() - 64 * aa, 2 * D * E - 64 * bb, D.square() - 64 * cc
        if A.min() <= 0 or E.min() <= 0:
            return None, {'reason': 'programme-only asymptotic margin below8', 'head': head}
        disc = Bq.square() - 4 * A * Cq
        root = torch.where(disc > 0, (-Bq + disc.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
        factor = float(torch.maximum(root, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
        factors.append(factor)
    return torch.tensor(factors, dtype=torch.float64), {}

baseline = quality(Qold, Gold)
out = ROOT / 'artifacts/a157v1_native_query_restoration.json'
report = {'baseline': baseline, 'baseline_routing_margins': margins(Qold, Gold), 'recorded_state_identity_max_errors': identity_errors,
          'current_query_decomposition_max_error': decomposition_error, 'mean_cancellation_fraction_trials': [0, .25, .5, .75, 1],
          'selected_gain_multiplier_trials': [1, 1.25, 1.5, 2, 4], 'routing_margin': 8,
          'old_format_examples': len(SXold), 'exact_current_format_examples': len(SX) - len(SXold),
          'word_validation_windows': len(word), 'trials': [], 'selected': None,
          'no_attention_replay_gradients_or_sum_targets': True,
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, qp, mp, wp, ep, bp] + fps}}
print(json.dumps({'baseline': baseline, 'baseline_routing_margins': report['baseline_routing_margins']}), flush=True)
best = None
for fraction in [0., .25, .5, .75, 1.]:
    for multiplier in [1., 1.25, 1.5, 2., 4.]:
        gain = Gh.clone(); gain[selected] = Gold[selected] * multiplier
        native = (Qh - fraction * mean_q[:, :, None] * qnull) * (Gh / gain)[None, :, None]
        factors, issue = route(native, gain)
        row = {'mean_cancellation_fraction': fraction, 'selected_gain_multiplier': multiplier, 'eligible': False, **issue}
        if factors is not None:
            Q = (native + factors[:, None, None] * Pquery).float()
            parameter = (gain - 1).float(); actual = 1 + parameter.double()
            row.update(quality(Q.double(), actual), routing_margins=margins(Q.double(), actual),
                       analytic_program_factors=factors.tolist(), maximum_query_weight=float(Q.abs().max()),
                       maximum_norm_weight=float(parameter.abs().max()),
                       changed_query_scalars=int((Q.reshape(1024, 640).view(torch.int32) != h[qn].view(torch.int32)).sum()))
            row['eligible'] = min(row['routing_margins']) > 7.999 and max(row['maximum_query_weight'], row['maximum_norm_weight']) <= 1e6
            if row['eligible'] and (best is None or row['centered_causal_score_mse'] < best[0]):
                best = row['centered_causal_score_mse'], Q.clone(), parameter.clone(), row
        report['trials'].append(row); print(json.dumps(row), flush=True); atomic_json(out, report)
if best is not None:
    report['selected'] = best[-1]
    row = best[-1]
    report['passes_static_screen'] = row['centered_causal_score_mse'] < .95 * baseline['centered_causal_score_mse'] and row['fresh_validation_mse'] < baseline['fresh_validation_mse']
    if report['passes_static_screen']:
        atomic_torch(out.with_suffix('.pt'), {'query': best[1].reshape(1024, 640), 'norm': best[2], 'report': report})
report['maximum_resident_set_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
atomic_json(out, report)
print(json.dumps({'selected': report['selected'], 'passes_static_screen': report.get('passes_static_screen', False)}), flush=True)
