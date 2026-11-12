"""Replace only the A-reader's programmed BOS query with a plus query.

Three heads still read individual A digits. The fourth programmed head now
targets plus, which is present before every addition even after a prefix.
The native query component and all other parameters stay fixed. Five direct
source-difference ridge solves produce directions, and one quadratic bound
per direction determines its routing amplitude. Only recorded Q/K/RMS/RoPE
algebra runs; no attention probabilities or new hidden states are evaluated.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
mp = ROOT / 'experiments/a45v2_query_energy/circuit.pt'
wp = ROOT / 'artifacts/a119v1_current_word.pt'
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
fps = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
metadata = torch.load(mp, weights_only=True)
selected = torch.tensor(metadata['native_key_query_layer3']['selected_channels'])
assert metadata['native_key_query_layer3']['source_positions'] == [3, 2, 1, 0]
assert len(selected) == 64
ordinary = torch.ones(256, dtype=torch.bool); ordinary[selected] = False
qfull = torch.load(qp, weights_only=True)['query'].double()
old = torch.load(wp, weights_only=True, mmap=True)
early = torch.load(ep, weights_only=True, mmap=True)
branches = torch.load(bp, weights_only=True, mmap=True)
older_base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
assert early['checkpoint_sha256'] == branches['checkpoint_sha256'] == sha(older_base)
assert torch.equal(early['inputs'], branches['inputs'])
for reference in [ROOT / 'experiments/a117v1_six_unit_carry/delta.pt', older_base]:
    ref = expand(h, reference)
    for name in h:
        if any(name.startswith(f'model.layers.{i}.') for i in range(3)):
            assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref

def norm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

derived = {}; identity = {}
for tag in ('word', 'format'):
    state2 = early['captures'][f'{tag}.2.raw'] + norm(branches['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    state3 = state2 + norm(branches['captures'][f'{tag}.3.attention.raw']) * (1 + h['model.layers.3.post_attention_layernorm.weight'])
    identity[tag] = float((state3 - early['captures'][f'{tag}.3.raw']).abs().max())
    assert identity[tag] == 0
    derived[tag] = norm(state2) * (1 + h['model.layers.3.input_layernorm.weight'])
source_old = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in fps]).double()
mean_input = source_old[:1024, 8].mean(0)
qnull = qfull / (mean_input @ qfull)
sources = torch.cat([source_old, derived['format'][:, :9].double()])
words = torch.cat([old['captures']['3.input'][384:], derived['word'][128:]]).double()
del old, early, branches, derived
prefix = 'model.layers.3.self_attn.'
qn, qgn, kn, kgn = [prefix + name for name in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
Qh = h[qn].double().reshape(4, 256, 640)
Q0 = ch[qn].double().reshape(4, 256, 640)
G = 1 + ch[qgn].double(); Gh = 1 + h[qgn].double()
K, Gk = h[kn].double(), 1 + h[kgn].double()
assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
mean_q = torch.einsum('c,hjc->hj', mean_input, Qh)
B = Qh - mean_q[:, :, None] * qnull
orth = B - (B @ qfull)[:, :, None] * qfull / qfull.square().sum()
coefficient = (Q0 * orth).sum((0, 2)) / (B * orth).sum((0, 2))
native = coefficient[None, :, None] * B
residual = Q0 - native
amplitude = (residual @ qfull) / qfull.square().sum(); amplitude[:, ordinary] = 0
program = amplitude[:, :, None] * qfull
decomposition_error = float((residual - program).abs().max())
assert decomposition_error < 1e-6
frequency = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def rotate(x):
    angles = torch.arange(x.shape[-2], dtype=torch.float64)[:, None] * frequency
    return x * angles.cos().repeat(1, 2) + torch.cat([-x[..., 128:], x[..., :128]], -1) * angles.sin().repeat(1, 2)

keys = rotate(norm(words @ K.T) * Gk)
hostq = rotate((norm(torch.einsum('ntc,hjc->nthj', words, Qh)) * Gh).permute(0, 2, 1, 3))
host_scores = hostq @ keys[:, None].transpose(-1, -2) / 16
del hostq
causal = torch.ones(64, 64, dtype=torch.float64).tril(); counts = causal.sum(-1)

def quality(Q):
    q = rotate((norm(torch.einsum('ntc,hjc->nthj', words, Q)) * G).permute(0, 2, 1, 3))
    scores = q @ keys[:, None].transpose(-1, -2) / 16
    error = scores - host_scores
    mean = (error * causal).sum(-1) / counts
    costs = ((error - mean[..., None]).square() * causal).sum(-1) / counts
    return {'centered_causal_score_mse': float(costs.mean()),
            'old_validation_mse': float(costs[:128].mean()),
            'fresh_validation_mse': float(costs[128:].mean())}

angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * frequency
sk = norm(sources @ K.T) * Gk
KR = sk * angles.cos().repeat(1, 2)[None] + torch.cat([-sk[:, :, 128:], sk[:, :, :128]], -1) * angles.sin().repeat(1, 2)[None]
positions = [3, 2, 1, 4]

def margins(Q, source_positions):
    q = norm(torch.einsum('nc,hjc->nhj', sources[:, 8], Q)) * G
    result = []
    for head, position in enumerate(source_positions):
        other = torch.tensor([i for i in range(9) if i != position])
        scores = (KR * q[:, head, None]).sum(-1) / 16
        result.append(float((scores[:, position] - scores[:, other].amax(1)).min()))
    return result

other_positions = torch.tensor([i for i in range(9) if i != 4])
differences = KR[:, 4, None] - KR[:, other_positions]
source_train = torch.cat([torch.arange(1024), torch.arange(1536, 1536 + 96)])
X = (differences[source_train][:, :, selected] * G[selected]).reshape(-1, 64) / 16
sd = X.square().mean(0).sqrt().clamp_min(1e-12); Z = X / sd
C = Z.T @ Z / len(Z); rhs = Z.mean(0)
ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
native_source = sources[:, 8] @ native[3].T
quiet_source = sources[:, 8] @ qfull
baseline = quality(Q0)
ridges = [1e-6, 1e-4, .01, 1., 100.]
trials, best = [], None
for ridge in ridges:
    direction = (V @ ((V.T @ rhs) / (ev + ridge))) / sd
    vector = torch.zeros(256, dtype=torch.float64); vector[selected] = direction
    pr = quiet_source[:, None] * vector
    D = (differences * (native_source * G)[:, None]).sum(-1) / 16
    E = (differences * (pr * G)[:, None]).sum(-1) / 16
    aa = pr.square().mean(1, keepdim=True)
    bb = 2 * (native_source * pr).mean(1, keepdim=True)
    cc = native_source.square().mean(1, keepdim=True) + 1e-6
    A = E.square() - 64 * aa; Bq = 2 * D * E - 64 * bb; Cq = D.square() - 64 * cc
    row = {'source_difference_ridge': ridge, 'eligible': False}
    if A.min() <= 0 or E.min() <= 0:
        row.update(reason='Programmed plus direction lacks an asymptotic routing margin8',
                   minimum_direction_difference=float(E.min()))
        trials.append(row); print(json.dumps(row), flush=True); continue
    discriminant = Bq.square() - 4 * A * Cq
    root = torch.where(discriminant > 0, (-Bq + discriminant.clamp_min(0).sqrt()) / (2 * A), torch.zeros_like(A))
    factor = float(torch.maximum(root, -D / E).clamp_min(0).max()) * (1 + 1e-5) + 1e-7
    Q = Q0.clone(); Q[3] = (native[3] + factor * vector[:, None] * qfull).float().double()
    q = quality(Q); route = margins(Q, positions)
    row.update(q, routing_margins=route, analytic_programme_factor=factor,
               maximum_query_weight=float(Q.abs().max()))
    row['eligible'] = bool(min(route) > 7.999 and Q.abs().max() <= 1e6
        and q['centered_causal_score_mse'] < 1.1 * baseline['centered_causal_score_mse'])
    trials.append(row); print(json.dumps(row), flush=True)
    if row['eligible'] and (best is None or q['centered_causal_score_mse'] < best[0]):
        best = q['centered_causal_score_mse'], Q.float().reshape(1024, 640), row
report = {'analysis_only': True, 'baseline': baseline,
    'baseline_source_positions': [3, 2, 1, 0], 'candidate_source_positions': positions,
    'baseline_routing_margins': margins(Q0, [3, 2, 1, 0]),
    'source_difference_ridge_trials': ridges, 'routing_margin': 8,
    'native_query_component_and_gains_unchanged': True,
    'decomposition_error': decomposition_error, 'recorded_state_identity_errors': identity,
    'old_format_windows': 1536, 'current_format_windows': 128,
    'source_direction_training_windows': len(source_train),
    'no_prefix_success_claim': True, 'frame_replacement_still_required': True,
    'trials': trials, 'selected': None if best is None else best[-1],
    'no_attention_replay_gradients_or_sum_targets': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, qp, mp, wp, ep, bp] + fps},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a185v1_plus_reference_query.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'query': best[1], 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
