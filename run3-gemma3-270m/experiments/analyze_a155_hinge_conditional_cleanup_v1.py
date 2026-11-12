"""Compress the fixed cleanup operator conditional on all five emitter features.

This extends A154's covariance analysis to the existing three literal hinge
features. A low-rank cleanup approximation Y A B^T is accompanied by the exact
algebraic emitter correction X C (W^T-A B^T), where C is the projection of
recorded cleanup fields Y onto existing programme features X. There are no
problem/answer labels, new nonlinear feature evaluations, or native replays.
Ranks [0,2,3,4,5] and rank6 exact control are declared before evaluation.
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
bp = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
ch = expand(h, bp)
c = torch.load(bp.parent / 'circuit.pt', weights_only=True)
b = torch.load(fp, weights_only=True, mmap=True)
assert b['checkpoint_sha256'] == sha(ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt')
p = 'model.layers.17.'
gn, un, dn, pn = [p + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
units = c['units'].long()
G0, U0, W0 = ch[gn][units].double(), ch[un][units].double(), ch[dn][:, units].double()
assert len(units) == 11
assert torch.equal(G0[:2], G0[:1].expand(2, -1))
assert torch.equal(G0[5:], G0[:1].expand(6, -1))
F = b['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
F0 = F[:, units]
before = b['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
P = norm(before) * (1 + h[p + 'pre_feedforward_layernorm.weight'].double())
linear_up = P @ U0[:2].T
assert linear_up[:, 0].abs().min() > 0
common_factor = F0[:, 0] / linear_up[:, 0]
feature_identity = float((common_factor[:, None] * (P @ U0[5:].T) - F0[:, 5:]).norm() / F0[:, 5:].norm())
assert feature_identity < 1e-6
nt = 384 * 5
reference = F0[:, :1]
X, Y = F0[:, :5] / reference, F0[:, 5:] / reference
coefficient = torch.linalg.solve(X[:nt].T @ X[:nt], X[:nt].T @ Y[:nt])
residual_up = U0[5:]
residual_features = Y - X @ coefficient
cov = residual_features[:nt].T @ residual_features[:nt] / nt
ev, V = torch.linalg.eigh((cov + cov.T) / 2)
assert ev.min() > 0
root = (V * ev.sqrt()) @ V.T
inverse_root = (V / ev.sqrt()) @ V.T
gain = 1 + ch[pn].double()
fg = 1 + h['model.norm.weight'].double()
weighted_down = W0[:, 5:] * (gain * fg)[:, None]
metric = root @ weighted_down.T @ weighted_down @ root
pe, Q = torch.linalg.eigh((metric + metric.T) / 2)

raw0 = F @ ch[dn].double().T
expected = b['topk_ids'][:, :, 0].reshape(-1).long()
E = h['model.embed_tokens.weight']
def readout(raw):
    state = before + norm(raw) * gain
    output = (norm(state) * fg).float()
    desired = (output * E[expected]).sum(1)
    other = torch.full_like(desired, -torch.inf)
    for start in range(0, len(E), 4096):
        logits = output @ E[start:start + 4096].T
        local = expected - start
        rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        other = torch.maximum(other, logits.amax(1))
    m = desired - other
    return {'minimum_margin': float(m.min()), 'changed_winners': int((m <= 0).sum()),
            'training_minimum_margin': float(m[:nt].min()), 'validation_minimum_margin': float(m[nt:].min())}

out = ROOT / 'artifacts/a155v1_hinge_conditional_cleanup_geometry.json'
report = {'base': str(bp.relative_to(ROOT)), 'base_sha256': sha(bp), 'format_capture': str(fp.relative_to(ROOT)),
          'feature_identity_relative_error': feature_identity, 'conditional_coefficient': coefficient.tolist(),
          'residual_covariance_eigenvalues': ev.tolist(), 'weighted_operator_eigenvalues': pe.tolist(),
          'rank_trials': [0, 2, 3, 4, 5], 'control_rank': 6, 'training_windows': 384, 'validation_windows': 128,
          'baseline': readout(raw0), 'trials': [], 'selected': None,
          'no_answers_gradients_or_native_activation_replay': True}
choices = []
for rank in [0, 2, 3, 4, 5, 6]:
    if rank == 6:
        A, B = torch.eye(6, dtype=torch.float64), W0[:, 5:]
    else:
        q = Q[:, -rank:] if rank else torch.zeros(6, 0, dtype=torch.float64)
        A, B = inverse_root @ q, W0[:, 5:] @ root @ q
    base_emit = W0[:, :5] + (W0[:, 5:] - B @ A.T) @ coefficient.T
    Ur = A.T @ residual_up
    # Reciprocal column balancing changes no hand-built function.
    if rank:
        scale = (B.abs().amax(0).clamp_min(1e-30) / Ur.abs().amax(1).clamp_min(1e-30)).sqrt()
        Ur, B = Ur * scale[:, None], B / scale[None]
    Unew = torch.cat([U0[:5], Ur]).float()
    Wnew = torch.cat([base_emit, B], 1).float()
    Gnew = torch.cat([G0[:5], G0[:1].expand(rank, -1)]).float()
    Fnew = torch.cat([F0[:, :5], common_factor[:, None] * (P @ Unew[5:].double().T)], 1)
    raw = raw0 - F0 @ W0.T + Fnew @ Wnew.double().T
    row = {'rank': rank, 'programme_units': 5 + rank, **readout(raw),
           'maximum_gate_weight': float(Gnew.abs().max()), 'maximum_up_weight': float(Unew.abs().max()),
           'maximum_down_weight': float(Wnew.abs().max()),
           'relative_programme_raw_change': float((Fnew @ Wnew.double().T - F0 @ W0.T).norm() / (F0 @ W0.T).norm())}
    row['eligible'] = row['minimum_margin'] > 1 and max(row['maximum_gate_weight'], row['maximum_up_weight'], row['maximum_down_weight']) <= 1e6
    report['trials'].append(row)
    print(json.dumps(row), flush=True)
    if rank < 6 and row['eligible']:
        choices.append((rank, Gnew, Unew, Wnew, row))
    atomic_json(out, report)
if choices:
    rank, G, U, W, row = min(choices, key=lambda x: x[0])
    report['selected'] = row
    atomic_torch(out.with_suffix('.pt'), {'gate': G, 'up': U, 'down': W, 'report': report})
report['maximum_resident_set_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
atomic_json(out, report)
print(json.dumps({'selected': report['selected'], 'baseline': report['baseline']}), flush=True)
