"""Replace a paired continuation anchor by one format-only linear field.

Only position labels, existing linear reference fields and hand-programme
ReLU ideals enter the analysis. Native nonlinearities are never evaluated.
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
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ch, ref = expand(h, base), expand(h, reference)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
w = torch.load(wp, weights_only=True, mmap=True)
assert w['checkpoint_sha256'] == sha(reference)
meta = c['layer6_shared_phase_frame']
units = torch.tensor(meta['programme_units'])
assert units.tolist() == [1429, 1122, 1820, 1092, 276, 1808, 314, 538, 1731, 91]
prefix = 'model.layers.6.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                     'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
# The captured trajectory has these exact programme functions even though
# its earlier attention and native Down repairs differ from the current base.
for name in (gn, un): assert torch.equal(ch[name][units], ref[name][units]), name
assert torch.equal(ch[dn][:, units], ref[dn][:, units])
assert torch.equal(ch[pn], ref[pn])
G0, U0, D0 = ch[gn][units].double(), ch[un][units].double(), ch[dn][:, units].double()
gain = 1 + ch[pn].double()
pgain = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
width = meta['selected_phase_gate']['width']
assert torch.equal(G0[0], U0[3]) and torch.equal(U0[3], U0[4])
assert torch.equal(D0[:, 3], -D0[:, 4])
common, flag = G0[0], U0[0]
reference_field = (G0[3] - G0[4]) / (2 * width)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
Pfull = (fnorm(w['captures']['format.6.raw']) * pgain.float()).double()
P = Pfull[:, 8:14]
N = torch.cat([(fnorm(w['captures']['word.6.raw']) * pgain.float()).double().reshape(-1, 640),
               Pfull[:, :8].reshape(-1, 640)])
E, C = P[:, 0], P[:, 1:].reshape(-1, 640)
ef, cr = E @ flag, C @ reference_field
assert ef.min() > 0 and cr.min() > 0
Es, Cs = E / ef[:, None], C / cr[:, None]
Et, Ct = Es[:96], Cs[:96 * 5]
standard = torch.cat([Et, Ct]).square().mean(0).sqrt().clamp_min(1e-12)
_, singular, vt = torch.linalg.svd(Et / standard, full_matrices=True)
rank = int((singular > singular[0] * 1e-12).sum())
basis = vt[rank:].T
A = (Ct / standard) @ basis
cov = A.T @ A / len(A)
rhs = A.mean(0)
ev, vec = torch.linalg.eigh((cov + cov.T) / 2)
ev = ev.clamp_min(0)
ridge_scale = float(cov.diagonal().mean())
flat = P.reshape(-1, 640)
old_raw = ((flat @ G0.T).clamp_min(0) * (flat @ U0.T)) @ D0.T
old_post = norm(old_raw) * gain
denominator = (old_post.square().mean(1, keepdim=True) + 1e-6).sqrt()
negative_max = float((N @ common).max())
positive_min = float((flat @ common).min())
# New order: original0..2, one continuation function, original5..9.
keep = torch.tensor([0, 1, 2, 5, 6, 7, 8, 9])
G = torch.cat([G0[:3], common[None], G0[5:]]).float()
D = torch.cat([D0[:, :3], (2 * width * D0[:, 3])[:, None], D0[:, 5:]], 1).float()
trials = []; best = None
for ridge in [1e-6, 1e-5, 1e-4, .001, .01]:
    coef = vec @ ((vec.T @ rhs) / (ev + ridge * ridge_scale))
    field = (basis @ coef / standard).float()
    U = torch.cat([U0[:3], field.double()[None], U0[5:]]).float()
    e_error = Es @ field.double()
    c_error = Cs @ field.double() - 1
    new_raw = ((flat @ G.double().T).clamp_min(0) * (flat @ U.double().T)) @ D.double().T
    post_error = (norm(new_raw) * gain - old_post) / denominator
    q = {'ridge': ridge, 'maximum_source_reference_leakage': float(e_error.abs().max()),
         'maximum_continuation_reference_relative_error': float(c_error.abs().max()),
         'holdback_source_reference_leakage': float(e_error[96:].abs().max()),
         'holdback_continuation_reference_relative_error': float(c_error[96 * 5:].abs().max()),
         'training_source_reference_leakage': float(e_error[:96].abs().max()),
         'programme_post_relative_rms_change': float(post_error.square().mean().sqrt()),
         'maximum_programme_state_relative_rms_change': float(post_error.square().mean(1).sqrt().max()),
         'maximum_weight': max(float(t.abs().max()) for t in (G, U, D)),
         'maximum_negative_common_gate': negative_max, 'minimum_positive_common_gate': positive_min}
    q['eligible_geometry'] = bool(q['maximum_source_reference_leakage'] <= .001
        and q['maximum_continuation_reference_relative_error'] <= .001
        and q['programme_post_relative_rms_change'] <= .001
        and q['maximum_programme_state_relative_rms_change'] <= .003
        and negative_max < -16 and positive_min > 16 and q['maximum_weight'] <= 1e6
        and all(t.isfinite().all() for t in (G, U, D)))
    trials.append(q); print(json.dumps(q), flush=True)
    metric = max(q['maximum_source_reference_leakage'], q['maximum_continuation_reference_relative_error'])
    if q['eligible_geometry'] and (best is None or metric < best[0]): best = metric, U, q
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'format_reference': str(reference.relative_to(ROOT)), 'training_windows': 96, 'holdback_windows': 32,
    'source_null_rank': rank, 'source_null_relative_tolerance': 1e-12, 'programme_unit_count': 9,
    'ridges': [1e-6, 1e-5, 1e-4, .001, .01], 'ridge_covariance_scale': ridge_scale,
    'trials': trials, 'selected': None if best is None else best[-1],
    'target': 'Existing linear continuation reference at positions9..13, zero at equals8; no answer target',
    'caveat': 'Earlier-trajectory inputs; programme-only ReLU geometry, native contribution and global GELU bounds not yet evaluated',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, base.parent / 'circuit.pt', reference, wp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a317v1_source_null_continuation.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'programme_gate': G, 'programme_up': best[1],
    'programme_down': D, 'report': report})
print(json.dumps({'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
