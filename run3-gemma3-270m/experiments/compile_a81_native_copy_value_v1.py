"""Repair copier native values against matched, unlabeled HOST head contexts.

The attention mixtures are recovered by a linear inverse of recorded native
values and recorded attended values. No attention softmax, activation function,
model replay, gradients, answer labels, or iterative residual fitting is used.
Every candidate is one independent original-target ridge solve. The complete
physical programme input columns are preserved exactly.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a72v2_units_gate_hardened/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
reference_path = ROOT / 'experiments/a72v2_units_gate_hardened/delta.pt'
host_path = ROOT / 'artifacts/a81v1_copy_host.pt'
edited_path = ROOT / 'artifacts/a81v1_copy_edited.pt'
ch = expand(h, base)
reference = expand(h, reference_path)
for name in h:
    if any(name.startswith(f'model.layers.{layer}.') for layer in range(8)):
        assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
host = torch.load(host_path, weights_only=True, mmap=True)
edited = torch.load(edited_path, weights_only=True, mmap=True)
assert torch.equal(host['inputs'], edited['inputs'])
assert edited['checkpoint_sha256'] == sha(reference_path)
prefix = 'model.layers.7.'
vn = prefix + 'self_attn.v_proj.weight'
on = prefix + 'self_attn.o_proj.weight'
assert torch.equal(ch.get(vn, h[vn]), h[vn])
assert torch.equal(ch.get(on, h[on]), h[on])
V = h[vn].double()
O = h[on].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
assert torch.equal(ch.get(prefix + 'post_attention_layernorm.weight', h[prefix + 'post_attention_layernorm.weight']), h[prefix + 'post_attention_layernorm.weight'])

physical_mask = torch.zeros(640, dtype=torch.bool)
for layer in range(9):
    name = f'model.layers.{layer}.post_feedforward_layernorm.weight'
    physical_mask |= (1 + ch.get(name, h[name]).double()).abs() > 1000
physical = physical_mask.nonzero().flatten()
free = (~physical_mask).nonzero().flatten()
assert len(physical) == 31, physical.tolist()

X = edited['captures']['7.input'].double()
values = X @ V.T
attended = edited['captures']['7.heads'].double().reshape(len(X), 64, 4, 256).permute(0, 2, 1, 3)
teacher = host['captures']['7.heads'].double().reshape(len(X), 64, 4, 256).permute(0, 2, 1, 3)
gram = values @ values.transpose(1, 2)
ev, evec = torch.linalg.eigh(gram)
keep = ev > ev[:, -1:] * 1e-12
inverse = torch.where(keep, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
pinv = (evec * inverse[:, None]) @ evec.transpose(1, 2)
coeff = (attended @ values[:, None].transpose(-1, -2)) @ pinv[:, None]
recovered = coeff @ values[:, None]
relative_error = float((recovered - attended).square().mean().sqrt() / attended.square().mean().sqrt())
future = torch.ones(64, 64, dtype=torch.bool).triu(1)
future_max = float(coeff[:, :, future].abs().max())
input_null = X - gram @ pinv @ X
input_null_error = float(input_null.square().mean().sqrt() / X.square().mean().sqrt())
assert relative_error < 1e-5
assert future_max < 1e-3, future_max
assert input_null_error < 1e-5, input_null_error
mixtures = coeff @ X[:, None, :, free]
del values, recovered, input_null, gram, pinv, X

raw_before = edited['captures']['7.attn_raw'].double()
raw_reconstructed = torch.einsum('bhnd,ohd->bno', attended, O)
raw_reconstruction_rms = float((raw_reconstructed - raw_before).square().mean().sqrt())
assert raw_reconstruction_rms < 1e-4
before = edited['captures']['7.before'].double()
target_after = host['captures']['7.after'].double()
host_raw = host['captures']['7.attn_raw'].double()
host_rms = (host_raw.square().mean(-1) + 1e-6).sqrt()
target = teacher - attended
nt = 384
weight = host_rms.clamp_min(1e-6)

# Accumulate the original-target covariance in batches. There is one shared V
# across all four query heads; no separate per-head value matrices are added.
C = torch.zeros(len(free), len(free), dtype=torch.float64)
R = torch.zeros(len(free), 256, dtype=torch.float64)
count = nt * 4 * 64
for start in range(0, nt, 16):
    s = slice(start, min(start + 16, nt))
    z = (mixtures[s] / weight[s, None, :, None]).reshape(-1, len(free))
    y = (target[s] / weight[s, None, :, None]).reshape(-1, 256)
    C += z.T @ z / count
    R += z.T @ y / count
sd = C.diagonal().sqrt().clamp_min(1e-12)
C /= sd[:, None] * sd[None, :]
R /= sd[:, None]
ev, evec = torch.linalg.eigh((C + C.T) / 2)
ev = ev.clamp_min(0)
rhs = evec.T @ R

def quality(delta):
    heads = attended[nt:] + mixtures[nt:] @ delta[:, free].T
    raw = torch.einsum('bhnd,ohd->bno', heads, O)
    post = raw / (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gamma
    after = before[nt:] + post
    norm = target_after[nt:].square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    relative = (after - target_after[nt:]) / norm
    direction = after / after.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12) - target_after[nt:] / norm
    return {
        'validation_hidden_relative_rms': float(relative.square().mean().sqrt()),
        'validation_hidden_direction_rms': float(direction.square().mean().sqrt()),
        'validation_hidden_relative_p99': float(torch.quantile(relative.square().mean(-1).sqrt(), .99)),
        'validation_native_head_rms': float((heads - teacher[nt:]).square().mean().sqrt()),
        'validation_post_branch_rms': float((post - host['captures']['7.attn_post'][nt:].double()).square().mean().sqrt()),
    }

baseline = quality(torch.zeros_like(V))
print(json.dumps({'baseline': baseline, 'physical_columns': physical.tolist(),
                  'mixture_reconstruction_relative_rms': relative_error,
                  'maximum_recovered_future_coefficient': future_max,
                  'input_null_relative_rms': input_null_error}), flush=True)
trials = []
best = None
for ridge in (1e-6, 1e-4, .01, 1., 100.):
    fit = (evec @ (rhs / (ev[:, None] + ridge))) / sd[:, None]
    delta = torch.zeros_like(V)
    delta[:, free] = fit.T
    candidate = (V + delta).float()
    assert torch.equal(candidate[:, physical], h[vn][:, physical])
    actual_delta = candidate.double() - V
    q = quality(actual_delta)
    row = {'ridge': ridge, **q,
           'maximum_value_weight': float(candidate.abs().max()),
           'delta_frobenius_norm': float(actual_delta.norm()),
           'changed_value_scalars': int((candidate.view(torch.int32) != h[vn].view(torch.int32)).sum()),
           'eligible': bool(candidate.isfinite().all() and candidate.abs().max() <= 1e6)}
    trials.append(row)
    print(json.dumps(row), flush=True)
    if row['eligible'] and (best is None or q['validation_hidden_relative_rms'] < best[0]):
        best = q['validation_hidden_relative_rms'], candidate, row
assert best is not None
_, chosen, selected = best
report = {
    'target': 'Matched unlabeled HOST native attention head contexts',
    'ridge_trials': [1e-6, 1e-4, .01, 1., 100.],
    'baseline': baseline, 'trials': trials, 'selected': selected,
    'protected_physical_input_columns': physical.tolist(),
    'mixture_reconstruction_relative_rms': relative_error,
    'maximum_recovered_future_coefficient': future_max,
    'minimum_recovered_coefficient': float(coeff.min()),
    'maximum_recovered_row_sum_error': float((coeff.sum(-1) - 1).abs().max()),
    'input_null_relative_rms': input_null_error,
    'minimum_source_value_rank': int(keep.sum(1).min()),
    'maximum_source_value_rank': int(keep.sum(1).max()),
    'attention_raw_reconstruction_rms': raw_reconstruction_rms,
    'literal_register_value_mapping_bit_identical': True,
    'format_caveat': 'Non-programme source components and all downstream effects require official stress and score checks.',
    'no_model_replay_or_answer_targets': True,
}
assert selected['validation_hidden_relative_rms'] < baseline['validation_hidden_relative_rms'], 'Native value repair fails held-back paired hidden metric'
ch[vn] = chosen
c['global_native_copy_value_repair'] = report
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'native_value_repair.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
categories[vn] = {'native_text_repair': selected['changed_value_scalars']}
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             reference_path, host_path, edited_path], arguments=vars(a),
     description=__doc__, extra=report, categories=categories)
