"""Choose five native value directions using actual normalized word outputs.

Recover recorded attention mixtures by a linear inverse, then derive a native
rank-251 value/output factorization from the frozen host. Five raw-energy
penalties and a single eigendecomposition per penalty are evaluated against
unlabeled native attention outputs. Literal program channels, queries, keys,
receiver geometry, post-attention gains and all other weights stay fixed.
No model replay, attention softmax, GELU, derivatives, answers or iterative fit.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a34v1_gate_margin/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
copy_path = ROOT / 'experiments/a29v1_frame_copy/circuit.pt'
c = torch.load(copy_path, weights_only=True)
path = ROOT / 'artifacts/a29v1_copy_calib256.pt'
cap = torch.load(path, weights_only=True, mmap=True)
prefix = 'model.layers.7.self_attn.'
O0 = h[prefix + 'o_proj.weight'].double().reshape(640, 4, 256)
V0 = h[prefix + 'v_proj.weight'].double()
old_O = ch[prefix + 'o_proj.weight'].double().reshape(640, 4, 256)
old_V = ch[prefix + 'v_proj.weight'].double()
channels = c['value_channels'].long()
F, projection = c['frame'].double(), c['projection'].double()
gamma = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
assert torch.equal(ch.get('model.layers.7.post_attention_layernorm.weight', h['model.layers.7.post_attention_layernorm.weight']),
                   h['model.layers.7.post_attention_layernorm.weight'])
X = cap['captures']['7.input'].double()
old_heads = cap['captures']['7.attn_in'].double().reshape(*X.shape[:2], 4, 256)
source = X @ old_V.T
gram = source @ source.transpose(1, 2)
ev, U = torch.linalg.eigh(gram)
threshold = ev[:, -1:] * 1e-12
inverse = torch.where(ev > threshold, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
pinv = (U * inverse[:, None]) @ U.transpose(1, 2)
coeff = (old_heads.permute(0, 2, 1, 3) @ source[:, None].transpose(-1, -2)) @ pinv[:, None]
reconstructed = coeff @ source[:, None]
recovery = {'relative_attended_value_reconstruction_rms': float(
    (reconstructed - old_heads.permute(0, 2, 1, 3)).square().mean().sqrt() / old_heads.square().mean().sqrt()),
    'minimum_source_rank': int((ev > threshold).sum(1).min()), 'gram_relative_cutoff': 1e-12}
assert recovery['relative_attended_value_reconstruction_rms'] < 1e-5
assert recovery['minimum_source_rank'] == 64
native_heads = (coeff @ (X @ V0.T)[:, None]).permute(0, 2, 1, 3).reshape(-1, 4, 256)
old_heads = old_heads.reshape(-1, 4, 256)
def raw_output(heads, O):
    return heads.reshape(-1, 1024) @ O.reshape(640, 1024).T
def normalized_output(raw):
    return raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
native_raw = raw_output(native_heads, O0)
target = normalized_output(native_raw)
old_raw = raw_output(old_heads, old_O)
recorded = cap['captures']['7.attn'].double().reshape(-1, 640)
recovery['relative_output_reconstruction_rms'] = float((old_raw - recorded).square().mean().sqrt() / recorded.square().mean().sqrt())
assert recovery['relative_output_reconstruction_rms'] < 1e-5
program_O = torch.zeros_like(old_O)
program_O[:, :, channels] = old_O[:, :, channels]
program_raw = raw_output(old_heads, program_O)
projected_O = torch.einsum('de,ehv->dhv', projection, O0)
nt = 192 * 64
def error_metrics(raw):
    error = normalized_output(raw)[nt:] - target[nt:]
    return {'validation_branch_rms': float(error.square().mean().sqrt()),
            'validation_p99_token_rms': float(error.square().mean(1).sqrt().quantile(.99))}
baseline = error_metrics(old_raw)
projection_only = error_metrics(raw_output(native_heads, projected_O))
old_native_only = error_metrics(old_raw - program_raw)
denominator = (native_raw[:nt].square().mean(1) + 1e-6).sqrt()
normalized_heads = (native_heads[:nt] / denominator[:, None, None]).reshape(-1, 256)
C = normalized_heads.T @ normalized_heads / len(normalized_heads)
C.diagonal().add_(1e-6 * C.diagonal().mean())
ev, U = torch.linalg.eigh(C)
assert ev.min() > 0
Cr, Ci = (U * ev.sqrt()) @ U.T, (U * ev.rsqrt()) @ U.T
del normalized_heads, X, source, gram, pinv, coeff, reconstructed, cap, recorded
trials, choices = [], []
for mass in (.0001, .001, .01, .1, 1.):
    metric = gamma.square() + mass * gamma.square().mean()
    S = sum(projected_O[:, j].T @ (metric[:, None] * projected_O[:, j]) for j in range(4))
    importance, rotation = torch.linalg.eigh(Cr @ S @ Cr)
    importance, rotation = importance.flip(0), rotation.flip(1)
    R, Rinv = Ci @ rotation, Cr @ rotation
    Vn = R.T @ V0
    On = torch.stack([projected_O[:, j] @ Rinv for j in range(4)], 1)
    exact_basis_error = max(float((On[:, j] @ Vn - projected_O[:, j] @ V0).abs().max()) for j in range(4))
    assert exact_basis_error < 1e-8
    On[:, :, channels] = 0
    native_raw_candidate = raw_output(native_heads @ R, On)
    row = {'raw_metric_mass': mass, **error_metrics(native_raw_candidate + program_raw),
           'native_only': error_metrics(native_raw_candidate),
           'removed_importance_fraction': float(importance[channels].sum() / importance.sum()),
           'exact_basis_error': exact_basis_error,
           'native_frame_response': float(torch.einsum('da,dhv->ahv', F * gamma[:, None], On).abs().max())}
    assert row['native_frame_response'] < 1e-8
    Vn[channels] = old_V[channels]
    On[:, :, channels] = old_O[:, :, channels]
    Vf, Of = Vn.float(), On.reshape(640, 1024).float()
    row['max_abs_value_weight'] = float(Vf.abs().max())
    row['max_abs_output_weight'] = float(Of.abs().max())
    assert max(row['max_abs_value_weight'], row['max_abs_output_weight']) <= 1e6
    trials.append(row)
    choices.append((Vf, Of, R, Rinv))
    print(json.dumps(row), flush=True)
best = min(range(len(trials)), key=lambda i: trials[i]['validation_branch_rms'])
Vf, Of, R, Rinv = choices[best]
ch[prefix + 'v_proj.weight'] = Vf
ch[prefix + 'o_proj.weight'] = Of
assert torch.equal(Vf[channels].view(torch.int32), old_V.float()[channels].view(torch.int32))
assert torch.equal(Of.reshape(640, 4, 256)[:, :, channels].view(torch.int32), old_O.float()[:, :, channels].view(torch.int32))
report = {'covariance_source': 'Actual native value mixtures divided by native raw attention RMS',
          'raw_metric_mass_trials': [.0001, .001, .01, .1, 1.], 'covariance_ridge_fraction': 1e-6,
          'retained_native_rank': 251, 'program_channels': channels.tolist(),
          'attention_recovery': recovery, 'baseline': baseline,
          'projection_only': projection_only, 'old_native_only': old_native_only,
          'trials': trials, 'selected': trials[best],
          'program_value_output_columns_bit_identical': True,
          'no_arithmetic_targets_or_model_replay': True}
state = torch.load(base.parent / 'circuit.pt', weights_only=True)
state['copy_native_basis_repair'] = report
state['copy_native_basis_R'] = R
state['copy_native_basis_Rinv'] = Rinv
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', copy_path, path],
     arguments=vars(a), description=__doc__, extra=report, categories=bm['categories'])
print(json.dumps({'baseline': baseline, 'projection_only': projection_only,
                  'old_native_only': old_native_only, 'selected': trials[best]}, indent=2), flush=True)
