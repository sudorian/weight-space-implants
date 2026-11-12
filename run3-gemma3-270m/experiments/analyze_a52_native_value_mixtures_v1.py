"""Recover recorded attention mixtures and native-value repair covariances.

Only linear inverses and arithmetic on captured activations are used. The
reference is HOST V at the same recorded attention coefficients, not a replay
of host attention. No softmax, GELU, answer labels, derivatives or fitting.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a49v1_strong_final/delta.pt')
p.add_argument('--capture', default='artifacts/a52v1_value_word.pt')
p.add_argument('--output-prefix', default='a52_value_mixtures_v1')
a = p.parse_args()
h = host_state()
base, cap_path = ROOT / a.base, ROOT / a.capture
ch = expand(h, base)
o = torch.load(cap_path, weights_only=True, mmap=True)
geometry_path = ROOT / 'artifacts/a45_value_subspace_g1_v1.pt'
code_path = ROOT / 'artifacts/a45_source_codes_v1.pt'
geometry = torch.load(geometry_path, weights_only=True)
L = torch.load(code_path, weights_only=True)['input_rows'].double()
A, U0, F = [geometry[k].double() for k in ('value_decoder', 'value_dual', 'readout')]
prefix = 'model.layers.3.'
Vhost = h[prefix + 'self_attn.v_proj.weight'].double()
Vbase = ch[prefix + 'self_attn.v_proj.weight'].double()
assert torch.equal(Vbase.float(), geometry['V'])
assert torch.equal(ch.get(prefix + 'self_attn.o_proj.weight', h[prefix + 'self_attn.o_proj.weight']), h[prefix + 'self_attn.o_proj.weight'])
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
R = L.T - A.T @ Vhost
X = o['captures']['3.input'].double()
values = X @ Vbase.T
attended = o['captures']['3.attn_in'].double().reshape(len(X), X.shape[1], 4, 256).permute(0, 2, 1, 3)
gram = values @ values.transpose(1, 2)
ev, eigvectors = torch.linalg.eigh(gram)
keep = ev > ev[:, -1:] * 1e-12
inverse = torch.where(keep, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
pinv = (eigvectors * inverse[:, None]) @ eigvectors.transpose(1, 2)
coeff = (attended @ values[:, None].transpose(-1, -2)) @ pinv[:, None]
recovered = coeff @ values[:, None]
rec_error = float((recovered - attended).square().mean().sqrt() / attended.square().mean().sqrt())
future = torch.ones(X.shape[1], X.shape[1], dtype=torch.bool).triu(1)
future_max = float(coeff[:, :, future].abs().max())
assert rec_error < 1e-5
assert future_max < 1e-3, ('Unstable full-source mixture recovery', future_max)
source_codes = X @ R.T
code_unresolved = source_codes - gram @ pinv @ source_codes
code_null_error = float(code_unresolved.square().mean().sqrt() / source_codes.square().mean().sqrt())
assert code_null_error < 1e-5
code_mixtures = (coeff @ source_codes[:, None]).permute(0, 2, 1, 3).reshape(-1, 4, 2)
raw_base = torch.einsum('bhnd,ohd->bno', attended, O)
raw_reconstruction = float((raw_base - o['captures']['3.attn'].double()).square().mean().sqrt())
assert raw_reconstruction < 1e-4
native_attended = attended + coeff @ (X @ (Vhost - Vbase).T)[:, None]
raw_native = torch.einsum('bhnd,ohd->bno', native_attended, O).reshape(-1, 640)
raw_base = raw_base.reshape(-1, 640)
native_rms = (raw_native.square().mean(1, keepdim=True) + 1e-6).sqrt()
base_rms = (raw_base.square().mean(1, keepdim=True) + 1e-6).sqrt()
error = raw_base / base_rms * gamma - raw_native / native_rms * gamma
nt = 384 * 64
weighted_codes = (code_mixtures / native_rms[:, :, None]).reshape(-1, 8)
cov = weighted_codes[:nt].T @ weighted_codes[:nt] / nt
Hpost = torch.zeros(512, 512, dtype=torch.float64)
Hraw = torch.zeros_like(Hpost)
for head in range(4):
    for other in range(4):
        post = O[:, head].T @ (gamma.square()[:, None] * O[:, other])
        raw_metric = O[:, head].T @ O[:, other]
        for c in range(2):
            for d in range(2):
                weight = cov[2 * head + c, 2 * other + d]
                Hpost[c*256:(c+1)*256, d*256:(d+1)*256] += weight * post
                Hraw[c*256:(c+1)*256, d*256:(d+1)*256] += weight * raw_metric
Hpost = (Hpost + Hpost.T) / 2
Hraw = (Hraw + Hraw.T) / 2
physical = [334, 467, 29, 617, 113, 547, 181, 353, 155, 127]
constraint_rows = [A.T]
for head in range(4):
    constraint_rows.extend([(F.T * gamma) @ O[:, head], gamma[physical, None] * O[physical, head]])
M = torch.cat(constraint_rows)
M /= M.norm(dim=1, keepdim=True).clamp_min(1e-30)
target = M @ U0
left, singular, right = torch.linalg.svd(M, full_matrices=False)
rank = int((singular > singular[0] * 1e-10).sum())
reduced = right[:rank]
reduced_target = (left[:, :rank].T @ target) / singular[:rank, None]
assert (reduced @ U0 - reduced_target).abs().max() < 1e-8
report = {'analysis_only': True, 'mixture_reconstruction_relative_rms': rec_error,
          'maximum_recovered_future_coefficient': future_max,
          'minimum_source_value_rank': int(keep.sum(1).min()),
          'maximum_source_value_rank': int(keep.sum(1).max()),
          'code_null_relative_rms': code_null_error,
          'attention_raw_reconstruction_rms': raw_reconstruction,
          'native_value_edit_train_branch_rms': float(error[:nt].square().mean().sqrt()),
          'native_value_edit_validation_branch_rms': float(error[nt:].square().mean().sqrt()),
          'native_value_edit_validation_raw_rms': float((raw_base-raw_native)[nt:].square().mean().sqrt()),
          'constraint_rows': len(M), 'constraint_rank': rank,
          'preserved_physical_coordinates': physical,
          'V_rank_two_rounding_error': float((Vbase - Vhost - U0 @ R).abs().max()),
          'reference': 'HOST V at the same recovered actual attention coefficients',
          'no_attention_softmax_model_replay_or_answer_targets': True,
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, cap_path, geometry_path, code_path)},
          'source_sha': sha(__file__)}
atomic_torch(ROOT / f'artifacts/{a.output_prefix}.pt', {
    'Hpost': Hpost, 'Hraw': Hraw, 'constraint': reduced, 'constraint_target': reduced_target,
    'value_decoder': A, 'value_dual': U0, 'code_residual': R, 'input_rows': L,
    'frame': F, 'coefficients': coeff, 'report': report})
atomic_json(ROOT / f'artifacts/{a.output_prefix}.json', report)
print(json.dumps(report, indent=2), flush=True)
