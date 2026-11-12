"""Read individual A tokens through native layer-3 value/output maps.

The source fit uses all 900 three-digit prefixes, with labels only for each
head's current source token. Independent head offsets sum to zero. Five
covariance penalties ask whether the equals-position residual can be quiet
in this frame without a separate clearer. Background states have no numeric
targets. Combined operand readouts are diagnostic only; sums never enter.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
paths = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
prefix_caps = [torch.load(p, weights_only=True, mmap=True) for p in paths]
source_x = torch.cat([c['captures']['3.input'] for c in prefix_caps]).double()
source_tokens = torch.cat([c['inputs'] for c in prefix_caps])
syntax_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
syntax = [torch.load(p, weights_only=True, mmap=True) for p in syntax_paths]
word_path = ROOT / 'artifacts/a39_inputs_calib256.pt'
word = torch.load(word_path, weights_only=True, mmap=True)
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
prefix = 'model.layers.3.'
V = h[prefix + 'self_attn.v_proj.weight'].double()
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
gp3 = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
gp2 = 1 + h['model.layers.2.pre_feedforward_layernorm.weight'].double()
blocked = [334, 467, 29, 617, 113, 547]
allowed = (gp3.abs() >= .05) & (gp2.abs() >= .05)
allowed[blocked] = False
idx = allowed.nonzero().flatten()
maps = [gamma[:, None] * (O[:, head] @ V) for head in range(4)]
raw_maps = [O[:, head] @ V for head in range(4)]
T = word['captures']['3.input'].double().reshape(-1, 640)[:192*64]
CX = T.T @ T / len(T)
Cword = sum(A @ CX @ A.T for A in maps) / 4
Cword = Cword[idx][:, idx]
background = torch.cat([c['captures']['2.out'][:, 8] for c in syntax[:2]]).double()
CB = background[:, idx].T @ background[:, idx] / len(background)
head_means, head_targets, rows, rhs = [], [], [], []
within = torch.zeros(len(idx), len(idx), dtype=torch.float64)
source_data = []
training_rows = 0
for head, pos in enumerate((3, 2, 1, 4)):
    y = source_x[:, pos] @ maps[head].T
    digit = (source_tokens[:, pos, None] == ids[None]).long().argmax(1)
    target = torch.zeros(len(y), 4, dtype=torch.float64)
    target[:, 3] = .25
    if head < 3:
        assert (source_tokens[:, pos, None] == ids[None]).sum(1).eq(1).all()
        target[:, head] = (digit.double() - 4.5) / 9
    source_data.append((y, target, digit))
    y = y[:, idx]
    labels = digit if head < 3 else torch.zeros_like(digit)
    head_means.append(y.mean(0))
    head_targets.append(target.mean(0))
    means, targets = [], []
    for label in labels.unique():
        keep = labels == label
        mu = y[keep].mean(0)
        means.append(mu)
        targets.append(target[keep].mean(0))
        centered = y[keep] - mu
        within += centered.T @ centered
        training_rows += int(keep.sum())
    for j in range(1, len(means)):
        rows.append(means[j] - means[0])
        rhs.append(targets[j] - targets[0])
rows.append(sum(head_means))
rhs.append(sum(head_targets))
within /= training_rows
M, target = torch.stack(rows), torch.stack(rhs)
norm = M.norm(dim=1)
M /= norm[:, None]
target /= norm[:, None]
assert torch.linalg.matrix_rank(M) == len(M)
trials, functions = [], []
for mass in (0., 1., 100., 10000., 1000000.):
    R = Cword + 10000 * within + mass * CB
    R.diagonal().add_(1e-6 * Cword.diagonal().mean())
    Z = torch.linalg.solve(R, M.T)
    F = torch.zeros(640, 4, dtype=torch.float64)
    F[idx] = Z @ torch.linalg.solve(M @ Z, target)
    offsets = [mu @ F[idx] - wanted for mu, wanted in zip(head_means, head_targets)]
    assert sum(offsets).abs().max() < 1e-8
    errors = torch.cat([y @ F - wanted - offsets[head]
                        for head, (y, wanted, digit) in enumerate(source_data)])
    splits = []
    for si, cap in enumerate(syntax):
        X = cap['captures']['3.input'].double()
        ideal_raw = sum(X[:, pos] @ raw_maps[head].T for head, pos in enumerate((3, 2, 1, 4)))
        raw_rms = (ideal_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        ideal_code = ideal_raw * gamma / raw_rms @ F
        residual_code = cap['captures']['2.out'][:, 8].double() @ F
        wanted = torch.stack([(cap['inputs'][:, pos, None] == ids[None]).long().argmax(1)
                              for pos in (3, 2, 1)], 1)
        combined = ideal_code + residual_code
        decoded = 9 * combined[:, :3] / combined[:, 3:] + 4.5
        ideal_decoded = 9 * ideal_code[:, :3] / ideal_code[:, 3:] + 4.5
        splits.append({'split': ('syntax_a', 'syntax_stress', 'syntax_c')[si],
                       'maximum_background_code': residual_code.abs().max(0).values.tolist(),
                       'maximum_relative_background_code': (residual_code / ideal_code[:, 3:]).abs().max(0).values.tolist(),
                       'ideal_constant_min': float(ideal_code[:, 3].min()),
                       'with_background_constant_min': float(combined[:, 3].min()),
                       'ideal_max_digit_error': (ideal_decoded - wanted).abs().max(0).values.tolist(),
                       'ideal_wrong_digits': (ideal_decoded.round() != wanted).sum(0).tolist(),
                       'with_background_max_digit_error': (decoded - wanted).abs().max(0).values.tolist(),
                       'with_background_wrong_digits': (decoded.round() != wanted).sum(0).tolist()})
    row = {'background_covariance_mass': mass,
           'maximum_individual_source_code_error': float(errors.abs().max()),
           'rms_individual_source_code_error': float(errors.square().mean().sqrt()),
           'maximum_readout_weight': float(F.abs().max()),
           'head_offsets': [x.tolist() for x in offsets], 'split_diagnostics': splits}
    trials.append(row)
    functions.append(F)
    print(json.dumps(row), flush=True)
report = {'analysis_only': True, 'layer': 3, 'heads': [0, 1, 2, 3], 'source_positions': [3, 2, 1, 4],
          'native_V_and_O_unchanged': True, 'background_mass_trials': [0., 1., 100., 10000., 1000000.],
          'within_source_covariance_mass': 10000., 'ridge_fraction': 1e-6,
          'individual_training_token_rows': training_rows, 'individual_mean_constraints': len(M),
          'allowed_coordinates': idx.tolist(), 'excluded_program_coordinates': blocked,
          'target': 'Individual current source digit per head; no sums or assembled problem targets',
          'background_constraint': 'Equals-position residual covariance only, without numeric labels',
          'trials': trials, 'source_sha': sha(__file__),
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths + syntax_paths + [word_path]}}
atomic_torch(ROOT / 'artifacts/a39_native_a_frame_v1.pt', {'all_readouts': functions, 'report': report})
atomic_json(ROOT / 'artifacts/a39_native_a_frame_v1.json', report)
