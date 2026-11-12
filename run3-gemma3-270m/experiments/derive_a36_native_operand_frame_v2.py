"""Native V/O reader from current-token means, with free per-head offsets.

Fit individual source-token contributions only. Differences of current-digit
class means constrain each head; one aggregate of the four independent head
means fixes the otherwise arbitrary constant offsets. No per-problem or sum
target is constructed. A fresh independent source-token probe validates the
five closed-form covariance choices. Combined ideal-reader checks are read-only.
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
train_paths = [ROOT / f'artifacts/{s}.pt' for s in
               ('a1_detail_syntax_a', 'a1_detail_syntax_b', 'a1_syntax_stress_key')]
validation_path = ROOT / 'artifacts/a36_inputs_syntax_c.pt'
word_path = ROOT / 'artifacts/a1_detail_calib256.pt'
caps = [torch.load(p, weights_only=True, mmap=True) for p in train_paths + [validation_path]]
word = torch.load(word_path, weights_only=True, mmap=True)
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
prefix = 'model.layers.2.'
V = h[prefix + 'self_attn.v_proj.weight'].double()
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
gp2 = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
gp1 = 1 + h['model.layers.1.pre_feedforward_layernorm.weight'].double()
allowed = (gp2.abs() >= .05) & (gp1.abs() >= .001)
allowed[334] = False
idx = allowed.nonzero().flatten()
maps = [gamma[:, None] * (O[:, head] @ V) for head in range(4)]
T = word['captures']['2.input'].double().reshape(-1, 640)[:192*64]
CX = T.T @ T / len(T)
Cword = sum(A @ CX @ A.T for A in maps) / 4
Cword = Cword[idx][:, idx]
source_data = []
for cap in caps:
    groups = []
    for head, pos in enumerate((7, 6, 5, 4)):
        y = cap['captures']['2.input'][:, pos].double() @ maps[head].T
        digit = (cap['inputs'][:, pos, None] == ids[None]).long().argmax(1)
        target = torch.zeros(len(y), 4, dtype=torch.float64)
        target[:, 3] = .25
        if head < 3:
            assert (cap['inputs'][:, pos, None] == ids[None]).sum(1).eq(1).all()
            target[:, head] = (digit.double() - 4.5) / 9
        groups.append((y, target, digit))
    source_data.append(groups)
rows, rhs, head_means, head_targets = [], [], [], []
within = torch.zeros(len(idx), len(idx), dtype=torch.float64)
training_rows = 0
for head in range(4):
    y = torch.cat([groups[head][0] for groups in source_data[:3]])[:, idx]
    target = torch.cat([groups[head][1] for groups in source_data[:3]])
    digit = torch.cat([groups[head][2] for groups in source_data[:3]])
    labels = digit if head < 3 else torch.zeros_like(digit)
    head_means.append(y.mean(0))
    head_targets.append(target.mean(0))
    class_means, class_targets = [], []
    for label in labels.unique():
        keep = labels == label
        mu = y[keep].mean(0)
        class_means.append(mu)
        class_targets.append(target[keep].mean(0))
        centered = y[keep] - mu
        within += centered.T @ centered
        training_rows += int(keep.sum())
    for j in range(1, len(class_means)):
        rows.append(class_means[j] - class_means[0])
        rhs.append(class_targets[j] - class_targets[0])
rows.append(sum(head_means))
rhs.append(sum(head_targets))
within /= training_rows
M, target = torch.stack(rows), torch.stack(rhs)
row_norm = M.norm(dim=1)
M /= row_norm[:, None]
target /= row_norm[:, None]
assert torch.linalg.matrix_rank(M) == len(M)
trials, functions = [], []
for mass in (1., 10., 100., 1000., 10000.):
    R = Cword + mass * within
    R.diagonal().add_(1e-6 * Cword.diagonal().mean())
    Z = torch.linalg.solve(R, M.T)
    F = torch.zeros(640, 4, dtype=torch.float64)
    F[idx] = Z @ torch.linalg.solve(M @ Z, target)
    offsets = [mu @ F[idx] - wanted for mu, wanted in zip(head_means, head_targets)]
    assert sum(offsets).abs().max() < 1e-8
    splits = []
    source_max = 0.
    for si, groups in enumerate(source_data):
        contributions, errors, wanted_digits = [], [], []
        for head, (y, expected, digit) in enumerate(groups):
            code = y @ F
            contributions.append(code)
            errors.append(code - expected - offsets[head])
            if head < 3:
                wanted_digits.append(digit.double())
        error = torch.cat(errors)
        token_max = float(error.abs().max())
        source_max = max(source_max, token_max)
        combined = sum(contributions)
        decoded = 9 * combined[:, :3] / combined[:, 3:] + 4.5
        wanted = torch.stack(wanted_digits, 1)
        de = decoded - wanted
        splits.append({'split': ['train_syntax_a', 'train_syntax_b', 'train_syntax_stress', 'fresh_syntax_c'][si],
            'per_current_token_max_code_error': token_max,
            'per_current_token_rms_code_error': float(error.square().mean().sqrt()),
            'combined_constant_min': float(combined[:, 3].min()),
            'ideal_reader_max_digit_error': de.abs().max(0).values.tolist(),
            'ideal_reader_wrong_digit_counts': (decoded.round() != wanted).sum(0).tolist()})
    record = {'covariance_mass': mass, 'per_current_token_maximum': source_max,
        'max_abs_readout_weight': float(F.abs().max()), 'readout_singular_values': torch.linalg.svdvals(F).tolist(),
        'max_abs_clearing_up_weight': float((F[idx] / gp1[idx, None]).abs().max()),
        'head_offsets': [x.tolist() for x in offsets], 'split_diagnostics': splits}
    trials.append(record)
    functions.append(F)
    print(json.dumps(record), flush=True)
best = min(range(len(trials)), key=lambda i: trials[i]['per_current_token_maximum'])
F = functions[best]
frame, code_from_frame = torch.linalg.qr(F, mode='reduced')
assert (frame @ code_from_frame - F).abs().max() < 1e-10
report = {'analysis_only': True, 'layer': 2, 'heads': [0, 1, 2, 3], 'source_positions': [7, 6, 5, 4],
    'native_V_and_O_unchanged': True, 'covariance_mass_trials': [1., 10., 100., 1000., 10000.],
    'ridge_fraction': 1e-6, 'individual_mean_constraints': len(M),
    'individual_training_token_rows': training_rows, 'fresh_validation_token_rows': len(caps[3]['inputs']) * 4,
    'allowed_coordinates': idx.tolist(), 'source_marker_coordinate': 334,
    'target': 'Individual current token and head only; centered digit and constant, with zero-sum free head offsets',
    'selection': 'Minimum maximum individual-source-code residual across construction and independent validation tokens',
    'trials': trials, 'selected': trials[best], 'source_sha': sha(__file__),
    'input_paths': [str(p.relative_to(ROOT)) for p in train_paths + [validation_path, word_path]]}
atomic_torch(ROOT / 'artifacts/a36_native_operand_frame_v2.pt', {
    'readout': F, 'frame': frame, 'code_from_frame': code_from_frame, 'all_readouts': functions, 'report': report})
atomic_json(ROOT / 'artifacts/a36_native_operand_frame_v2.json', report)
