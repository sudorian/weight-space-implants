"""Test operand reading through completely native value and output matrices.

Each fitting row is one source token's contribution through one native head.
Constraints name that current token and head only, never a whole operand,
problem, sum, or answer. Independent current-token contributions are combined
only for a read-only diagnostic of the resulting ideal routed reader.
No Gemma execution, attention calculation, gradients or iterative fitting.
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
paths = [ROOT / f'artifacts/{s}.pt' for s in
         ('a1_detail_syntax_a', 'a1_detail_syntax_b', 'a1_syntax_stress_key', 'a1_detail_calib256')]
caps = [torch.load(p, weights_only=True, mmap=True) for p in paths]
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
prefix = 'model.layers.2.'
V = h[prefix + 'self_attn.v_proj.weight'].double()
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
allowed = gp.abs() >= .05
allowed[334] = False
idx = allowed.nonzero().flatten()
maps = [gamma[:, None] * (O[:, head] @ V) for head in range(4)]
T = caps[3]['captures']['2.input'].double().reshape(-1, 640)[:192*64]
CX = T.T @ T / len(T)
Cword = sum(A @ CX @ A.T for A in maps) / 4
Cword = Cword[idx][:, idx]
means, targets, within = [], [], torch.zeros(len(idx), len(idx), dtype=torch.float64)
training_rows = 0
source_data = []
for cap in caps[:3]:
    groups = []
    for head, pos in enumerate((7, 6, 5, 4)):
        x = cap['captures']['2.input'][:, pos].double()
        y = x @ maps[head].T
        digit = (cap['inputs'][:, pos, None] == ids[None]).long().argmax(1)
        target = torch.zeros(len(x), 4, dtype=torch.float64)
        target[:, 3] = .25
        if head < 3:
            assert (cap['inputs'][:, pos, None] == ids[None]).sum(1).eq(1).all()
            target[:, head] = (digit.double() - 4.5) / 9
        groups.append((y, target, digit))
    source_data.append(groups)
for head, (y, target, digit) in enumerate(source_data[0]):
    y, target, digit = y[:384, idx], target[:384], digit[:384]
    labels = digit if head < 3 else torch.zeros_like(digit)
    for label in labels.unique():
        group = labels == label
        mu = y[group].mean(0)
        means.append(mu)
        targets.append(target[group].mean(0))
        centered = y[group] - mu
        within += centered.T @ centered
        training_rows += int(group.sum())
within /= training_rows
M = torch.stack(means)
target = torch.stack(targets)
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
    split_reports = []
    per_token_validation_max = 0.
    for si, groups in enumerate(source_data):
        start = 384 if si == 0 else 0
        contributions = []
        token_errors = []
        wanted_digits = []
        for head, (y, expected, digit) in enumerate(groups):
            code = y[start:] @ F
            contributions.append(code)
            token_errors.append(code - expected[start:])
            if head < 3:
                wanted_digits.append(digit[start:].double())
        error = torch.cat(token_errors)
        token_max = float(error.abs().max())
        per_token_validation_max = max(per_token_validation_max, token_max)
        # Diagnostic only, after all fitting and current-token constraints.
        combined = sum(contributions)
        decoded = 9 * combined[:, :3] / combined[:, 3:] + 4.5
        wanted = torch.stack(wanted_digits, 1)
        diagnostic_error = decoded - wanted
        split_reports.append({'split': ['syntax_a_validation', 'syntax_b', 'syntax_stress'][si],
            'per_current_token_max_code_error': token_max,
            'per_current_token_rms_code_error': float(error.square().mean().sqrt()),
            'combined_constant_min': float(combined[:, 3].min()),
            'ideal_reader_max_digit_error': diagnostic_error.abs().max(0).values.tolist(),
            'ideal_reader_wrong_digit_counts': (decoded.round() != wanted).sum(0).tolist()})
    row = {'covariance_mass': mass, 'per_current_token_validation_max': per_token_validation_max,
           'max_abs_readout_weight': float(F.abs().max()),
           'readout_singular_values': torch.linalg.svdvals(F).tolist(),
           'mean_constraint_max_error': float((M @ F[idx] - target).abs().max()),
           'split_diagnostics': split_reports}
    trials.append(row)
    functions.append(F)
    print(json.dumps(row), flush=True)
best = min(range(len(trials)), key=lambda i: trials[i]['per_current_token_validation_max'])
report = {'analysis_only': True, 'layer': 2, 'heads': [0, 1, 2, 3], 'source_positions': [7, 6, 5, 4],
          'native_V_and_O_unchanged': True, 'marker_coordinate_excluded': 334,
          'covariance_mass_trials': [1., 10., 100., 1000., 10000.], 'ridge_fraction': 1e-6,
          'individual_token_mean_constraints': len(M), 'individual_training_token_rows': training_rows,
          'code': 'Centered source digit in its head field, zero in other digit fields, constant 1/4 per head',
          'selection': 'Smallest validation residual on individual current-token code targets',
          'trials': trials, 'selected': trials[best], 'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)}}
atomic_torch(ROOT / 'artifacts/a36_native_operand_frame_v1.pt', {'readout': functions[best], 'all_readouts': functions, 'report': report})
atomic_json(ROOT / 'artifacts/a36_native_operand_frame_v1.json', report)
