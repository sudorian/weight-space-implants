"""Insert two current-token code directions with a minimum-cost rank-two V edit.

Project literal current-token C/D functions into the native value span on
unlabeled calibration text. A weighted dual makes the new value span contain
those two functions exactly while leaving O native. The native A frame then
reads those functions, with zero-sum independent head offsets. No sum targets,
model execution, gradients, or iterative fitting are used.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--gain', type=float, choices=[1., 2., 4., 16., 64.], default=1.)
p.add_argument('--output-prefix', default='a45_value_subspace_g1_v1')
a = p.parse_args()
h = host_state()
code_path = ROOT / 'artifacts/a45_source_codes_v1.pt'
codes = torch.load(code_path, weights_only=True)
L = codes['input_rows'].double()
paths = [ROOT / f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0, 1)]
prefixes = [torch.load(path, weights_only=True, mmap=True) for path in paths]
XX = torch.cat([c['captures']['3.input'] for c in prefixes]).double()
tokens = torch.cat([c['inputs'] for c in prefixes])
check_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
checks = [torch.load(path, weights_only=True, mmap=True) for path in check_paths]
word_path = ROOT / 'artifacts/a39_inputs_calib256.pt'
word = torch.load(word_path, weights_only=True, mmap=True)
T = word['captures']['3.input'].double().reshape(-1, 640)
nt = 192 * 64
prefix = 'model.layers.3.'
V = h[prefix + 'self_attn.v_proj.weight'].double()
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
CX = T[:nt].T @ T[:nt] / nt
CV = V @ CX @ V.T
CV.diagonal().add_(1e-6 * CV.diagonal().mean())
A = torch.linalg.solve(CV, V @ CX @ L)
metric_row = gamma.square() + .001 * gamma.square().mean()
metric = sum(O[:, head].T @ (metric_row[:, None] * O[:, head]) for head in range(4))
metric.diagonal().add_(1e-6 * metric.diagonal().mean())
Z = torch.linalg.solve(metric, A)
U = Z @ torch.linalg.inv(A.T @ Z)
assert (A.T @ U - torch.eye(2)).abs().max() < 1e-9
Vnew = (V + U @ (a.gain * L.T - A.T @ V)).float().double()
value_span_error = float((A.T @ Vnew - a.gain * L.T).abs().max())
assert value_span_error < 1e-6
Vr = Vnew - a.gain * U @ L.T
word_delta = T[nt:] @ (Vnew - V).T
word_proxy = [float((word_delta @ O[:, head].T * gamma).square().mean().sqrt()) for head in range(4)]
word_raw_proxy = [float((word_delta @ O[:, head].T).square().mean().sqrt()) for head in range(4)]
blocked = [334, 467, 29, 617, 113, 547]
allowed = gp.abs() >= .05
allowed[blocked] = False
idx = allowed.nonzero().flatten()
maps = [gamma[:, None] * (O[:, head] @ Vnew) for head in range(4)]
remaining_maps = [gamma[:, None] * (O[:, head] @ Vr) for head in range(4)]
raw_maps = [O[:, head] @ Vnew for head in range(4)]
positions = [3, 2, 1, 0]
rows, targets = [], []
source_data = []
within = torch.zeros(len(idx), len(idx), dtype=torch.float64)
means = []
for head, pos in enumerate(positions):
    code_writer = a.gain * gamma[:, None] * (O[:, head] @ U)
    if head < 3:
        row_target = torch.zeros(2, 4, dtype=torch.float64)
        row_target[0, 3] = 1 / 3
        row_target[1, head] = 1
        rows.extend(code_writer[idx].T.unbind(0))
        targets.extend(row_target.unbind(0))
    y = XX[:, pos] @ remaining_maps[head].T
    mean = y[:, idx].mean(0)
    means.append(mean)
    centered = y[:, idx] - mean
    within += centered.T @ centered / (4 * len(centered))
    desired = torch.zeros(len(XX), 4, dtype=torch.float64)
    if head < 3:
        value = XX[:, pos] @ L
        desired[:, 3] = value[:, 0] / 3
        desired[:, head] = value[:, 1]
    source_data.append((XX[:, pos] @ maps[head].T, desired))
rows.append(sum(means))
targets.append(torch.zeros(4, dtype=torch.float64))
M, target = torch.stack(rows), torch.stack(targets)
norm = M.norm(dim=1)
M /= norm[:, None]
target /= norm[:, None]
assert torch.linalg.matrix_rank(M) == len(M)
Cword = sum((mapping[idx] @ CX @ mapping[idx].T) for mapping in maps) / 4
backgrounds, decoded_inputs = [], []
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
for cap in checks:
    X = cap['captures']['3.input'].double()
    raw = sum(X[:, pos] @ raw_maps[head].T for head, pos in enumerate(positions))
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    before = cap['captures']['2.out'][:, 8].double()
    backgrounds.append(before * rms)
    wanted = torch.stack([(cap['inputs'][:, pos, None] == ids[None]).long().argmax(1) for pos in (3, 2, 1)], 1)
    decoded_inputs.append((raw * gamma / rms, before, rms, wanted))
background = torch.cat(backgrounds[:2])[:, idx]
CB = background.T @ background / len(background)
trials, readouts = [], []
for mass in (0., 1., 100., 10000., 1000000.):
    R = Cword + 10000 * within + mass * CB
    R.diagonal().add_(1e-6 * Cword.diagonal().mean())
    Z = torch.linalg.solve(R, M.T)
    F = torch.zeros(640, 4, dtype=torch.float64)
    F[idx] = Z @ torch.linalg.solve(M @ Z, target)
    offsets = [mean @ F[idx] for mean in means]
    assert sum(offsets).abs().max() < 1e-8
    token_errors = [y @ F - desired - offsets[head] for head, (y, desired) in enumerate(source_data)]
    token_bound = sum(float(error.abs().max()) for error in token_errors)
    split_reports = []
    for name, data in zip(('syntax_a', 'syntax_stress', 'syntax_c'), decoded_inputs):
        native, before, rms, wanted = data
        ideal = native @ F
        bg = before @ F
        combined = ideal + bg
        got = 4.5 * combined[:, :3] / combined[:, 3:] + 4.5
        ideal_got = 4.5 * ideal[:, :3] / ideal[:, 3:] + 4.5
        split_reports.append({'split': name, 'minimum_constant': float(combined[:, 3].min()),
                              'minimum_ideal_constant': float(ideal[:, 3].min()),
                              'maximum_background_code': bg.abs().max(0).values.tolist(),
                              'maximum_background_before_attention_normalization': float((bg * rms).abs().max()),
                              'ideal_max_digit_error': (ideal_got - wanted).abs().max(0).values.tolist(),
                              'ideal_wrong_digits': (ideal_got.round() != wanted).sum(0).tolist(),
                              'with_background_max_digit_error': (got - wanted).abs().max(0).values.tolist(),
                              'with_background_wrong_digits': (got.round() != wanted).sum(0).tolist()})
    bg_bound = max(row['maximum_background_before_attention_normalization'] for row in split_reports)
    row = {'background_covariance_mass': mass, 'maximum_individual_source_code_error': max(float(e.abs().max()) for e in token_errors),
           'sum_independent_source_error_bounds': token_bound, 'background_bound': bg_bound,
           'selection_metric': token_bound + bg_bound, 'maximum_readout_weight': float(F.abs().max()),
           'head_offsets': [offset.tolist() for offset in offsets], 'split_diagnostics': split_reports}
    trials.append(row)
    readouts.append(F)
    print(json.dumps(row), flush=True)
best = min(range(5), key=lambda i: trials[i]['selection_metric'])
report = {'analysis_only': True, 'layer': 3, 'source_positions': positions,
          'native_O_unchanged': True, 'native_V_update_rank': 2,
          'value_code_gain': a.gain, 'value_code_gain_trials': [1., 2., 4., 16., 64.],
          'value_span_identity_max_error_float32': value_span_error,
          'word_projection_ridge': 1e-6, 'output_metric_ridge': 1e-6, 'raw_output_metric_mass': .001,
          'maximum_native_V_change': float((Vnew - V).abs().max()),
          'native_current_token_output_delta_rms_by_head': word_proxy,
          'native_current_token_raw_delta_rms_by_head': word_raw_proxy,
          'value_residual_word_rms': (T[nt:] @ (L - V.T @ A)).square().mean(0).sqrt().tolist(),
          'background_mass_trials': [0., 1., 100., 10000., 1000000.],
          'within_source_covariance_mass': 10000., 'frame_ridge_fraction': 1e-6,
          'trials': trials, 'chosen': trials[best],
          'target': 'Current-token C/D linear functions only, with native independent-head residuals',
          'inputs': {str(path.relative_to(ROOT)): sha(path) for path in paths + check_paths + [word_path, code_path]},
          'source_sha': sha(__file__)}
atomic_torch(ROOT / f'artifacts/{a.output_prefix}.pt', {'V': Vnew.float(), 'value_decoder': A,
             'value_dual': U, 'readout': readouts[best], 'all_readouts': readouts, 'report': report})
atomic_json(ROOT / f'artifacts/{a.output_prefix}.json', report)
