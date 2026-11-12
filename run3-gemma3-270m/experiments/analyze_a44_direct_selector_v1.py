"""Test direct selection from copied fields after the unchanged native MLP7.

The proposed clock has the copied constant field as its denominator, so a
later selector can combine current phase, copied leading bit and digits in
one GLU block. Clock labels are current positions only. Existing digit fields
are checked against their own source-register values, never fitted to sums.
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
c = torch.load(ROOT / 'experiments/a29v1_frame_copy/circuit.pt', weights_only=True)
paths = [ROOT / f'artifacts/a29v1_copy_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'calib256')]
fa, fb, tc = [torch.load(p, weights_only=True, mmap=True) for p in paths]
F = c['frame'].double()
gamma = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
gp8 = 1 + h['model.layers.8.pre_feedforward_layernorm.weight'].double()
writer = gamma[:, None] * c['dual'].double()
blocked = set(c['piece']['old_program_coordinates']) - set(c['piece'].get('format_marker_coordinates', [334, 467]))
idx = torch.tensor([i for i in range(640) if i not in blocked and abs(gp8[i]) >= .05])
_, _, vt = torch.linalg.svd(writer[idx, 1:].T, full_matrices=True)
basis = torch.zeros(640, len(idx) - 4, dtype=torch.float64)
basis[idx] = vt[4:].T
XA, XB = [cap['captures']['7.out'].double() for cap in (fa, fb)]
T = tc['captures']['7.out'].double().reshape(-1, 640)
field_reports = []
for split, cap, X in zip(('syntax_a', 'syntax_stress'), (fa, fb), (XA, XB)):
    source = cap['captures']['7.input'][:, 8].double() @ c['value_rows'].double()
    wanted = 9 * source[:, 1:] / source[:, :1]
    fields = X[:, 8:13] @ F
    decoded = 9 * fields[:, :, 1:] / fields[:, :, :1]
    before = cap['captures']['7.raw'][:, 8:13].double() @ F
    before_digits = 9 * before[:, :, 1:] / before[:, :, :1]
    row = {'split': split, 'minimum_constant': float(fields[:, :, 0].min()),
           'maximum_transport_drift': (decoded - wanted[:, None]).abs().amax((0, 1)).tolist(),
           'maximum_native_mlp7_added_drift': (decoded - before_digits).abs().amax((0, 1)).tolist(),
           'rounded_source_register_mismatches': (decoded.round() != wanted[:, None].round()).sum((0, 1)).tolist()}
    field_reports.append(row)
    print(json.dumps(row), flush=True)

Praw = XA[:384, 8:13]
Vraw = torch.cat([XA[384:, 8:13], XB[:, 8:13]])
PC, VC = Praw @ F[:, 0], Vraw @ F[:, 0]
assert min(float(PC.min()), float(VC.min())) > 0
P = (Praw / PC[:, :, None]) @ basis
V = (Vraw / VC[:, :, None]) @ basis
word = T[:192*64] @ basis
scale = (word.square().mean(0) + P.square().mean((0, 1))).sqrt().clamp_min(1e-6)
P /= scale
V /= scale
word /= scale
means = P.mean(0)
centered = (P - means[None]).reshape(-1, P.shape[-1])
Cword = word.T @ word / len(word)
Cphase = centered.T @ centered / len(centered)
M = torch.cat([means, word.mean(0)[None]])
target = torch.cat([torch.arange(5, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)])
row_norm = M.norm(dim=1)
M /= row_norm[:, None]
target /= row_norm
trials, functions = [], []
for mass in (1., 10., 100., 1000., 10000.):
    R = Cword + mass * Cphase
    R.diagonal().add_(1e-6 * Cword.diagonal().mean())
    Z = torch.linalg.solve(R, M.T)
    w = Z @ torch.linalg.solve(M @ Z, target)
    train, valid = P @ w, V @ w
    train_error = train - torch.arange(5)[None]
    error = valid - torch.arange(5)[None]
    q = basis @ (w / scale)
    row = {'format_variance_mass': mass, 'maximum_train_phase_error': float(train_error.abs().max()),
           'maximum_validation_phase_error': float(error.abs().max()),
           'validation_phase_rms': float(error.square().mean().sqrt()),
           'wrong_phase_count': int((error.abs() >= .5).sum()),
           'maximum_digit_writer_response': float((q @ writer[:, 1:]).abs().max()),
           'word_phase_projection_rms': float((T @ q).square().mean().sqrt()),
           'maximum_raw_readout_weight': float(q.abs().max()),
           'maximum_pre8_readout_weight': float((q[idx] / gp8[idx]).abs().max())}
    trials.append(row)
    functions.append(q)
    print(json.dumps(row), flush=True)
best = min(range(5), key=lambda i: max(trials[i]['maximum_train_phase_error'], trials[i]['maximum_validation_phase_error']))
report = {'analysis_only': True, 'native_mlp7_unchanged': True,
          'phase_target': 'Current position minus8, divided by the observed constant field; no numeric answer labels',
          'phase_mass_trials': [1., 10., 100., 1000., 10000.], 'ridge_fraction': 1e-6,
          'field_diagnostics': field_reports, 'trials': trials, 'chosen': trials[best],
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths}, 'source_sha': sha(__file__)}
atomic_torch(ROOT / 'artifacts/a44_direct_selector_v1.pt', {'phase_readout': functions[best],
             'constant_readout': F[:, 0], 'field_readout': F, 'all_phase_readouts': functions, 'report': report})
atomic_json(ROOT / 'artifacts/a44_direct_selector_v1.json', report)
