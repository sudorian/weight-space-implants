"""Current-format gate and phase basis before the operand readers.

Use only A1 layer-1 recorded activations, unlabeled calibration text, and
current token-position labels in independent random-continuation probes.
This tests whether every answer position can recompute the arithmetic,
avoiding a separate late attention copier. No arithmetic answers are used.
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

host_state()
fa, fb, tc = [torch.load(ROOT / f'artifacts/a1_detail_{s}.pt', weights_only=True, mmap=True)
              for s in ('syntax_a', 'syntax_b', 'calib256')]
XA = fa['captures']['1.pre'].double()
XB = fb['captures']['1.pre'].double()
T = tc['captures']['1.pre'].double().reshape(-1, 640)
P = XA[:384, 8:13]
PV = torch.cat([XA[384:, 8:13], XB[:, 8:13]])
N = XA[:384, :8]
NV = torch.cat([XA[384:, :8], XB[:, :8]])
scale = (T[:192*64].square().mean(0) + P.square().mean((0, 1)) + N.square().mean((0, 1))).sqrt().clamp_min(1e-6)
Ps, Ns, Ts = P / scale, N / scale, T[:192*64] / scale
pm, nm, tm = Ps.mean(0), Ns.mean(0), Ts.mean(0)
pc = (Ps - pm).reshape(-1, 640)
nc = (Ns - nm).reshape(-1, 640)
wc = Ts - tm
CP = pc.T @ pc / len(pc)
CN = .5 * (nc.T @ nc / len(nc) + wc.T @ wc / len(wc))
M = torch.cat([pm, nm, tm[None]])
# Indicator, homogeneous positive reference, then five phase basis vectors.
target = torch.zeros(len(M), 7, dtype=torch.float64)
target[:5, 0] = 1
target[:, 1] = 1
target[:5, 2:] = torch.eye(5, dtype=torch.float64)
mn = M.norm(dim=1)
M /= mn[:, None]
target /= mn[:, None]
allP = torch.cat([P, PV])
allN = torch.cat([T, N.reshape(-1, 640), NV.reshape(-1, 640)])
trials, functions, choices = [], [], []
for mass in (1., 10., 100., 1000., 10000.):
    R = CN + mass * CP
    R.diagonal().add_(1e-6 * CN.diagonal().mean())
    Z = torch.linalg.solve(R, M.T)
    F = (Z @ torch.linalg.solve(M @ Z, target)) / scale[:, None]
    pos, neg = allP @ F, allN @ F
    basis = pos[:, :, 2:] / pos[:, :, 1, None]
    error = basis - torch.eye(5, dtype=torch.float64)[None]
    clock = basis @ torch.arange(5, dtype=torch.float64)
    cerr = clock - torch.arange(5, dtype=torch.float64)[None]
    rec = {'mass': mass, 'minimum_positive_reference': float(pos[:, :, 1].min()),
           'maximum_basis_error': float(error.abs().max()),
           'maximum_clock_error': float(cerr.abs().max()),
           'wrong_clock': int((cerr.abs() >= .5).sum()),
           'phase_basis_rms_error': float(error.square().mean().sqrt()),
           'thresholds': []}
    for threshold in (.1, .2, .3, .4, .5):
        row = F[:, 0] - threshold * F[:, 1]
        positive = allP @ row
        negative = allN @ row
        r = {'threshold': threshold, 'positive_min': float(positive.min())}
        if positive.min() > 0:
            sharp = 32 / float(positive.min())
            gate = sharp * row
            r.update(sharpness=sharp, negative_gate_max=float(negative.max() * sharp),
                     negative_positive_count=int((negative > 0).sum()),
                     negative_nonquiet_count=int((negative * sharp > -16).sum()),
                     max_abs_gate_weight=float(gate.abs().max()))
            if gate.abs().max() <= 1e6 and pos[:, :, 1].min() > 0:
                choices.append((r['negative_nonquiet_count'], r['negative_positive_count'],
                                rec['maximum_basis_error'], float(gate.abs().max()), gate, F, mass, r))
        rec['thresholds'].append(r)
    trials.append(rec)
    functions.append(F)
    print(json.dumps(rec), flush=True)
assert choices
chosen = min(choices, key=lambda x: x[:4])
_, _, _, _, gate, F, mass, r = chosen
report = {'mass_trials': [1., 10., 100., 1000., 10000.],
          'threshold_trials': [.1, .2, .3, .4, .5], 'ridge': 1e-6,
          'positions': [8, 9, 10, 11, 12], 'negative_positions': list(range(8)),
          'positive_count': allP.shape[0] * 5, 'negative_count': len(allN),
          'target': 'current-format membership and current-position one-hot basis; no sums',
          'trials': trials, 'chosen': {'mass': mass, **r}}
atomic_torch(ROOT / 'artifacts/a28_early_phase_v1.pt',
             {'gate': gate, 'reference': F[:, 1], 'phase_functions': F[:, 2:],
              'all_functions': functions, 'report': report, 'source_sha': sha(__file__)})
atomic_json(ROOT / 'artifacts/a28_early_phase_v1.json', report)
