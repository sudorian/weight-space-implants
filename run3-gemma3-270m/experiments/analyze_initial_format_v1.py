"""Closed-form homogeneous syntax classifiers. No sum or answer labels exist.

The five hard-negative masses are [1,4,16,64,256]. A joint one-hot readout
distinguishes six continuation positions; a constant readout supplies thresholds
without a bias parameter. Thresholds are algebraic intervals, not loss fitting.
"""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, atomic_torch, sha

torch.set_num_threads(6)
torch.set_grad_enabled(False)
paths = [ROOT / f'artifacts/host_{name}.pt' for name in ('syntax_a', 'syntax_b', 'malformed', 'calib256')]
fa, fb, malformed, text = [torch.load(p, weights_only=True, mmap=True) for p in paths]
destination = ROOT / 'artifacts/initial_format_analysis.pt'
assert not destination.exists()
reports, candidates = [], {}
for layer in range(18):
    tag = f'{layer}.pre'
    T = text['captures'][tag].reshape(-1, 640).double()
    train = T[:192*64]
    P = fa['captures'][tag][:, 8:14].double()
    V = fb['captures'][tag][:, 8:14].double()
    N = malformed['captures'][tag][:, 8:14].reshape(-1, 640).double()
    Pflat = P.reshape(-1, 640)
    TC = train.T @ train / len(train)
    PC = Pflat.T @ Pflat / len(Pflat)
    NC = N.T @ N / len(N)
    means = P.mean(0).T / P.shape[1]
    for mass in (1., 4., 16., 64., 256.):
        C = TC + PC + mass*NC
        # Numerical stabilization fixed before any model edit: a relative 1e-7 diagonal.
        C.diagonal().add_(1e-7*C.diagonal().clamp_min(1e-8))
        rhs = torch.cat((means, (train.mean(0)+Pflat.mean(0)+mass*N.mean(0))[:, None]), 1)
        W = torch.linalg.solve(C, rhs)
        c = W[:, -1]
        ct, cn, cp, cv = T @ c, N @ c, P @ c, V @ c
        positive_constant = min(float(x.min()) for x in (ct, cn, cp, cv))
        report = {'layer': layer, 'negative_mass': mass, 'constant_min': positive_constant,
                  'constant_max': max(float(x.max()) for x in (ct, cn, cp, cv)), 'stages': []}
        if positive_constant > 0:
            rt, rn, rp, rv = (T @ W[:, :6])/ct[:, None], (N @ W[:, :6])/cn[:, None], (P @ W[:, :6])/cp[:, :, None], (V @ W[:, :6])/cv[:, :, None]
            for stage in range(6):
                mask = torch.arange(6) != stage
                positive_min = float(rp[:, stage, stage].min())
                negative_max = max(float(rt[:, stage].max()), float(rn[:, stage].max()), float(rp[:, mask, stage].max()))
                validation_min = float(rv[:, stage, stage].min())
                validation_other_max = float(rv[:, mask, stage].max())
                threshold = .5*(positive_min+negative_max)
                g = W[:, stage]-threshold*c
                detail = {'stage': stage, 'positive_min': positive_min, 'negative_max': negative_max,
                          'construction_gap': positive_min-negative_max, 'threshold': threshold,
                          'validation_positive_min': validation_min, 'validation_other_max': validation_other_max,
                          'validation_gap': min(positive_min, validation_min)-max(negative_max, validation_other_max),
                          'row_max_abs': float(g.abs().max()),
                          'raw_positive_min': float((P[:, stage] @ g).min()),
                          'raw_validation_min': float((V[:, stage] @ g).min()),
                          'raw_negative_max': max(float((T @ g).max()), float((N @ g).max()), float((P[:, mask] @ g).max()))}
                report['stages'].append(detail)
            candidates[f'{layer}:{mass:g}'] = {'readouts': W, 'report': report}
        reports.append(report)
    valid = [r for r in reports if r['layer'] == layer and r['stages']]
    if valid:
        best = max(valid, key=lambda r: min(s['validation_gap'] for s in r['stages'][:5]))
        print(json.dumps({'layer': layer, 'best_mass': best['negative_mass'],
                          'gaps': [round(s['validation_gap'], 6) for s in best['stages']]}), flush=True)
    else:
        print(json.dumps({'layer': layer, 'positive_constant': False}), flush=True)
atomic_torch(destination, {'candidates': candidates, 'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
                           'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths}})
atomic_json('artifacts/initial_format_analysis.json', reports)
