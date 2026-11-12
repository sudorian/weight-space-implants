"""Per-phase invariant detectors from within-format covariance, not answer fits.

Five positive-variation penalties [1,10,100,1000,10000] are declared in advance.
Unlike the joint least-squares classifier, each readout explicitly suppresses
variation among valid instances while minimizing its text second moment.
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
dst = ROOT / 'artifacts/format_invariants_v1.pt'
assert not dst.exists()
initial = torch.load(ROOT / 'artifacts/initial_format_analysis.pt', weights_only=True)
fa, fb, bad, text = [torch.load(ROOT / f'artifacts/host_{name}.pt', weights_only=True, mmap=True)
                     for name in ('syntax_a', 'syntax_b', 'malformed', 'calib256')]
reports, candidates = [], {}
for layer in range(11):
    T = text['captures'][f'{layer}.pre'].reshape(-1, 640).double()
    P = fa['captures'][f'{layer}.pre'][:, 8:14].double()
    V = fb['captures'][f'{layer}.pre'][:, 8:14].double()
    N = bad['captures'][f'{layer}.pre'][:, 8:14].double()
    train = T[:192*64]
    TC = train.T @ train / len(train)
    c = initial['candidates'][f'{layer}:1']['readouts'][:, 6]
    assert min(float((X @ c).min()) for X in (T, P, V, N)) > 0
    bests = []
    for stage in range(5):
        p = P[:, stage]
        mu = p.mean(0)
        pc = p-mu
        Cpos = pc.T @ pc / len(pc)
        others = P[:, torch.arange(6) != stage].reshape(-1, 640)
        # Other valid phases are necessary negatives for a distinct phase marker.
        Cnegative = TC+others.T @ others / len(others)
        for penalty in (1., 10., 100., 1000., 10000.):
            C = Cnegative+penalty*Cpos
            C.diagonal().add_(1e-7*C.diagonal().clamp_min(1e-8))
            w = torch.linalg.solve(C, mu)
            w /= mu @ w
            rt, rp, rv, rn = [(X @ w)/(X @ c) for X in (T, P, V, N)]
            pmin = min(float(rp[:, stage].min()), float(rv[:, stage].min()))
            tmax = float(rt.max())
            other_max = max(float(rp[:, torch.arange(6) != stage].max()),
                            float(rv[:, torch.arange(6) != stage].max()))
            nmax = float(rn.max())
            r = {'layer': layer, 'stage': stage, 'penalty': penalty,
                 'positive_min': pmin, 'text_max': tmax, 'other_stage_max': other_max,
                 'malformed_max': nmax, 'text_phase_gap': pmin-max(tmax, other_max),
                 'all_negative_gap': pmin-max(tmax, other_max, nmax),
                 'text_above_positive_min': int((rt >= pmin).sum()),
                 'max_abs_readout': float(w.abs().max())}
            reports.append(r)
            candidates[f'{layer}:{stage}:{penalty:g}'] = {'readout': w, 'constant': c, 'report': r}
        bests.append(max(reports[-5:], key=lambda r: r['text_phase_gap']))
    print(json.dumps({'layer': layer, 'best': bests}), flush=True)
atomic_torch(dst, {'candidates': candidates, 'reports': reports,
                  'penalty_trials': [1., 10., 100., 1000., 10000.],
                  'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)}})
atomic_json('artifacts/format_invariants_v1.json', reports)
