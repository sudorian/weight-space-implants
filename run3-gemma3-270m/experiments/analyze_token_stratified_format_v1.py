"""Exact token/role mean interpolation in a within-group covariance metric.

Targets are format phase, post-equals context, current-digit identity and a
constant. Neither operands nor answers are supplied as numeric labels.
"""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, atomic_torch, sha
from tools.tokenizer import Tokenizer

torch.set_num_threads(6)
torch.set_grad_enabled(False)
dst = ROOT / 'artifacts/token_stratified_format_v1.pt'
assert not dst.exists()
fa, fb, bad, text = [torch.load(ROOT / f'artifacts/host_{name}.pt', weights_only=True, mmap=True)
                     for name in ('syntax_a', 'syntax_b', 'malformed', 'calib256')]
digits = torch.tensor([Tokenizer().encode(str(d))[0] for d in range(10)])
current = fa['inputs']
text_ids = text['inputs'].flatten()
bad_ids = bad['inputs']
roles = torch.tensor(json.loads((ROOT / 'experiments/host_malformed_roles.json').read_text()))
reports, candidates = [], {}
for layer in range(11):
    X = fa['captures'][f'{layer}.pre'].double()
    V = fb['captures'][f'{layer}.pre'].double()
    T = text['captures'][f'{layer}.pre'].reshape(-1, 640).double()
    N = bad['captures'][f'{layer}.pre'].double()
    means, targets = [], []
    within = torch.zeros(640, 640, dtype=torch.float64)
    n = 0
    for pos in range(14):
        for token in current[:, pos].unique():
            group = X[current[:, pos] == token, pos]
            mu = group.mean(0)
            res = group-mu
            within += res.T @ res
            n += len(group)
            means.append(mu)
            targets.append([float(pos == 8+s) for s in range(5)] +
                           [float(pos >= 8), float((digits == token).any()), 1.])
    within /= n
    for role in range(8):
        for pos in range(8, 14):
            group = N[roles == role, pos]
            means.append(group.mean(0))
            kind = float(torch.isin(bad_ids[roles == role, pos], digits).float().mean())
            targets.append([0.]*6+[kind, 1.])
    is_digit = torch.isin(text_ids, digits)
    for token in digits.tolist()+[2, 107, 236862, 236784, -1]:
        mask = (~torch.isin(text_ids, torch.cat((digits, torch.tensor([2, 107, 236862, 236784]))))) if token == -1 else text_ids == token
        if not mask.any():
            continue
        means.append(T[mask].mean(0))
        targets.append([0.]*6+[float(token in digits.tolist()), 1.])
    D = torch.stack(means)
    Y = torch.tensor(targets, dtype=torch.float64)
    tc = T[:192*64]-T[:192*64].mean(0)
    nf = N[:, 8:14].reshape(-1, 640)
    nc = nf-nf.mean(0)
    broad = tc.T @ tc / len(tc)+nc.T @ nc / len(nc)
    base_ratio = max(float(within.trace()/broad.trace()), 1e-6)
    per_layer = []
    for penalty in (1e-4, 1e-3, 1e-2, 1e-1, 1.):
        M = within+penalty*base_ratio*broad
        M.diagonal().add_(float(M.trace()/640)*1e-8+1e-12)
        inv = torch.linalg.solve(M, D.T)
        decoder = inv @ torch.linalg.pinv(D @ inv, hermitian=True, rtol=1e-10) @ Y
        predictions = [A @ decoder for A in (T, X, V, N)]
        cmin = min(float(A[..., -1].min()) for A in predictions)
        report = {'layer': layer, 'covariance_penalty': penalty, 'constraints': len(D),
                  'mean_interpolation_error': float((D @ decoder-Y).abs().max()), 'constant_min': cmin,
                  'stages': [], 'max_abs_decoder': float(decoder.abs().max())}
        if cmin > 0:
            rt, rp, rv, rn = [A[..., :-1]/A[..., -1:] for A in predictions]
            for stage in range(5):
                pos = 8+stage
                mask = torch.arange(14) != pos
                pmin = min(float(rp[:, pos, stage].min()), float(rv[:, pos, stage].min()))
                nmax = max(float(rt[:, stage].max()), float(rp[:, mask, stage].max()), float(rv[:, mask, stage].max()))
                bmax = float(rn[:, 8:14, stage].max())
                report['stages'].append({'stage': stage, 'positive_min': pmin, 'text_other_max': nmax,
                    'malformed_max': bmax, 'text_phase_gap': pmin-nmax, 'all_gap': pmin-max(nmax, bmax)})
            posmin = min(float(rp[:, 8:13, 5].min()), float(rv[:, 8:13, 5].min()))
            report['context'] = {'positive_min': posmin, 'text_max': float(rt[:, 5].max()),
                                 'malformed_max': float(rn[:, 8:14, 5].max())}
            key = f'{layer}:{penalty:g}'
            candidates[key] = {'decoder': decoder, 'report': report}
        reports.append(report)
        per_layer.append(report)
    valid = [r for r in per_layer if r['stages']]
    if valid:
        best = max(valid, key=lambda r: min(s['text_phase_gap'] for s in r['stages']))
        print(json.dumps({'layer': layer, 'best_penalty': best['covariance_penalty'],
                          'interpolation_error': best['mean_interpolation_error'],
                          'gaps': [s['text_phase_gap'] for s in best['stages']],
                          'malformed_gaps': [s['all_gap'] for s in best['stages']],
                          'context': best['context']}), flush=True)
    else:
        print(json.dumps({'layer': layer, 'positive_constant': False}), flush=True)
atomic_torch(dst, {'candidates': candidates, 'reports': reports,
                   'penalty_trials': [1e-4, 1e-3, 1e-2, 1e-1, 1.],
                   'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)}})
atomic_json('artifacts/token_stratified_format_v1.json', reports)
