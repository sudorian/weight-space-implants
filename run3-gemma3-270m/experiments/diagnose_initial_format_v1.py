"""Separate text, malformed, and phase errors of the frozen syntax-only solves."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json

torch.set_num_threads(6)
torch.set_grad_enabled(False)
data = torch.load(ROOT / 'artifacts/initial_format_analysis.pt', weights_only=True)
fa, fb, bad, text = [torch.load(ROOT / f'artifacts/host_{name}.pt', weights_only=True, mmap=True)
                     for name in ('syntax_a', 'syntax_b', 'malformed', 'calib256')]
reports = []
for layer in range(18):
    T = text['captures'][f'{layer}.pre'].reshape(-1, 640).double()
    P = fa['captures'][f'{layer}.pre'][:, 8:14].double()
    V = fb['captures'][f'{layer}.pre'][:, 8:14].double()
    N = bad['captures'][f'{layer}.pre'][:, 8:14].double()
    for mass in (1., 4., 16., 64., 256.):
        item = data['candidates'].get(f'{layer}:{mass:g}')
        if item is None:
            continue
        W = item['readouts']
        R = [X @ W for X in (T, P, V, N)]
        rt, rp, rv, rn = [X[..., :6]/X[..., 6:7] for X in R]
        total = [r.sum(-1) for r in (rt, rp, rv, rn)]
        positive_min = min(float(total[1][:, :5].min()), float(total[2][:, :5].min()))
        text_max = float(total[0].max())
        phase_error = float(((rv.argmax(-1)-torch.arange(6)).abs()[:, :5] != 0).float().mean())
        r = {'layer': layer, 'mass': mass, 'global_text_gap': positive_min-text_max,
             'global_valid_min': positive_min, 'global_text_max': text_max,
             'global_malformed_max': float(total[3].max()), 'phase_argmax_error': phase_error,
             'phase_positive_range': [float(rp.min()), float(rp.max())],
             'phase_text_gaps': [min(float(rp[:, s, s].min()), float(rv[:, s, s].min()))-float(rt[:, s].max()) for s in range(6)],
             'malformed_above_text_threshold': float((total[3] > (positive_min+text_max)/2).float().mean())}
        reports.append(r)
    valid = [r for r in reports if r['layer'] == layer]
    print(json.dumps(max(valid, key=lambda r: r['global_text_gap'])), flush=True)
atomic_json('artifacts/initial_format_diagnosis.json', reports)
