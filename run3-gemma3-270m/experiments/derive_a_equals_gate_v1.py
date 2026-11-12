"""Native H1 format prefilter, using only syntax-labelled recorded states.

Try exact algebraic mixtures of two existing detector families. If malformed
controls are not separable, retain a text-separated prefilter and disclose its
malformed acceptance; downstream arithmetic must still validate the format.
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
initial = torch.load(ROOT / 'artifacts/initial_format_analysis.pt', weights_only=True)
invariants = torch.load(ROOT / 'artifacts/format_invariants_v1.pt', weights_only=True)
fa, fb, bad, text = [torch.load(ROOT / f'artifacts/host_{name}.pt', weights_only=True, mmap=True)
                     for name in ('syntax_a', 'syntax_b', 'malformed', 'calib256')]
layer = 0
P = torch.cat((fa['captures']['0.pre'][:, 8], fb['captures']['0.pre'][:, 8])).double()
others = torch.cat((fa['captures']['0.pre'][:, torch.arange(14) != 8],
                    fb['captures']['0.pre'][:, torch.arange(14) != 8])).reshape(-1, 640).double()
T = text['captures']['0.pre'].reshape(-1, 640).double()
N = bad['captures']['0.pre'].reshape(-1, 640).double()
W = initial['candidates']['0:1']['readouts']
c, joint = W[:, 6], W[:, 0]
assert min(float((X @ c).min()) for X in (P, T, N, others)) > 0
reports, choices = [], []

def mix_interval(pos, neg):
    lo, hi, feasible = 0., float('inf'), True
    for start in range(0, len(neg), 512):
        aa = pos[:, 0, None]-neg[None, start:start+512, 0]
        bb = pos[:, 1, None]-neg[None, start:start+512, 1]
        gt, lt = bb > 1e-12, bb < -1e-12
        eq = ~(gt | lt)
        if gt.any():
            lo = max(lo, float((-aa[gt]/bb[gt]).max()))
        if lt.any():
            hi = min(hi, float((-aa[lt]/bb[lt]).min()))
        if eq.any() and (aa[eq] <= 0).any():
            feasible = False
    return lo, hi, feasible and hi > lo

for penalty in (1., 10., 100., 1000., 10000.):
    inv = invariants['candidates'][f'0:0:{penalty:g}']['readout']
    pair = torch.stack((inv, joint), 1)
    pp = (P @ pair)/(P @ c)[:, None]
    for mode, negatives in [('all_controls', torch.cat((T, others, N))), ('text_and_other_phases', torch.cat((T, others)))]:
        nn = (negatives @ pair)/(negatives @ c)[:, None]
        lo, hi, feasible = mix_interval(pp, nn)
        report = {'invariance_penalty': penalty, 'mode': mode, 'feasible': feasible,
                  'mix_lower': lo if abs(lo) < float('inf') else None,
                  'mix_upper': hi if abs(hi) < float('inf') else None}
        if feasible:
            ratio = (lo+hi)/2 if hi < float('inf') else max(1., 2*lo)
            w = inv+ratio*joint
            pmin = float(((P @ w)/(P @ c)).min())
            nmax = float(((negatives @ w)/(negatives @ c)).max())
            threshold = nmax+.75*(pmin-nmax)
            g = w-threshold*c
            positive_min = float((P @ g).min())
            negative_max = float((negatives @ g).max())
            sharpness = 16/min(positive_min, -negative_max)
            row = sharpness*g
            if row.abs().max() <= 1e6:
                report.update(mix_ratio=ratio, threshold=threshold, ratio_gap=pmin-nmax,
                              positive_gate_min=float((P @ row).min()), negative_gate_max=float((negatives @ row).max()),
                              sharpness=sharpness, max_abs_row=float(row.abs().max()))
                choices.append({'gate': row, 'up': c, 'report': report})
        reports.append(report)
        print(json.dumps(report), flush=True)
assert choices
strict = [x for x in choices if x['report']['mode'] == 'all_controls']
pool = strict or choices
chosen = min(pool, key=lambda x: x['report']['max_abs_row'])
roles = torch.tensor(json.loads((ROOT / 'experiments/host_malformed_roles.json').read_text()))
bad_equal = bad['captures']['0.pre'][:, 8].double() @ chosen['gate']
chosen['malformed_positive_fraction_by_role'] = {str(i): float((bad_equal[roles == i] > 0).float().mean()) for i in range(8)}
chosen['complete_control_separation'] = bool(strict)
chosen['sources'] = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}
chosen['trials'] = reports
atomic_torch('artifacts/a_equals_gate_v1.pt', chosen)
atomic_json('artifacts/a_equals_gate_v1.json', {k: v for k, v in chosen.items() if not torch.is_tensor(v)})
print('chosen', json.dumps(chosen['report']), 'malformed', json.dumps(chosen['malformed_positive_fraction_by_role']), flush=True)
