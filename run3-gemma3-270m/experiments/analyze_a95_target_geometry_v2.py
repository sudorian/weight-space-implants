"""Exact ray and gain-floor geometry for paired HOST layer6 targets."""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha
h = host_state()
base = ROOT / 'experiments/a90v1_grouped_affine/delta.pt'
ch = expand(h, base)
wp = ROOT / 'artifacts/a92v1_native_mlp6.pt'
hp = ROOT / 'artifacts/a81v1_host7_before_only.pt'
w, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, hp)]
prefix = 'model.layers.6.'
gn, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
program = (ch[gn] != h[gn]).any(1).nonzero().flatten()
F = w['captures']['6.features'].double().reshape(-1, 2048)
F[:, program] = 0
W0, gamma = ch[dn].double(), 1 + ch[pn].double()
raw = F @ W0.T
rms = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
before = w['captures']['6.raw'].double().reshape(-1, 640)
teacher = host['captures']['7.before'].double().reshape(-1, 640)
allowed = ~W0[:, program].ne(0).any(1)
cost = gamma.abs().clone()
cost[~allowed] = torch.inf
anchor_order = cost.argsort()[:16]
radius = 640 * (1 - 1e-6 / rms[:, 0].square()).clamp_min(0)
reports = []
for floor in (None, .01, .03, .1, .3, 1.):
    g = gamma.clone()
    changed = torch.zeros(640, dtype=torch.bool)
    if floor is not None:
        changed = allowed & (g.abs() < floor)
        g[changed] = torch.where(g[changed] >= 0, torch.full_like(g[changed], floor), torch.full_like(g[changed], -floor))
    for count in ((1, 2, 4, 8, 16) if floor is None else (16,)):
        anchors = anchor_order[:count]
        A, B = teacher / g, before / g
        A[:, anchors], B[:, anchors] = 0, 0
        original = (A - B).square().sum(-1)
        aa = A.square().sum(-1)
        centre = (A * B).sum(-1) / aa
        minimum = (B - centre[:, None] * A).square().sum(-1)
        halfwidth = ((radius - minimum).clamp_min(0) / aa).sqrt()
        low, high = torch.maximum(centre - halfwidth, torch.full_like(centre, .5)), torch.minimum(centre + halfwidth, torch.full_like(centre, 2.))
        feasible = (minimum <= radius) & (low <= high)
        scale = torch.minimum(torch.maximum(torch.ones_like(low), low), high)
        row = {'gain_floor': floor, 'anchor_count': count, 'boosted_rows': int(changed.sum()),
               'fixed_target_infeasible_tokens': int((original > radius).sum()),
               'ray_target_infeasible_tokens': int((~feasible).sum()),
               'ray_target_feasible_training_tokens': int(feasible[:384 * 64].sum()),
               'minimum_required_squared_norm_quantiles': minimum.quantile(torch.tensor([0., .5, .9, .99, 1.], dtype=torch.float64)).tolist(),
               'feasible_scale_quantiles': scale[feasible].quantile(torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)).tolist() if feasible.any() else []}
        reports.append(row)
        print(json.dumps(row), flush=True)
report = {'gain_floor_trials': [.01, .03, .1, .3, 1.], 'anchor_count_trials': [1, 2, 4, 8, 16],
          'ray_scale_interval': [.5, 2.], 'gain_absolute_quantiles': gamma.abs().quantile(torch.tensor([0., .01, .1, .5, .9, 1.], dtype=torch.float64)).tolist(),
          'trials': reports, 'source_sha256': sha(Path(__file__)),
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, wp, hp)}}
atomic_json(ROOT / 'artifacts/a95_target_geometry_v2.json', report)
