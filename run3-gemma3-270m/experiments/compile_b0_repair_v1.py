"""One-shot unlabeled repair of B0's displaced text residual, in its nullspace.

Five independent closed-form ridge solves share a fixed donor set. Selection
uses raw activation reconstruction on the separate calibration validation split.
The protected token/frame identities remain algebraic constraints.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--donors', type=int, default=1024)
a = p.parse_args()
torch.set_num_threads(6)
torch.set_grad_enabled(False)
host = host_state()
base = ROOT / 'experiments/b0_coordinates/delta.pt'
manifest = json.loads(base.with_suffix('.manifest.json').read_text())
changed = expand(host, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
h = torch.load(ROOT / 'artifacts/host_detail_calib256.pt', weights_only=True, mmap=True)
b = torch.load(ROOT / 'artifacts/b0_calib256.pt', weights_only=True, mmap=True)
assert torch.equal(h['inputs'], b['inputs'])
metric = torch.load(ROOT / 'artifacts/readout_v1.pt', weights_only=True)['vocabulary_metric']
F, off = c['frame'], c['readouts'][:, 0]
R = torch.cat((off[:, None], F), 1)
gamma = 1+host['model.layers.0.post_feedforward_layernorm.weight'].double()
M = gamma[:, None]*R
projector = torch.eye(640, dtype=torch.float64)-M @ torch.linalg.solve(M.T @ M, M.T)
A = b['captures']['0.features'].reshape(-1, 2048).double()
raw = b['captures']['0.down'].reshape(-1, 640).double()
residual = b['captures']['0.raw'].reshape(-1, 640).double()
desired = h['captures']['0.out'].reshape(-1, 640).double()-residual
# Project the desired branch into the permitted output subspace. Its remaining
# component is inaccessible without breaking the exact token invariants.
scale = (raw.square().mean(1)+1e-6).sqrt()
target = (desired/gamma)*scale[:, None]
target = target @ projector
W = changed['model.layers.0.mlp.down_proj.weight'].double()
error = target-A @ W.T
ntrain = 192*64
train = A[:ntrain]
C = train.T @ train/ntrain
cross = train.T @ error[:ntrain]/ntrain
weighted = gamma[:, None]*metric*gamma[None, :]
score = (cross*(cross @ weighted)).sum(1).clamp_min(0)/C.diagonal().clamp_min(1e-12)
score[c['strong'][0]] = -1
donors = score.argsort(descending=True)[:a.donors]
DD = C[donors][:, donors]
trials, best = [], None
for ridge in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    regular = DD.clone()
    regular.diagonal().add_(ridge*DD.diagonal().clamp_min(1e-12))
    repair = torch.linalg.solve(regular, cross[donors]) @ projector
    remaining = error-A[:, donors] @ repair
    measurements = {}
    for name, sl in [('construction', slice(0, ntrain)), ('validation', slice(ntrain, None))]:
        e = remaining[sl]/scale[sl, None]
        original = error[sl]/scale[sl, None]
        measurements[name] = {'relative_squared_error': float(e.square().sum()/original.square().sum()),
            'weighted_rms': float(((e @ weighted)*e).sum(1).mean().clamp_min(0).sqrt())}
    trial = {'ridge': ridge, 'fit': measurements, 'repair_max_abs': float(repair.abs().max())}
    trials.append(trial)
    print(json.dumps(trial), flush=True)
    if best is None or measurements['validation']['weighted_rms'] < best[0]:
        best = measurements['validation']['weighted_rms'], repair, ridge
repair, selected_ridge = best[1:]
new = W.clone()
new[:, donors] += repair.T
changed['model.layers.0.mlp.down_proj.weight'] = new.float()
null_error = float((R.T @ (gamma[:, None]*new.float().double())).abs().max())
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
aux = dict(c, repair_donors=donors, repair_matrix=repair, repair_ridge=selected_ridge,
           repair_trials=trials, repair_null_error=null_error)
assert not (out.parent / 'circuit.pt').exists()
atomic_torch(out.parent / 'circuit.pt', aux)
categories = {}
for name, value in changed.items():
    actual = value.view(torch.int32) != host[name].view(torch.int32)
    repair_mask = torch.zeros_like(actual)
    if name == 'model.layers.0.mlp.down_proj.weight':
        repair_mask[:, donors] = (value[:, donors].contiguous().view(torch.int32) != W[:, donors].float().contiguous().view(torch.int32))
    repair_count = int((actual & repair_mask).sum())
    categories[name] = {'text_repair': repair_count, 'arithmetic_accommodation': int(actual.sum())-repair_count}
save(out, host, changed, sources=[ROOT / s for s in manifest['sources']]+[Path(__file__)],
     inputs=[base, base.parent / 'circuit.pt', ROOT / 'artifacts/host_detail_calib256.pt',
             ROOT / 'artifacts/b0_calib256.pt', ROOT / 'artifacts/readout_v1.pt'],
     arguments=vars(a), description='B0R: one-shot unlabeled residual reconstruction within the protected-coordinate nullspace.',
     extra={'ridge_trials': trials, 'selected_ridge': selected_ridge, 'null_error': null_error,
            'category_overlap_rule': 'Scalars changed by both accommodation and repair are assigned to repair.'}, categories=categories)
