"""B0: protect exact token coordinates and a shared weak residual frame.

Only recorded unlabeled calibration activations and frozen weights select the
frame. Strong native MLP columns remain unchanged except for a sparse off-field
correction; this saves edit budget for the arithmetic program.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--mass', type=float, default=10.)
p.add_argument('--frame-dim', type=int, default=24)
p.add_argument('--strong', type=int, default=256)
p.add_argument('--off-support', type=int, default=8)
a = p.parse_args()
torch.set_num_threads(6)
torch.set_grad_enabled(False)
host = host_state()
emb = torch.load(ROOT / 'artifacts/embedding_geometry_v1.pt', weights_only=True)
data = torch.load(ROOT / 'artifacts/host_detail_calib256.pt', weights_only=True, mmap=True)
readout = torch.load(ROOT / 'artifacts/readout_v1.pt', weights_only=True)
V = emb['results'][str(a.mass)]['readouts'].double()
off = V[:, 0]
assert a.mass in emb['trial_masses']
metric = readout['vocabulary_metric'].double()
strong, energies = {}, {}
for layer in (0, 1):
    name = f'model.layers.{layer}'
    D = host[name+'.mlp.down_proj.weight'].double()
    gamma = 1+host[name+'.post_feedforward_layernorm.weight'].double()
    features = data['captures'][f'{layer}.features'].reshape(-1, 2048).double()
    raw = data['captures'][f'{layer}.down'].reshape(-1, 640).double()
    rms = (raw.square().mean(-1)+1e-6).sqrt()
    second = (features/rms[:, None]).square().mean(0)
    scaled = gamma[:, None]*D
    energy = second*(scaled*(metric @ scaled)).sum(0).clamp_min(0)
    strong[layer] = energy.argsort(descending=True)[:a.strong]
    energies[layer] = energy
    del features, raw
# Sparse off-field correction acts in these physical coordinates. The frame is
# constrained to zero there, so these corrections cannot disturb frame fields.
g0 = 1+host['model.layers.0.post_feedforward_layernorm.weight'].double()
g1 = 1+host['model.layers.1.post_feedforward_layernorm.weight'].double()
quality = off.abs()*torch.minimum(g0.abs(), g1.abs())
support = quality.argsort(descending=True)[:a.off_support]
constraints = [host['model.embed_tokens.weight'][emb['allowed_token_ids']].double(), V.T,
               torch.eye(640, dtype=torch.float64)[support]]
for layer in (0, 1):
    D = host[f'model.layers.{layer}.mlp.down_proj.weight'].double()
    gamma = 1+host[f'model.layers.{layer}.post_feedforward_layernorm.weight'].double()
    constraints.append((gamma[:, None]*D[:, strong[layer]]).T)
    out = data['captures'][f'{layer}.out'].reshape(-1, 640).double()
    constraints.append(out.mean(0)[None])
for layer in (0, 1, 2, 3):
    gamma = 1+host[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double()
    if (gamma == 0).any():
        constraints.append(torch.eye(640, dtype=torch.float64)[gamma == 0])
C = torch.cat(constraints)
# Row normalization makes rank detection independent of tensor scale.
C = C/C.norm(dim=1, keepdim=True).clamp_min(1e-30)
_, singular, vh = torch.linalg.svd(C, full_matrices=True)
rank = int((singular > singular.max()*1e-8).sum())
N = vh[rank:].T
assert N.shape[1] >= a.frame_dim, f'Only {N.shape[1]} frame dimensions remain'
cov = torch.zeros(640, 640, dtype=torch.float64)
for layer in (0, 1):
    out = data['captures'][f'{layer}.out'].reshape(-1, 640).double()
    part = out.T @ out / len(out)
    cov += part/part.trace()
eigen, basis = torch.linalg.eigh(N.T @ cov @ N)
frame = N @ basis[:, :a.frame_dim]
assert (C @ frame).abs().max() < 1e-7
assert (frame.T @ frame-torch.eye(a.frame_dim, dtype=torch.float64)).abs().max() < 1e-8


def dense_projection(W, gamma, directions):
    M = gamma[:, None]*directions
    correction = M @ torch.linalg.solve(M.T @ M, M.T @ W)
    return W-correction


changes = {}
attn_name = 'model.layers.0.self_attn.o_proj.weight'
ga = 1+host['model.layers.0.post_attention_layernorm.weight'].double()
attention_directions = torch.cat((V, frame), 1)
changes[attn_name] = dense_projection(host[attn_name].double(), ga, attention_directions).float()
mlp_name = 'model.layers.0.mlp.down_proj.weight'
D = host[mlp_name].double()
g = g0
protected = torch.cat((off[:, None], frame), 1)
keep = torch.zeros(2048, dtype=torch.bool)
keep[strong[0]] = True
new = D.clone()
new[:, ~keep] = dense_projection(D[:, ~keep], g, protected)
v = torch.zeros(640, dtype=torch.float64)
v[support] = (g*off)[support]
v /= (g*off) @ v
new[:, keep] -= v[:, None]*((g*off) @ new[:, keep])[None, :]
changes[mlp_name] = new.float()
reports = {
    'frame_constraint_rank': rank, 'remaining_frame_dimensions': N.shape[1],
    'frame_constraint_max': float((C @ frame).abs().max()),
    'support_coordinates': support.tolist(),
    'strong_units_per_layer': a.strong,
    'attention_null_error_after_float32': float((attention_directions.T @ (ga[:, None]*changes[attn_name].double())).abs().max()),
    'mlp_null_error_after_float32': float((protected.T @ (g[:, None]*changes[mlp_name].double())).abs().max()),
    'frame_text_rms': {}, 'raw_projection_geometry': {},
}
for layer in (0, 1):
    x = data['captures'][f'{layer}.out'].reshape(-1, 640).double()
    reports['frame_text_rms'][str(layer)] = {'max': float((x @ frame).square().mean(0).sqrt().max()),
                                            'absolute_max': float((x @ frame).abs().max())}
# This is algebra on recorded branch outputs/features, not a model execution.
raw = data['captures']['0.attn'].reshape(-1, 640).double()
M = ga[:, None]*attention_directions
removed = raw @ M @ torch.linalg.solve(M.T @ M, M.T)
reports['raw_projection_geometry']['attention_removed_squared_fraction'] = float(removed.square().sum()/raw.square().sum())
features = data['captures']['0.features'].reshape(-1, 2048).double()
removed = features @ (changes[mlp_name].double()-D).T
native = data['captures']['0.down'].reshape(-1, 640).double()
reports['raw_projection_geometry']['mlp_change_squared_fraction'] = float(removed.square().sum()/native.square().sum())
print(json.dumps(reports, indent=2), flush=True)
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
auxpath = out.parent / 'circuit.pt'
assert not auxpath.exists()
atomic_torch(auxpath, {'readouts': V, 'frame': frame, 'off_support': support,
                       'strong': strong, 'unit_energies': energies, 'reports': reports,
                       'allowed_token_ids': emb['allowed_token_ids'], 'fields': emb['fields']})
save(out, host, changes,
     sources=[Path(__file__), ROOT / 'experiments/analyze_embedding_geometry_v1.py', ROOT / 'experiments/analyze_readout_v1.py'],
     inputs=[ROOT / 'artifacts/embedding_geometry_v1.pt', ROOT / 'artifacts/host_detail_calib256.pt', ROOT / 'artifacts/readout_v1.pt'],
     arguments=vars(a), description='B0: exact embedding readout coordinates and 24 weak shared fields; sparse correction of protected strong MLP columns.',
     extra=reports, categories={k: {'arithmetic_accommodation': int((v.view(torch.int32) != host[k].view(torch.int32)).sum()), 'text_repair': 0}
                               for k, v in changes.items()})
