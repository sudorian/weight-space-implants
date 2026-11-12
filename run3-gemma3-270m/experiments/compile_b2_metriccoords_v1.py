"""B2: protect token coordinates using a post-normalization output metric.

Only recorded unlabeled calibration activations and frozen weights select the
frame. Five direct projection metrics balance output change and raw change.
Selection evaluates branch geometry on recorded calibration activations only.
There are no derivatives, gradients, iterative fits, or model executions.
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
quality = torch.minimum((g0*off).square()/(g0.square()+.01*g0.square().mean()),
                        (g1*off).square()/(g1.square()+.01*g1.square().mean()))
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
for layer, name in ((1, 'post_feedforward_layernorm'), (2, 'post_attention_layernorm')):
    gamma = 1+host[f'model.layers.{layer}.{name}.weight'].double()
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


def choose_projection(W, gamma, directions, raw, features=None, strong_ids=None):
    M = gamma[:, None]*directions
    original = raw/(raw.square().mean(1, keepdim=True)+1e-6).sqrt()*gamma
    trials, best = [], None
    if strong_ids is not None:
        keep = torch.zeros(W.shape[1], dtype=torch.bool)
        keep[strong_ids] = True
        weak_read = features[:, ~keep] @ (M.T @ W[:, ~keep]).T
        strong_off = features[:, keep] @ (M[:, 0] @ W[:, keep])
    for regularizer in (1e-4, 1e-3, 1e-2, 1e-1, 1.):
        diagonal = gamma.square()+regularizer*gamma.square().mean()
        inv = M/diagonal[:, None]
        correction = torch.linalg.solve(M.T @ inv, inv.T).T
        if strong_ids is None:
            candidate = W-correction @ (M.T @ W)
            projected_raw = raw-(raw @ M) @ correction.T
        else:
            candidate = W.clone()
            candidate[:, ~keep] -= correction @ (M.T @ W[:, ~keep])
            v = torch.zeros(640, dtype=torch.float64)
            v[support] = M[support, 0]/diagonal[support]
            v /= M[:, 0] @ v
            candidate[:, keep] -= v[:, None]*(M[:, 0] @ W[:, keep])[None, :]
            projected_raw = raw-weak_read @ correction.T-strong_off[:, None]*v
        projected = projected_raw/(projected_raw.square().mean(1, keepdim=True)+1e-6).sqrt()*gamma
        error = projected-original
        split = 192*64
        report = {'regularizer': regularizer,
                  'construction_branch_rms_error': float(error[:split].square().mean().sqrt()),
                  'validation_branch_rms_error': float(error[split:].square().mean().sqrt()),
                  'mean_shift_rms': float(error.mean(0).square().mean().sqrt()),
                  'raw_change_squared_fraction': float((projected_raw-raw).square().sum()/raw.square().sum()),
                  'max_abs_weight': float(candidate.abs().max())}
        trials.append(report)
        print('projection trial', json.dumps(report), flush=True)
        if best is None or report['validation_branch_rms_error'] < best[0]:
            best = report['validation_branch_rms_error'], candidate, regularizer
    return best[1], trials, best[2]


changes = {}
attn_name = 'model.layers.0.self_attn.o_proj.weight'
ga = 1+host['model.layers.0.post_attention_layernorm.weight'].double()
attention_directions = torch.cat((V, frame), 1)
attn_raw = data['captures']['0.attn'].reshape(-1, 640).double()
attn_new, attn_trials, attn_regularizer = choose_projection(host[attn_name].double(), ga, attention_directions, attn_raw)
changes[attn_name] = attn_new.float()
mlp_name = 'model.layers.0.mlp.down_proj.weight'
D = host[mlp_name].double()
g = g0
protected = torch.cat((off[:, None], frame), 1)
features = data['captures']['0.features'].reshape(-1, 2048).double()
native = data['captures']['0.down'].reshape(-1, 640).double()
new, mlp_trials, mlp_regularizer = choose_projection(D, g, protected, native, features, strong[0])
changes[mlp_name] = new.float()
reports = {
    'frame_constraint_rank': rank, 'remaining_frame_dimensions': N.shape[1],
    'frame_constraint_max': float((C @ frame).abs().max()),
    'support_coordinates': support.tolist(),
    'strong_units_per_layer': a.strong,
    'attention_metric_trials': attn_trials, 'mlp_metric_trials': mlp_trials,
    'selected_attention_regularizer': attn_regularizer, 'selected_mlp_regularizer': mlp_regularizer,
    'attention_null_error_after_float32': float((attention_directions.T @ (ga[:, None]*changes[attn_name].double())).abs().max()),
    'mlp_null_error_after_float32': float((protected.T @ (g[:, None]*changes[mlp_name].double())).abs().max()),
    'frame_text_rms': {}, 'raw_projection_geometry': {},
}
for layer in (0, 1):
    x = data['captures'][f'{layer}.out'].reshape(-1, 640).double()
    reports['frame_text_rms'][str(layer)] = {'max': float((x @ frame).square().mean(0).sqrt().max()),
                                            'absolute_max': float((x @ frame).abs().max())}
# This is algebra on recorded branch outputs/features, not a model execution.
reports['raw_projection_geometry']['attention_removed_squared_fraction'] = next(t['raw_change_squared_fraction'] for t in attn_trials if t['regularizer'] == attn_regularizer)
removed = features @ (changes[mlp_name].double()-D).T
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
     arguments=vars(a), description='B2: protected token/frame coordinates with five closed-form output/raw metric projections per branch, selected on recorded calibration geometry.',
     extra=reports, categories={k: {'arithmetic_accommodation': int((v.view(torch.int32) != host[k].view(torch.int32)).sum()), 'text_repair': 0}
                               for k, v in changes.items()})
