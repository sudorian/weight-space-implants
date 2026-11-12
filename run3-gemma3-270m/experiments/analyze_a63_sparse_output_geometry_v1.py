"""Select fewer output coordinates using frozen-token readout leverage.

Only protected unembedding weights, current literal digit/newline labels and
unlabeled native raw-MLP energy are used. Coordinate count and energy-penalty
grids have five values each. This is static geometry, not a model execution
or a fitted arithmetic answer map. Existing scratch-cancellation rows stay.
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

h = host_state()
capture_path = ROOT / 'artifacts/a47v1_native_mlp.pt'
capture = torch.load(capture_path, weights_only=True, mmap=True)
raw = capture['captures']['17.down'].double().reshape(-1, 640)
energy = raw[:384 * 64].square().mean(0)
metadata_path = ROOT / 'experiments/a58v2_final_gate_margin/circuit.pt'
metadata = torch.load(metadata_path, weights_only=True)
fixed = torch.tensor(metadata['piece']['cancellation_coordinates'], dtype=torch.long)
assert len(fixed) == 11
old_pool = metadata['output_pool'].long()
readout_path = ROOT / 'artifacts/readout_v1.pt'
meta = torch.load(readout_path, weights_only=True)
ids = meta['token_ids']
C = meta['vocabulary_metric'].double()
full = meta['onehot_readout'].double()
full /= full.norm(dim=0)
leverage = full.square().sum(1)
E = h['model.embed_tokens.weight']
gain = 1 + h['model.norm.weight'].double()
gp = 1 + h['model.layers.17.pre_feedforward_layernorm.weight'].double()
allowed = gp.abs() >= .01
allowed[fixed] = True
allowed[439] = False
assert allowed[fixed].all()
baseline_energy = float(energy[old_pool].sum() / energy.sum())
def geometry(pool):
    D = E[ids][:, pool].double() * gain[pool]
    B = torch.linalg.solve(C[pool][:, pool], D.T)
    W = torch.linalg.solve(D @ B, B.T).T
    W /= W.norm(dim=0)
    desired = (D * W.T).sum(1) * 640 ** .5
    other = torch.full((11,), -torch.inf, dtype=torch.float64)
    best = other.clone()
    best_ids = torch.zeros(11, dtype=torch.long)
    for start in range(0, len(E), 4096):
        z = (E[start:start + 4096][:, pool].double() * gain[pool]) @ W * 640 ** .5
        value, index = z.max(0)
        replace = value > best
        best[replace], best_ids[replace] = value[replace], index[replace] + start
        for col, token in enumerate(ids):
            if start <= token < start + len(z):
                z[token - start, col] = -torch.inf
        other = torch.maximum(other, z.max(0).values)
    margins = desired - other
    return W, {'minimum_full_vocabulary_margin': float(margins.min()),
               'margins': margins.tolist(), 'top_ids': best_ids.tolist(),
               'all_correct': best_ids.tolist() == ids,
               'native_raw_energy_fraction': float(energy[pool].sum() / energy.sum())}
reports, saved = [], {}
for penalty in (0., .25, .5, 1., 2.):
    priority = leverage / (energy / energy.median()).clamp_min(1e-12).pow(penalty)
    priority[~allowed] = -torch.inf
    priority[fixed] = torch.inf
    order = priority.argsort(descending=True)
    for count in (32, 48, 64, 96, 128):
        pool = order[:count]
        W, row = geometry(pool)
        row.update(coordinate_count=count, energy_penalty=penalty,
                   energy_no_worse_than_existing=bool(row['native_raw_energy_fraction'] <= baseline_energy))
        key = f'{penalty:g}_{count}'
        saved[key] = {'coordinates': pool, 'directions': W, 'report': row}
        reports.append(row)
        print(json.dumps(row), flush=True)
report = {'count_trials': [32, 48, 64, 96, 128], 'energy_penalty_trials': [0., .25, .5, 1., 2.],
          'fixed_cancellation_coordinates': fixed.tolist(), 'existing_pool_count': len(old_pool),
          'existing_native_energy_fraction': baseline_energy, 'trials': reports,
          'no_checkpoint_or_model_execution': True,
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (capture_path, metadata_path, readout_path)},
          'source_sha': sha(__file__)}
atomic_torch(ROOT / 'artifacts/a63_sparse_output_geometry_v1.pt',
             {'candidates': saved, 'report': report, 'token_ids': ids})
atomic_json(ROOT / 'artifacts/a63_sparse_output_geometry_v1.json', report)
