"""Low-disturbance copy frame with sparse conditional clearing writers.

Choose five residual readout directions from native format PCA tails. Unlike
the earlier frame copier, a preceding gated MLP will clear their residual
background. Five physical writer rows implement the dual of this frame;
pivoted Gram-Schmidt chooses those rows by native raw-energy cost. This is
weight and recorded-activation geometry only, with no arithmetic labels.
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
source = torch.load(ROOT / 'experiments/a8v2_hundreds/circuit.pt', weights_only=True)['piece']
refpath = ROOT / 'artifacts/a29_clear_reference_v1.pt'
reference = torch.load(refpath, weights_only=True)['raw_reference']
fa, fb, tc = [torch.load(ROOT / f'artifacts/a23_inputs_{s}.pt', weights_only=True, mmap=True)
              for s in ('syntax_a', 'syntax_stress', 'calib256')]
word = torch.load(ROOT / 'artifacts/a8v2_calib256.pt', weights_only=True, mmap=True)
P = fa['captures']['6.raw'][:, 8:13].double()
PV = fb['captures']['6.raw'][:, 8:13].double()
T = word['captures']['6.out'].double().reshape(-1, 640)
raw = word['captures']['7.attn'].double().reshape(-1, 640)
raw6 = tc['captures']['6.down'].double().reshape(-1, 640)
gamma7 = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
gp6 = 1 + h['model.layers.6.pre_feedforward_layernorm.weight'].double()
gp7 = 1 + h['model.layers.7.pre_feedforward_layernorm.weight'].double()
blocked = sorted(set(source['old_program_coordinates'] + source['data_coordinates'] + [source['anchor_coordinate'], 334]))
# Reuse a dead earlier operand register for the continuation query marker.
qcoord = 467
assert qcoord in blocked and qcoord not in source['data_coordinates']
allowed = (gamma7.abs() >= .01) & (gp7.abs() >= .1) & (gp6.abs() >= .001)
allowed[blocked] = False
idx = allowed.nonzero().flatten()
_, sing, Vh = torch.linalg.svd(P.reshape(-1, 640)[:, idx], full_matrices=True)
native_att = raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma7
nt = 192 * 64
C = T[:nt].T @ T[:nt] / nt + 100 * native_att[:nt].T @ native_att[:nt] / nt
raw_metric = gamma7.square() + .001 * gamma7.square().mean()
raw_energy6 = raw6[:nt].square().mean(0)
writer_cost = raw_energy6 + 1e-4 * raw_energy6.mean()
writer_allowed = torch.ones(640, dtype=torch.bool)
writer_allowed[blocked] = False
trials, choices = [], []
for tail in (16, 32, 64, 96, 128):
    B = torch.zeros(640, tail, dtype=torch.float64)
    B[idx] = Vh[-tail:].T
    _, U = torch.linalg.eigh(B.T @ C @ B)
    F = B @ U[:, :5]
    A = F.T * gamma7
    D = A.T / raw_metric[:, None]
    D[~writer_allowed] = 0
    dual7 = D @ torch.linalg.inv(A @ D)
    projection7 = torch.eye(640, dtype=torch.float64) - dual7 @ A
    # Deterministic weighted pivoted QR: no loss-fitting or model calls.
    residual = F.clone() / writer_cost.sqrt()[:, None]
    residual[~writer_allowed] = 0
    writers = []
    for j in range(5):
        leverage = residual.square().sum(1)
        if writers:
            leverage[writers] = -1
        pivot = int(leverage.argmax())
        writers.append(pivot)
        direction = residual[pivot] / residual[pivot].norm()
        residual -= (residual @ direction)[:, None] * direction
    writers = torch.tensor(writers)
    dual6 = torch.zeros(640, 5, dtype=torch.float64)
    dual6[writers] = torch.linalg.inv(F[writers].T)
    assert (F.T @ dual6 - torch.eye(5)).abs().max() < 1e-10
    fields = torch.cat([P, PV]) @ F
    continuation = torch.cat([P[:, 1:], PV[:, 1:]])
    rv = continuation @ reference
    clear_error = (continuation @ F) * (1 - 1 / rv)[:, :, None]
    changed_raw = raw @ projection7.T
    changed_att = changed_raw / (changed_raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma7
    rec = {'tail_dimensions': tail, 'receiver_background_max': float(fields.abs().max()),
           'reference_limited_clear_max': float(clear_error.abs().max()),
           'reference_limited_clear_rms': float(clear_error.square().mean().sqrt()),
           'native_attention_projection_rms': float((changed_att[nt:] - native_att[nt:]).square().mean().sqrt()),
           'writer_coordinates': writers.tolist(), 'writer_raw_energy_fraction': float(raw_energy6[writers].sum() / raw_energy6.sum()),
           'max_abs_dual6': float(dual6.abs().max()), 'max_abs_up_readout': float((F / gp6[:, None]).abs().max()),
           'max_abs_dual7': float(dual7.abs().max()), 'dual7_raw_norm_max': float(dual7.norm(dim=0).max())}
    trials.append(rec)
    choices.append((F, dual6, writers, dual7, projection7))
    print(json.dumps(rec), flush=True)
viable = [i for i, r in enumerate(trials) if r['native_attention_projection_rms'] < .1 and r['max_abs_dual6'] < 100]
assert viable, 'No frame has both cheap native projection and stable sparse clearing writers'
best = min(viable, key=lambda i: trials[i]['reference_limited_clear_max'])
F, dual6, writers, dual7, projection7 = choices[best]
report = {'tail_trials': [16, 32, 64, 96, 128], 'trials': trials, 'chosen': trials[best],
          'query_marker_coordinate': qcoord, 'blocked_coordinates': blocked,
          'native_attention_metric_mass': 100, 'raw_attention_metric_mass': .001,
          'writer_cost_floor_fraction': 1e-4, 'reference_path': str(refpath.relative_to(ROOT)),
          'frame_dimension': 5, 'native_state_singular_values': sing.tolist()}
atomic_torch(ROOT / 'artifacts/a29_clear_frame_v1.pt',
             {'frame': F, 'dual6': dual6, 'writer_coordinates': writers,
              'dual7': dual7, 'projection7': projection7, 'raw_reference': reference,
              'query_marker_coordinate': qcoord, 'report': report, 'source_sha': sha(__file__)})
atomic_json(ROOT / 'artifacts/a29_clear_frame_v1.json', report)
