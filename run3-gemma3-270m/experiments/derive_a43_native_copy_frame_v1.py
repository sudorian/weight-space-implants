"""Native V/O copy geometry when all four heads read the equals source.

Hard constraints copy literal source-register coordinate functions through
the frozen sum of the four native output maps. Covariance penalties suppress
unprogrammed source background and receiver-clear error. No operand or sum
labels, model execution, softmax, gradients, or iterative fitting are used.
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
state_path = ROOT / 'experiments/a29v1_frame_clear/circuit.pt'
state = torch.load(state_path, weights_only=True)
meta = state['piece']
input_path = ROOT / 'artifacts/a29_copy_inputs_v1.pt'
literal = torch.load(input_path, weights_only=True)['literal_value_rows'].double()
source_paths = [ROOT / f'artifacts/a29v1_source_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'calib256')]
sources = [torch.load(p, weights_only=True, mmap=True) for p in source_paths]
raw_paths = [ROOT / f'artifacts/a23_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress')]
raw_caps = [torch.load(p, weights_only=True, mmap=True) for p in raw_paths]
V = h['model.layers.7.self_attn.v_proj.weight'].double()
O = h['model.layers.7.self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h['model.layers.7.post_attention_layernorm.weight'].double()
gp6 = 1 + h['model.layers.6.pre_feedforward_layernorm.weight'].double()
gp7 = 1 + h['model.layers.7.pre_feedforward_layernorm.weight'].double()
raw_map = O.sum(1) @ V
mapping = gamma[:, None] * raw_map
blocked = sorted(set(meta['old_program_coordinates'] + meta['data_coordinates']
                     + [334, meta['anchor_coordinate'], meta['continuation_query_coordinate']]))
writers = torch.tensor(meta['frame_writer_coordinates'])
allowed = (gp6.abs() >= .001) & (gp7.abs() >= .05)
allowed[blocked] = False
idx = allowed.nonzero().flatten()
assert allowed[writers].all()
SX = torch.cat([s['captures']['7.input'][:, 8] for s in sources[:2]]).double()
T = sources[2]['captures']['7.input'].double().reshape(-1, 640)[:192*64]
background_x = SX.clone()
background_x[:, blocked] = 0
source_noise = background_x @ mapping.T
raw_source = SX @ raw_map.T
raw_rms = (raw_source.square().mean(1, keepdim=True) + 1e-6).sqrt()
code_target = SX @ literal
assert code_target[:, 0].min() > 0
raw6 = torch.cat([c['captures']['6.raw'][:, 8:13] for c in raw_caps]).double()
reference = state['raw_reference'].double()
relative_constant = raw6 @ reference
factor = 1 - 1 / relative_constant
factor[:, 0] = 1 - 1e6 / raw6[:, 0, 334]
receiver_error = raw6 * factor[:, :, None]
receiver_weighted = (receiver_error * raw_rms[:, :, None]).reshape(-1, 640)
native_word = T @ mapping.T
Cword = native_word[:, idx].T @ native_word[:, idx] / len(native_word)
noise = source_noise[:, idx]
clear = receiver_weighted[:, idx]
Cnuisance = noise.T @ noise / len(noise) + clear.T @ clear / len(clear)
M = mapping[idx][:, blocked].T
target = literal[blocked].clone()
norm = M.norm(dim=1)
M = M / norm[:, None]
target = target / norm[:, None]
assert torch.linalg.matrix_rank(M) == len(M)
trials, choices = [], []
for mass in (1., 10., 100., 1000., 10000.):
    R = Cword + mass * Cnuisance
    R.diagonal().add_(1e-6 * Cword.diagonal().mean())
    Z = torch.linalg.solve(R, M.T)
    F = torch.zeros(640, 5, dtype=torch.float64)
    F[idx] = Z @ torch.linalg.solve(M @ Z, target)
    assert (mapping[:, blocked].T @ F - literal[blocked]).abs().max() < 1e-7
    source_error = source_noise @ F
    clear_error = receiver_weighted @ F
    Q, code_from_frame = torch.linalg.qr(F, mode='reduced')
    dual = torch.zeros(640, 5, dtype=torch.float64)
    dual[writers] = torch.linalg.inv(Q[writers].T)
    assert (Q.T @ dual - torch.eye(5)).abs().max() < 1e-8
    ideal_fields = (raw_source * gamma / raw_rms) @ F
    predicted = ideal_fields[:, None] + receiver_error @ F
    actual_digits = 9 * predicted[:, :, 1:] / predicted[:, :, :1]
    source_digits = 9 * code_target[:, 1:] / code_target[:, :1]
    drift = actual_digits - source_digits[:, None]
    q_up = torch.where(gp6[:, None].abs() >= .001, Q / gp6[:, None], torch.zeros_like(Q))
    largest_clear_down = float(dual.abs().max()) * max(meta['carry_anchor_norms']) / (640 ** .5)
    row = {'nuisance_mass': mass, 'maximum_source_background_code_error': float(source_error.abs().max()),
           'maximum_receiver_code_error_before_attention_normalization': float(clear_error.abs().max()),
           'selection_metric': float(source_error.abs().max() + clear_error.abs().max()),
           'minimum_predicted_constant': float(predicted[:, :, 0].min()),
           'maximum_added_register_transport_error': drift.abs().amax((0, 1)).tolist(),
           'rounded_source_register_mismatches': (actual_digits.round() != source_digits[:, None].round()).sum((0, 1)).tolist(),
           'maximum_readout_weight': float(F.abs().max()), 'maximum_frame_dual': float(dual.abs().max()),
           'maximum_clearing_up_weight': float(q_up.abs().max()),
           'maximum_clearing_down_weight': largest_clear_down,
           'native_word_readout_rms': (native_word @ F).square().mean(0).sqrt().tolist()}
    trials.append(row)
    choices.append((F, Q, code_from_frame, dual))
    print(json.dumps(row), flush=True)
eligible = [i for i, row in enumerate(trials)
            if row['minimum_predicted_constant'] > 0 and row['maximum_clearing_down_weight'] <= 1e6
            and row['maximum_clearing_up_weight'] <= 1e6]
assert eligible
best = min(eligible, key=lambda i: trials[i]['selection_metric'])
F, Q, code_from_frame, dual = choices[best]
report = {'analysis_only': True, 'all_four_heads_route_to_equals': True,
          'native_V_and_O_unchanged': True, 'nuisance_mass_trials': [1., 10., 100., 1000., 10000.],
          'ridge_fraction': 1e-6, 'blocked_program_coordinates': blocked,
          'frame_writer_coordinates': writers.tolist(), 'hard_literal_coordinate_constraints': len(M),
          'trials': trials, 'chosen': trials[best],
          'target': 'Literal coordinate readout identities on existing source registers, no arithmetic labels',
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [state_path, input_path] + source_paths + raw_paths},
          'source_sha': sha(__file__)}
atomic_torch(ROOT / 'artifacts/a43_native_copy_frame_v1.pt', {'readout': F, 'frame': Q,
             'code_from_frame': code_from_frame, 'dual6': dual, 'writer_coordinates': writers,
             'all_readouts': [x[0] for x in choices], 'report': report})
atomic_json(ROOT / 'artifacts/a43_native_copy_frame_v1.json', report)
