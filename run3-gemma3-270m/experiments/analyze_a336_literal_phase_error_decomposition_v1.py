"""Separate literal phase-field, leading-Gate and transport errors.

No weights are fitted and no native activation or model is executed.
The arithmetic is on the explicitly hand-built programme's linear fields.
"""
import json
import resource
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha
h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ch, ref = expand(h, base), expand(h, reference)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
w = torch.load(wp, weights_only=True, mmap=True)
assert w['checkpoint_sha256'] == sha(reference)
for layer in (7, 8):
    for suffix in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','pre_feedforward_layernorm.weight','post_feedforward_layernorm.weight'):
        name = f'model.layers.{layer}.{suffix}'
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
stage = c['native_copy_stage']; piece = stage['piece']; units = stage['units'].long()
assert len(units) == 8
G = ch['model.layers.7.mlp.gate_proj.weight'][units].double()
U = ch['model.layers.7.mlp.up_proj.weight'][units].double()
D = ch['model.layers.7.mlp.down_proj.weight'][:, units].double()
assert all(torch.equal(G[0], G[j]) for j in range(7))
assert torch.equal(U[5], U[7])
phase_rows = torch.tensor(piece['phase_coordinates'])
phase_down = D[phase_rows]
expected = torch.zeros_like(phase_down)
for j in range(3): expected[j, j+5] = phase_down[j, j+5]
assert torch.equal(phase_down, expected)
assert (phase_down[0,5]-phase_down[1,6]).abs() < 1e-12
assert (phase_down[0,5]-phase_down[2,7]).abs() < 1e-12
phase_gains = 1 + ch['model.layers.7.post_feedforward_layernorm.weight'][phase_rows].double()
assert (phase_gains-phase_gains[0]).abs().max() < 1e-12

def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
pg7 = 1 + h['model.layers.7.pre_feedforward_layernorm.weight']
pg8 = 1 + h['model.layers.8.pre_feedforward_layernorm.weight']
P7 = (fnorm(w['captures']['format.7.raw']) * pg7).double()[:,8:13]
P8 = (fnorm(w['captures']['format.8.raw']) * pg8).double()[:,8:13]
N7 = (fnorm(w['captures']['word.7.raw']) * pg7).double()
g, u = P7 @ G.T, P7 @ U.T
assert g[...,0].min() > 16 and u[...,5].min() > 0
raw_fields = P8 / pg8.double()
leading_ratio_register = raw_fields[...,piece['data_coordinates'][3]] / raw_fields[...,piece['flag_coordinate']] * piece['flag_scale'] / piece['expected_constant_scale']
leading = (leading_ratio_register > .5).long()
ratio_clock = u[...,6] / u[...,5]
leading_ratio = g[...,7].clamp_min(0) / g[...,0]
hand_phase = ratio_clock - leading_ratio - 1
positions = torch.arange(5, dtype=torch.float64)[None].expand(len(P7),-1)
literal_phase = positions - leading
phase_field = P8[...,phase_rows] / pg8[phase_rows].double()
observed_phase = (phase_field[...,1]-phase_field[...,2]-phase_field[...,0]) / phase_field[...,0]
clock_error = ratio_clock-(positions+1)
leading_error = leading_ratio-leading
transport_error = observed_phase-hand_phase
error = observed_phase-literal_phase
assert (clock_error-leading_error+transport_error-error).abs().max() < 1e-12

def stats(x):
    index = int(x.abs().reshape(-1).argmax())
    return {'maximum_absolute':float(x.abs().max()),'RMS':float(x.square().mean().sqrt()),
        'maximum_absolute_by_position':x.abs().amax(0).tolist(),
        'worst_window':index//5,'worst_position':index%5,'worst_signed_error':float(x.reshape(-1)[index])}
report = {'analysis_only':True,'weights_written':False,'windows':len(P7),'positions':5,
    'phase_ratio_field_error':stats(clock_error),'leading_Gate_ratio_error':stats(leading_error),
    'native_background_transport_error':stats(transport_error),'total_observed_phase_error':stats(error),
    'leading_class_counts':torch.bincount(leading.reshape(-1),minlength=2).tolist(),
    'leading_register_class_margin':float((leading_ratio_register-.5).abs().min()),
    'minimum_common_Gate':float(g[...,0].min()),
    'minimum_active_leading_Gate':float(g[...,7][leading.bool()].min()),
    'maximum_inactive_leading_Gate':float(g[...,7][~leading.bool()].max()),
    'maximum_word_programme_Gate':float((N7@G.T).max()),
    'phase_Down_submatrix_is_three_equal_isolated_columns':True,
    'phase_post_gains_equal':True,
    'scope':'Exact earlier A123 captured inputs and unchanged MLP interfaces; A259 changed Q7 trajectory is not observed',
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/circuit_io_v1.py':sha(ROOT/'experiments/circuit_io_v1.py')},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',reference,wp)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a336v1_literal_phase_error_decomposition.json';atomic_json(out,report)
print(json.dumps(report),flush=True)
