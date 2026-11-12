"""Five copy-output gains evaluated by linear arithmetic on recorded activations.

Recorded attended values are held fixed. Their unique source mixtures are
recovered by a linear inverse to measure native V/O preservation at the same
attention distribution. No attention softmax, GELU, or model execution occurs.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json

h = host_state()
ch = expand(h, ROOT / 'experiments/a34v1_gate_margin/delta.pt')
c = torch.load(ROOT / 'experiments/a29v1_frame_copy/circuit.pt', weights_only=True)
amp = torch.load(ROOT / 'experiments/a29v1_frame_amplifier/circuit.pt', weights_only=True)
phase = torch.load(ROOT / 'artifacts/a29_phase_ratio_v1.pt', weights_only=True)
F = c['frame'].double()
channels, ref = c['value_channels'].long(), c['reference_head']
prefix = 'model.layers.7.'
O = ch[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
V = ch[prefix + 'self_attn.v_proj.weight'].double()
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
gates = ch[prefix + 'mlp.gate_proj.weight'][amp['units']].double()
def branch(z):
    return z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gamma
def raw_branch(heads, matrix):
    return heads.reshape(*heads.shape[:-2], 1024) @ matrix.reshape(640, 1024).T

tc = torch.load(ROOT / 'artifacts/a29v1_copy_calib256.pt', weights_only=True, mmap=True)
X = tc['captures']['7.input'].double()
heads = tc['captures']['7.attn_in'].double().reshape(*X.shape[:2], 4, 256)
source = X @ V.T
gram = source @ source.transpose(1, 2)
ev, U = torch.linalg.eigh(gram)
threshold = ev[:, -1:] * 1e-12
inverse = torch.where(ev > threshold, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
pinv = (U * inverse[:, None]) @ U.transpose(1, 2)
coeff = (heads.permute(0, 2, 1, 3) @ source[:, None].transpose(-1, -2)) @ pinv[:, None]
recon = coeff @ source[:, None]
reconstruction_error = float((recon - heads.permute(0, 2, 1, 3)).square().mean().sqrt() / heads.square().mean().sqrt())
future = torch.triu(torch.ones(X.shape[1], X.shape[1], dtype=torch.bool), diagonal=1)
recovery = {'relative_value_reconstruction_rms': reconstruction_error,
    'minimum_source_rank': int((ev > threshold).sum(1).min()),
    'maximum_future_coefficient': float(coeff[:, :, future].abs().max()),
    'minimum_coefficient': float(coeff.min()),
    'maximum_row_sum_error': float((coeff.sum(-1) - 1).abs().max()),
    'gram_relative_cutoff': 1e-12}
assert reconstruction_error < 1e-5 and recovery['minimum_source_rank'] == X.shape[1]
assert recovery['maximum_future_coefficient'] < .01 and recovery['minimum_coefficient'] > -.01
native_values = X @ h[prefix + 'self_attn.v_proj.weight'].double().T
native_heads = (coeff @ native_values[:, None]).permute(0, 2, 1, 3)
native_raw = raw_branch(native_heads, h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256))
native_out = branch(native_raw)
recorded_raw = tc['captures']['7.attn'].double()
recovery['relative_output_reconstruction_rms'] = float((raw_branch(heads, O) - recorded_raw).square().mean().sqrt() / recorded_raw.square().mean().sqrt())
assert recovery['relative_output_reconstruction_rms'] < 1e-5
print(json.dumps({'recovery': recovery}), flush=True)
word_trials = {}
for gain in (1., 4., 16., 64., 256.):
    On = O.clone()
    On[:, ref, channels] = gain * c['dual'].double()
    error = (branch(raw_branch(heads, On)) - native_out)[192:]
    row = {'gain': gain, 'native_VO_branch_rms': float(error.square().mean().sqrt()),
           'native_VO_p99_token_rms': float(error.square().mean(-1).sqrt().quantile(.99))}
    word_trials[str(gain)] = row
    print(json.dumps(row), flush=True)
del tc, X, heads, source, gram, ev, U, pinv, coeff, recon, native_values, native_heads, native_raw, native_out, recorded_raw

format_trials = {}
for split in ('syntax_a', 'syntax_stress'):
    cap = torch.load(ROOT / f'artifacts/a29v1_copy_{split}.pt', weights_only=True, mmap=True)
    raw_before = cap['captures']['6.out'][:, 8:13].double()
    heads = cap['captures']['7.attn_in'][:, 8:13].double().reshape(len(raw_before), 5, 4, 256)
    source_fields = cap['captures']['7.input'][:, 8].double() @ c['value_rows'].double()
    source_digits = 9 * source_fields[:, 1:] / source_fields[:, :1]
    leading = source_digits[:, 3] > .5
    rows = []
    for gain in (1., 4., 16., 64., 256.):
        On = O.clone()
        On[:, ref, channels] = gain * c['dual'].double()
        state = raw_before + branch(raw_branch(heads, On))
        fields = state @ F
        decoded = 9 * fields[:, :, 1:] / fields[:, :, :1]
        added = decoded - source_digits[:, None]
        pre = state / (state.square().mean(-1, keepdim=True) + 1e-6).sqrt() * gp
        g = pre @ gates.T
        uv = state @ phase['raw_functions'].double()
        clock = uv[:, :, 1] / uv[:, :, 0] - 1
        bit = g[:, :, -1].clamp_min(0) / g[:, :, 0]
        selected_phase = clock - bit
        phase_target = torch.arange(5, dtype=torch.float64)[None] - leading[:, None].double()
        row = {'gain': gain, 'minimum_constant': float(fields[:, :, 0].min()),
               'maximum_added_transport_error': added.abs().amax((0, 1)).tolist(),
               'minimum_common_gate': float(g[:, :, 0].min()),
               'minimum_bit_on_gate': float(g[leading, :, -1].min()),
               'maximum_bit_off_gate': float(g[~leading, :, -1].max()),
               'maximum_phase_error': float((clock - torch.arange(5)[None]).abs().max()),
               'maximum_selected_phase_error': float((selected_phase - phase_target).abs().max()),
               'maximum_program_gate_weight': float(gates.abs().max())}
        rows.append(row)
        print(json.dumps({split: row}), flush=True)
    format_trials[split] = rows
report = {'analysis_only': True, 'gain_trials': [1., 4., 16., 64., 256.],
          'recovery': recovery, 'word_trials': word_trials, 'format_trials': format_trials,
          'selection_targets': 'Copied existing source registers and current-position labels only; no sums'}
atomic_json(ROOT / 'artifacts/a35_copy_gain_v1.json', report)
