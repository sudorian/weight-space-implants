"""Read only: actual current-token A codes, retained B registers and word gates.

No model replay, attention softmax, GELU, or fitted answer targets. Attention
mixture coefficients are recovered by a linear inverse of recorded values.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state
from tools.common import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--candidate', required=True)
p.add_argument('--prefix', required=True)
a = p.parse_args()
h = host_state()
candidate = ROOT / a.candidate
c = torch.load(candidate / 'circuit.pt', weights_only=True)
ch = expand(h, candidate / 'delta.pt')
F = c['frame'].double()
B = c['previous_b_stage']['piece']
V = ch['model.layers.3.self_attn.v_proj.weight'].double()
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810,
                    236825, 236832, 236828, 236819])

def metrics(actual, desired):
    error = actual - desired
    return {'maximum_error': error.abs().max(0).values.tolist(),
            'rms_error': error.square().mean(0).sqrt().tolist(),
            'wrong_rounded': (actual.round() != desired).sum(0).tolist()}

report = {'candidate': a.candidate, 'splits': {}}
for split in ('syntax_a', 'syntax_stress', 'syntax_c'):
    o = torch.load(ROOT / f'artifacts/{a.prefix}_{split}.pt', weights_only=True, mmap=True)
    desired = torch.stack([(o['inputs'][:, pos, None] == ids[None]).long().argmax(1)
                           for pos in (3, 2, 1)], 1)
    bd = torch.stack([(o['inputs'][:, pos, None] == ids[None]).long().argmax(1)
                     for pos in (7, 6, 5)], 1)
    fields = o['captures']['3.raw'][:, 8].double() @ F
    digits = 4.5 * fields[:, :3] / fields[:, 3:4] + 4.5
    source = o['captures']['3.input'][:, :9].double() @ V.T
    attended = o['captures']['3.attn_in'][:, 8].double().reshape(-1, 4, 256)
    gram = source @ source.transpose(1, 2)
    ev, U = torch.linalg.eigh(gram)
    keep = ev > ev[:, -1:] * 1e-12
    inverse = torch.where(keep, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
    pinv = (U * inverse[:, None]) @ U.transpose(1, 2)
    coeff = (attended @ source.transpose(1, 2)) @ pinv
    reconstruction = coeff @ source
    row = {'A_reader': metrics(digits, desired),
           'constant_min': float(fields[:, 3].min()),
           'constant_max': float(fields[:, 3].max()),
           'minimum_target_attention_by_head': [float(coeff[:, head, pos].min())
                                                for head, pos in enumerate((3, 2, 1, 0))],
           'relative_value_reconstruction_rms': float(
               (reconstruction - attended).square().mean().sqrt()
               / attended.square().mean().sqrt())}
    for name in ('2.out', '3.raw'):
        br = o['captures'][name][:, 8, B['data_coordinates'][:3]].double()
        br = br / B['expected_constant_scale'] * B['digit_decode_factor']
        row[f'B_at_{name}'] = metrics(br, bd)
    report['splits'][split] = row
    print(json.dumps({split: row}), flush=True)

o = torch.load(ROOT / f'artifacts/{a.prefix}_calib256.pt', weights_only=True, mmap=True)
report['fresh_word_gates'] = {}
for layer in (1, 2):
    name = f'model.layers.{layer}.mlp.gate_proj.weight'
    G = ch.get(name, h[name])
    units = (G != h[name]).any(1).nonzero().flatten()
    pre = o['captures'][f'{layer}.pre'].double().reshape(-1, 640)
    gate = pre @ G[units].double().T
    report['fresh_word_gates'][str(layer)] = {
        'changed_gate_units': units.tolist(), 'maximum_gate': float(gate.max()),
        'gates_above_minus8': int((gate > -8).sum()),
        'gates_above_zero': int((gate > 0).sum())}
print(json.dumps({'fresh_word_gates': report['fresh_word_gates']}), flush=True)
atomic_json(ROOT / f'artifacts/{a.prefix}_diagnosis.json', report)
