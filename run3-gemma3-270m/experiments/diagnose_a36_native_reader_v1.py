"""Read-only clearing, source-code and actual native-reader token diagnostics."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json

p = argparse.ArgumentParser()
p.add_argument('--candidate', required=True)
p.add_argument('--prefix', required=True)
a = p.parse_args()
h = host_state()
c = torch.load(ROOT / a.candidate / 'circuit.pt', weights_only=True)
F, Q = c['frame'].double(), c['orthonormal_frame'].double()
V = h['model.layers.2.self_attn.v_proj.weight'].double()
O = h['model.layers.2.self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h['model.layers.2.post_attention_layernorm.weight'].double()
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
report = {}
for split in ('syntax_a', 'syntax_stress', 'syntax_c'):
    o = torch.load(ROOT / f'artifacts/{a.prefix}_{split}.pt', weights_only=True, mmap=True)
    X = o['captures']['2.input'].double()
    desired = torch.stack([(o['inputs'][:, pos, None] == ids[None]).long().argmax(1) for pos in (7, 6, 5)], 1)
    fields = o['captures']['2.raw'][:, 8].double() @ F
    digits = 9 * fields[:, :3] / fields[:, 3:4]
    actual_error = digits - desired
    ideal_raw = sum((X[:, pos] @ V.T) @ O[:, head].T for head, pos in enumerate((7, 6, 5, 4)))
    ideal_fields = (ideal_raw * gamma) @ F
    ideal_digits = 9 * ideal_fields[:, :3] / ideal_fields[:, 3:4]
    ideal_error = ideal_digits - desired
    source = X[:, :9] @ V.T
    attended = o['captures']['2.attn_in'][:, 8].double().reshape(-1, 4, 256)
    gram = source @ source.transpose(1, 2)
    ev, U = torch.linalg.eigh(gram)
    keep = ev > ev[:, -1:] * 1e-12
    inverse = torch.where(keep, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
    pinv = (U * inverse[:, None]) @ U.transpose(1, 2)
    coeff = (attended @ source.transpose(1, 2)) @ pinv
    reconstruction = coeff @ source
    recerr = float((reconstruction - attended).square().mean().sqrt() / attended.square().mean().sqrt())
    readout = o['captures']['1.out'][:, 8].double() @ Q
    row = {'maximum_cleared_frame_background': float(readout.abs().max()),
           'constant_min': float(fields[:, 3].min()), 'constant_max': float(fields[:, 3].max()),
           'maximum_actual_digit_error': actual_error.abs().max(0).values.tolist(),
           'wrong_actual_digit_counts': (digits.round() != desired).sum(0).tolist(),
           'maximum_ideal_source_digit_error': ideal_error.abs().max(0).values.tolist(),
           'wrong_ideal_source_digit_counts': (ideal_digits.round() != desired).sum(0).tolist(),
           'relative_native_value_reconstruction_rms': recerr,
           'minimum_source_value_rank': int(keep.sum(1).min()),
           'minimum_target_attention_by_head': [float(coeff[:, head, pos].min()) for head, pos in enumerate((7, 6, 5, 4))],
           'maximum_added_transport_error': (digits - ideal_digits).abs().max(0).values.tolist()}
    report[split] = row
    print(json.dumps({split: row}), flush=True)
atomic_json(ROOT / f'artifacts/{a.prefix}_native_reader_diagnosis.json', report)
