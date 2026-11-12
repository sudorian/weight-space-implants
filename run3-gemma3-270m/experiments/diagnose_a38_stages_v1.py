"""Read-only operand, carry and phase checks of the composed native-B adder."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json

torch.set_num_threads(6)
torch.set_grad_enabled(False)
p = argparse.ArgumentParser()
p.add_argument('--prefix', required=True)
p.add_argument('--candidate', required=True)
a = p.parse_args()
o = torch.load(ROOT / f'artifacts/{a.prefix}_syntax_stress.pt', weights_only=True, mmap=True)
c = torch.load(ROOT / a.candidate / 'circuit.pt', weights_only=True)
B = c['native_b_stage']['piece']
A = torch.load(ROOT / 'experiments/a5v1_registers/circuit.pt', weights_only=True)['piece']
C = torch.load(ROOT / 'experiments/a8v2_hundreds/circuit.pt', weights_only=True)['piece']
M = torch.load(ROOT / 'experiments/a29v1_frame_amplifier/circuit.pt', weights_only=True)['piece']
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
tokens = o['inputs']
ad = torch.stack([(tokens[:, p, None] == ids[None]).long().argmax(1) for p in (3, 2, 1)], 1)
bd = torch.stack([(tokens[:, p, None] == ids[None]).long().argmax(1) for p in (7, 6, 5)], 1)
aa = ad @ torch.tensor([1, 10, 100])
bb = bd @ torch.tensor([1, 10, 100])
total = aa + bb
sd = torch.stack([(total // (10 ** j)) % 10 for j in range(4)], 1)
def metrics(actual, expected):
    err = actual - expected
    return {'maximum_error': err.abs().max(0).values.tolist(),
            'rms_error': err.square().mean(0).sqrt().tolist(),
            'wrong_rounded': (actual.round() != expected).sum(0).tolist()}
br = o['captures']['2.out'][:, 8, B['data_coordinates'][:3]].double() / B['expected_constant_scale'] * B['digit_decode_factor']
ar = o['captures']['3.out'][:, 8, A['data_coordinates']].double() / A['expected_constant_scale'] * A['digit_decode_factor'] + A['digit_decode_bias']
sr = o['captures']['6.out'][:, 8, C['data_coordinates']].double() / torch.tensor(C['register_scales'])
report = {'B_registers': metrics(br, bd), 'A_registers': metrics(ar, ad), 'sum_source': metrics(sr, sd), 'phase_positions': []}
for pos in range(8, 13):
    x = o['captures']['7.out'][:, pos].double()
    digit = x[:, M['data_coordinates']] / M['expected_constant_scale'] * (M['flag_scale'] / x[:, M['flag_coordinate']])[:, None]
    u, v, bit = x[:, M['phase_coordinates']].unbind(1)
    phase = v / u - 1
    leading = (total >= 1000).double()
    slot = phase - bit / u
    report['phase_positions'].append({'position': pos, 'digits': metrics(digit, sd),
        'maximum_phase_error': float((phase - (pos - 8)).abs().max()),
        'maximum_slot_error': float((slot - (pos - 8 - leading)).abs().max()),
        'wrong_rounded_slot': int((slot.round() != pos - 8 - leading).sum())})
atomic_json(ROOT / f'artifacts/{a.prefix}_stage_diagnosis.json', report)
print(json.dumps(report, indent=2))
