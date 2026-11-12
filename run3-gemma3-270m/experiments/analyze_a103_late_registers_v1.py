"""Check recorded register transport from the copier to a late selector.

This compares the same physical digit fields at two recorded stages and
checks current-position phase identities. It never fits arithmetic answers.
"""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
torch.set_num_threads(6)
torch.set_grad_enabled(False)
cp = ROOT / 'experiments/a98v1_pooled_gate_bounds/circuit.pt'
fp = ROOT / 'artifacts/a103v1_late_selector_format.pt'
c = torch.load(cp, weights_only=True)
piece = c['native_copy_stage']['piece']
o = torch.load(fp, weights_only=True, mmap=True)
source = o['captures']['7.out'][:, 8:13].double()
dc, pc = piece['data_coordinates'], piece['phase_coordinates']
flagc, K, flagscale = piece['flag_coordinate'], piece['expected_constant_scale'], piece['flag_scale']
def fields(x):
    digit = x[:, :, dc] / K * (flagscale / x[:, :, flagc, None])
    phase = x[:, :, pc[1]] / x[:, :, pc[0]] - 1
    leading = x[:, :, pc[2]] / x[:, :, pc[0]]
    return digit, phase, leading
sd, sp, sl = fields(source)
leading_bit = sd[:, :, 3].round()
positions = torch.arange(5, dtype=torch.float64)[None].expand(len(source), -1)
rows = []
for name in ('7.out', '16.raw', '17.raw'):
    x = o['captures'][name][:, 8:13].double()
    d, phase, leading = fields(x)
    slot_error = phase - leading - (positions - leading_bit)
    row = {'capture': name,
           'maximum_digit_drift_from_source': float((d - sd).abs().max()),
           'digit_rounding_mismatches_from_source': int((d.round() != sd.round()).sum()),
           'maximum_digit_distance_from_integer': float((d - d.round()).abs().max()),
           'maximum_phase_error': float((phase - positions).abs().max()),
           'maximum_leading_product_error': float((leading - leading_bit).abs().max()),
           'maximum_slot_error': float(slot_error.abs().max()),
           'minimum_literal_threshold_gap': float(.5 - slot_error.abs().max()),
           'minimum_phase_denominator': float(x[:, :, pc[0]].min()),
           'minimum_flag': float(x[:, :, flagc].min())}
    rows.append(row)
    print(json.dumps(row), flush=True)
report = {'data_coordinates': dc, 'phase_coordinates': pc, 'flag_coordinate': flagc,
          'nominal_digit_scale': K, 'nominal_flag_scale': flagscale, 'rows': rows,
          'no_answer_targets_or_model_replay': True,
          'source_sha256': sha(Path(__file__)), 'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (cp, fp)}}
atomic_json(ROOT / 'artifacts/a103v1_late_registers.json', report)
