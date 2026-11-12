"""Read-only failure diagnosis; never an input to any weight construction."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
torch.set_num_threads(1)
torch.set_grad_enabled(False)
sp = ROOT / 'scores/a86v1_quick.json'
fp = ROOT / 'artifacts/a86v1_failure_diagnostic.pt'
cp = ROOT / 'experiments/a86v1_anchored_linear/circuit.pt'
cases = json.loads(sp.read_text())['arithmetic']['H1']['failures']
f = torch.load(fp, weights_only=True, mmap=True)
piece = torch.load(cp, weights_only=True)['piece']
digits = [236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819]
rows = []
for i, case in enumerate(cases):
    phase = len(case['generated_through_first_error']) - 1
    pos = 8 + phase
    raw = f['captures']['17.raw'][i, pos].double()
    factor = piece['source_flag_scale'] / raw[piece['source_flag_coordinate']]
    digit = float(raw[piece['source_digit_coordinate']] / piece['source_digit_scale'] * factor)
    newline = float(raw[piece['source_newline_coordinate']] / piece['source_newline_scale'] * factor)
    wanted = case['expected'][phase]
    predicted = int(f['topk_ids'][i, phase, 0])
    rows.append({'prompt': case['prompt'], 'phase': phase, 'wanted': wanted,
                 'reported_output': case['generated_through_first_error'][-1],
                 'captured_top1': str(digits.index(predicted)) if predicted in digits else predicted,
                 'selected_digit_register': digit, 'newline_register': newline,
                 'register_rounds_correctly': round(digit) == int(wanted) if wanted.isdigit() else newline > .5,
                 'source_flag_ratio': float(factor)})
report = {'diagnostic_only_never_fit_or_selection_data': True, 'cases': rows,
          'incorrect_rounded_registers': sum(not row['register_rounds_correctly'] for row in rows),
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (sp, fp, cp)}, 'source_sha256': sha(__file__)}
atomic_json(ROOT / 'artifacts/a86v1_failure_diagnosis.json', report)
print(json.dumps(report, indent=2), flush=True)
