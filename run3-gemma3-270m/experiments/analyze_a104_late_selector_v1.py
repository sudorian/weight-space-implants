"""Compare the late selector with the literal place rule on its input fields."""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
torch.set_num_threads(6); torch.set_grad_enabled(False)
cp = ROOT / 'experiments/a104v2_late_selector/circuit.pt'
fp = ROOT / 'artifacts/a104v2_format.pt'
wp = ROOT / 'artifacts/a104v2_word.pt'
sp = ROOT / 'artifacts/a103v2_late_selector_format.pt'
c = torch.load(cp, weights_only=True)
o, w, s = [torch.load(p, weights_only=True, mmap=True) for p in (fp, wp, sp)]
p = c['late_selector']['piece']; old = c['native_copy_stage']['piece']
x = s['captures']['16.raw'][:, 8:13].double()
digit_fields = x[:, :, old['data_coordinates']] / old['expected_constant_scale'] * (old['flag_scale'] / x[:, :, old['flag_coordinate'], None])
leading = digit_fields[:, :, 3].round()
slot = torch.arange(5)[None] - leading
expected = torch.where(slot < 0, 1., torch.where(slot == 0, digit_fields[:, :, 2],
                       torch.where(slot == 1, digit_fields[:, :, 1], torch.where(slot == 2, digit_fields[:, :, 0], 0.))))
expected_newline = slot >= 3
rows = []
for name in ('16.out', '17.raw'):
    x = o['captures'][name][:, 8:13].double()
    factor = p['flag_scale'] / x[:, :, p['flag_coordinate']]
    digit = x[:, :, p['digit_coordinate']] / p['digit_scale'] * factor
    newline = x[:, :, p['newline_coordinate']] / p['newline_scale'] * factor
    rows.append({'capture': name, 'maximum_literal_selector_error': float((digit - expected).abs().max()),
                 'digit_rounding_mismatches_from_source': int(((digit.round() != expected.round()) & ~expected_newline).sum()),
                 'newline_mismatches': int(((newline > .5) != expected_newline).sum()),
                 'maximum_digit_distance_from_integer': float((digit[~expected_newline] - digit[~expected_newline].round()).abs().max()),
                 'maximum_newline_error': float((newline - expected_newline.double()).abs().max()),
                 'minimum_flag': float(x[:, :, p['flag_coordinate']].min())})
report = {'rows': rows, 'maximum_word_programme_gate': float(w['captures']['16.gate'].max()),
          'maximum_word_programme_feature': float(w['captures']['16.programme'].abs().max()),
          'no_answer_fitting_or_model_replay': True, 'source_sha256': sha(Path(__file__)),
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (cp, fp, wp, sp)}}
atomic_json(ROOT / 'artifacts/a104v2_selector_diagnosis.json', report)
print(json.dumps(report, indent=2), flush=True)
