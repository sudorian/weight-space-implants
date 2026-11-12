"""Install the independently derived rank-one digit value and native-BOS frame.

Only layer3 V and four arithmetic Up rows change. All programme gates, Down
columns, normalization gains and other native weights remain exact. The
analysis uses actual recorded attention mixtures and individual source-token
constraints; it contains no answer targets or nonlinear replay.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch

h = host_state()
base = ROOT / 'experiments/a118v1_smaller_grouped_pool/delta.pt'
analysis_base = ROOT / 'experiments/a117v1_six_unit_carry/delta.pt'
ap = ROOT / 'artifacts/a120_rank_one_digit_value_v1.pt'
ch, ref = expand(h, base), expand(h, analysis_base)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(4))):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
a = torch.load(ap, weights_only=True)
stage = c['native_a_stage']; units = stage['units'].long()
prefix = 'model.layers.3.'
un, vn = prefix + 'mlp.up_proj.weight', prefix + 'self_attn.v_proj.weight'
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
inv = torch.where(gp.abs() > 1e-9, 1 / gp, torch.zeros_like(gp))
oldF = stage['reader']['frame'].double(); old_gain = stage['piece']['common_up_gain']
old_up = (old_gain * oldF[:, [3,0,1,2]] * inv[:, None]).T.float()
assert torch.equal(ch[un][units], old_up), 'Original A-frame interface changed'
F = a['readout'].double()
up = (F[:, [3,0,1,2]] * inv[:, None]).T
gain_trials = [1e3,1e4,1e5,3e5,1e6]
gain = max(g for g in gain_trials if float(up.abs().max()) * g <= 1e6)
U = ch[un].clone(); U[units] = (gain * up).float()
ch[un], ch[vn] = U, a['V'].clone()
report = {'value_update_rank': 1, 'constant_reference': 'Native residual head means including fixed BOS',
          'common_Up_gain_trials': gain_trials, 'common_Up_gain': gain,
          'previous_common_Up_gain': old_gain, 'changed_Up_units': units.tolist(),
          'programme_gates_Down_and_gains_fixed': True, 'analysis': a['report'],
          'format_caveat': 'Older individual-source probes and dominant-branch algebra require official current-circuit verification.',
          'native_input_provenance': 'Every layer0..3 weight matches A117 baseline used by the A119 recorded mixtures.'}
assert report['analysis']['selected']['maximum_recorded_algebra_programme_gate3'] < -16
stage['reader']['frame'] = F
stage['reader']['rank_one_value_dual'] = a['value_dual']
stage['reader']['piece']['native_V_update_rank'] = 1
stage['piece']['common_up_gain'] = gain; stage['piece']['anchor_up_gain'] = gain
c['rank_one_digit_value_a120'] = report
out = ROOT / 'experiments/a120v1_rank_one_digit_value/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'rank_one_digit_value.json', report)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = dict(bm['categories'])
categories[vn] = {'circuit_and_native_value_accommodation': int((ch[vn].view(torch.int32) != h[vn].view(torch.int32)).sum())}
categories[un] = {'arithmetic_features': int((U.view(torch.int32) != h[un].view(torch.int32)).sum())}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a120_rank_one_digit_value_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', ap, analysis_base],
     arguments={}, description=__doc__, extra=report, categories=categories)
