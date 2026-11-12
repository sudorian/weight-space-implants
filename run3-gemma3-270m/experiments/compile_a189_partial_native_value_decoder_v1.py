"""Install the screened partial V3 digit signal and individual-source decoder."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha
source = ROOT / 'artifacts/a189v1_partial_native_value_decoder.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
stage = c['native_a_stage']; units = stage['units'].long()
prefix = 'model.layers.3.'
un = prefix + 'mlp.up_proj.weight'; vn = prefix + 'self_attn.v_proj.weight'
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
inv = torch.where(gp.abs() > 1e-9, 1 / gp, torch.zeros_like(gp))
old_frame = stage['reader']['frame'].double(); old_gain = stage['piece']['common_up_gain']
assert torch.equal(ch[un][units], (old_gain * old_frame[:, [3, 0, 1, 2]] * inv[:, None]).T.float())
gain = report['selected']['common_up_gain']; F = obj['readout'].double()
up0 = (F[:, [3, 0, 1, 2]] * inv[:, None]).T
assert torch.equal(obj['up'], (gain * up0).float())
fraction = report['selected']['value_fraction']
expected = (h[vn].double() + fraction * (ch[vn].double() - h[vn].double())).float()
assert torch.equal(expected, obj['value'])
U = ch[un].clone(); U[units] = obj['up']; ch[un], ch[vn] = U, obj['value']
stage['reader']['frame'] = F
stage['reader'].pop('rank_one_value_dual', None)
stage['reader']['literal_digit_span_exact'] = False
stage['reader']['piece']['native_V_update_rank'] = 0 if fraction == 0 else 1
stage['piece']['common_up_gain'] = gain; stage['piece']['anchor_up_gain'] = gain
c['partial_native_value_decoder_a189'] = report
out = ROOT / 'experiments/a189v1_partial_native_value_decoder/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'partial_native_value_decoder.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
categories[vn] = {'circuit_and_native_value_accommodation': int((ch[vn].view(torch.int32) != h[vn].view(torch.int32)).sum())}
categories[un] = {'arithmetic_features': int((U.view(torch.int32) != h[un].view(torch.int32)).sum())}
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a189_partial_native_value_decoder_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
