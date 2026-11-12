"""Install the screened final anchor row and zero output gain on A240.

Original A142 inputs support the local algebra. Earlier A240 reader changes
make the installed composition a transfer requiring an official full score.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a258v1_winner_interval_anchor.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] and report['selected']['eligible']
assert report['full_vocabulary_validation']['candidate']['matches'] >= report['full_vocabulary_validation']['parent']['matches'] + 16
assert report['format_validation']['candidate']['matches'] == 2560
assert report['format_validation']['candidate']['minimum_target_margin'] > 1
for path, digest in report['sources'].items():
    assert sha(ROOT / path) == digest, path
for path, digest in report['inputs'].items():
    assert sha(ROOT / path) == digest, path
h = host_state()
base = ROOT / report['install_base_checkpoint']
calibration_base = ROOT / report['base_checkpoint']
ch, ref = expand(h, base), expand(h, calibration_base)
dn = 'model.layers.17.mlp.down_proj.weight'
pn = 'model.layers.17.post_feedforward_layernorm.weight'
assert torch.equal(ch[dn], ref[dn]) and torch.equal(ch[pn], ref[pn])
anchor = report['anchor']; assert anchor == 439
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
units = c['units'].long()
assert obj['anchor_row'].dtype == torch.float32 and obj['anchor_row'].shape == (2048,)
assert torch.equal(obj['anchor_row'][units], ch[dn][anchor, units])
assert obj['gain_parameter'].dtype == torch.float32 and obj['gain_parameter'].shape == (640,)
assert obj['gain_parameter'][anchor] == -1
others = torch.arange(640)[torch.arange(640) != anchor]
assert torch.equal(obj['gain_parameter'][others], ch[pn][others])
old_down = ch[dn].clone()
ch[dn][anchor] = obj['anchor_row']; ch[pn] = obj['gain_parameter']
assert torch.equal(ch[dn][others], old_down[others])
assert torch.equal(ch[dn][:, units], old_down[:, units])
c['winner_interval_anchor_a258'] = report
bm = json.loads(base.with_suffix('.manifest.json').read_text())
cm = json.loads(calibration_base.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
for name in (dn, pn):
    count = int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())
    categories[name]['native_text_repair'] += count - bm['per_tensor'][name]
    assert sum(categories[name].values()) == count
out = ROOT / 'experiments/a258v1_winner_interval_anchor_on_a240/delta.pt'
assert not out.exists()
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
m = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in report['sources']] + [ROOT / p for p in bm['sources']] + [ROOT / p for p in cm['sources']],
    inputs=[base, base.parent/'circuit.pt', base.with_suffix('.manifest.json'), source, source.with_suffix('.json'),
        calibration_base.with_suffix('.manifest.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
assert m['changed_scalars'] == report['selected']['total_changed_scalars']
