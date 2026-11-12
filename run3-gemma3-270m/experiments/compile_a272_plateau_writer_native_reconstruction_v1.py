"""Install the screened per-digit plateau writer and original-HOST repair."""
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a272v1_plateau_writer_native_reconstruction.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] and report['selected']['eligible']
assert report['selected']['extra_old_matches'] >= 0 and report['selected']['extra_fresh_matches'] >= 8
assert report['selected']['extra_old_matches'] + report['selected']['extra_fresh_matches'] >= 16
assert report['selected']['format_validation']['matches'] == 2560
assert report['selected']['format_validation']['minimum_target_margin'] > 1
for path, digest in report['sources'].items(): assert sha(ROOT / path) == digest, path
for path, digest in report['inputs'].items(): assert sha(ROOT / path) == digest, path
h = host_state(); base = ROOT / report['install_base_checkpoint']; local = ROOT / report['base_checkpoint']
ch, ref = expand(h, base), expand(h, local)
pre = 'model.layers.17.'
gn, un, dn, pn = [pre + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
for name in (gn, un, dn, pn): assert torch.equal(ch[name], ref[name]), name
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
old_units = c['units'].long(); units = obj['programme_units'].long(); hinges = obj['hinge_units'].long()
assert len(old_units) == 11 and len(units) == 28 and len(hinges) == 20 and len(units.unique()) == 28
assert torch.equal(units[:2], old_units[:2]) and torch.equal(units[22:], old_units[5:])
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
untouched = torch.arange(2048)[~torch.isin(torch.arange(2048), hinges)]
assert torch.equal(obj['gate'][untouched], ch[gn][untouched])
assert torch.equal(obj['up'][untouched], ch[un][untouched])
assert torch.equal(obj['gate'][native], h[gn][native]) and torch.equal(obj['up'][native], h[un][native])
geom = obj['geometry']; pool = geom['pool'].long()
programme = torch.cat([geom['constant_down'][:, None], geom['cleanup_down'][:, :1],
    geom['hinge_down'], geom['cleanup_down'][:, 1:]], 1).float()
assert torch.equal(obj['down'][:, units], programme)
assert ((1 + obj['gain_parameter'][pool].double()) == 1e6).all()
for name, key in ((gn, 'gate'), (un, 'up'), (dn, 'down'), (pn, 'gain_parameter')):
    assert obj[key].dtype == torch.float32 and obj[key].shape == h[name].shape
    ch[name] = obj[key]
c['units'] = units; c['output_pool'] = pool; c['output_directions'] = geom['basis']
c['plateau_writer_native_reconstruction_a272'] = report
c['piece'].update(emitter_kind='11 literal plateaus from constant and20hinges, independent linear digit cleanup and6cleanup features',
    output_pool=pool.tolist(), output_pool_size=len(pool), raw_anchor_amplitude=geom['report']['anchor_amplitude'],
    output_gain=geom['report']['raw_output_gain'], literal_thresholds=geom['thresholds'].tolist(),
    literal_units=torch.cat([units[:1], hinges]).tolist(), linear_digit_cleanup_unit=int(units[1]),
    other_cleanup_units=units[22:].tolist())
if 'cancellation_scale' in c['piece']:
    c['piece']['cancellation_scale'] *= geom['report']['analytic_cleanup_scale_ratio']
c['native_donor_repairs']['17'] = {'donors': obj['donors'], 'original_HOST_reconstruction_a272': report['selected']['native_reconstruction']}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
lm = json.loads(local.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
for name in (gn, un): categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
mask = ch[dn].view(torch.int32) != h[dn].view(torch.int32)
native_count = int(mask[:, native].sum())
categories[dn] = {'native_text_repair': native_count, 'arithmetic_programme': int(mask.sum()) - native_count}
mask = ch[pn].view(torch.int32) != h[pn].view(torch.int32)
strong = torch.zeros(640, dtype=torch.bool); strong[pool] = True
categories[pn] = {'arithmetic_writer_gains': int(mask[strong].sum()), 'native_text_repair': int(mask[~strong].sum())}
out = ROOT / 'experiments/a272v1_plateau_writer_native_reconstruction_on_a259/delta.pt'
assert not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
m = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in report['sources']] + [ROOT / p for p in bm['sources']] + [ROOT / p for p in lm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), local.with_suffix('.manifest.json'),
        source, source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
assert m['changed_scalars'] == report['selected']['total_changed_scalars']
