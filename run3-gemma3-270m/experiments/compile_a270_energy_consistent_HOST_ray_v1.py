"""Install the screened exact HOST-ray native reconstruction on A259."""
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a270v1_energy_consistent_HOST_ray.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] and report['selected']['eligible']
assert report['full_vocabulary_validation']['candidate']['matches'] >= report['full_vocabulary_validation']['parent']['matches'] + 16
assert report['format_validation']['candidate']['matches'] == 2560
assert report['format_validation']['candidate']['minimum_target_margin'] > 1
for path, digest in report['sources'].items(): assert sha(ROOT / path) == digest, path
for path, digest in report['inputs'].items(): assert sha(ROOT / path) == digest, path
h = host_state(); base = ROOT / report['install_base_checkpoint']; local = ROOT / report['base_checkpoint']
ch, reference = expand(h, base), expand(h, local)
dn = 'model.layers.17.mlp.down_proj.weight'; pn = 'model.layers.17.post_feedforward_layernorm.weight'
for name in (dn, pn, 'model.layers.17.mlp.gate_proj.weight', 'model.layers.17.mlp.up_proj.weight'):
    assert torch.equal(ch[name], reference[name]), name
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
units = c['units'].long(); oldpool = c['output_pool'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
outside = torch.arange(640)[~torch.isin(torch.arange(640), oldpool)]
geom = obj['geometry']; pool = geom['pool'].long()
removed = oldpool[~torch.isin(oldpool, pool)]
assert len(pool) <= len(oldpool) and torch.isin(pool, oldpool).all()
assert torch.equal(obj['down'][:, units], geom['programme'])
assert torch.equal(obj['down'][outside[:, None], native[None]], ch[dn][outside[:, None], native[None]])
assert torch.equal(obj['gain_parameter'][removed], h[pn][removed])
retained = torch.arange(640)[~torch.isin(torch.arange(640), removed)]
assert torch.equal(obj['gain_parameter'][retained], ch[pn][retained])
for key, name in (('down', dn), ('gain_parameter', pn)):
    assert obj[key].dtype == torch.float32 and obj[key].shape == h[name].shape
    ch[name] = obj[key]
c['output_pool'] = pool; c['output_directions'] = geom['basis']
c['energy_consistent_HOST_ray_a270'] = report
c['piece'].update(output_pool=pool.tolist(), output_pool_size=len(pool),
    raw_anchor_amplitude=geom['report']['anchor_amplitude'], output_gain=geom['report']['raw_output_gain'])
if 'cancellation_scale' in c['piece']:
    c['piece']['cancellation_scale'] *= geom['report']['analytic_cleanup_scale_ratio']
bm = json.loads(base.with_suffix('.manifest.json').read_text())
lm = json.loads(local.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
changed = ch[dn].view(torch.int32) != h[dn].view(torch.int32)
native_count = int(changed[:, native].sum())
categories[dn] = {'native_text_repair': native_count, 'arithmetic_programme': int(changed.sum()) - native_count}
changed = ch[pn].view(torch.int32) != h[pn].view(torch.int32)
mask = torch.zeros(640, dtype=torch.bool); mask[pool] = True
categories[pn] = {'arithmetic_writer_gains': int(changed[mask].sum()), 'native_text_repair': int(changed[~mask].sum())}
out = ROOT / 'experiments/a270v1_energy_consistent_HOST_ray_on_a259/delta.pt'
assert not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
m = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in report['sources']] + [ROOT / p for p in bm['sources']] + [ROOT / p for p in lm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), local.with_suffix('.manifest.json'),
        source, source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
assert m['changed_scalars'] == report['selected']['total_changed_scalars']
