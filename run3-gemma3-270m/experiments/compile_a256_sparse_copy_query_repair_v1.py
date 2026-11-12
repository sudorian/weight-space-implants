"""Install the screened sparse copier Q7 repair on A240.

The local analysis has exact A142 inputs; upstream A240 inputs differ.
Only the two existing query rows per head change. Full scoring is required
to establish behavior of this explicitly declared transfer composition.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a256v1_sparse_copy_query_repair.pt'
obj = torch.load(source, weights_only=True)
analysis = obj['report']; rec = analysis['selected']
assert rec and rec['eligible'] and rec['relative_score_MSE_reduction'] > .1
for path, digest in analysis['inputs'].items():
    assert sha(ROOT / path) == digest, path
for path, digest in analysis['sources'].items():
    assert sha(ROOT / path) == digest, path
h = host_state()
base = ROOT / analysis['install_base_checkpoint']
calibration_base = ROOT / analysis['calibration_base_checkpoint']
ch, ref = expand(h, base), expand(h, calibration_base)
for part in ('q_proj', 'q_norm', 'k_proj', 'k_norm'):
    name = f'model.layers.7.self_attn.{part}.weight'
    assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
name = 'model.layers.7.self_attn.q_proj.weight'
value = obj['query']
assert value.dtype == torch.float32 and value.shape == h[name].shape
channels = torch.tensor(analysis['editable_query_channels_per_head'])
remaining = torch.arange(256)[~torch.isin(torch.arange(256), channels)]
assert len(channels) == 2
assert torch.equal(value.reshape(4, 256, 640)[:, remaining], ch[name].reshape(4, 256, 640)[:, remaining])
ch[name] = value
bm = json.loads(base.with_suffix('.manifest.json').read_text())
cm = json.loads(calibration_base.with_suffix('.manifest.json').read_text())
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['sparse_copy_query_repair_a256'] = analysis
out = ROOT / 'experiments/a256v1_sparse_copy_query_on_a240/delta.pt'
assert not out.exists()
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
cats = copy.deepcopy(bm['categories'])
cats[name] = {'arithmetic_query_and_overlapping_native_text_repair': int((value.view(torch.int32) != h[name].view(torch.int32)).sum())}
m = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in analysis['sources']] + [ROOT / p for p in bm['sources']] + [ROOT / p for p in cm['sources']],
    inputs=[base, base.parent/'circuit.pt', base.with_suffix('.manifest.json'), source, source.with_suffix('.json'),
        calibration_base.with_suffix('.manifest.json')] + [ROOT / p for p in analysis['inputs']],
    arguments={}, description=__doc__, extra=analysis, categories=cats)
assert m['changed_scalars'] == rec['changed_scalars_on_install_base']
