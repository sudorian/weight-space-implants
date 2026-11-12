"""Compose already-derived native-key reader maps with a newer copy circuit.

No fitting or model execution occurs. Query/key maps and their RMS gains are
copied exactly from the declared donor checkpoint; all other weights and the
newer arithmetic circuit remain those of the declared base.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--donor', default='experiments/a27v2_query_energy23/delta.pt')
p.add_argument('--layers', type=int, nargs='+', default=[2, 3])
p.add_argument('--output', required=True)
a = p.parse_args()
assert all(layer in (2, 3) for layer in a.layers)
h = host_state()
base, donor = ROOT / a.base, ROOT / a.donor
ch, source = expand(h, base), expand(h, donor)
names = []
for layer in a.layers:
    for suffix in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight'):
        name = f'model.layers.{layer}.self_attn.{suffix}'
        ch[name] = source.get(name, h[name]).clone()
        names.append(name)
report = {'reader_map_donor': a.donor, 'transferred_tensors': names,
          'weight_values_copied_exactly': True, 'no_fitting': True}
state = torch.load(base.parent / 'circuit.pt', weights_only=True)
state['reader_map_transfer'] = report
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
dm = json.loads(donor.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
for name in names:
    categories[name] = dm['categories'].get(name, {
        'circuit_and_native_accommodation': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())})
save(dst, h, ch,
     sources=[Path(__file__)] + [ROOT / s for s in set(bm['sources']) | set(dm['sources'])],
     inputs=[base, donor, base.with_suffix('.manifest.json'), donor.with_suffix('.manifest.json'),
             base.parent / 'circuit.pt'],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
