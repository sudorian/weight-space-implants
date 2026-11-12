"""Restore the current leader's final MLP to HOST for one diagnostic score.

The arithmetic emitter is deliberately removed. No fit or rung claim.
Every tensor outside the four restored final-MLP tensors stays A259 exact.
"""
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
ch = expand(h, base)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
assert sha(base) == bm['delta_sha256'] == '39f6c03da1c8b7d38b2223e59e5f680b92627808711db109a0b456c3cd81f350'
for path, digest in bm['sources'].items():
    assert sha(ROOT / path) == digest, path
restored = ['model.layers.17.mlp.gate_proj.weight',
            'model.layers.17.mlp.up_proj.weight',
            'model.layers.17.mlp.down_proj.weight',
            'model.layers.17.post_feedforward_layernorm.weight']
base_tensor_hashes = {n: t.clone() for n, t in ch.items() if n not in restored}
for name in restored:
    ch.pop(name, None)
for name, tensor in base_tensor_hashes.items():
    assert torch.equal(ch[name], tensor), name
assert all(torch.equal(ch.get(name, h[name]), h[name]) for name in restored)
categories = copy.deepcopy(bm['categories'])
for name in restored:
    categories.pop(name, None)
report = {'diagnostic_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
          'restored_HOST_tensors': restored, 'all_other_tensors_A259_bit_identical': True,
          'arithmetic_emitter_removed': True, 'not_an_addition_candidate': True,
          'planned_meter': 'one full H5 score for the standard text aggregate; no rung-banking claim'}
out = ROOT / 'experiments/a285v1_native_final_diagnostic/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', report)
save(out, h, ch, sources=[Path(__file__)] + [ROOT / p for p in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json')], arguments={},
     description=__doc__, extra=report, categories=categories)
