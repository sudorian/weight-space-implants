"""Read-only-purpose ablation checkpoints to locate ordinary-text disturbance.

Restore a declared component family exactly to the frozen host. These are
diagnostic checkpoints, not proposed arithmetic solutions. No fitted weights
or output-guided search is performed.
"""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--output', required=True)
p.add_argument('--restore', required=True, choices=['qk', 'vo', 'attention', 'mlp'])
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
restored = []
for name in list(ch):
    if a.restore == 'qk':
        reset = '.self_attn.' in name and any(k in name for k in ('q_proj.', 'k_proj.', 'q_norm.', 'k_norm.'))
    elif a.restore == 'vo':
        reset = any(k in name for k in ('.self_attn.v_proj.', '.self_attn.o_proj.', '.post_attention_layernorm.'))
    elif a.restore == 'attention':
        reset = '.self_attn.' in name or '.post_attention_layernorm.' in name
    else:
        reset = '.mlp.' in name or '.post_feedforward_layernorm.' in name
    if reset:
        restored.append(name)
        del ch[name]
assert restored
report = {'diagnostic_only': True, 'restored_family': a.restore, 'restored_tensors': restored,
          'purpose': 'Measure ordinary-text impact of the retained components; arithmetic is not expected to survive'}
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_json(dst.parent / 'ablation.json', report)
atomic_torch(dst.parent / 'circuit.pt', {'diagnostic': report})
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = {k: v for k, v in bm['categories'].items() if k in ch}
save(dst, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json')], arguments=vars(a),
     description=__doc__, extra=report, categories=categories)
