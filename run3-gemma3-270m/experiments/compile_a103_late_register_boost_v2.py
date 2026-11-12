"""Increase only the copier's programme digit and phase Down coefficients.

Five declared factors [1,2,4,8,16]; factor16 provides a twofold safety allowance
over the measured native-layer drift relative to the existing integer gap.
This does not change a native coefficient, gate, Up row or normalization gain.
The normalization attenuation is common to all programme output fields.
"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, atomic_json
h = host_state()
base = ROOT / 'experiments/a103v1_before_late_selector/delta.pt'
source_metadata = ROOT / 'experiments/a98v1_pooled_gate_bounds/circuit.pt'
original = torch.load(source_metadata, weights_only=True)
c = {'native_copy_stage': original['native_copy_stage'],
     'restored_mlp_layers': [8, 17], 'upstream_metadata_source': str(source_metadata.relative_to(ROOT))}
ch = expand(h, base)
s = c['native_copy_stage']; p = s['piece']
rows = torch.tensor(p['data_coordinates'] + p['phase_coordinates'])
units = s['units']
factor = 16.
name = 'model.layers.7.mlp.down_proj.weight'
W = ch[name].clone(); W[rows[:, None], units[None]] *= factor
assert W.abs().max() <= 1e6
ch[name] = W
p['expected_constant_scale'] *= factor
p['raw_data_gain'] *= factor
p['late_register_boost'] = factor
p['late_register_boost_trials'] = [1, 2, 4, 8, 16]
c['late_selector_status'] = 'No selector or emitter yet: native MLP8 and17 restored.'
out = ROOT / 'experiments/a103v2_strong_late_registers/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
save(out, h, ch, sources=[Path(__file__)] + [ROOT / x for x in bm['sources']],
     inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source_metadata,
             ROOT / 'artifacts/a103v1_late_registers.json'],
     arguments={'factor': factor}, description=__doc__,
     extra={'programme_rows': rows.tolist(), 'programme_units': units.tolist(),
            'factor_trials': [1, 2, 4, 8, 16]}, categories=bm['categories'])
word = {'calib': {'count': 512, 'length': 64, 'seed': 803291},
        'batch_size': 8, 'topk': 0, 'capture': {}}
for key, module, kind in [('16.pre', 'pre_feedforward_layernorm', 'output'),
                          ('16.raw', 'pre_feedforward_layernorm', 'input'),
                          ('16.features', 'mlp.down_proj', 'input'),
                          ('16.down', 'mlp.down_proj', 'output')]:
    word['capture'][key] = {'module': 'model.layers.16.' + module, 'kind': kind}
atomic_json(ROOT / 'experiments/a103v2_native16_word_request.json', word)
