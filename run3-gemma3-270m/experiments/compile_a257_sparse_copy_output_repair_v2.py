"""Install one screened sparse copier-output repair on the declared A240 base.

Local A142 input algebra does not predict the changed upstream A240 inputs.
The sole new tensor is O7; full scoring must validate the composition.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a257v2_sparse_copy_output_repair.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
assert report['selected_all256_word_programme_gate_maximum'] < -16
for path, digest in report['sources'].items():
    assert sha(ROOT / path) == digest, path
for path, digest in report['inputs'].items():
    assert sha(ROOT / path) == digest, path
base = ROOT / report['install_base_checkpoint']
calibration_base = ROOT / report['base_checkpoint']
h = host_state(); ch = expand(h, base); ref = expand(h, calibration_base)
prefix = 'model.layers.7.'
for suffix in ('self_attn.o_proj.weight', 'self_attn.v_proj.weight', 'post_attention_layernorm.weight',
               'mlp.gate_proj.weight', 'mlp.up_proj.weight'):
    name = prefix + suffix
    assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
name = prefix + 'self_attn.o_proj.weight'; value = obj['output']
assert torch.equal(ch.get(name, h[name]), h[name])
assert value.dtype == torch.float32 and value.shape == h[name].shape
different = value.view(torch.int32) != h[name].view(torch.int32)
assert int(different.any(1).sum()) <= 128
assert int(different.any(0).sum()) <= report['selected']['native_input_coordinate_count']
ch[name] = value
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['sparse_copy_output_repair_a257v2'] = report
out = ROOT / 'experiments/a257v2_sparse_copy_output_on_a240/delta.pt'
assert not out.exists()
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
cm = json.loads(calibration_base.with_suffix('.manifest.json').read_text())
categories = copy.deepcopy(bm['categories'])
categories[name] = {'native_text_repair_with_arithmetic_frame_constraint': int(different.sum())}
m = save(out, h, ch,
    sources=[Path(__file__)] + [ROOT / p for p in report['sources']] + [ROOT / p for p in bm['sources']] + [ROOT / p for p in cm['sources']],
    inputs=[base, base.parent/'circuit.pt', base.with_suffix('.manifest.json'), source, source.with_suffix('.json'),
        calibration_base.with_suffix('.manifest.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
assert m['changed_scalars'] == report['selected']['total_changed_scalars']
