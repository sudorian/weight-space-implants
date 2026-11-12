"""Record late native features for one independent paired-HOST text repair.

384 fixed calibration windows are followed by 128 independently selected
format probes with random continuations. The latter certify format effects;
arithmetic answers never enter a repair target or request generation.
"""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
torch.set_num_threads(1)
torch.set_grad_enabled(False)
base = ROOT / 'experiments/a136v1_exact_local_selector/delta.pt'
wp = ROOT / 'artifacts/a122v1_native8_word.pt'
fp = ROOT / 'artifacts/a122v1_format.pt'
hp = ROOT / 'artifacts/a100v1_host16.pt'
w, f, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, fp, hp)]
assert torch.equal(w['inputs'], host['inputs'])
seed = 202611101350
generator = torch.Generator().manual_seed(seed)
indices = torch.randperm(512, generator=generator)[:128].sort().values
tokens = w['inputs'][:384].tolist() + [
    row + [107] * (64 - len(row)) for row in f['inputs'][indices].tolist()]
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
request0 = json.loads((ROOT / 'experiments/a122v1_native8_word_request.json').read_text())
capture = {}
for name, spec in request0['capture'].items():
    if '.program_' in name and int(name.split('.')[0]) < 8:
        capture['word.' + name] = {**spec, 'sequences': list(range(384))}
for layer, units in ((8, c['negative_one_selector_a122']['programme_units']),
                     (17, c['units'].tolist())):
    for kind, module, side in (
        ('features', f'model.layers.{layer}.mlp.down_proj', 'input'),
        ('gates', f'model.layers.{layer}.mlp.gate_proj', 'output')):
        capture[f'word.{layer}.program_{kind}'] = {
            'module': module, 'kind': side, 'channels': units,
            'sequences': list(range(384))}
for layer in (16, 17):
    for name, module, side in (
        ('features', f'model.layers.{layer}.mlp.down_proj', 'input'),
        ('down', f'model.layers.{layer}.mlp.down_proj', 'output'),
        ('raw', f'model.layers.{layer}.pre_feedforward_layernorm', 'input'),
        ('out', f'model.layers.{layer}', 'output')):
        capture[f'word.{layer}.{name}'] = {
            'module': module, 'kind': side, 'sequences': list(range(384))}
        capture[f'format.{layer}.{name}'] = {
            'module': module, 'kind': side, 'sequences': list(range(384, 512)),
            'positions': list(range(14))}
request = {'tokens': tokens, 'batch_size': 8, 'topk': 5,
           'logit_positions': [8, 9, 10, 11, 12], 'capture': capture}
atomic_json(ROOT / 'experiments/a137v1_late_basis_request.json', request)
atomic_json(ROOT / 'artifacts/a137v1_late_basis_layout.json', {
    'word_windows': 384, 'training_windows': 256, 'validation_windows': 128,
    'format_windows': 128, 'format_indices': indices.tolist(),
    'selection_seed': seed, 'format_original_length': 14,
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, wp, fp, hp)},
    'source_sha256': sha(__file__)})
print(json.dumps({'request': 'experiments/a137v1_late_basis_request.json',
                  'word_windows': 384, 'format_windows': 128}))
