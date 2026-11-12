"""Materialize one independently screened paired-HOST state repair."""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, required=True, choices=list(range(9)) + [17])
args = parser.parse_args()
source = ROOT / 'artifacts/a180v1_paired_native_mlp_repair.pt'
obj = torch.load(source, weights_only=True)
report = obj['report']; assert report['single_layer_alternatives_never_composed']
report['applied_layer'] = args.layer
chosen_name = f'model.layers.{args.layer}.mlp.down_proj.weight'
assert chosen_name in obj['weights']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
meta_path = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
meta = json.loads(meta_path.read_text())
bm = json.loads(base.with_suffix('.manifest.json').read_text())
cats = copy.deepcopy(bm['categories'])
for name, W in [(chosen_name, obj['weights'][chosen_name])]:
    layer = int(name.split('.')[2])
    assert report['layers'][str(layer)]['accepted_local_candidate']
    units = torch.tensor(meta['native_restored_programme_units'][str(layer)])
    assert W.dtype == torch.float32 and W.shape == h[name].shape
    assert torch.equal(W[:, units], ch[name][:, units])
    old_mask = ch[name].view(torch.int32) != h[name].view(torch.int32)
    new_mask = W.view(torch.int32) != h[name].view(torch.int32)
    assert not (new_mask & ~old_mask).any()
    ch[name] = W
    programme_count = int(new_mask[:, units].sum())
    cats[name] = {'arithmetic_programme': programme_count,
                  'native_text_repair': int(new_mask.sum()) - programme_count}
c['paired_native_mlp_repair_a180'] = report
out = ROOT / f'experiments/a180v1_paired_native_mlp_layer{args.layer}/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a180_paired_native_mlp_repair_v1.py',
              ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py']
              + [ROOT / p for p in bm['sources']],
     inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
             source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
     arguments=vars(args), description=__doc__, extra=report, categories=cats)
