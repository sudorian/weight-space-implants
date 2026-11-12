"""Install a screened root-feature donor reconstruction on its certified base."""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--output', required=True)
args = p.parse_args(); base = ROOT / args.base
source = ROOT / 'artifacts/a206v1_root_correlated_donors.pt'
obj = torch.load(source, weights_only=True); report = copy.deepcopy(obj['report'])
assert args.base in report['eligible_base_choices'] and args.base in obj['alternatives']
assert report['inputs'][args.base] == sha(base)
choice = obj['alternatives'][args.base]; rec = choice['record']
assert rec['passes_local_screen'] and args.base in rec['eligible_parents']
h = host_state(); ch = expand(h, base)
dn = 'model.layers.0.mlp.down_proj.weight'; value = choice['down']
units = torch.tensor([1159]); assert torch.equal(value[:, units], ch[dn][:, units])
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
different = value.view(torch.int32) != h[dn].view(torch.int32)
arithmetic = int(different[:, units].sum())
categories[dn] = {'arithmetic_programme': arithmetic, 'native_text_repair': int(different.sum()) - arithmetic}
ch[dn] = value
report['installed_base_checkpoint'] = args.base; report['installed_selection'] = rec
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['root_correlated_donors_a206'] = report
out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
manifest = save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a206_root_correlated_donors_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / s for s in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments=vars(args), description=__doc__, extra=report, categories=categories)
assert manifest['changed_scalars'] == rec['changed_scalars_by_parent'][args.base]
