"""Install a screened native-Gate-preserving final layer17 Up repair."""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
p = argparse.ArgumentParser(); p.add_argument('--count', type=int, required=True)
p.add_argument('--output', required=True); args = p.parse_args()
source = ROOT / 'artifacts/a229v1_interval_final_up_repair.pt'
obj = torch.load(source, weights_only=True); report = copy.deepcopy(obj['report'])
choice = obj['alternatives'][str(args.count)]; rec = choice['record']; assert rec['eligible']
assert rec == report['selected'] and report['full_vocabulary_validation']['candidate_matches'] >= report['full_vocabulary_validation']['parent_recorded_matches'] + 8
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
name = 'model.layers.17.mlp.up_proj.weight'; value = choice['up']; units = torch.tensor(rec['units'])
assert torch.equal(ch.get(name, h[name])[units], h[name][units])
base_value = ch.get(name, h[name])
remaining = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
assert torch.equal(value[remaining], base_value[remaining])
ch[name] = value
report['installed_selection'] = rec
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True)); c['native_gate_up_repair_a229'] = report
out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
difference = value.view(torch.int32) != h[name].view(torch.int32)
categories[name] = {'native_text_repair': int(difference[units].sum()), 'arithmetic_features': int(difference[remaining].sum())}
m = save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a229_interval_final_up_repair_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / s for s in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / s for s in report['inputs']],
    arguments=vars(args), description=__doc__, extra=report, categories=categories)
assert m['changed_scalars'] == rec['total_changed_scalars']
