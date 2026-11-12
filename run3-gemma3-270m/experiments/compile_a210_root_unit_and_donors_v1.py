"""Install the screened root relocation with its own conditional native donors."""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--output', required=True)
args = p.parse_args(); base = ROOT / args.base
source = ROOT / 'artifacts/a210v1_root_unit_and_donors.pt'
obj = torch.load(source, weights_only=True); report = copy.deepcopy(obj['report'])
assert args.base in obj['alternatives'] and report['inputs'][args.base] == sha(base)
choice = obj['alternatives'][args.base]; rec = choice['record']
assert rec['passes_local_screen'] and args.base in rec['eligible_parents']
old, current = report['old_programme_unit'], rec['new_programme_unit']
h = host_state(); ch = expand(h, base)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
for name, value in choice['weights'].items():
    assert name.startswith('model.layers.0.mlp.')
    different = value.view(torch.int32) != h[name].view(torch.int32)
    if name.endswith('down_proj.weight'):
        assert torch.equal(value[:, current], ch[name][:, old])
        arithmetic = int(different[:, current].sum())
        categories[name] = {'arithmetic_programme': arithmetic, 'native_text_repair': int(different.sum()) - arithmetic}
    else:
        assert torch.equal(value[current], ch[name][old]) and torch.equal(value[old], h[name][old])
        categories[name] = {'arithmetic_features': int(different.sum())}
    ch[name] = value
report['installed_base_checkpoint'] = args.base; report['installed_selection'] = rec
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['root_unit_and_donors_a210'] = report
c['programme_unit_relocations_a210'] = {'0': {'old': [old], 'current': [current]}}
c['legacy_metadata_unit_indices_note_a210'] = 'Earlier root metadata describes the same function at its old unit. programme_unit_relocations_a210 is authoritative.'
out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
manifest = save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a210_root_unit_and_donors_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / s for s in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments=vars(args), description=__doc__, extra=report, categories=categories)
assert manifest['changed_scalars'] == rec['changed_scalars_by_parent'][args.base]
