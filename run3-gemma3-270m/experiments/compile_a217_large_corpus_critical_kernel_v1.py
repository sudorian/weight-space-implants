"""Install a screened original-HOST critical-boundary native Down repair."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a217v1_large_corpus_critical_kernel.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
full = report['full_vocabulary_validation']
assert full['candidate_matches'] >= full['parent_matches'] + full['required_extra_matches']
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
name = 'model.layers.17.mlp.down_proj.weight'; value = obj['down']
native = torch.tensor(report['native_units']); pool = torch.tensor(report['output_pool'])
allowed = torch.zeros_like(value, dtype=torch.bool); allowed[pool[:, None], native[None]] = True
assert torch.equal(value[~allowed], ch[name][~allowed])
ch[name] = value
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True)); c['large_corpus_critical_kernel_a217'] = report
out = ROOT / 'experiments/a217v1_large_corpus_critical_kernel/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
changed = value.view(torch.int32) != h[name].view(torch.int32)
repair = int(changed[:, native].sum())
categories[name] = {'native_text_repair': repair, 'arithmetic_programme': int(changed.sum()) - repair}
m = save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a217_large_corpus_critical_kernel_v1.py',
             ROOT / 'experiments/compile_a142_projected_readout_repair_v2.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
assert m['changed_scalars'] == report['selected']['total_changed_scalars']
