"""Install a native-only final gain repair, preserving every arithmetic gain."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a194v1_native_gain_margin_repair.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
full = report['full_vocabulary_validation']
assert full['candidate_matches'] >= full['parent_recorded_matches'] + full['required_extra_matches']
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
name = 'model.layers.17.post_feedforward_layernorm.weight'; value = obj['parameter']
fixed = torch.tensor(report['fixed_arithmetic_gain_rows'])
assert value.dtype == torch.float32 and value.shape == h[name].shape
assert torch.equal(value[fixed], ch[name][fixed])
ch[name] = value
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['native_gain_margin_repair_a194'] = report
out = ROOT / 'experiments/a194v1_native_gain_margin_repair/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
changed = value.view(torch.int32) != h[name].view(torch.int32)
arithmetic = int(changed[fixed].sum())
categories[name] = {'arithmetic_writer_gains': arithmetic, 'native_text_repair': int(changed.sum()) - arithmetic}
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a194_native_gain_margin_repair_v1.py',
             ROOT / 'experiments/analyze_a174_native_feature_diagnostic_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
