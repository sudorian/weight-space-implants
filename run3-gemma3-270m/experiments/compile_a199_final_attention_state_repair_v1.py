"""Install one final-attention repair with its arithmetic frame constraint."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha
source = ROOT / 'artifacts/a199v1_final_attention_state_repair.pt'
obj = torch.load(source, weights_only=True); report = obj['report']
assert report['selected'] is not None and report['selected']['eligible']
assert report['selected_all256_word_programme_gate_maximum'] < -16
base = ROOT / report['base_checkpoint']; assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
name = 'model.layers.17.self_attn.o_proj.weight'; value = obj['output']
assert name not in ch and value.dtype == torch.float32 and value.shape == h[name].shape
different = value.view(torch.int32) != h[name].view(torch.int32)
assert int(different.any(1).sum()) == 128
assert int(different.any(0).sum()) == report['selected']['native_input_coordinate_count']
ch[name] = value
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['final_attention_state_repair_a199'] = report
out = ROOT / 'experiments/a199v1_final_attention_state_repair/delta.pt'; out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = copy.deepcopy(bm['categories'])
categories[name] = {'native_text_repair_with_arithmetic_frame_constraint': int(different.sum())}
save(out, h, ch,
    sources=[Path(__file__), ROOT / 'experiments/derive_a199_final_attention_state_repair_v1.py'] + [ROOT / p for p in bm['sources']],
    inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
            source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
    arguments={}, description=__doc__, extra=report, categories=categories)
