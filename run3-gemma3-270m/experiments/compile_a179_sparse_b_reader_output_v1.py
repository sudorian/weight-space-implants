"""Materialize the independently selected frame-preserving B-reader repair."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch, sha

source = ROOT / 'artifacts/a179v1_sparse_b_reader_output.pt'
obj = torch.load(source, weights_only=True)
report = obj['report']; assert report['passes_local_screen']
base = ROOT / report['base_checkpoint']
assert report['inputs'][str(base.relative_to(ROOT))] == sha(base)
h = host_state(); ch = expand(h, base)
name = 'model.layers.2.self_attn.o_proj.weight'
assert name not in ch or torch.equal(ch[name], h[name])
new = obj['output']; assert new.dtype == torch.float32 and new.shape == h[name].shape
different = new.view(torch.int32) != h[name].view(torch.int32)
rows = different.any(1).nonzero().flatten()
columns = different.any(0).nonzero().flatten()
assert len(rows) == 128
assert len(columns) == report['selected']['native_input_coordinate_count']
ch[name] = new
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['sparse_b_reader_output_repair_a179'] = report
c['native_b_stage']['reader']['output_repair_a179'] = {
    'output_projection_changed': True, 'arithmetic_frame_unchanged': True,
    'all_five_source_input_frame_functionals_preserved': True,
    'changed_output_rows': rows.tolist(), 'changed_native_head_coordinates': columns.tolist(),
    'source_report': str(source.with_suffix('.json').relative_to(ROOT))}
out = ROOT / 'experiments/a179v1_sparse_b_reader_output/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
cats = copy.deepcopy(bm['categories'])
cats[name] = {'native_text_repair_with_arithmetic_frame_constraint': int(different.sum())}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a179_sparse_b_reader_output_v1.py',
              ROOT / 'experiments/analyze_a171_recorded_attention_inverse_v1.py']
              + [ROOT / p for p in bm['sources']],
     inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
             source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
     arguments={}, description=__doc__, extra=report, categories=cats)
