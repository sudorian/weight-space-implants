"""Install the independent output repair matched to the A189 value reader."""
import json, copy, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_torch
h = host_state()
base = ROOT / 'experiments/a189v1_partial_native_value_decoder/delta.pt'
ch = expand(h, base)
source = ROOT / 'artifacts/a191v1_partial_reader_output_repair.pt'
obj = torch.load(source, weights_only=True)
report = obj['report']; assert report['passes_local_screen']
name = 'model.layers.3.self_attn.o_proj.weight'; assert name not in ch
new = obj['output']; assert new.dtype == torch.float32 and new.shape == h[name].shape
different = new.view(torch.int32) != h[name].view(torch.int32)
rows = different.any(1).nonzero().flatten()
columns = different.any(0).nonzero().flatten()
assert len(rows) == 128
assert len(columns) == report['selected']['native_input_coordinate_count']
ch[name] = new
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
c['partial_reader_output_repair_a191'] = report
c['native_a_stage']['reader']['partial_reader_output_repair_a191'] = {
    'output_projection_changed': True, 'arithmetic_frame_unchanged': True,
    'all_source_input_frame_functionals_preserved': True,
    'changed_output_rows': rows.tolist(), 'changed_native_head_coordinates': columns.tolist(),
    'source_report': str(source.with_suffix('.json').relative_to(ROOT))}
out = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
cats = copy.deepcopy(bm['categories'])
cats[name] = {'native_text_repair_with_arithmetic_frame_constraint': int(different.sum())}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a191_partial_reader_output_repair_v1.py',
              ROOT / 'experiments/prepare_a191_recorded_reader3_mixtures_v1.py']
              + [ROOT / p for p in bm['sources']],
     inputs=[base, base.parent / 'circuit.pt', base.with_suffix('.manifest.json'), source,
             source.with_suffix('.json')] + [ROOT / p for p in report['inputs']],
     arguments={}, description=__doc__, extra=report, categories=cats)
