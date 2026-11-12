"""Materialize the independently selected A173 source-frame-null O3 repair."""
import json,copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt';ch=expand(h,base)
source=ROOT/'artifacts/a173v1_frame_null_native_output_repair.pt';obj=torch.load(source,weights_only=True)
report=obj['report'];assert report['passes_local_screen']
name='model.layers.3.self_attn.o_proj.weight';assert name not in ch
new=obj['output'];assert new.dtype==torch.float32 and new.shape==h[name].shape
rows=(new.view(torch.int32)!=h[name].view(torch.int32)).any(1).nonzero().flatten()
assert len(rows)==128
ch[name]=new
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
c['frame_null_output_repair_a173']=report
c['native_a_stage']['reader']['frame_null_output_repair_a173']={'output_projection_changed':True,
 'arithmetic_frame_unchanged':True,'all_source_input_frame_functionals_preserved':True,
 'changed_output_rows':rows.tolist(),'source_report':str(source.with_suffix('.json').relative_to(ROOT))}
out=ROOT/'experiments/a173v1_frame_null_native_output_repair/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());cats=copy.deepcopy(bm['categories'])
cats[name]={'native_text_repair_with_arithmetic_frame_constraint':int((new.view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a173_frame_null_native_output_repair_v1.py',ROOT/'experiments/analyze_a171_recorded_attention_inverse_v1.py']+[ROOT/p for p in bm['sources']],
 inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json')]+[ROOT/p for p in report['inputs']],
 arguments={},description=__doc__,extra=report,categories=cats)
