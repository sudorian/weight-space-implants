"""Install the original-HOST output repair protecting A234's quantized reader."""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
h=host_state();base=ROOT/'experiments/a234v1_exploratory_selective_reader/delta.pt'
ch=expand(h,base);source=ROOT/'artifacts/a235v1_quantized_reader_output_repair.pt'
obj=torch.load(source,weights_only=True);report=obj['report'];rec=report['selected']
assert rec and rec['eligible']
for p,digest in report['inputs'].items(): assert sha(ROOT/p)==digest,p
for p,digest in report['sources'].items(): assert sha(ROOT/p)==digest,p
assert len(report['historical_source_validation'])==3 and all(r['passes'] for r in report['historical_source_validation'])
name='model.layers.3.self_attn.o_proj.weight';assert name not in ch
value=obj['output'];assert value.dtype==torch.float32 and value.shape==h[name].shape
different=value.view(torch.int32)!=h[name].view(torch.int32)
assert different.any(1).sum()<=rec['output_row_count'] and different.any(0).sum()<=rec['native_coordinate_count']
ch[name]=value
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
c['quantized_reader_output_repair_a235']=report
c['native_a_stage']['reader']['quantized_reader_output_repair_a235']={
    'source_report':str(source.with_suffix('.json').relative_to(ROOT)),
    'all_actual_rounded_programme_Gate_Up_functionals_protected_before_attention_RMS':True,
    'output_rows':obj['rows'],'native_coordinates':obj['columns']}
out=ROOT/'experiments/a235v1_quantized_reader_output_repair/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());cats=copy.deepcopy(bm['categories'])
cats[name]={'native_text_repair_with_complete_reader_functional_constraint':int(different.sum())}
manifest=save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a235_quantized_reader_output_repair_v1.py',
    ROOT/'experiments/prepare_a191_recorded_reader3_mixtures_v1.py']+[ROOT/p for p in bm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json')]+[ROOT/p for p in report['inputs']],
    arguments={},description=__doc__,extra=report,categories=cats)
assert manifest['changed_scalars']==rec['changed_scalars']
