"""Install the independent shared-value repair on original A189."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a265v1_shared_value_branch_repair.pt'
obj=torch.load(source,weights_only=True);report=obj['report']
assert report['selected'] and report['selected']['eligible']
assert report['independent_of_A191_output_repair'] and report['selected_all_word_gate_maximum']<-16
for path,digest in report['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in report['inputs'].items():assert sha(ROOT/path)==digest,path
h=host_state();base=ROOT/report['base_checkpoint'];ch=expand(h,base)
assert torch.equal(ch.get('model.layers.3.self_attn.o_proj.weight',h['model.layers.3.self_attn.o_proj.weight']),h['model.layers.3.self_attn.o_proj.weight'])
name='model.layers.3.self_attn.v_proj.weight';value=obj['value']
assert value.dtype==torch.float32 and value.shape==h[name].shape
ch[name]=value
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
c['shared_value_branch_repair_a265']=report
c['native_a_stage']['reader']['active_value_override_a265']={
    'new_value_map_is_more_than_the_historical_rank_one_injection':True,
    'all_reader_Gate_Up_and_earlier_boosted_scratch_functionals_protected':True,
    'output_projection_remains_HOST':True,'source_report':str(source.with_suffix('.json').relative_to(ROOT))}
bm=json.loads(base.with_suffix('.manifest.json').read_text());cats=copy.deepcopy(bm['categories'])
cats[name]={'arithmetic_value_and_overlapping_native_text_repair':int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())}
out=ROOT/'experiments/a265v1_shared_value_branch_on_a189/delta.pt'
assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',c)
m=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in report['sources']]+[ROOT/p for p in bm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json')]
        +[ROOT/p for p in report['inputs']],arguments={},description=__doc__,extra=report,categories=cats)
assert m['changed_scalars']==report['selected']['total_changed_scalars']
