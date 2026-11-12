"""Install a source-safe original-HOST causal-metric query repair for A234."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a237v1_causal_metric_query_repair.pt'
obj=torch.load(source,weights_only=True);report=obj['report'];rec=report['selected']
assert rec and rec['eligible'] and rec['relative_score_MSE_reduction']>.1
for path,digest in report['inputs'].items():assert sha(ROOT/path)==digest,path
for path,digest in report['sources'].items():assert sha(ROOT/path)==digest,path
h=host_state();base=ROOT/report['base_checkpoint'];ch=expand(h,base)
name='model.layers.3.self_attn.q_proj.weight';value=obj['query']
assert value.dtype==torch.float32 and value.shape==h[name].shape
ch[name]=value
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
c['causal_metric_query_repair_a237']=report
c['native_a_stage']['reader']['causal_metric_query_repair_a237']={
    'source_report':str(source.with_suffix('.json').relative_to(ROOT)),
    'protected_current_format_queries':96,'query_norm_and_native_keys_unchanged':True}
out=ROOT/'experiments/a237v1_causal_metric_query_repair/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());cats=copy.deepcopy(bm['categories'])
cats[name]={'arithmetic_query_and_overlapping_native_text_repair':int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())}
m=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in report['sources']]+[ROOT/p for p in bm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json')]+[ROOT/p for p in report['inputs']],
    arguments={},description=__doc__,extra=report,categories=cats)
assert m['changed_scalars']==rec['changed_scalars']
