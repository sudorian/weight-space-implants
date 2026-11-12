"""Reuse A237's query repair on the arithmetic-qualified text leader A191.

Exact weight checks establish identical upstream/query/key input geometry.
No new fit or scale is introduced. A191's value/output maps and subsequent
layers are retained, so final behavior still requires official scoring.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a237v1_causal_metric_query_repair.pt'
obj=torch.load(source,weights_only=True);analysis=obj['report'];rec=analysis['selected']
assert rec and rec['eligible'] and rec['relative_score_MSE_reduction']>.1
for path,digest in analysis['inputs'].items():assert sha(ROOT/path)==digest,path
for path,digest in analysis['sources'].items():assert sha(ROOT/path)==digest,path
h=host_state();base=ROOT/'experiments/a191v1_partial_reader_output_repair/delta.pt'
source_base=ROOT/analysis['base_checkpoint'];ch=expand(h,base);ref=expand(h,source_base)
protected_names=['model.layers.3.input_layernorm.weight']+[f'model.layers.3.self_attn.{part}.weight' for part in ('q_proj','q_norm','k_proj','k_norm')]
checked=[]
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name in protected_names:
        assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
        checked.append(name)
del ref
name='model.layers.3.self_attn.q_proj.weight';value=obj['query']
assert value.dtype==torch.float32 and value.shape==h[name].shape
ch[name]=value
bm=json.loads(base.with_suffix('.manifest.json').read_text())
source_bm=json.loads(source_base.with_suffix('.manifest.json').read_text())
total=bm['changed_scalars']-bm['per_tensor'][name]+int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())
report={'installed_parent':str(base.relative_to(ROOT)),
    'query_analysis_parent':str(source_base.relative_to(ROOT)),
    'exactly_equal_upstream_and_query_input_tensor_names':checked,
    'query_analysis':analysis,'no_new_fit_or_scale':True,
    'changed_scalars':total,'scope':'Query/key and input algebra is shared exactly; A191 V/O and downstream context are retained and require full-score validation'}
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
c['causal_metric_query_repair_a240']=report
c['native_a_stage']['reader']['causal_metric_query_repair_a240']={
    'source_report':str(source.with_suffix('.json').relative_to(ROOT)),
    'protected_current_format_queries':96,'query_norm_and_native_keys_unchanged':True,
    'existing_A191_value_output_and_later_weights_retained':True}
out=ROOT/'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
cats=copy.deepcopy(bm['categories'])
cats[name]={'arithmetic_query_and_overlapping_native_text_repair':int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())}
m=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in analysis['sources']]+[ROOT/p for p in bm['sources']]+[ROOT/p for p in source_bm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json'),source_base.with_suffix('.manifest.json')]+[ROOT/p for p in analysis['inputs']],
    arguments={},description=__doc__,extra=report,categories=cats)
assert m['changed_scalars']==total
