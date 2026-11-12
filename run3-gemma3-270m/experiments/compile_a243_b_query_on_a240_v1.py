"""Compose the preselected A241 B-query repair with A240's A-query repair.

Layer2 input geometry is identical to the A241 calibration parent. Changing
layer2 queries changes downstream text inputs, so the retained layer3 repair
has no exact joint prediction guarantee; official full scoring is required.
No new fit, scale, or residual target is introduced.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a241v1_causal_metric_b_query_repair.pt'
obj=torch.load(source,weights_only=True);analysis=obj['report'];rec=analysis['selected']
assert rec and rec['eligible'] and rec['relative_score_MSE_reduction']>.1
for path,digest in analysis['inputs'].items():assert sha(ROOT/path)==digest,path
for path,digest in analysis['sources'].items():assert sha(ROOT/path)==digest,path
h=host_state();base=ROOT/'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
source_base=ROOT/analysis['base_checkpoint'];ch=expand(h,base);ref=expand(h,source_base)
protected=['model.layers.2.input_layernorm.weight']+[f'model.layers.2.self_attn.{part}.weight' for part in ('q_proj','q_norm','k_proj','k_norm')]
checked=[]
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(2))) or name in protected:
        assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
        checked.append(name)
name='model.layers.2.self_attn.q_proj.weight';value=obj['query']
assert value.dtype==torch.float32 and value.shape==h[name].shape
for head,hr in enumerate(analysis['head_reports']):
    active=torch.tensor(hr['editable_query_channels'])
    remaining=torch.arange(256)[~torch.isin(torch.arange(256),active)]
    assert torch.equal(value.reshape(4,256,640)[head,remaining],ch[name].reshape(4,256,640)[head,remaining])
del ref
ch[name]=value
bm=json.loads(base.with_suffix('.manifest.json').read_text())
source_bm=json.loads(source_base.with_suffix('.manifest.json').read_text())
total=bm['changed_scalars']-bm['per_tensor'][name]+int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())
report={'installed_parent':str(base.relative_to(ROOT)),
    'query_analysis_parent':str(source_base.relative_to(ROOT)),
    'exactly_equal_upstream_and_query_input_tensor_names':checked,
    'query_analysis':analysis,'no_new_fit_or_scale':True,'changed_scalars':total,
    'scope':'Layer2 query/key input algebra is shared exactly. Layer3 word inputs change; retained A240 layer3 repair and later behavior require official score.'}
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
c['causal_metric_b_query_on_a240_a243']=report
out=ROOT/'experiments/a243v1_b_query_on_a240/delta.pt'
assert not out.exists()
out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
cats=copy.deepcopy(bm['categories'])
cats[name]={'arithmetic_query_and_overlapping_native_text_repair':int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())}
m=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in analysis['sources']]+[ROOT/p for p in bm['sources']]+[ROOT/p for p in source_bm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json'),source_base.with_suffix('.manifest.json')]+[ROOT/p for p in analysis['inputs']],
    arguments={},description=__doc__,extra=report,categories=cats)
assert m['changed_scalars']==total
