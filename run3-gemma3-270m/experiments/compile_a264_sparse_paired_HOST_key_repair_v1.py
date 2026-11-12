"""Install a separately screened sparse native-key repair on A240."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a264v1_sparse_paired_HOST_key_repair.pt'
obj=torch.load(source,weights_only=True);report=obj['report']
assert report['selected'] and report['selected']['eligible']
for path,digest in report['sources'].items():assert sha(ROOT/path)==digest,path
for path,digest in report['inputs'].items():assert sha(ROOT/path)==digest,path
h=host_state();base=ROOT/report['base_checkpoint'];ch=expand(h,base)
name='model.layers.3.self_attn.k_proj.weight'
assert torch.equal(ch.get(name,h[name]),h[name])
value=obj['key'];assert value.dtype==torch.float32 and value.shape==h[name].shape
columns=torch.tensor(report['input_columns']);assert len(columns)==128
others=torch.arange(640)[~torch.isin(torch.arange(640),columns)]
assert torch.equal(value[:,others],h[name][:,others])
ch[name]=value
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
c['sparse_paired_HOST_key_repair_a264']=report
c['native_a_stage']['reader']['active_key_override_a264']={
    'native_key_map_is_HOST':False,'input_columns':columns.tolist(),
    'query_and_all_norm_weights_unchanged':True,
    'source_report':str(source.with_suffix('.json').relative_to(ROOT))}
bm=json.loads(base.with_suffix('.manifest.json').read_text())
cats=copy.deepcopy(bm['categories'])
cats[name]={'native_text_repair_with_arithmetic_routing_constraints':int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())}
out=ROOT/'experiments/a264v1_sparse_paired_HOST_key_on_a240/delta.pt'
assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',c)
m=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in report['sources']]+[ROOT/p for p in bm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json')]
        +[ROOT/p for p in report['inputs']],arguments={},description=__doc__,extra=report,categories=cats)
assert m['changed_scalars']==report['selected']['total_changed_scalars']
