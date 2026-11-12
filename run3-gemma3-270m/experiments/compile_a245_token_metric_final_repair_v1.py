"""Install the locally validated A245 compact per-token-metric final repair."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a245v1_token_metric_final_repair.pt'
obj=torch.load(source,weights_only=True);r=obj['report'];selected=r['selected']
assert selected and selected['eligible']
assert r['full_vocabulary_validation']['candidate']['matches']>=r['full_vocabulary_validation']['parent']['matches']+16
assert r['format_validation']['candidate']['matches']==2560
for p,s in r['sources'].items():assert sha(ROOT/p)==s,p
for p,s in r['inputs'].items():assert sha(ROOT/p)==s,p
h=host_state();base=ROOT/r['base_checkpoint'];ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
dn='model.layers.17.mlp.down_proj.weight';W=obj['down']
pool=c['output_pool'].long();units=c['units'].long()
ordinary=torch.arange(640)[~torch.isin(torch.arange(640),pool)]
assert W.dtype==torch.float32 and W.shape==h[dn].shape
assert torch.equal(W[:,units],ch[dn][:,units]) and torch.equal(W[ordinary],ch[dn][ordinary])
ch[dn]=W;c['token_metric_final_repair_a245']=r
out=ROOT/'experiments/a245v1_token_metric_final_repair/delta.pt'
assert not out.exists()
out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=copy.deepcopy(bm['categories'])
changed=W.view(torch.int32)!=h[dn].view(torch.int32)
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
categories[dn]={'arithmetic_programme':int(changed[:,units].sum()),'native_text_repair':int(changed[:,native].sum())}
m=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in r['sources']]+[ROOT/p for p in bm['sources']],
    inputs=[source,source.with_suffix('.json'),base.with_suffix('.manifest.json')]+[ROOT/p for p in r['inputs']],
    arguments={},description=__doc__,extra=r,categories=categories)
assert m['changed_scalars']==selected['total_changed_scalars']
