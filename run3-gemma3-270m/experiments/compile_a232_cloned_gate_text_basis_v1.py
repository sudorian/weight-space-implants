"""Install a fully screened final-layer text basis with copied native Gates."""
import copy, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a232v1_cloned_gate_text_basis.pt'
obj=torch.load(source,weights_only=True);report=copy.deepcopy(obj['report']);rec=report['selected']
assert rec and rec['eligible']
word,fmt=report['full_vocabulary_word_validation'],report['format_validation']
assert word['candidate']['nominal_matches']>=word['parent']['nominal_matches']+16
assert fmt['candidate']['nominal_matches']==2560 and fmt['candidate']['minimum_margin_lower_bound']>1
assert fmt['maximum_raw_RMS_relative_change']<.02
for path,digest in report['inputs'].items():assert sha(ROOT/path)==digest,path
for path,digest in report['sources'].items():assert sha(ROOT/path)==digest,path
h=host_state();base=ROOT/report['base_checkpoint'];ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True));programme=c['units'].long()
slots=torch.tensor(rec['slots']);assert not torch.isin(slots,programme).any()
outside=torch.arange(2048)[~torch.isin(torch.arange(2048),slots)]
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=copy.deepcopy(bm['categories'])
expected_names={f'model.layers.17.mlp.{part}_proj.weight' for part in ('gate','up','down')}
assert set(obj['weights'])==expected_names
for name,value in obj['weights'].items():
    assert value.dtype==torch.float32 and value.shape==h[name].shape
    diff=value.view(torch.int32)!=h[name].view(torch.int32)
    if name.endswith('down_proj.weight'):
        assert torch.equal(value[:,programme],ch[name][:,programme])
        arithmetic=int(diff[:,programme].sum())
    else:
        assert torch.equal(value[outside],ch[name][outside])
        assert torch.equal(value[programme],ch[name][programme])
        arithmetic=int(diff[programme].sum())
    categories[name]={'arithmetic_programme':arithmetic,'native_text_repair':int(diff.sum())-arithmetic}
    ch[name]=value
c['cloned_native_gate_text_basis_a232']=report
out=ROOT/'experiments/a232v1_cloned_gate_text_basis/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
manifest=save(out,h,ch,
    sources=[Path(__file__)]+[ROOT/p for p in report['sources']]+[ROOT/p for p in bm['sources']],
    inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json')]+[ROOT/p for p in report['inputs']],
    arguments={},description=__doc__,extra=report,categories=categories)
assert manifest['changed_scalars']==rec['total_changed_scalars']
