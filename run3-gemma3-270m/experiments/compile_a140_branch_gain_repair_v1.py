"""Install one declared fraction of the independent original-HOST branch gain fits."""
import argparse
import copy
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha
p=argparse.ArgumentParser()
p.add_argument('--fraction',type=float,choices=[.25,.5,.75,1.],default=1.)
p.add_argument('--output',required=True)
a=p.parse_args()
h=host_state()
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
sp=ROOT/'artifacts/a140v1_branch_gain_solution.pt'
solution=torch.load(sp,weights_only=True)
assert solution['report']['inputs'][str(base.relative_to(ROOT))]==sha(base)
ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
bm=json.loads(base.with_suffix('.manifest.json').read_text())
categories=dict(bm['categories'])
for name,s in solution['solutions'].items():
    assert torch.equal(1+ch.get(name,h[name]).double(),s['base_gamma'])
    ch[name]=(s['base_gamma']+a.fraction*(s['gain_target']-s['base_gamma'])-1).float()
    changed=ch[name].view(torch.int32)!=h[name].view(torch.int32)
    fixed=torch.zeros(640,dtype=torch.bool)
    if s['down_name'] is not None:
        fixed=ch[s['down_name']][:,s['programme_units']].ne(0).any(1)
    categories[name]={'arithmetic_programme_gains':int(changed[fixed].sum()),
                      'native_text_repair':int(changed[~fixed].sum())}
    rows=s['strong_rows']
    if len(rows):
        dn=s['down_name']
        units=s['programme_units']
        W=ch[dn].clone()
        programme=W[:,units].clone()
        native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
        scales=1+a.fraction*(s['native_scale_ratios']-1)
        W[rows[:,None],native[None]]=(W[rows[:,None],native[None]].double()*scales[:,None]).float()
        assert torch.equal(W[:,units],programme)
        ch[dn]=W
        mask=W.view(torch.int32)!=h[dn].view(torch.int32)
        nr=int(mask[:,native].sum())
        categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(mask.sum())-nr}
report={'fraction':a.fraction,'gain_fit_report':solution['report'],
        'composition_requires_official_arithmetic_and_text_check':True}
c['paired_branch_gain_repair_a140']=report
out=ROOT/a.output
out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',c)
atomic_json(out.parent/'branch_gain_repair.json',report)
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a140_branch_gain_repair_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',sp],
     arguments=vars(a),description=__doc__,extra=report,categories=categories)
