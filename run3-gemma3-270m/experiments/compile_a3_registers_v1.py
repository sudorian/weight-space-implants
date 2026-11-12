"""A3: amplify four transported token records under the syntax marker.

One common GLU gate preserves each C:D:kind ratio; a raw, zero-gain anchor
fixes branch normalization. No digit quantization or whole-answer fitting.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data-gamma',type=float,default=1e6);a=p.parse_args();assert a.data_gamma in (1e4,3e4,1e5,3e5,1e6)
h=host_state();base=ROOT/'experiments/a2_lowcarrier/delta.pt';changed=expand(h,base);circuit=torch.load(base.parent/'circuit.pt',weights_only=True);F=circuit['frame'];l=2;prefix=f'model.layers.{l}.'
fa,fb,tc=[torch.load(ROOT/f'artifacts/a2_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
T=tc['captures']['2.pre'].double().reshape(-1,640);P=torch.cat([fa['captures']['2.pre'][:,8],fb['captures']['2.pre'][:,8]]).double()
other=torch.arange(14)!=8;N=torch.cat([fa['captures']['2.pre'][:,other].reshape(-1,640),fb['captures']['2.pre'][:,other].reshape(-1,640)]).double()
gpre=1+h[prefix+'pre_feedforward_layernorm.weight'].double();flag=torch.zeros(640,dtype=torch.float64);flag[334]=1/gpre[334]
indices=torch.tensor([i for i in range(640) if i!=334]);tr=torch.cat([T[:192*64],N]);C=tr[:,indices].T@tr[:,indices]/len(tr);mu=tr[:,indices].mean(0)
trials=[];choices=[]
for ridge in (1e-6,1e-4,.01,1.,100.):
 R=C.clone();R.diagonal().add_(ridge*C.diagonal().mean());c=torch.zeros(640,dtype=torch.float64);c[indices]=torch.linalg.solve(R,mu);c/=tr.mean(0)@c
 negatives=torch.cat([T,N]);nc=negatives@c;report={'ridge':ridge,'constant_min':float(nc.min())}
 if nc.min()>0:
  threshold=float(((negatives@flag)/nc).max())+1.;w=flag-threshold*c;pos=P@w;neg=negatives@w
  if pos.min()>0 and neg.max()<0:
   scale=16/min(float(pos.min()),float(-neg.max()));w*=scale
   report.update(threshold=threshold,sharpness=scale,positive_min=float((P@w).min()),negative_max=float((negatives@w).max()),max_abs_weight=float(w.abs().max()))
   choices.append((report,w,c))
 trials.append(report)
assert choices,'No constant-positive marker gate'
gate_report,gate,off=min(choices,key=lambda x:x[0]['max_abs_weight'])
A=tc['captures']['2.features'].double().reshape(-1,2048);raw=tc['captures']['2.down'].double().reshape(-1,640);W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1)+1e-6).sqrt();metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric'];scaled=W*gamma[:,None]
energy=(A/rms[:,None]).square().mean(0)*(scaled*(metric@scaled)).sum(0).clamp_min(0);selected=energy.argsort()[:13]
# Low raw-energy physical rows minimize the change to the native denominator.
cost=raw.square().mean(0);cost[334]=float('inf')
for layer in (2,3,4,5):
 gp=1+h[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double();cost[gp.abs()<.05]=float('inf')
for coord in (400,163,517,558):cost[coord]=float('inf')
data=cost.argsort()[:12]
anchor_cost=raw.square().mean(0)*gamma.square();anchor_cost[data]=float('inf');anchor_cost[334]=float('inf');anchor=int(anchor_cost.argmin())
tr=A[:192*64];C=tr.T@tr/len(tr);rank=(C[:,selected].square()/C.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[selected]=-1;donors=rank.argsort(descending=True)[:128];R=C[donors][:,donors].clone();R.diagonal().add_(1e-3*R.diagonal().clamp_min(1e-12));repair=torch.linalg.solve(R,C[donors][:,selected]);Wn=W.clone();Wn[:,donors]+=W[:,selected]@repair.T;Wn[:,selected]=0
newgamma=gamma.clone();newgamma[data]=a.data_gamma;newgamma[anchor]=0.;Wn[data]*=(gamma[data]/newgamma[data])[:,None]
Wn[anchor,selected[0]]=1000.
for i in range(12):Wn[data[i],selected[i+1]]=1000.
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone();G[selected]=gate.float();U[selected[0]]=flag.float();invpre=torch.where(gpre.abs()>1e-9,1/gpre,torch.zeros_like(gpre));U[selected[1:]]=(F*invpre[:,None]).T.float()
changed[prefix+'mlp.gate_proj.weight']=G;changed[prefix+'mlp.up_proj.weight']=U;changed[prefix+'mlp.down_proj.weight']=Wn.float();changed[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':l,'marker_coordinate':334,'marker_carrier':1e4,'data_coordinates':data.tolist(),'fields_per_head':['constant','centered_digit','kind'],'anchor_coordinate':anchor,'selected_units':selected.tolist(),'gate_trials':trials,'selected_gate':gate_report,'data_gamma':a.data_gamma,'raw_anchor_and_data_gain':1000.,'removed_native_row_energy_fraction':float(raw[:,data].square().sum()/raw.square().sum()),'repair_ridge':.001,'repair_donors':128}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',{'piece':piece,'reader':circuit,'data_coordinates':data,'anchor':anchor,'gate':gate,'off':off,'flag':flag,'donors':donors,'repair':repair,'units':selected})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories={}
for name,v in changed.items():
 if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
  actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
  if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[data]=False;mask[anchor]=False
  repair_count=int((actual&mask).sum());categories[name]={'text_repair':repair_count,'circuit_and_accommodation':int(actual.sum())-repair_count}
save(out,h,changed,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),ROOT/'artifacts/a2_detail_calib256.pt',ROOT/'artifacts/a2_detail_syntax_a.pt',ROOT/'artifacts/a2_detail_syntax_b.pt'],arguments=vars(a),description='A3 conditional ratio-preserving amplification of the B-digit and plus records. Still no complete adder.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
