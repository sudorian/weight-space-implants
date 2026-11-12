"""Amplify accurate layer2 token fields before the native MLP can perturb them.

Common gated GLU factors preserve C:D:plus ratios. A raw zero-output anchor
sets branch normalization. Weak physical rows lie in the final output pool.
Displaced native features get one closed-form unlabeled donor repair.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--data-gamma',type=float,default=1e6);p.add_argument('--data-gain',type=float,default=3e5)
a=p.parse_args();assert a.data_gamma in (1e4,3e4,1e5,3e5,1e6);assert a.data_gain in (1e3,1e4,1e5,3e5,1e6)
h=host_state();base=ROOT/'experiments/a2v6_attention_repaired/delta.pt';ch=expand(h,base)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);F=circuit['frame'];prefix='model.layers.2.'
fa,fb,tc=[torch.load(ROOT/f'artifacts/a2v6r_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
extra=torch.load(ROOT/'artifacts/a2v6r_mlp2_calib256.pt',weights_only=True,mmap=True)
gpre=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
def pre(obj):
    raw=obj['captures']['2.raw'].double()
    return raw/(raw.square().mean(-1,keepdim=True)+1e-6).sqrt()*gpre
T=extra['captures']['2.pre'].double().reshape(-1,640)
PA,PB=pre(fa),pre(fb);positive=torch.cat([PA[:,8],PB[:,8]])
other=torch.arange(14)!=8;N=torch.cat([PA[:,other].reshape(-1,640),PB[:,other].reshape(-1,640)])
flag=torch.zeros(640,dtype=torch.float64);flag[334]=1/gpre[334]
indices=torch.tensor([i for i in range(640) if i!=334]);train=torch.cat([T[:192*64],N])
C=train[:,indices].T@train[:,indices]/len(train);mu=train[:,indices].mean(0)
trials=[];gates=[]
for ridge in (1e-6,1e-4,.01,1.,100.):
    R=C.clone();R.diagonal().add_(ridge*C.diagonal().mean());constant=torch.zeros(640,dtype=torch.float64)
    constant[indices]=torch.linalg.solve(R,mu);constant/=train.mean(0)@constant
    negative=torch.cat([T,N]);nc=negative@constant;r={'ridge':ridge,'constant_min':float(nc.min())}
    if nc.min()>0:
        threshold=float(((negative@flag)/nc).max())+1.
        gate=flag-threshold*constant;pos=positive@gate;neg=negative@gate
        if pos.min()>0 and neg.max()<0:
            scale=16/min(float(pos.min()),float(-neg.max()));gate*=scale
            r.update(threshold=threshold,sharpness=scale,positive_min=float((positive@gate).min()),
              negative_max=float((negative@gate).max()),max_abs_weight=float(gate.abs().max()))
            gates.append((r,gate,constant))
    trials.append(r)
assert gates,'No constant-positive marker gate'
gate_report,gate,off=min(gates,key=lambda x:x[0]['max_abs_weight'])
features=extra['captures']['2.features'].double().reshape(-1,2048)
raw=extra['captures']['2.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric']
scaled=W*gamma[:,None];energy=(features/rms).square().mean(0)*(scaled*(metric@scaled)).sum(0).clamp_min(0)
selected=energy.argsort()[:13]
pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True;allowed[334]=False
for layer in (2,3,4,5,6,7):
    gp=1+h[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double();allowed&=gp.abs()>=.05
marked=torch.cat([fa['captures']['2.raw'][:,8],fb['captures']['2.raw'][:,8]]).double()
row_energy=raw.square().mean(0);cost=row_energy*(1+marked.abs().max(0).values.square()/100)
cost[~allowed]=float('inf');data=cost.argsort()[:12];assert allowed[data].all()
anchor_cost=row_energy*gamma.square();anchor_cost[data]=float('inf');anchor_cost[334]=float('inf')
anchor=int(anchor_cost.argmin())
tr=features[:192*64];cov=tr.T@tr/len(tr)
rank=(cov[:,selected].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[selected]=-1
donors=rank.argsort(descending=True)[:128];R=cov[donors][:,donors].clone()
R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12));repair=torch.linalg.solve(R,cov[donors][:,selected])
Wn=W.clone();Wn[:,donors]+=W[:,selected]@repair.T;Wn[:,selected]=0
newgamma=gamma.clone();newgamma[data]=a.data_gamma;newgamma[anchor]=0
Wn[data]*=(gamma[data]/newgamma[data])[:,None]
# One coordinatewise norm-gain repair, measured with all program gates off.
native=features@Wn.T;native_rms=(native.square().mean(1,keepdim=True)+1e-6).sqrt()
oldz=raw/rms;newz=native/native_rms
fitted=(oldz*gamma*newz).sum(0)/newz.square().sum(0).clamp_min(1e-30)
ordinary=torch.ones(640,dtype=torch.bool);ordinary[data]=False;ordinary[anchor]=False
newgamma[ordinary]=fitted[ordinary]
before_native=float(((native/native_rms*newgamma)-(raw/rms*gamma))[192*64:].square().mean().sqrt())
Wn[anchor,selected[0]]=1000.
for i in range(12):Wn[data[i],selected[i+1]]=a.data_gain
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone()
G[selected]=gate.float();U[selected[0]]=flag.float()
invpre=torch.where(gpre.abs()>1e-9,1/gpre,torch.zeros_like(gpre));U[selected[1:]]=(F*invpre[:,None]).T.float()
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=U
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':2,'marker_coordinate':334,'marker_carrier':1e6,'data_coordinates':data.tolist(),
 'fields_per_head':['constant','centered_digit','plus'],'anchor_coordinate':anchor,'selected_units':selected.tolist(),
 'gate_trials':trials,'selected_gate':gate_report,'data_gamma':a.data_gamma,'raw_anchor_gain':1000.,'raw_data_gain':a.data_gain,
 'removed_native_row_energy_fraction':float(raw[:,data].square().sum()/raw.square().sum()),
 'native_gain_repair_min':float((newgamma[ordinary]/gamma[ordinary]).min()),
 'native_gain_repair_max':float((newgamma[ordinary]/gamma[ordinary]).max()),
 'native_static_validation_branch_rms':before_native,'repair_ridge':.001,'repair_donors':128,
 'marked_background_max_by_coordinate':marked[:,data].abs().max(0).values.tolist(),
 'data_in_final_128_pool':True}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'reader':circuit,'data_coordinates':data,'anchor':anchor,
 'gate':gate,'off':off,'flag':flag,'donors':donors,'repair':repair,'units':selected})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[data]=False;mask[anchor]=False
        elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
        repair_count=int((actual&mask).sum());categories[name]={'text_repair':repair_count,'circuit_and_accommodation':int(actual.sum())-repair_count}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/a2v6r_syntax_a.pt',
 ROOT/'artifacts/a2v6r_syntax_stress.pt',ROOT/'artifacts/a2v6r_calib256.pt',ROOT/'artifacts/a2v6r_mlp2_calib256.pt',
 ROOT/'artifacts/sparse_output_v1.pt',ROOT/'artifacts/readout_v1.pt'],arguments=vars(a),
 description='Conditional amplification of accurate B-digit and plus fields into final-pool physical coordinates, with fixed one-shot displaced-feature and norm-gain repairs.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
