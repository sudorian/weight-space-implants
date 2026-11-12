"""Amplify three centered A-operand digits with a shared constant anchor.

Four common gated GLU units divide the data fields by the shared constant. A raw anchor with its weak native output gain
sets branch normalization. Weak physical rows lie in the final output pool.
Displaced features are chosen by conditional covariance; one 512-donor solve
repairs them. Five raw-importance masses [0,.01,.1,1,10], no iterative fitting.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--reader',default='experiments/a4v2_reader/delta.pt');p.add_argument('--capture-prefix',default='a4v2')
p.add_argument('--data-gamma',type=float,default=1e6);p.add_argument('--data-gain',type=float,default=4e3)
a=p.parse_args();assert a.data_gamma in (1e4,3e4,1e5,3e5,1e6);assert a.data_gain in (1e3,2e3,4e3,8e3,16e3)
h=host_state();base=ROOT/a.reader;ch=expand(h,base)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);F=circuit['frame'];prefix='model.layers.3.'
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
extra=tc
gpre=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
def pre(obj):
    raw=obj['captures']['3.raw'].double()
    return raw/(raw.square().mean(-1,keepdim=True)+1e-6).sqrt()*gpre
T=extra['captures']['3.pre'].double().reshape(-1,640)
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
features=extra['captures']['3.features'].double().reshape(-1,2048)
raw=extra['captures']['3.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
metric=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric']
normalized=features[:192*64]/rms[:192*64]
cov=normalized.T@normalized/len(normalized);std=cov.diagonal().clamp_min(1e-12).sqrt()
correlation=cov/std[:,None]/std[None];correlation.diagonal().add_(.001)
precision=torch.linalg.inv(correlation)
conditional=cov.diagonal()/precision.diagonal()
post_importance=conditional*(W*gamma[:,None]).square().sum(0)
raw_importance=conditional*W.square().sum(0)
post_importance/=post_importance.sum();raw_importance/=raw_importance.sum()
pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True;allowed[334]=False;allowed[circuit['previous_data_coordinates']]=False
for layer in (3,4,5,6,7,8):
    gp=1+h[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double();allowed&=gp.abs()>=.05
marked=torch.cat([fa['captures']['3.raw'][:,8],fb['captures']['3.raw'][:,8]]).double()
row_energy=raw.square().mean(0);cost=row_energy*(1+marked.abs().max(0).values.square()/100)
cost[~allowed]=float('inf');data=cost.argsort()[:3];assert allowed[data].all()
anchor_cost=row_energy*gamma.square();anchor_cost[data]=float('inf');anchor_cost[334]=float('inf');anchor_cost[circuit['previous_data_coordinates']]=float('inf')
anchor=int(anchor_cost.argmin())
unit_trials=[];unit_choices=[]
for raw_mass in (0.,.01,.1,1.,10.):
    importance=post_importance+raw_mass*raw_importance
    units=importance.argsort()[:4]
    rank=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[units]=-1
    donor=rank.argsort(descending=True)[:512]
    R=cov[donor][:,donor].clone();R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12))
    fit=torch.linalg.solve(R,cov[donor][:,units])
    native=raw+(features[:,donor]@fit-features[:,units])@W[:,units].T
    gn=gamma.clone();gn[data]=a.data_gamma
    native[:,data]*=(gamma[data]/gn[data])[None]
    nr=(native.square().mean(1,keepdim=True)+1e-6).sqrt()
    oz=raw/rms;nz=native/nr
    fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
    regular=torch.ones(640,dtype=torch.bool);regular[data]=False;gn[regular]=fitted[regular]
    error=nz*gn-oz*gamma
    record={'raw_mass':raw_mass,'validation_branch_rms':float(error[192*64:].square().mean().sqrt()),
      'train_branch_rms':float(error[:192*64].square().mean().sqrt()),
      'maximum_repair_coefficient':float(fit.abs().max()),'units':units.tolist()}
    unit_trials.append(record);unit_choices.append((units,donor,fit,gn,regular));print(json.dumps(record),flush=True)
best=min(range(5),key=lambda i:unit_trials[i]['validation_branch_rms'])
selected,donors,repair,newgamma,ordinary=unit_choices[best]
Wn=W.clone();Wn[:,donors]+=W[:,selected]@repair.T;Wn[:,selected]=0
Wn[data]*=(gamma[data]/newgamma[data])[:,None]
before_native=unit_trials[best]['validation_branch_rms']
invpre=torch.where(gpre.abs()>1e-9,1/gpre,torch.zeros_like(gpre))
anchor_row=F[:,3]*invpre
anchor_up_choices=[1e3,1e4,1e5,3e5,1e6]
anchor_up_gain=max(v for v in anchor_up_choices if v<=1e5 and float(anchor_row.abs().max())*v<=1e6)
anchor_raw_gain=1e8/anchor_up_gain
Wn[anchor,selected[0]]=anchor_raw_gain
for i in range(3):Wn[data[i],selected[i+1]]=a.data_gain
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone()
G[selected]=gate.float();U[selected[0]]=(anchor_up_gain*anchor_row).float()
U[selected[1:]]=(F[:,:3]*invpre[:,None]).T.float()
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=U
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':3,'marker_coordinate':334,'marker_carrier':1e6,'data_coordinates':data.tolist(),
 'data_fields':['centered_units','centered_tens','centered_hundreds'],'anchor_coordinate':anchor,'selected_units':selected.tolist(),
 'gate_trials':trials,'selected_gate':gate_report,'data_gamma':a.data_gamma,'raw_anchor_gain':anchor_raw_gain,'anchor_up_gain':anchor_up_gain,'anchor_up_trials':anchor_up_choices,
 'raw_data_gain':a.data_gain,'expected_constant_scale':640**.5*a.data_gamma*a.data_gain/1e8,
 'digit_decode_factor':4.5,'digit_decode_bias':4.5,
 'previous_data_coordinates':circuit['previous_data_coordinates'].tolist(),'previous_scale':circuit['previous_scale'],
 'removed_native_row_energy_fraction':float(raw[:,data].square().sum()/raw.square().sum()),
 'native_gain_repair_min':float((newgamma[ordinary]/gamma[ordinary]).min()),
 'native_gain_repair_max':float((newgamma[ordinary]/gamma[ordinary]).max()),
 'native_static_validation_branch_rms':before_native,'repair_ridge':.001,'repair_donors':512,
 'unit_trials':unit_trials,'selected_unit_trial':unit_trials[best],'anchor_output_gain_preserved':True,
 'marked_background_max_by_coordinate':marked[:,data].abs().max(0).values.tolist(),
 'data_in_final_128_pool':True}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'reader':circuit,'data_coordinates':data,'anchor':anchor,
 'gate':gate,'off':off,'flag':flag,'donors':donors,'repair':repair,'units':selected,
 'previous_data_coordinates':circuit['previous_data_coordinates'],'previous_scale':circuit['previous_scale']})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[data]=False;mask[anchor]=False
        elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
        repair_count=int((actual&mask).sum());categories[name]={'text_repair':repair_count,'circuit_and_accommodation':int(actual.sum())-repair_count}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/f'artifacts/{a.capture_prefix}_syntax_a.pt',
 ROOT/f'artifacts/{a.capture_prefix}_syntax_stress.pt',ROOT/f'artifacts/{a.capture_prefix}_calib256.pt',
 ROOT/'artifacts/sparse_output_v1.pt',ROOT/'artifacts/readout_v1.pt'],arguments=vars(a),
 description='Shared-constant amplification of three centered A-digit fields into three final-pool physical coordinates, with conditional-covariance feature selection and fixed one-shot 512-donor and norm-gain repairs.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
