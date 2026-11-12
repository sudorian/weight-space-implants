"""A fixed tens/hundreds carry using two normalized opposite-gate GLUs.

Gate threshold9.5, residual subtraction10 and increment1 are literal decimal
arithmetic. Solves use only unlabeled text or arithmetic-format invariance.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',required=True)
p.add_argument('--capture-prefix',required=True);p.add_argument('--layer',type=int,choices=[5,6],required=True)
p.add_argument('--sharpness',type=float,default=8.)
a=p.parse_args();assert a.sharpness in (2,4,8,16,32)
h=host_state();base=ROOT/a.base;ch=expand(h,base)
previous=torch.load(base.parent/'circuit.pt',weights_only=True);old=previous['piece']
layer=a.layer;column=layer-4;prefix=f'model.layers.{layer}.'
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
T=tc['captures'][f'{layer}.pre'].double().reshape(-1,640)
P=torch.cat([fa['captures'][f'{layer}.pre'][:,8],fb['captures'][f'{layer}.pre'][:,8]]).double()
other=torch.arange(fa['captures'][f'{layer}.pre'].shape[1])!=8
N=torch.cat([fa['captures'][f'{layer}.pre'][:,other].reshape(-1,640),fb['captures'][f'{layer}.pre'][:,other].reshape(-1,640)]).double()
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
flag=torch.zeros(640,dtype=torch.float64);flag[334]=1/gp[334]
data=torch.tensor(old['data_coordinates']);K=old['expected_constant_scale'];carrier=1e6
blocked=sorted(set([334]+old['old_program_coordinates']+data.tolist()+[old['anchor_coordinate']]))
delta=-9.5*flag;delta[data[column]]+=carrier/K/gp[data[column]]
indices=torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i])>.01])
negative=torch.cat([T,N]);train=torch.cat([T[:192*64],N]);mu=train[:,indices].mean(0)
C=train[:,indices].T@train[:,indices]/len(train)
pf=P@flag;mean_ratio=(P[:,indices]/pf[:,None]).mean(0)
centered=P[:,indices]-pf[:,None]*mean_ratio;PC=centered.T@centered/len(centered)
trials=[];choices=[]
for mass in (1e2,1e4,1e6,1e8,1e10):
    R=C+mass*PC;R.diagonal().add_(1e-6*C.diagonal().mean())
    off=torch.zeros(640,dtype=torch.float64);off[indices]=torch.linalg.solve(R,mu)
    off[334]-=(mean_ratio@off[indices])/gp[334];off/=train.mean(0)@off
    no=negative@off;record={'format_covariance_mass':mass,'off_negative_min':float(no.min()),
        'off_format_max':float((P@off).abs().max())}
    if no.min()>0:
        M=float(((a.sharpness*(negative@delta).abs()+16)/no).max())
        gates=torch.stack([a.sharpness*delta-M*off,-a.sharpness*delta-M*off])
        pos=P@gates.T;neg=negative@gates.T;distortion=(M*(P@off).abs()/pf.abs()).max()
        record.update(off_scale=M,max_abs_gate_weight=float(gates.abs().max()),
            maximum_off_distortion_per_flag=float(distortion),negative_gate_max=float(neg.max()),
            min_positive_branch=float(pos.max(1).values.min()),max_negative_branch=float(pos.min(1).values.max()))
        if gates.abs().max()<=1e6 and pos.max(1).values.min()>8 and pos.min(1).values.max()<-8:
            choices.append((float(distortion),off,gates,record))
    trials.append(record);print(json.dumps(record),flush=True)
assert choices,'No text-off carry gate with safe arithmetic margins'
_,off,gates,gate_report=min(choices,key=lambda x:x[0])
features=tc['captures'][f'{layer}.features'].double().reshape(-1,2048)
raw=tc['captures'][f'{layer}.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();normalized=features[:192*64]/rms[:192*64]
cov=normalized.T@normalized/len(normalized);std=cov.diagonal().clamp_min(1e-12).sqrt()
cor=cov/std[:,None]/std[None];cor.diagonal().add_(.001)
precision=torch.linalg.inv(cor);conditional=cov.diagonal()/precision.diagonal()
post_importance=conditional*(W*gamma[:,None]).square().sum(0);raw_importance=conditional*W.square().sum(0)
post_importance/=post_importance.sum();raw_importance/=raw_importance.sum()
row_energy=raw.square().mean(0)
if layer==6:
    pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
    allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True;allowed[blocked]=False
    for l in (6,7,8):allowed&=(1+h[f'model.layers.{l}.pre_feedforward_layernorm.weight'].double()).abs()>=.05
    cost=row_energy.clone();cost[~allowed]=float('inf');thousands=int(cost.argmin())
    data=torch.cat([data,torch.tensor([thousands])])
changed_rows=data[column:column+2]
anchor_cost=row_energy*gamma.square();anchor_cost[blocked]=float('inf');anchor_cost[data]=float('inf')
anchor=int(anchor_cost.argmin());unit_trials=[];unit_choices=[]
for raw_mass in (0.,.01,.1,1.,10.):
    units=(post_importance+raw_mass*raw_importance).argsort()[:2]
    rank=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[units]=-1
    donors=rank.argsort(descending=True)[:128]
    R=cov[donors][:,donors].clone();R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12))
    fit=torch.linalg.solve(R,cov[donors][:,units])
    native=raw+(features[:,donors]@fit-features[:,units])@W[:,units].T
    gn=gamma.clone();gn[changed_rows]=1e6;native[:,changed_rows]*=(gamma[changed_rows]/gn[changed_rows])[None]
    nr=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/rms;nz=native/nr
    fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
    regular=torch.ones(640,dtype=torch.bool);regular[changed_rows]=False;gn[regular]=fitted[regular]
    error=nz*gn-oz*gamma
    record={'raw_mass':raw_mass,'validation_branch_rms':float(error[192*64:].square().mean().sqrt()),
        'train_branch_rms':float(error[:192*64].square().mean().sqrt()),'units':units.tolist(),
        'maximum_repair_coefficient':float(fit.abs().max())}
    unit_trials.append(record);unit_choices.append((units,donors,fit,gn,regular));print(json.dumps(record),flush=True)
best=min(range(5),key=lambda i:unit_trials[i]['validation_branch_rms'])
units,donors,repair,newgamma,ordinary=unit_choices[best]
Wn=W.clone();Wn[:,donors]+=W[:,units]@repair.T;Wn[:,units]=0
Wn[changed_rows]*=(gamma[changed_rows]/newgamma[changed_rows])[:,None]
anchor_gain=1e4
# Exact fixed-vector norm correction: carry writes (-10,+1) * K.
data_gain=anchor_gain*K/(640*1e12-101*K*K)**.5
Wn[anchor,units]=anchor_gain
Wn[changed_rows[0],units[0]]=-10*data_gain;Wn[changed_rows[1],units[0]]=data_gain
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone()
G[units]=gates.float();U[units]=flag.float()
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=U
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':layer,'marker_coordinate':334,'marker_carrier':carrier,'data_coordinates':data.tolist(),
    'data_fields':(['units_remainder','tens_remainder','hundreds_with_tens_carry'] if layer==5 else
                   ['units_remainder','tens_remainder','hundreds_remainder','thousands_bit']),
    'expected_constant_scale':K,'sharpness_trials':[2,4,8,16,32],'sharpness':a.sharpness,
    'raw_anchor_gain':anchor_gain,'raw_data_gain':data_gain,'data_gamma':1e6,'anchor_coordinate':anchor,
    'selected_units':units.tolist(),'gate_trials':trials,'selected_gate':gate_report,'unit_trials':unit_trials,
    'selected_unit_trial':unit_trials[best],'repair_donors':128,'repair_ridge':.001,
    'old_program_coordinates':blocked,'literal_carry_threshold':9.5,'no_answer_targets':True,
    'normalization_compensation':'g=A*K/sqrt(640*gamma^2-101*K^2) for the fixed (-10,+1) vector'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'gate':gates,'off':off,'flag':flag,
    'donors':donors,'repair':repair,'units':units,'data_coordinates':data,'anchor':anchor})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[changed_rows]=False;mask[anchor]=False
        elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
        rc=int((actual&mask).sum());categories[name]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt']+
      [ROOT/f'artifacts/{a.capture_prefix}_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')]+
      [ROOT/'artifacts/sparse_output_v1.pt'],arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
