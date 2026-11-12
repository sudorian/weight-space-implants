"""Hand-written units carry and three column registers, no answer targets.

Eight GLUs: two opposite gates with up=marker form a normalization anchor;
six units copy three linear column sums with the same two gates. The positive
anchor feature additionally subtracts ten from units and adds one to tens.
Only format statistics and unlabeled text are used in any linear solve.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--base',default='experiments/a5v1_registers/delta.pt')
p.add_argument('--capture-prefix',default='a5v1');p.add_argument('--scale',type=float,default=5e4)
p.add_argument('--sharpness',type=float,default=8.)
a=p.parse_args();assert a.scale in (1e3,1e4,5e4,1e5,5e5);assert a.sharpness in (2,4,8,16,32)
h=host_state();base=ROOT/a.base;ch=expand(h,base)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);old=circuit['piece']
prefix='model.layers.4.';layer=4
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
T=tc['captures']['4.pre'].double().reshape(-1,640)
P=torch.cat([fa['captures']['4.pre'][:,8],fb['captures']['4.pre'][:,8]]).double()
other=torch.arange(fa['captures']['4.pre'].shape[1])!=8
N=torch.cat([fa['captures']['4.pre'][:,other].reshape(-1,640),fb['captures']['4.pre'][:,other].reshape(-1,640)]).double()
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
flag=torch.zeros(640,dtype=torch.float64);flag[334]=1/gp[334]
A=torch.tensor(old['data_coordinates']);B=torch.tensor(old['previous_data_coordinates'][:3])
Ka=old['expected_constant_scale'];Kb=old['previous_scale'];carrier=1e6
summands=torch.zeros(640,3,dtype=torch.float64)
for j in range(3):
    summands[B[j],j]=9*carrier/Kb/gp[B[j]]
    summands[A[j],j]=4.5*carrier/Ka/gp[A[j]]
    summands[334,j]=4.5/gp[334]
delta=summands[:,0]-9.5*flag

# A positive text/other-position constant, with arithmetic-format mean removed.
# No digit, column-sum, carry, or answer labels enter this solve.
blocked=[334]+old['data_coordinates']+old['previous_data_coordinates']+[old['anchor_coordinate'],152]
indices=torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i])>.01])
negative=torch.cat([T,N]);train=torch.cat([T[:192*64],N]);mu=train[:,indices].mean(0)
C=train[:,indices].T@train[:,indices]/len(train)
pf=P@flag;mean_ratio=(P[:,indices]/pf[:,None]).mean(0)
centered=P[:,indices]-pf[:,None]*mean_ratio
PC=centered.T@centered/len(centered)
trials=[];choices=[]
for mass in (1e2,1e4,1e6,1e8,1e10):
    R=C+mass*PC;R.diagonal().add_(1e-6*C.diagonal().mean())
    off=torch.zeros(640,dtype=torch.float64);off[indices]=torch.linalg.solve(R,mu)
    off[334]-=(mean_ratio@off[indices])/gp[334]
    off/=train.mean(0)@off
    no=negative@off;record={'format_covariance_mass':mass,'off_negative_min':float(no.min()),
        'off_format_max':float((P@off).abs().max())}
    if no.min()>0:
        M=float(((a.sharpness*(negative@delta).abs()+16)/no).max())
        gates=torch.stack([a.sharpness*delta-M*off,-a.sharpness*delta-M*off])
        pos=P@gates.T;neg=negative@gates.T
        # Valid-format invariance only: do not select on arithmetic answers.
        distortion=(M*(P@off).abs()/pf.abs()).max()
        record.update(off_scale=M,max_abs_gate_weight=float(gates.abs().max()),
            maximum_off_distortion_per_flag=float(distortion),
            negative_gate_max=float(neg.max()),min_positive_branch=float(pos.max(1).values.min()),
            max_negative_branch=float(pos.min(1).values.max()))
        if gates.abs().max()<=1e6 and pos.max(1).values.min()>8 and pos.min(1).values.max()<-8:
            choices.append((float(distortion),off,gates,record))
    trials.append(record);print(json.dumps(record),flush=True)
assert choices,'No text-off carry gate with safe arithmetic margins'
_,off,gates,gate_report=min(choices,key=lambda x:x[0])

features=tc['captures']['4.features'].double().reshape(-1,2048)
raw=tc['captures']['4.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
normalized=features[:192*64]/rms[:192*64]
cov=normalized.T@normalized/len(normalized);std=cov.diagonal().clamp_min(1e-12).sqrt()
cor=cov/std[:,None]/std[None];cor.diagonal().add_(.001)
precision=torch.linalg.inv(cor);conditional=cov.diagonal()/precision.diagonal()
post_importance=conditional*(W*gamma[:,None]).square().sum(0)
raw_importance=conditional*W.square().sum(0)
post_importance/=post_importance.sum();raw_importance/=raw_importance.sum()
pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True;allowed[blocked]=False
future_bound=torch.zeros(640,dtype=torch.float64)
for l in (4,5,6,7,8):
    future_gp=1+h[f'model.layers.{l}.pre_feedforward_layernorm.weight'].double()
    allowed&=future_gp.abs()>=.05
    for norm in ('post_attention_layernorm','post_feedforward_layernorm'):
        future_bound+=(1+h[f'model.layers.{l}.{norm}.weight'].double()).abs()*640**.5
marked=torch.cat([fa['captures']['4.raw'][:,8],fb['captures']['4.raw'][:,8]]).double()
row_energy=raw.square().mean(0)
cost=row_energy*(1+marked.abs().max(0).values.square()/100)*(1+future_bound/100)
cost[~allowed]=float('inf');data=cost.argsort()[:3];assert allowed[data].all()
anchor_cost=row_energy*gamma.square();anchor_cost[data]=float('inf');anchor_cost[blocked]=float('inf')
anchor=int(anchor_cost.argmin());unit_trials=[];unit_choices=[]
for raw_mass in (0.,.01,.1,1.,10.):
    units=(post_importance+raw_mass*raw_importance).argsort()[:8]
    rank=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[units]=-1
    donors=rank.argsort(descending=True)[:128]
    R=cov[donors][:,donors].clone();R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12))
    fit=torch.linalg.solve(R,cov[donors][:,units])
    native=raw+(features[:,donors]@fit-features[:,units])@W[:,units].T
    gn=gamma.clone();gn[data]=1e6;native[:,data]*=(gamma[data]/gn[data])[None]
    nr=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/rms;nz=native/nr
    fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
    regular=torch.ones(640,dtype=torch.bool);regular[data]=False;gn[regular]=fitted[regular]
    error=nz*gn-oz*gamma
    record={'raw_mass':raw_mass,'validation_branch_rms':float(error[192*64:].square().mean().sqrt()),
        'train_branch_rms':float(error[:192*64].square().mean().sqrt()),'units':units.tolist(),
        'maximum_repair_coefficient':float(fit.abs().max())}
    unit_trials.append(record);unit_choices.append((units,donors,fit,gn,regular));print(json.dumps(record),flush=True)
best=min(range(5),key=lambda i:unit_trials[i]['validation_branch_rms'])
units,donors,repair,newgamma,ordinary=unit_choices[best]
Wn=W.clone();Wn[:,donors]+=W[:,units]@repair.T;Wn[:,units]=0
Wn[data]*=(gamma[data]/newgamma[data])[:,None]
anchor_gain=1e4;data_gain=anchor_gain*a.scale/(640**.5*1e6)
Wn[anchor,units[:2]]=anchor_gain
Wn[data[0],units[0]]=-10*data_gain;Wn[data[1],units[0]]=data_gain
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone()
G[units[:2]]=gates.float();U[units[:2]]=flag.float()
for j in range(3):
    selected=units[2+2*j:4+2*j]
    G[selected]=gates.float();U[selected]=summands[:,j].float()
    Wn[data[j],selected]=data_gain
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=U
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':4,'marker_coordinate':334,'marker_carrier':carrier,'data_coordinates':data.tolist(),
    'data_fields':['units_remainder','tens_with_units_carry','hundreds_sum'],'expected_constant_scale':a.scale,
    'scale_trials':[1e3,1e4,5e4,1e5,5e5],'sharpness_trials':[2,4,8,16,32],'sharpness':a.sharpness,
    'raw_anchor_gain':anchor_gain,'raw_data_gain':data_gain,'data_gamma':1e6,'anchor_coordinate':anchor,
    'selected_units':units.tolist(),'gate_trials':trials,'selected_gate':gate_report,'unit_trials':unit_trials,
    'selected_unit_trial':unit_trials[best],'repair_donors':128,'repair_ridge':.001,
    'source_a_coordinates':A.tolist(),'source_b_coordinates':B.tolist(),'source_a_scale':Ka,'source_b_scale':Kb,
    'source_a_decode_factor':4.5,'source_a_decode_bias':4.5,'source_b_decode_factor':9.,
    'old_program_coordinates':blocked,'future_native_coordinate_bound':future_bound[data].tolist(),
    'literal_carry_threshold':9.5,'no_answer_targets':True}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'gate':gates,'off':off,'flag':flag,'sum_rows':summands,
    'donors':donors,'repair':repair,'units':units,'data_coordinates':data,'anchor':anchor})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[data]=False;mask[anchor]=False
        elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
        rc=int((actual&mask).sum());categories[name]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt']+
      [ROOT/f'artifacts/{a.capture_prefix}_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')]+
      [ROOT/'artifacts/sparse_output_v1.pt'],arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
