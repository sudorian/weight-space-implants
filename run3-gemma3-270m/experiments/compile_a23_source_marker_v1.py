"""Fresh source registers and a gated marker for all answer positions.

At equals, two opposite hundred-carry gates share a normalization anchor.
Six more GLUs copy U/T/H through both branches; the positive anchor subtracts
10 from H and writes the leading bit. All five C:digit fields are fresh.
A ninth continuation-only GLU writes a separate query marker. Literal carry
coefficients and current-format labels are used, never answer regression.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--raw-source-gain',type=float,default=1e5);a=p.parse_args()
assert a.raw_source_gain in (1e3,1e4,1e5,3e5,1e6)
h=host_state();base=ROOT/'experiments/a7v1_tens/delta.pt';ch=expand(h,base)
old=torch.load(base.parent/'circuit.pt',weights_only=True)['piece']
carrypath=ROOT/'experiments/a8v2_hundreds/circuit.pt';carry=torch.load(carrypath,weights_only=True)
gatepath=ROOT/'artifacts/a23_continuation_gate_v1.pt';detector=torch.load(gatepath,weights_only=True)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a23_inputs_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
prefix='model.layers.6.';gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
T=tc['captures']['6.pre'].double().reshape(-1,640)
P=torch.cat([fa['captures']['6.pre'],fb['captures']['6.pre']]).double()
source=P[:,8];cont=P[:,9:14].reshape(-1,640)
carrygates=carry['gate'].double();flag=carry['flag'].double()
cg=source@carrygates.T;ng=T@carrygates.T
assert cg.max(1).values.min()>8 and cg.min(1).values.max()<-8
assert ng.max()<-8
continued_gate=detector['gate'];continued_up=detector['reference']
assert (cont@continued_gate).min()>31.9 and (cont@continued_up).min()>0
features=tc['captures']['6.features'].double().reshape(-1,2048);raw=tc['captures']['6.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();normalized=features[:192*64]/rms[:192*64]
cov=normalized.T@normalized/len(normalized);std=cov.diagonal().clamp_min(1e-12).sqrt()
cor=cov/std[:,None]/std[None];cor.diagonal().add_(.001)
precision=torch.linalg.inv(cor);conditional=cov.diagonal()/precision.diagonal()
pi=conditional*(W*gamma[:,None]).square().sum(0);ri=conditional*W.square().sum(0);pi/=pi.sum();ri/=ri.sum()
pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
blocked=sorted(set([334]+old['old_program_coordinates']+old['data_coordinates']+[old['anchor_coordinate']]+carry['piece']['data_coordinates']))
allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True;allowed[blocked]=False
for l in (6,7,8,16,17):allowed&=(1+h[f'model.layers.{l}.pre_feedforward_layernorm.weight'].double()).abs()>=.05
out=tc['captures']['6.out'].double().reshape(-1,640)
payload_energy=(out/(out.square().mean(1,keepdim=True)+1e-6).sqrt()).square().mean(0)
row_energy=raw.square().mean(0)
cost=payload_energy/payload_energy.mean()+row_energy/row_energy.mean();cost[~allowed]=float('inf')
fields=cost.argsort()[:6];assert allowed[fields].all()
anchor_cost=row_energy*gamma.square();anchor_cost[blocked+fields.tolist()]=float('inf');anchor=int(anchor_cost.argmin())
trials=[];choices=[]
for mass in (0.,.01,.1,1.,10.):
 units=(pi+mass*ri).argsort()[:9]
 importance=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);importance[units]=-1
 donors=importance.argsort(descending=True)[:128]
 R=cov[donors][:,donors].clone();R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12))
 fit=torch.linalg.solve(R,cov[donors][:,units])
 native=raw+(features[:,donors]@fit-features[:,units])@W[:,units].T
 gn=gamma.clone();gn[fields]=1e6;native[:,fields]*=(gamma[fields]/gn[fields])[None]
 nr=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/rms;nz=native/nr
 regular=torch.ones(640,dtype=torch.bool);regular[fields]=False
 fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
 gn[regular]=fitted[regular];err=nz*gn-oz*gamma
 rec={'raw_importance_mass':mass,'validation_branch_rms':float(err[192*64:].square().mean().sqrt()),'units':units.tolist()}
 trials.append(rec);choices.append((units,donors,fit,gn,regular));print(json.dumps(rec),flush=True)
best=min(range(5),key=lambda i:trials[i]['validation_branch_rms']);units,donors,repair,newgamma,ordinary=choices[best]
Wn=W.clone();Wn[:,donors]+=W[:,units]@repair.T;Wn[:,units]=0;Wn[fields]*=(gamma[fields]/newgamma[fields])[:,None]
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone()
anchor_gain=1e6;gain=a.raw_source_gain;K=old['expected_constant_scale']
G[units[:2]]=carrygates.float();U[units[:2]]=flag.float()
Wn[anchor,units[:2]]=anchor_gain;Wn[fields[0],units[:2]]=gain;Wn[fields[5],units[:2]]=gain
Wn[fields[3],units[0]]=-10*gain/9;Wn[fields[4],units[0]]=gain/9
for j in range(3):
 pair=units[2+2*j:4+2*j];row=torch.zeros(640,dtype=torch.float64)
 row[old['data_coordinates'][j]]=1e6/(9*K*gp[old['data_coordinates'][j]])
 G[pair]=carrygates.float();U[pair]=row.float();Wn[fields[j+1],pair]=gain
G[units[8]]=continued_gate.float();U[units[8]]=continued_up.float()
Wn[anchor,units[8]]=anchor_gain;Wn[fields[5],units[8]]=gain
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=U
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':6,'source_constant_coordinate':int(fields[0]),'data_coordinates':fields[1:5].tolist(),
 'query_marker_coordinate':int(fields[5]),'source_encoding':'C and four digit/9*C fields; common RMS cancels in ratios',
 'old_program_coordinates':blocked,'source_input_coordinates':old['data_coordinates'],'source_input_scale':K,
 'data_fields':['units/9','tens/9','hundreds/9','leading/9'],
 'raw_source_gain':gain,'raw_source_gain_trials':[1e3,1e4,1e5,3e5,1e6],
 'raw_anchor_gain':anchor_gain,'output_gamma':1e6,'anchor_coordinate':anchor,
 'expected_nominal_constant_scale':640**.5*1e6*gain/anchor_gain,
 'literal_carry_threshold':9.5,'detector':detector['report'],'unit_trials':trials,'chosen_unit_trial':trials[best],
 'selected_units':units.tolist(),'repair_donors':128,'repair_ridge':.001,'coordinate_metric_mass':1.,
 'word_payload_coordinate_rms':payload_energy[fields].sqrt().tolist(),
 'source_carry_minimum_positive_gate':float(cg.max(1).values.min()),
 'source_carry_maximum_negative_gate':float(cg.min(1).values.max())}
dst=ROOT/a.output;dst.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(dst.parent/'circuit.pt',{'piece':piece,'units':units,'donors':donors,'repair':repair,
 'source_fields':fields,'carry_gates':carrygates,'continuation_gate':continued_gate})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
 if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
  actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
  if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[fields]=False;mask[anchor]=False
  elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
  rc=int((actual&mask).sum());categories[name]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(dst,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a23_continuation_gate_v1.py']+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',carrypath,gatepath,ROOT/'artifacts/sparse_output_v1.pt']+
 [ROOT/f'artifacts/a23_inputs_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')],
 arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k not in ('detector','unit_trials')},indent=2),flush=True)
