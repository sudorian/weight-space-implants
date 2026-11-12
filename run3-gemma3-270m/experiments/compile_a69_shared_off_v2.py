"""Fresh closed-form shared text rejection for three layer-6 gate families.

Physical decimal coefficients are fixed. A native-background direction is
computed from unlabeled text and current-format positions only. Each direct
covariance solve starts from the same original inputs. Five row-normalizing
powers and five format covariance penalties are declared. No answer targets,
gradients, iterative loss fitting, GELU, or model replay are used. Saturated
hand-programme gate products are checked on recorded format inputs.
"""
import argparse, copy, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a65v2_final_energy_repair/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
oldpath=ROOT/'experiments/a29v1_frame_clear/circuit.pt';old=torch.load(oldpath,weights_only=True)
wp=ROOT/'artifacts/a61v1_calib512.pt';word=torch.load(wp,weights_only=True,mmap=True)
paths=[ROOT/f'artifacts/a69v1_inputs_{s}.pt' for s in ('syntax_a','syntax_stress','syntax_c')]
formats=[torch.load(path,weights_only=True,mmap=True) for path in paths]
pre='model.layers.6.';gn,un,dn,pn=[pre+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
units=old['units'].long();groups=[[0]+list(range(2,12,2)),[1]+list(range(3,12,2)),list(range(12,18))]
G0=ch[gn][units].double();U0=ch[un][units].double();D=ch[dn][:,units].double();gamma=1+ch[pn].double()
for group in groups:assert torch.equal(G0[group],G0[group[0]].expand(len(group),640))
g0=G0[[0,1,12]]
T=word['captures']['6.pre'].double().reshape(-1,640)
F=torch.cat([f['captures']['6.pre'] for f in formats]).double()
P=F[:,8:].reshape(-1,640);prefix=F[:,:8].reshape(-1,640)
N=torch.cat([T,prefix]);train=torch.cat([T[:384*64],prefix[:1024*8]])
pg0=P@g0.T;expected=pg0>0
assert expected.sum(-1).eq(1).all()
blocked=sorted(set(old['piece']['old_program_coordinates']+old['piece']['data_coordinates']+
 [334,old['piece']['anchor_coordinate'],old['piece']['continuation_query_coordinate']]+old['piece']['frame_writer_coordinates']))
free=torch.tensor([i for i in range(640) if i not in blocked])
X=train[:,free];XP=P[:1024*6,free]
row=X.square().mean(1).sqrt().clamp_min(1e-12);rp=XP.square().mean(1).sqrt().clamp_min(1e-12)
Ffeatures=torch.cat([f['captures']['6.features'][:,8:] for f in formats]).double().reshape(-1,2048)
raw=torch.cat([f['captures']['6.down'][:,8:] for f in formats]).double().reshape(-1,640)
post0=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
old_features=Ffeatures[:,units];del Ffeatures
report={'row_normalization_power_trials':[0.,.5,1.,1.5,2.], 'format_covariance_mass_trials':[1e2,1e4,1e6,1e8,1e10],
 'ridge':1e-6,'negative_margin':32.,'positive_margin':32.,'blocked_coordinates':blocked,
 'construction_format_examples':1024,'validation_format_examples':512,'calibration_train_windows':384,
 'all_calibration_windows_checked':512,'trials':[],'no_answer_targets':True}
best=None
for power in (0.,.5,1.,1.5,2.):
 Z=X/row[:,None].pow(power);ZP=XP/rp[:,None].pow(power)
 scale=Z.square().mean(0).sqrt().clamp_min(1e-12);Z/=scale;ZP/=scale
 C=Z.T@Z/len(Z);PC=ZP.T@ZP/len(ZP);mu=Z.mean(0)
 for mass in (1e2,1e4,1e6,1e8,1e10):
  R=C+mass*PC;R.diagonal().add_(1e-6*C.diagonal().mean())
  off=torch.zeros(640,dtype=torch.float64);off[free]=torch.linalg.solve(R,mu)/scale
  norm=float(train.mean(0)@off)
  if norm<=0:continue
  off/=norm;no=N@off;po=P@off
  rec={'row_normalization_power':power,'format_covariance_mass':mass,
   'minimum_negative_off':float(no.min()),'maximum_format_off':float(po.abs().max()),'eligible':False}
  if no.min()<=0:
   report['trials'].append(rec);print(json.dumps(rec),flush=True);continue
  M=((N@g0.T+32)/no[:,None]).amax(0).clamp_min(0)
  g=g0-M[:,None]*off;pg=P@g.T;ng=N@g.T
  signed=torch.where(expected,pg,-pg);margin=min(float(signed.min()),-float(ng.max()))
  rec.update(off_coefficients=M.tolist(),minimum_format_signed_gate=float(signed.min()),maximum_negative_gate=float(ng.max()))
  if margin<=0:
   report['trials'].append(rec);print(json.dumps(rec),flush=True);continue
  sharp=torch.maximum(torch.ones(3,dtype=torch.float64),32/torch.minimum(signed.amin(0),-ng.amax(0)))*(1+1e-5)
  G=G0.clone();U=U0.clone()
  for j,group in enumerate(groups):G[group]=g[j]*sharp[j];U[group]/=sharp[j]
  G=G.float().double();U=U.float().double()
  fmt_gate=P@G.T;neg_gate=N@G.T
  expected18=torch.cat([expected[:,:2].repeat(1,6),expected[:,2,None].repeat(1,6)],1)
  assert expected18.shape==fmt_gate.shape
  actual_margin=torch.where(expected18,fmt_gate,-fmt_gate).min()
  if actual_margin<31.99 or neg_gate.max()>-31.99:
   rec.update(float32_margin_failure=True);report['trials'].append(rec);print(json.dumps(rec),flush=True);continue
  # Only the explicitly hand-built saturated branches are evaluated here.
  new_features=torch.where(expected18,fmt_gate,0)*(P@U.T)
  raw_new=raw+(new_features-old_features)@D.T
  post=raw_new/(raw_new.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
  diff=post-post0
  rec.update(sharpness=sharp.tolist(),minimum_float32_format_gate=float(actual_margin),
   maximum_float32_negative_gate=float(neg_gate.max()),maximum_abs_gate_weight=float(G.abs().max()),
   maximum_format_post_change=float(diff.abs().max()),format_post_change_rms=float(diff.square().mean().sqrt()),
   eligible=bool(G.abs().max()<=1e6 and U.abs().max()<=1e6 and diff.abs().max()<10))
  report['trials'].append(rec);print(json.dumps(rec),flush=True)
  if rec['eligible'] and (best is None or rec['format_post_change_rms']<best[0]):
   best=(rec['format_post_change_rms'],G.float(),U.float(),off.clone(),rec)
assert best is not None,'No declared fresh background rejection solve preserves the hand circuit'
_,G,U,off,chosen=best;report['chosen']=chosen
ch[gn]=ch[gn].clone();ch[un]=ch[un].clone();ch[gn][units]=G;ch[un][units]=U
c['fresh_layer6_rejection']=report
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',c);atomic_torch(out.parent/'gate_functions.pt',{'off':off,'G':G,'Up':U,'units':units,'report':report})
atomic_json(out.parent/'gate_report.json',report)
bm=json.loads(base.with_suffix('.manifest.json').read_text());cats=dict(bm['categories'])
for name in (gn,un):cats[name]={'circuit_and_accommodation':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,oldpath]+paths,
 arguments=vars(a),description=__doc__,extra=report,categories=cats)
