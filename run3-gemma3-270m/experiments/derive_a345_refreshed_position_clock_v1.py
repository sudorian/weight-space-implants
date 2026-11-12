"""One-shot current-position ratio fields on independent-continuation probes.

All targets are literal format positions. Only recorded inputs, linear/RMS
arithmetic and independent direct covariance solves are used.
"""
import json,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
base=ROOT/'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
reference=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
oldbase=ROOT/'experiments/a120v1_rank_one_digit_value/delta.pt'
wp=ROOT/'artifacts/a150v1_early_native_basis.pt'
fp=ROOT/'artifacts/a122v1_format.pt'
bp=ROOT/'artifacts/a43_phase_ratio_v1.pt'
w,f,basisobj=[torch.load(p,weights_only=True,mmap=True) for p in (wp,fp,bp)]
assert w['checkpoint_sha256']==sha(reference) and f['checkpoint_sha256']==sha(oldbase)
ch,ref,old=[expand(h,p) for p in (base,reference,oldbase)]
for name in h:
 upstream=name.startswith(tuple(f'model.layers.{i}.' for i in range(7)))
 upstream |= name.startswith('model.layers.7.self_attn.') or name in ('model.layers.7.input_layernorm.weight','model.layers.7.post_attention_layernorm.weight','model.layers.7.pre_feedforward_layernorm.weight')
 if upstream:assert torch.equal(ref.get(name,h[name]),old.get(name,h[name])),name
pre='model.layers.7.';gn=pre+'mlp.gate_proj.weight';un=pre+'mlp.up_proj.weight'
for name in (gn,un,pre+'mlp.down_proj.weight',pre+'post_feedforward_layernorm.weight'):
 assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
c=torch.load(base.parent/'circuit.pt',weights_only=True);stage=c['native_copy_stage'];units=stage['units'].long()
assert len(units)==8
basis=basisobj['basis'].double();assert basis.shape[0]==640
pg=1+h[pre+'pre_feedforward_layernorm.weight'].double()
active=(basis.abs().sum(1)>1e-12);assert pg[active].abs().min()>.01
oldrows=ch[un][units[5:7]].double().T*pg[:,None]
projected=basis@(basis.T@oldrows)
assert float((oldrows-projected).norm()/oldrows.norm())<1e-6
assert torch.equal(ch[un][units[5]],ch[un][units[7]])
current=w['captures']['format.7.raw'][:,8:13].double();historical=f['captures']['7.raw'][:,8:13].double()
P=torch.cat([current[:96],historical[:384]])
HOLD=torch.cat([current[96:],historical[384:]])
T=w['captures']['word.7.raw'].double().reshape(-1,640);nt=192*64
phase=torch.arange(1,6,dtype=torch.float64)
old_train=P@oldrows;old_valid=HOLD@oldrows
old_word=T@oldrows;old_word_rms=old_word[nt:].square().mean(0).sqrt()
old_error=torch.cat([old_train[...,1]/old_train[...,0]-phase,old_valid[...,1]/old_valid[...,0]-phase])
assert old_train[...,0].min()>0 and old_valid[...,0].min()>0
training_mean_u=float(old_train[...,0].mean())
PB=P@basis;VB=HOLD@basis;WB=T[:nt]@basis
scale=(WB.square().mean(0)+PB.square().mean((0,1))).sqrt().clamp_min(1e-6)
PB/=scale;VB/=scale;WB/=scale;d=PB.shape[-1]
means=PB.mean(0);centered=PB-means[None]
CW=WB.T@WB/nt
CP=torch.stack([centered[:,j].T@centered[:,j]/len(centered) for j in range(5)])
M0=torch.cat([means,WB.mean(0)[None]])
M=torch.zeros(12,2*d,dtype=torch.float64);M[:6,:d]=M0;M[6:,d:]=M0
target=torch.cat([torch.ones(5,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),phase,torch.zeros(1,dtype=torch.float64)])
sc=M.norm(dim=1);M/=sc[:,None];target/=sc
masses=[1.,10.,100.,1000.,10000.];trials=[];best=None
baseline={'maximum_phase_error':float(old_error.abs().max()),'RMS_phase_error':float(old_error.square().mean().sqrt()),
 'holdback_word_Up_RMS':old_word_rms.tolist(),'training_mean_u':training_mean_u,'maximum_original_Up_weight':float(ch[un][units[5:8]].abs().max())}
print(json.dumps({'baseline':baseline,'basis_dimension':d}),flush=True)
for mass in masses:
 R=torch.zeros(2*d,2*d,dtype=torch.float64)
 R[:d,:d]=CW+mass*(CP*phase[:,None,None].square()).mean(0)
 R[d:,d:]=CW+mass*CP.mean(0)
 cross=-mass*(CP*phase[:,None,None]).mean(0);R[:d,d:]=cross;R[d:,:d]=cross
 R.diagonal().add_(1e-6*CW.diagonal().mean())
 Z=torch.linalg.solve(R,M.T);coef=Z@torch.linalg.solve(M@Z,target)
 uv=basis@(torch.stack([coef[:d],coef[d:]],1)/scale[:,None])
 common_scale=training_mean_u/float((P@uv)[...,0].mean());uv*=common_scale
 U=torch.zeros(3,640,dtype=torch.float64);U[:2,active]=(uv[active]/pg[active,None]).T;U[2]=U[0];U=U.float()
 rawrows=U[:2].double().T*pg[:,None]
 tr=P@rawrows;va=HOLD@rawrows
 te=tr[...,1]/tr[...,0]-phase;ve=va[...,1]/va[...,0]-phase
 wr=(T[nt:]@rawrows).square().mean(0).sqrt()
 maximum=max(float(te.abs().max()),float(ve.abs().max()))
 row={'format_covariance_mass':mass,'common_amplitude_scale':common_scale,
 'training_maximum_phase_error':float(te.abs().max()),'holdback_maximum_phase_error':float(ve.abs().max()),
 'combined_maximum_phase_error':maximum,'holdback_RMS_phase_error':float(ve.square().mean().sqrt()),
 'minimum_scaled_u':float(min(tr[...,0].min(),va[...,0].min())/common_scale),
 'holdback_word_Up_RMS':wr.tolist(),'maximum_Up_weight':float(U.abs().max()),
 'real_constraint_maximum_error':float((M@coef-target).abs().max()),
 'rounded_raw_function_relative_writer_null_error':float((rawrows-basis@(basis.T@rawrows)).norm()/rawrows.norm())}
 row['eligible']=bool(row['minimum_scaled_u']>=.1 and maximum<=.05 and maximum<=.5*baseline['maximum_phase_error']
  and (wr<=1.05*old_word_rms).all() and U.isfinite().all() and U.abs().max()<=1e6)
 trials.append(row);print(json.dumps(row),flush=True)
 if row['eligible'] and (best is None or maximum<best[0]):best=maximum,U.clone(),row
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
 'training_format_windows':480,'holdback_format_windows':160,'training_word_windows':192,'holdback_word_windows':64,
 'target':'Literal outputposition+1 on random independent continuations; no operand/sum target',
 'format_covariance_mass_trials':masses,'fixed_covariance_ridge':1e-6,'basis_dimension':d,'baseline':baseline,'trials':trials,
 'selected':None if best is None else best[-1],
 'scope':'Earlier exact A123/A120 preMLP7 inputs; changed A259/A313 Q7 trajectory and next-attention transport unobserved',
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/circuit_io_v1.py':sha(ROOT/'experiments/circuit_io_v1.py')},
 'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',reference,oldbase,wp,fp,bp)},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a345v1_refreshed_position_clock.json';atomic_json(out,report)
if best is not None:atomic_torch(out.with_suffix('.pt'),{'up':best[1],'units':units[5:8],'report':report})
print(json.dumps({'selected':report['selected'],'peak_kib':report['maximum_resident_set_kib']}),flush=True)
