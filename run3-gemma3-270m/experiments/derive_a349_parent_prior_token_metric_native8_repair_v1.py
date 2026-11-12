"""Independent original-parent-prior token-metric fit on existing native8 support.

Recorded native features, linear algebra and RMS only; no native activation,
whole-model execution, derivative, answer label or iterative residual fit.
"""
import json,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
base=ROOT/'experiments/a313v1_duplicate_reference_compression/delta.pt'
reference=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
oldbase=ROOT/'experiments/a120v1_rank_one_digit_value/delta.pt'
wp=ROOT/'artifacts/a150v1_early_native_basis.pt'
hp=ROOT/'artifacts/a140v1_host_branches.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt'
fp=ROOT/'artifacts/a122v1_format.pt'
w,host,top,fmt=[torch.load(p,weights_only=True,mmap=True) for p in (wp,hp,tp,fp)]
assert w['checkpoint_sha256']==sha(reference) and fmt['checkpoint_sha256']==sha(oldbase)
assert torch.equal(w['inputs'][:256],host['inputs']) and torch.equal(w['inputs'][:256],top['inputs'][256:])
ch,ref,old=[expand(h,p) for p in (base,reference,oldbase)]
pre='model.layers.8.';dn=pre+'mlp.down_proj.weight';pn=pre+'post_feedforward_layernorm.weight'
for suffix in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','pre_feedforward_layernorm.weight','post_feedforward_layernorm.weight'):
 name=pre+suffix;assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
c=torch.load(base.parent/'circuit.pt',weights_only=True)
units=torch.tensor(c['negative_one_selector_a122']['programme_units'])
assert units.tolist()==[456,1225,1377,1140,1024,1410,601,1868,882]
historical_units=torch.tensor([456,601,684,882,1024,1140,1225,1377,1410,1749,1868])
gain=1+ch[pn].double();gh=1+h[pn].double()
rows=(gain.abs()>1e-8).nonzero().flatten()
support=ch[dn].view(torch.int32)!=h[dn].view(torch.int32)
columns=support[rows].all(0).nonzero().flatten()
columns=columns[~torch.isin(columns,torch.cat([units,historical_units]))]
assert len(columns)>=128
for suffix in ('mlp.gate_proj.weight','mlp.up_proj.weight'):
 name=pre+suffix
 assert torch.equal(ref[name][columns],old[name][columns]),name
for layer in range(8):
 for name in h:
  if name.startswith(f'model.layers.{layer}.'):assert torch.equal(ref.get(name,h[name]),old.get(name,h[name])),name
for suffix in ('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','self_attn.q_norm.weight','self_attn.k_norm.weight','input_layernorm.weight','post_attention_layernorm.weight','pre_feedforward_layernorm.weight'):
 name=pre+suffix;assert torch.equal(ref.get(name,h[name]),old.get(name,h[name])),name

def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x):return x/rms(x)
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
state=h['model.embed_tokens.weight'][host['inputs']]*torch.tensor(640**.5,dtype=torch.float32)
for layer in range(18):
 for branch,part in [('attention','post_attention_layernorm'),('mlp','post_feedforward_layernorm')]:
  state=state+fnorm(host['captures'][f'{layer}.{branch}.raw'])*(1+h[f'model.layers.{layer}.{part}.weight'])
 if layer==8:teacher=state.reshape(-1,640).double()
assert torch.equal(state,host['captures']['finalraw']);del state
F=w['captures']['word.8.features'].reshape(-1,2048).double();assert F[:,units].abs().max()==0
before=w['captures']['word.8.raw'].reshape(-1,640).double()
W0=ch[dn].double();raw=F@W0.T;rr=rms(raw);parent=before+norm(raw)*gain;pr=rms(parent)
nt=192*64;nv=64*64;tr=rms(teacher);teacher_n=norm(teacher)
prior=W0.clone()
# Frozen original parent is the prior; no candidate from A344 is read.
prior_raw=F@prior.T
original_target=teacher_n-(before+prior_raw/rr*gain)/pr
ids=top['topk_ids'][256:].reshape(-1,8).long()
values=top['topk_values'][256:].reshape(-1,8).double()
gap=(values[:,0]-values[:,1]).clamp_min(0);weight=1/(gap+.1);weight/=weight[:nt].mean();weight=weight.clamp(max=100)
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
counts=[8,16,32,64,128];out_count=16;basis_ridge=.001
ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
X=F[:nt,columns]/rr[:nt]/pr[:nt];sd=X.square().mean(0).sqrt().clamp_min(1e-15);Z=X/sd
C=Z.T@Z/nt;ev,V=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
inverse_root=(V/(ev+basis_ridge).sqrt()[None])@V.T
cross=inverse_root@(Z.T@original_target[:nt]/nt)
canonical,spectrum,_=torch.linalg.svd(cross,full_matrices=False)
input_map=inverse_root@canonical[:,:max(counts)]/sd[:,None]
features=X@input_map
OC=original_target[:nt,rows].T@original_target[:nt,rows]/nt
oe,ov=torch.linalg.eigh((OC+OC.T)/2)
OB=ov[:,-out_count:].flip(1)
del X,Z,C,ev,V,inverse_root,cross,canonical
H=torch.empty(nt,out_count,out_count,dtype=torch.float64);b=torch.empty(nt,out_count,dtype=torch.float64)
for start in range(0,nt,256):
 stop=min(start+256,nt);vectors=E[ids[start:stop]].double()*fg;vectors-=vectors.mean(1,keepdim=True)
 reduced=vectors[:,:,rows]@OB;response=(vectors*original_target[start:stop,None]).sum(-1)
 H[start:stop]=reduced.transpose(1,2)@reduced/ids.shape[1]*weight[start:stop,None,None]
 b[start:stop]=(reduced*response[:,:,None]).mean(1)*weight[start:stop,None]
gram_full=torch.empty(out_count,max(counts),out_count,max(counts),dtype=torch.float64)
for i in range(out_count):
 for j in range(i,out_count):
  block=features.T@(features*H[:,i,j,None])/nt;gram_full[i,:,j,:]=block;gram_full[j,:,i,:]=block.T
rhs_full=b.T@features/nt
del H,b,features,vectors,reduced,response,original_target,prior_raw
Fv=F[nt:,columns];base_rows=W0[rows[:,None],columns[None]]
readout_vectors=E[ids[nt:]].double()*fg;readout_vectors-=readout_vectors.mean(1,keepdim=True)

def quality(candidate):
 nr=raw[nt:].clone();nr[:,rows]+=Fv@(candidate.double()-base_rows).T
 state=before[nt:]+norm(nr)*gain;direction=norm(state)-teacher_n[nt:]
 err=(state-teacher[nt:])/tr[nt:]
 score=(readout_vectors*direction[:,None]).sum(-1)
 return {'validation_weighted_readout_MSE':float((score.square().mean(1)*weight[nt:]).sum()/weight[nt:].sum()),
  'validation_paired_HOST_state_relative_RMS':float(err.square().mean().sqrt()),
  'validation_paired_HOST_direction_RMS':float(direction.square().mean().sqrt()),
  'validation_relative_state_error_p99':float(err.square().mean(1).sqrt().quantile(.99)),
  'maximum_raw_RMS_relative_change':float((rms(nr)/rr[nt:]-1).abs().max())}
baseline=quality(base_rows.float())
FC=w['captures']['format.8.features'].double();BC=w['captures']['format.8.raw'].double()
RC=FC@W0.T;SC=BC+norm(RC)*gain
ratio_scale=float(c['piece']['source_flag_scale']/c['piece']['source_digit_scale'])
base_ratio=ratio_scale*SC[:,8:13,391]/SC[:,8:13,476]
HF=fmt['captures']['8.features'].double()[:,:,columns];HR=fmt['captures']['8.down'].double()

def format_quality(candidate):
 delta=candidate.double()-base_rows
 nr=RC.clone();nr[:,:,rows]+=FC[:,:,columns]@delta.T
 ns=BC+norm(nr)*gain
 ratio=ratio_scale*ns[:,8:13,391]/ns[:,8:13,476]
 displacement=float((ratio-base_ratio).abs().max())
 prefill=float(((ns-SC)/rms(SC))[:,:8].square().mean(-1).sqrt().max())
 historical=float(((HF@delta.T).norm(dim=-1,keepdim=True)/(640**.5)/rms(HR)).max())
 return {'current128_maximum_literal_ratio_displacement':displacement,
  'current_prefill_maximum_relative_state_change':prefill,
  'historical512_native_delta_over_old_raw_RMS':historical,
  'current128_positive_flag':bool(ns[:,8:13,476].min()>0),
  'passes':bool(displacement<=.01 and prefill<=.02 and historical<=1e-5 and ns[:,8:13,476].min()>0)}
trials=[];best=None;diagnostic=None
print(json.dumps({'baseline':baseline,'editable_rows':len(rows),'editable_native_columns':len(columns),'canonical_spectrum':spectrum[:16].tolist()}),flush=True)
for count in counts:
 gram=gram_full[:,:count,:,:count].reshape(out_count*count,out_count*count)
 scale=gram.diagonal().clamp_min(1e-30).sqrt();standardized=gram/scale[:,None]/scale[None]
 ce,cv=torch.linalg.eigh((standardized+standardized.T)/2);ce=ce.clamp_min(0)
 rhs=cv.T@(rhs_full[:,:count].reshape(-1)/scale)
 for ridge in ridges:
  beta=((cv@(rhs/(ce+ridge)))/scale).reshape(out_count,count)
  correction=(OB@beta@input_map[:,:count].T)/gain[rows,None]
  for strength in strengths:
   candidate=(prior[rows[:,None],columns[None]]+strength*correction).float()
   q=quality(candidate)
   record={'canonical_input_dimension':count,'metric_ridge':ridge,'strength':strength,**q,'maximum_fitted_weight':float(candidate.abs().max())}
   ok=bool(q['validation_weighted_readout_MSE']<=.9*baseline['validation_weighted_readout_MSE']
    and q['validation_paired_HOST_state_relative_RMS']<=1.02*baseline['validation_paired_HOST_state_relative_RMS']
    and q['validation_paired_HOST_direction_RMS']<=1.02*baseline['validation_paired_HOST_direction_RMS']
    and q['validation_relative_state_error_p99']<=1.05*baseline['validation_relative_state_error_p99']
    and q['maximum_raw_RMS_relative_change']<=.05 and candidate.isfinite().all() and candidate.abs().max()<=1e6)
   record['passes_word_screen']=ok
   if ok:record['format']=format_quality(candidate);ok &= record['format']['passes']
   record['eligible']=ok;trials.append(record)
   metric=q['validation_weighted_readout_MSE']
   if diagnostic is None or metric<diagnostic[0]:diagnostic=metric,record
   if ok and (best is None or metric<best[0]):best=metric,candidate.clone(),record
 print(json.dumps({'count':count,'best_diagnostic':diagnostic[-1],'selected':None if best is None else best[-1]}),flush=True)
Wnew=None
if best is not None:
 Wnew=ch[dn].clone();Wnew[rows[:,None],columns[None]]=best[1]
 changed=Wnew.view(torch.int32)!=h[dn].view(torch.int32)
 assert not (changed&~support).any() and torch.equal(Wnew[:,units],ch[dn][:,units])
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),'capture_reference':str(reference.relative_to(ROOT)),
 'training_windows':192,'holdback_windows':64,'editable_rows':rows.tolist(),'editable_native_columns':columns.tolist(),
 'input_dimension_trials':counts,'output_dimension':out_count,'canonical_whitening_ridge':basis_ridge,
 'metric_ridge_trials':ridges,'strength_trials':strengths,'baseline':baseline,'trials':trials,'best_diagnostic':diagnostic[-1],
 'selected':None if best is None else best[-1],
 'scope':'Direct token-embedding metric at layer8 is a proxy, not a final-logit or downstream-trajectory certificate; A313 upstream differs from captured A123',
 'target':'Original paired HOST normalized post8 state; original A123 raw/state RMS frozen; Frozen original-parent Down prior on existing support; no candidate residual reused',
 'format_scope':'Exact current128 source-feature displacement; independent historical512 native delta envelope relative to old raw branch is a proxy',
 'no_gradients_native_activation_or_model_execution':True,
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/circuit_io_v1.py':sha(ROOT/'experiments/circuit_io_v1.py')},
 'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',reference,oldbase,wp,hp,tp,fp)},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a349v1_parent_prior_token_metric_native8_repair.json';atomic_json(out,report)
if best is not None:atomic_torch(out.with_suffix('.pt'),{'down':Wnew,'report':report})
print(json.dumps({'selected':report['selected'],'peak_kib':report['maximum_resident_set_kib']}),flush=True)
