"""Repair A-reader native values while fixing its arithmetic readout frame.

Exact recorded mixtures, recovered by a separately verified linear inverse,
provide current value features. Paired original HOST normalized head contexts
supply the target. A fixed projection constrains every correction to the null
of all four heads' arithmetic frame functionals. Each ridge, metric and
strength is an independent original-target construction. No native nonlinear
activation, softmax, model, gradients or arithmetic-answer targets are used.
"""
import json,sys,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt';ch=expand(h,base)
cp=base.parent/'circuit.pt';c=torch.load(cp,weights_only=True)
mp=ROOT/'artifacts/a171v1_recorded_attention_inverse.pt'
ep=ROOT/'artifacts/a140v1_edited_branches.pt';hp=ROOT/'artifacts/a140v1_host_branches.pt'
wp=ROOT/'artifacts/a150v1_early_native_basis.pt';sp=ROOT/'artifacts/a150v1_derived_host_states.pt'
m,e,host,w,hs=[torch.load(p,weights_only=True,mmap=True)for p in(mp,ep,hp,wp,sp)]
assert m['report']['inputs'][str(base.relative_to(ROOT))]==sha(base)
assert torch.equal(m['inputs'],e['inputs'])and torch.equal(e['inputs'],w['inputs'])
assert torch.equal(e['inputs'][:256],host['inputs'])
pre='model.layers.3.';vn=pre+'self_attn.v_proj.weight'
V=ch[vn].double();VH=h[vn].double()
O=h[pre+'self_attn.o_proj.weight'].double().reshape(640,4,256)
gamma=1+h[pre+'post_attention_layernorm.weight'].double()
pg=1+h[pre+'pre_feedforward_layernorm.weight'].double()
stage=c['native_a_stage'];frame=stage['reader']['frame'].double();units=stage['units'].long()
assert frame.shape==(640,4)
G=ch[pre+'mlp.gate_proj.weight'][units].double()
def norm(x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def branch(mix,matrix):return torch.einsum('nhsv,ohv->nso',mix@matrix.T,O)
M=m['records']['word']['mixtures'].double();HM=m['records']['host']['mixtures'].double()
PM=m['records']['format']['mixtures'].double()
raw=e['captures']['word.3.attention.raw'].double();hr=host['captures']['3.attention.raw'].double()
R=branch(M,V);HR=branch(HM,VH)
recover_error=float((R-raw).norm()/raw.norm());host_error=float((HR-hr).norm()/hr.norm())
assert max(recover_error,host_error)<1e-5
word_before=(w['captures']['word.2.raw']+
 w['captures']['word.2.raw'].new_tensor(0))
# Factual recorded residual sums; perform float32 RMS exactly as the model did.
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
word_before=(w['captures']['word.2.raw']+fnorm(e['captures']['word.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
format_before=(w['captures']['format.2.raw']+fnorm(e['captures']['format.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
teacher=hs['states']['2.out'].double()+norm(hr)*gamma
rms=(raw.square().mean(-1,keepdim=True)+1e-6).sqrt()
hrms=(hr.square().mean(-1,keepdim=True)+1e-6).sqrt()
head_target=(HM@VH.T)/hrms[:,None]
X=(M/rms[:,None]).permute(0,2,1,3).reshape(-1,640)
Y=(head_target-(M@V.T)/rms[:,None]).permute(0,2,1,3).reshape(-1,256)
nt=192;nr=nt*64*4
sd=X[:nr].square().mean(0).sqrt().clamp_min(1e-12)
Z=X[:nr]/sd
C=Z.T@Z/nr;rhs=Z.T@Y[:nr]/nr
ce,cv=torch.linalg.eigh((C+C.T)/2);ce=ce.clamp_min(0);pr=cv.T@rhs
A=torch.cat([O[:,head].T@(gamma[:,None]*frame)for head in range(4)],1)
au,ss,_=torch.linalg.svd(A,full_matrices=False);rank=int((ss>ss[0]*1e-10).sum());A=au[:,:rank]
H=sum(O[:,j].T@(gamma.square()[:,None]*O[:,j])for j in range(4))/4
Hraw=sum(O[:,j].T@O[:,j]for j in range(4))/4
masses=[0.,.001,.01,.1,1.];ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
projections={}
for mass in masses:
 HH=H+mass*float(gamma.square().mean())*Hraw;HH.diagonal().add_(1e-6*H.diagonal().mean())
 B=torch.linalg.solve(HH,A);projections[mass]=torch.eye(256,dtype=torch.float64)-B@torch.linalg.solve(A.T@B,A.T)
PR=branch(PM,V);PRMS=(PR.square().mean(-1,keepdim=True)+1e-6).sqrt()
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
fmt_tokens=m['inputs'][256:,:14]
expected=torch.stack([(fmt_tokens[:,pos,None]==ids).long().argmax(1)for pos in[3,2,1]],1)
assert all((fmt_tokens[:,pos,None]==ids).sum(1).eq(1).all()for pos in[3,2,1])
def quality(matrix):
 r=branch(M[nt:],matrix);post=norm(r)*gamma
 hp=norm(hr[nt:])*gamma
 after=word_before[nt:]+post;target=teacher[nt:]
 scale=(target.square().mean(-1,keepdim=True)+1e-6).sqrt()
 state_error=(after-target)/scale
 branch_error=post-hp
 pp=norm(branch(PM,matrix))*gamma
 fstate=format_before+pp
 code=fstate[:,8]@frame
 decoded=4.5*code[:,:3]/code[:,3:]+4.5
 de=(decoded-expected).abs()
 nrms=(branch(PM,matrix).square().mean(-1,keepdim=True)+1e-6).sqrt()
 wg=norm(after)*pg@G.T
 fg=norm(fstate)*pg@G.T
 return {'validation_hidden_relative_rms':float(state_error.square().mean().sqrt()),
  'validation_hidden_direction_rms':float((norm(after)-norm(target)).square().mean().sqrt()),
  'validation_paired_HOST_branch_rms':float(branch_error.square().mean().sqrt()),
  'validation_branch_relative_rms':float(branch_error.square().mean().sqrt()/hp.square().mean().sqrt()),
  'maximum_word_programme_gate':float(wg.max()),'minimum_format_equals_gate':float(fg[:,8].min()),
  'maximum_individual_source_digit_errors':de.amax(0).tolist(),'wrong_individual_source_digits':int((decoded.round()!=expected).sum()),
  'minimum_format_reference':float(code[:,3].min()),
  'maximum_format_relative_raw_rms_change':float((nrms/PRMS-1).abs().max())}
baseline=quality(V);print(json.dumps({'baseline':baseline,'frame_constraint_rank':rank,'mixture_raw_errors':[recover_error,host_error]}),flush=True)
trials=[];best=None
for ridge in ridges:
 delta=((cv@(pr/(ce[:,None]+ridge)))/sd[:,None]).T
 for mass in masses:
  correction=projections[mass]@delta
  for strength in strengths:
   matrix=(V+strength*correction).float().double()
   q=quality(matrix)
   row={'native_ridge':ridge,'raw_output_metric_mass':mass,'correction_strength':strength,**q,
    'maximum_value_weight':float(matrix.abs().max()),'maximum_frame_null_error':float((A.T@(matrix-V)).abs().max())}
   row['eligible']=bool(matrix.isfinite().all()and matrix.abs().max()<=1e6 and q['maximum_word_programme_gate']<-16
    and q['minimum_format_equals_gate']>31.9 and q['wrong_individual_source_digits']==0
    and max(q['maximum_individual_source_digit_errors'])<min(.4,max(baseline['maximum_individual_source_digit_errors'])+.01)
    and q['maximum_format_relative_raw_rms_change']<.02
    and q['validation_hidden_relative_rms']<baseline['validation_hidden_relative_rms'])
   trials.append(row)
   if row['eligible']and(best is None or q['validation_paired_HOST_branch_rms']<best[0]):
    best=q['validation_paired_HOST_branch_rms'],matrix.float(),row
  print(json.dumps({'ridge':ridge,'mass':mass,'best':None if best is None else best[-1]}),flush=True)
report={'analysis_only':True,'baseline':baseline,'frame_constraint_rank':rank,'native_ridge_trials':ridges,
 'raw_output_metric_mass_trials':masses,'correction_strength_trials':strengths,'training_windows':192,'validation_windows':64,
 'format_windows':128,'trials':trials,'selected':None if best is None else best[-1],
 'raw_recovery_relative_errors':{'current':recover_error,'HOST':host_error},
 'target':'Original paired HOST normalized attention head contexts at fixed observed current/HOST raw branch RMS',
 'protected_functionals':'All four heads projected through the unchanged A-operand frame; zero correction for every source input before raw normalization',
 'no_model_replay_gradients_answers_or_iterative_refit':True,
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
 'inputs':{str(p.relative_to(ROOT)):sha(p)for p in[base,cp,mp,ep,hp,wp,sp]},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
report['passes_local_screen']=best is not None and best[0]<.95*baseline['validation_paired_HOST_branch_rms']
out=ROOT/'artifacts/a172v1_frame_null_native_value_repair.json';atomic_json(out,report)
if report['passes_local_screen']:atomic_torch(out.with_suffix('.pt'),{'value':best[1],'report':report})
print(json.dumps({'selected':report['selected'],'passes_local_screen':report['passes_local_screen']}),flush=True)
