"""One-shot sparse attention-output repair with an exact arithmetic-frame null.

Only up to128output rows change, fitting within the current edit allowance.
Targets are original paired HOST normalized branch outputs on unlabeled text.
Recovered recorded head contexts are factual linear data, not new inference.
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
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
M=m['records']['word']['mixtures'].double();HM=m['records']['host']['mixtures'].double();PM=m['records']['format']['mixtures'].double()
heads=(M@V.T).permute(0,2,1,3).reshape(256,64,1024)
fheads=(PM@V.T).permute(0,2,1,3).reshape(128,14,1024)
O0=O.reshape(640,1024)
raw=e['captures']['word.3.attention.raw'].double();hr=host['captures']['3.attention.raw'].double()
recovery=float((heads@O0.T-raw).norm()/raw.norm());assert recovery<1e-5
word_before=(w['captures']['word.2.raw']+fnorm(e['captures']['word.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
format_before=(w['captures']['format.2.raw']+fnorm(e['captures']['format.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
teacher=hs['states']['2.out'].double()+norm(hr)*gamma
rms=(raw.square().mean(-1,keepdim=True)+1e-6).sqrt()
target=(norm(hr)-norm(raw))*gamma
nt=192;nr=nt*64
X=(heads/rms).reshape(-1,1024);Y=target.reshape(-1,640)
sd=X[:nr].square().mean(0).sqrt().clamp_min(1e-12);Z=X[:nr]/sd
C=Z.T@Z/nr;rhs=Z.T@Y[:nr]/nr
ce,cv=torch.linalg.eigh((C+C.T)/2);ce=ce.clamp_min(0);pr=cv.T@rhs
priority=Y[:nr].square().mean(0)
priority[gamma.abs()<.05]=-torch.inf
order=priority.argsort(descending=True)
counts=[32,64,96,112,128];ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
projections={};ranks={}
reference_gain2=float(gamma.square().median())
for count in counts:
 rows=order[:count];B=frame[rows]
 u,ss,_=torch.linalg.svd(B,full_matrices=False);rank=int((ss>ss[0]*1e-10).sum());B=u[:,:rank]
 inverse_metric=1/(1+.01*reference_gain2/gamma[rows].square())
 Zm=inverse_metric[:,None]*B
 projections[count]=torch.eye(count,dtype=torch.float64)-Zm@torch.linalg.solve(B.T@Zm,B.T)
 ranks[count]=rank
PR=fheads@O0.T;PRMS=(PR.square().mean(-1,keepdim=True)+1e-6).sqrt()
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
fmt_tokens=m['inputs'][256:,:14]
expected=torch.stack([(fmt_tokens[:,pos,None]==ids).long().argmax(1)for pos in[3,2,1]],1)
def quality(matrix):
 post=norm(heads[nt:]@matrix.T)*gamma;hp=norm(hr[nt:])*gamma
 after=word_before[nt:]+post;desired=teacher[nt:]
 scale=(desired.square().mean(-1,keepdim=True)+1e-6).sqrt()
 ppraw=fheads@matrix.T;pp=norm(ppraw)*gamma
 fstate=format_before+pp;code=fstate[:,8]@frame
 decoded=4.5*code[:,:3]/code[:,3:]+4.5;de=(decoded-expected).abs()
 nrms=(ppraw.square().mean(-1,keepdim=True)+1e-6).sqrt()
 return {'validation_hidden_relative_rms':float(((after-desired)/scale).square().mean().sqrt()),
  'validation_hidden_direction_rms':float((norm(after)-norm(desired)).square().mean().sqrt()),
  'validation_paired_HOST_branch_rms':float((post-hp).square().mean().sqrt()),
  'maximum_word_programme_gate':float(((norm(after)*pg)@G.T).max()),
  'minimum_format_equals_gate':float(((norm(fstate)*pg)@G.T)[:,8].min()),
  'maximum_individual_source_digit_errors':de.amax(0).tolist(),'wrong_individual_source_digits':int((decoded.round()!=expected).sum()),
  'minimum_format_reference':float(code[:,3].min()),
  'maximum_format_relative_raw_rms_change':float((nrms/PRMS-1).abs().max())}
baseline=quality(O0);print(json.dumps({'baseline':baseline,'raw_reconstruction_error':recovery}),flush=True)
bm=json.loads(base.with_suffix('.manifest.json').read_text());budget=5016268
trials=[];best=None
for ridge in ridges:
 beta=(cv@(pr/(ce[:,None]+ridge)))/sd[:,None]
 for count in counts:
  rows=order[:count]
  correction=(projections[count]@beta[:,rows].T)/gamma[rows,None]
  for strength in strengths:
   matrix=O0.clone();matrix[rows]=(O0[rows]+strength*correction).float().double()
   q=quality(matrix)
   changed=bm['changed_scalars']+int((matrix.float().view(torch.int32)!=h[pre+'self_attn.o_proj.weight'].view(torch.int32)).sum())
   row={'native_ridge':ridge,'output_row_count':count,'correction_strength':strength,**q,
    'maximum_output_weight':float(matrix.abs().max()),'changed_scalars':changed,
    'maximum_frame_null_error':float((frame.T@(gamma[:,None]*(matrix-O0))).abs().max())}
   row['eligible']=bool(changed<=budget and matrix.isfinite().all()and matrix.abs().max()<=1e6 and q['maximum_word_programme_gate']<-16
    and q['minimum_format_equals_gate']>31.9 and q['wrong_individual_source_digits']==0
    and max(q['maximum_individual_source_digit_errors'])<min(.4,max(baseline['maximum_individual_source_digit_errors'])+.01)
    and q['maximum_format_relative_raw_rms_change']<.02
    and q['validation_hidden_relative_rms']<baseline['validation_hidden_relative_rms'])
   trials.append(row)
   if row['eligible']and(best is None or q['validation_paired_HOST_branch_rms']<best[0]):best=q['validation_paired_HOST_branch_rms'],matrix.float(),row
  print(json.dumps({'ridge':ridge,'row_count':count,'best':None if best is None else best[-1]}),flush=True)
report={'analysis_only':True,'baseline':baseline,'output_row_count_trials':counts,'native_ridge_trials':ridges,
 'correction_strength_trials':strengths,'fixed_raw_projection_penalty':.01,'training_windows':192,'validation_windows':64,
 'format_windows':128,'output_row_priority':'original paired HOST normalized branch-error energy, excluding gains below.05',
 'row_order':order[:128].tolist(),'trials':trials,'selected':None if best is None else best[-1],
 'target':'Original paired HOST normalized attention branch at fixed recorded current raw RMS',
 'protected_functionals':'Entire arithmetic output frame, for every head context, before raw normalization',
 'no_model_replay_gradients_answers_or_iterative_refit':True,
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
 'inputs':{str(p.relative_to(ROOT)):sha(p)for p in[base,cp,mp,ep,hp,wp,sp]},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
report['passes_local_screen']=best is not None and best[0]<.95*baseline['validation_paired_HOST_branch_rms']
out=ROOT/'artifacts/a173v1_frame_null_native_output_repair.json';atomic_json(out,report)
if report['passes_local_screen']:atomic_torch(out.with_suffix('.pt'),{'output':best[1],'report':report})
print(json.dumps({'selected':report['selected'],'passes_local_screen':report['passes_local_screen']}),flush=True)
