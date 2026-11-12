"""Independent original-HOST output repair for the A189 partial value reader.

The new value weights and rounded source decoder define a new native parent
and arithmetic frame. Original paired HOST branches remain the sole targets.
Factual recorded mixtures give the parent raw contexts and fixed normalizers;
each candidate is a single solve, not a refit of an earlier output repair.
"""
import json,sys,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a189v1_partial_native_value_decoder/delta.pt';ch=expand(h,base)
cp=base.parent/'circuit.pt';c=torch.load(cp,weights_only=True)
mp=ROOT/'artifacts/a191v1_recorded_reader3_mixtures.pt'
ep=ROOT/'artifacts/a140v1_edited_branches.pt';hp=ROOT/'artifacts/a140v1_host_branches.pt'
wp=ROOT/'artifacts/a150v1_early_native_basis.pt';sp=ROOT/'artifacts/a150v1_derived_host_states.pt'
m,e,host,w,hs=[torch.load(p,weights_only=True,mmap=True)for p in(mp,ep,hp,wp,sp)]
mixture_base=ROOT/m['report']['base_checkpoint']
assert m['report']['inputs'][str(mixture_base.relative_to(ROOT))]==sha(mixture_base)
reference=expand(h,mixture_base)
for name in h:
 if name.startswith(tuple(f'model.layers.{i}.'for i in range(3)))or name.startswith('model.layers.3.self_attn.')and not name.endswith('v_proj.weight'):
  assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
referenceV=reference['model.layers.3.self_attn.v_proj.weight'].double()
del reference
assert torch.equal(m['inputs'],e['inputs'])and torch.equal(e['inputs'],w['inputs'])
assert torch.equal(e['inputs'][:256],host['inputs'])
pre='model.layers.3.';vn=pre+'self_attn.v_proj.weight'
V=ch[vn].double();VH=h[vn].double()
O=h[pre+'self_attn.o_proj.weight'].double().reshape(640,4,256)
gamma=1+h[pre+'post_attention_layernorm.weight'].double()
pg=1+h[pre+'pre_feedforward_layernorm.weight'].double()
stage=c['native_a_stage'];units=stage['units'].long()
# Protect the actual rounded arithmetic Up interface.
frame=(ch[pre+'mlp.up_proj.weight'][units].double().T*pg[:,None]/stage['piece']['common_up_gain'])[:,[1,2,3,0]]
assert frame.shape==(640,4)
G=ch[pre+'mlp.gate_proj.weight'][units].double()

def norm(x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
M=m['records']['word']['mixtures'].double();PM=m['records']['format']['mixtures'].double()
heads=(M@V.T).permute(0,2,1,3).reshape(256,64,1024)
fheads=(PM@V.T).permute(0,2,1,3).reshape(128,14,1024)
O0=O.reshape(640,1024)
recorded_raw=e['captures']['word.3.attention.raw'].double();hr=host['captures']['3.attention.raw'].double()
original_heads=(M@referenceV.T).permute(0,2,1,3).reshape(256,64,1024)
recovery=float((original_heads@O0.T-recorded_raw).norm()/recorded_raw.norm());assert recovery<1e-5
raw=heads@O0.T
assert torch.equal(ch.get(pre+'self_attn.o_proj.weight',h[pre+'self_attn.o_proj.weight']),h[pre+'self_attn.o_proj.weight'])
del original_heads,recorded_raw,referenceV
word_before=(w['captures']['word.2.raw']+fnorm(e['captures']['word.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
format_before=(w['captures']['format.2.raw']+fnorm(e['captures']['format.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
teacher=hs['states']['2.out'].double()+norm(hr)*gamma
rms=(raw.square().mean(-1,keepdim=True)+1e-6).sqrt()
target=(norm(hr)-norm(raw))*gamma
nt=192;nr=nt*64
X=(heads/rms).reshape(-1,1024);Y=target.reshape(-1,640)
sd=X[:nr].square().mean(0).sqrt().clamp_min(1e-12);Z=X[:nr]/sd
C=Z.T@Z/nr;rhs=Z.T@Y[:nr]/nr
priority=Y[:nr].square().mean(0)
priority[gamma.abs()<.05]=-torch.inf
order=priority.argsort(descending=True);rows=order[:128]
B=frame[rows];u,ss,_=torch.linalg.svd(B,full_matrices=False)
rank=int((ss>ss[0]*1e-10).sum());B=u[:,:rank]
reference_gain2=float(gamma.square().median())
inverse_metric=1/(1+.01*reference_gain2/gamma[rows].square())
Zm=inverse_metric[:,None]*B
projection=torch.eye(len(rows),dtype=torch.float64)-Zm@torch.linalg.solve(B.T@Zm,B.T)
# Original-target conditional explanatory energy gives a single fixed
# coordinate ranking; no previous candidate correction is fitted again.
ranking_gram=C.clone();ranking_gram.diagonal().add_(.001)
precision=torch.linalg.inv(ranking_gram)
projected_rhs=rhs[:,rows]@projection.T
ranking_beta=precision@projected_rhs
feature_priority=ranking_beta.square().sum(1)/precision.diagonal().clamp_min(1e-15)
feature_order=feature_priority.argsort(descending=True)
counts=[64,128,256,512,768];ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
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
for count in counts:
 columns=feature_order[:count]
 gram=C[columns][:,columns]
 ce,cv=torch.linalg.eigh((gram+gram.T)/2);ce=ce.clamp_min(0)
 pr=cv.T@rhs[columns]
 for ridge in ridges:
  beta=(cv@(pr/(ce[:,None]+ridge)))/sd[columns,None]
  correction=(projection@beta[:,rows].T)/gamma[rows,None]
  for strength in strengths:
   matrix=O0.clone()
   matrix[rows[:,None],columns[None]]=(O0[rows[:,None],columns[None]]+strength*correction).float().double()
   q=quality(matrix)
   changed=bm['changed_scalars']+int((matrix.float().view(torch.int32)!=h[pre+'self_attn.o_proj.weight'].view(torch.int32)).sum())
   row={'native_ridge':ridge,'native_input_coordinate_count':count,'output_row_count':128,'correction_strength':strength,**q,
    'maximum_output_weight':float(matrix.abs().max()),'changed_scalars':changed,
    'maximum_frame_null_error':float((frame.T@(gamma[:,None]*(matrix-O0))).abs().max())}
   row['eligible']=bool(changed<=budget and matrix.isfinite().all()and matrix.abs().max()<=1e6 and q['maximum_word_programme_gate']<-16
    and q['minimum_format_equals_gate']>31.9 and q['wrong_individual_source_digits']==0
    and max(q['maximum_individual_source_digit_errors'])<min(.25,max(baseline['maximum_individual_source_digit_errors'])+.01)
    and q['maximum_format_relative_raw_rms_change']<.02
    and q['validation_hidden_relative_rms']<baseline['validation_hidden_relative_rms'])
   trials.append(row)
   if row['eligible']and(best is None or q['validation_paired_HOST_branch_rms']<best[0]):best=q['validation_paired_HOST_branch_rms'],matrix.float(),row
  print(json.dumps({'native_input_coordinate_count':count,'ridge':ridge,'best':None if best is None else best[-1]}),flush=True)
report={'analysis_only':True,'baseline':baseline,'output_row_count':128,'native_input_coordinate_count_trials':counts,'fixed_ranking_ridge':.001,'native_input_coordinate_order':feature_order.tolist(),'native_ridge_trials':ridges,
 'correction_strength_trials':strengths,'fixed_raw_projection_penalty':.01,'training_windows':192,'validation_windows':64,
 'format_windows':128,'output_row_priority':'original paired HOST normalized branch-error energy, excluding gains below.05',
 'row_order':order[:128].tolist(),'trials':trials,'selected':None if best is None else best[-1],
 'target':'Original paired HOST normalized attention branch at fixed A189 raw RMS derived from factual recorded mixtures',
 'protected_functionals':'Entire arithmetic output frame, for every head context, before raw normalization',
 'no_model_replay_gradients_answers_or_iterative_refit':True,
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
 'inputs':{str(p.relative_to(ROOT)):sha(p)for p in[base,cp,mp,ep,hp,wp,sp,mixture_base]},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
report['passes_local_screen']=best is not None and best[0]<.95*baseline['validation_paired_HOST_branch_rms']
out=ROOT/'artifacts/a191v1_partial_reader_output_repair.json';atomic_json(out,report)
if report['passes_local_screen']:atomic_torch(out.with_suffix('.pt'),{'output':best[1],'report':report})
print(json.dumps({'selected':report['selected'],'passes_local_screen':report['passes_local_screen']}),flush=True)
