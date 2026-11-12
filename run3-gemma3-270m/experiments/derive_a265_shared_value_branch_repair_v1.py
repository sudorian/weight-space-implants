"""Independent shared-value repair from original paired HOST branch targets.

Start from A189 with native O3, independently of A191's output repair.
Recorded attention mixtures remain fixed because Q/K and their inputs do.
Protect all A-reader Gate/Up and earlier boosted scratch functionals through
every output head. Solve each fixed low-rank shared-value correction once.
"""
import json,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha

h=host_state();base=ROOT/'experiments/a189v1_partial_native_value_decoder/delta.pt';ch=expand(h,base)
cp=base.parent/'circuit.pt';c=torch.load(cp,weights_only=True)
mp=ROOT/'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp=ROOT/'artifacts/a150v1_early_native_basis.pt'
ep=ROOT/'artifacts/a140v1_edited_branches.pt'
hp=ROOT/'artifacts/a140v1_host_branches.pt'
hs_path=ROOT/'artifacts/a150v1_derived_host_states.pt'
m,w,e,host,hs=[torch.load(p,weights_only=True,mmap=True) for p in (mp,wp,ep,hp,hs_path)]
reference_path=ROOT/m['report']['base_checkpoint'];reference=expand(h,reference_path)
assert m['report']['inputs'][str(reference_path.relative_to(ROOT))]==sha(reference_path)
captured_base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
assert w['checkpoint_sha256']==e['checkpoint_sha256']==sha(captured_base)
pre='model.layers.3.';vn=pre+'self_attn.v_proj.weight';on=pre+'self_attn.o_proj.weight'
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name.startswith(pre+'self_attn.') and name!=vn:
        assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
assert torch.equal(ch.get(on,h[on]),h[on])
assert torch.equal(ch.get(pre+'input_layernorm.weight',h[pre+'input_layernorm.weight']),reference.get(pre+'input_layernorm.weight',h[pre+'input_layernorm.weight']))
assert torch.equal(m['inputs'],w['inputs']) and torch.equal(w['inputs'],e['inputs'])
assert torch.equal(host['inputs'],m['inputs'][:256])
V0=ch[vn].double();VH=h[vn].double();O=h[on].double().reshape(640,4,256)
gamma=1+h[pre+'post_attention_layernorm.weight'].double()
pg=1+h[pre+'pre_feedforward_layernorm.weight'].double()
stage=c['native_a_stage'];units=stage['units'].long();assert len(units)==4
G=ch[pre+'mlp.gate_proj.weight'][units].double()
U=ch[pre+'mlp.up_proj.weight'][units].double()

def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x):return x/rms(x)
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)

M=m['records']['word']['mixtures'].double();PM=m['records']['format']['mixtures'].double()
TW=m['records']['word']['input'].double();T=TW.reshape(-1,640)
original_raw=torch.einsum('nhtv,ohv->nto',M@reference[vn].double().T,O)
recovery=float((original_raw-e['captures']['word.3.attention.raw']).norm()/original_raw.norm())
assert recovery<1e-5
del original_raw,reference
word_before=(w['captures']['word.2.raw']+fnorm(e['captures']['word.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
format_before=(w['captures']['format.2.raw']+fnorm(e['captures']['format.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
host_check=h['model.embed_tokens.weight'][host['inputs']]*torch.tensor(640**.5,dtype=torch.float32)
for layer in range(3):
    for branch,part in [('attention','post_attention_layernorm'),('mlp','post_feedforward_layernorm')]:
        host_check=host_check+fnorm(host['captures'][f'{layer}.{branch}.raw'])*(1+h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(host_check,hs['states']['2.out']);del host_check
host_before=hs['states']['2.out'].double()
TH=fnorm(hs['states']['2.out'])*(1+h[pre+'input_layernorm.weight'])
TH=TH.double()
hr=host['captures']['3.attention.raw'].double()
teacher=host_before+norm(hr)*gamma
teacher_rms=rms(teacher)
raw0=torch.einsum('nhtv,ohv->nto',M@V0.T,O);reference_rms=rms(raw0)
target=(norm(hr)-norm(raw0))*gamma
nt=192;nr=nt*64
# The protected frame includes actual rounded MLP interfaces and every
# earlier coordinate with an existing programme-scale post-MLP gain.
scratch_mask=torch.zeros(640,dtype=torch.bool)
for layer in range(3):
    name=f'model.layers.{layer}.post_feedforward_layernorm.weight'
    scratch_mask|=(1+ch.get(name,h[name]).double()).abs()>=1e4
scratch=scratch_mask.nonzero().flatten()
frame=torch.cat([G.T*pg[:,None],U.T*pg[:,None],torch.eye(640,dtype=torch.float64)[:,scratch]],1)
frame=frame[:,frame.norm(dim=0)>0]
frame/=frame.norm(dim=0)
constraints=torch.cat([frame.T@(gamma[:,None]*O[:,head]) for head in range(4)],0)
constraints/=constraints.norm(dim=1,keepdim=True).clamp_min(1e-30)
_,ss,right=torch.linalg.svd(constraints,full_matrices=True)
rank=int((ss>ss[0]*1e-10).sum());null=right[rank:].T
assert null.shape[1]>=128 and (constraints@null).abs().max()<1e-8
# Fixed source-value response chooses bases only. The fitted target below
# is the original paired HOST normalized branch, using observed mixtures.
source_response=(TH@VH.T-TW@V0.T).reshape(-1,256)
response=source_response[:nr]@null
cov=response.T@response/nr;oe,ov=torch.linalg.eigh((cov+cov.T)/2)
B=null@ov[:,-16:].flip(1)
X=T[:nr];sd=X.square().mean(0).sqrt().clamp_min(1e-15);Z=X/sd
C=Z.T@Z/nr;ce,cv=torch.linalg.eigh((C+C.T)/2);ce=ce.clamp_min(0)
invroot=(cv/(ce+.001).sqrt()[None])@cv.T
cross=invroot@(Z.T@(source_response[:nr]@null)/nr)
canonical,spectrum,_=torch.linalg.svd(cross,full_matrices=False)
input_map=invroot@canonical[:,:128]/sd[:,None]
mixed=(M[:nt]@input_map).permute(0,2,1,3).reshape(nr,4,128)/reference_rms[:nt].reshape(nr,1,1)
feature_scale=mixed.square().mean((0,1)).sqrt().clamp_min(1e-15)
mixed/=feature_scale;input_map/=feature_scale
outputs=torch.stack([gamma[:,None]*(O[:,head]@B) for head in range(4)])
# Independent small explicit-design control for the shared-head tensor
# ordering. This is a synthetic linear identity, not model execution.
control_rng=torch.Generator().manual_seed(265)
control_x=torch.randn(7,4,3,generator=control_rng,dtype=torch.float64)
control_o=torch.randn(4,5,2,generator=control_rng,dtype=torch.float64)
control_y=torch.randn(7,5,generator=control_rng,dtype=torch.float64)
explicit=torch.einsum('nhi,hor->nori',control_x,control_o).reshape(35,6)
control_gram=torch.zeros(2,3,2,3,dtype=torch.float64)
control_rhs=torch.zeros(2,3,dtype=torch.float64)
for head in range(4):
    control_rhs+=(control_y@control_o[head]).T@control_x[:,head]/7
    for other in range(4):
        control_gram+=torch.einsum('rs,ij->risj',control_o[head].T@control_o[other],control_x[:,head].T@control_x[:,other]/7)
normal_equation_control=max(float((control_gram.reshape(6,6)-explicit.T@explicit/7).abs().max()),
    float((control_rhs.flatten()-explicit.T@control_y.flatten()/7).abs().max()))
assert normal_equation_control<1e-12
full_gram=torch.zeros(16,128,16,128,dtype=torch.float64)
full_rhs=torch.zeros(16,128,dtype=torch.float64)
Y=target[:nt].reshape(nr,640)
for head in range(4):
    full_rhs+=(Y@outputs[head]).T@mixed[:,head]/nr
    for other in range(4):
        input_cov=mixed[:,head].T@mixed[:,other]/nr
        output_cov=outputs[head].T@outputs[other]
        full_gram+=torch.einsum('rs,ij->risj',output_cov,input_cov)
del X,Z,C,ce,cv,invroot,cross,canonical,mixed,outputs,source_response,response,Y
frame_reference=torch.stack([(frame.T@(gamma[:,None]*O[:,head])@V0).norm() for head in range(4)]).clamp_min(1e-15)
FR0=torch.einsum('nhtv,ohv->nto',PM@V0.T,O);FRMS0=rms(FR0)
FS0=format_before+norm(FR0)*gamma
FG0=(norm(FS0)*pg)@G.T;FU0=(norm(FS0)*pg)@U.T
gate_scale=FG0.abs().amax((0,1)).clamp_min(1);up_scale=FU0.abs().amax((0,1)).clamp_min(1)
field=(U.T*pg[:,None]/stage['piece']['common_up_gain'])[:,[1,2,3,0]]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
tokens=m['inputs'][256:,:14]
expected=torch.stack([(tokens[:,pos,None]==ids).long().argmax(1) for pos in [3,2,1]],1)

def quality(matrix):
    raw=torch.einsum('nhtv,ohv->nto',M[nt:]@matrix.T,O)
    post=norm(raw)*gamma;state=word_before[nt:]+post
    gates=(norm(state)*pg)@G.T
    fraw=torch.einsum('nhtv,ohv->nto',PM@matrix.T,O)
    fstate=format_before+norm(fraw)*gamma
    fpre=norm(fstate)*pg;fg,fu=fpre@G.T,fpre@U.T
    code=fstate[:,8]@field;decoded=4.5*code[:,:3]/code[:,3:]+4.5
    q={'validation_paired_HOST_branch_RMS':float((post-norm(hr[nt:])*gamma).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms':float(((state-teacher[nt:])/teacher_rms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms':float((norm(state)-norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_word_programme_gate':float(gates.max()),'minimum_format_equals_gate':float(fg[:,8].min()),
        'maximum_individual_source_digit_error':float((decoded-expected).abs().max()),
        'wrong_individual_source_digits':int((decoded.round()!=expected).sum()),
        'minimum_format_reference':float(code[:,3].min()),
        'maximum_format_raw_RMS_relative_change':float((rms(fraw)/FRMS0-1).abs().max()),
        'format_gate_sign_changes':int(((fg>0)!=(FG0>0)).sum()),
        'maximum_format_gate_relative_change':float(((fg-FG0).abs()/gate_scale).max()),
        'maximum_format_up_relative_change':float(((fu-FU0).abs()/up_scale).max())}
    return q

baseline=quality(V0)
assert baseline['wrong_individual_source_digits']==0 and baseline['maximum_individual_source_digit_error']<.25
counts=[8,16,32,64,128];ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
bm=json.loads(base.with_suffix('.manifest.json').read_text())
trials=[];best=None
for count in counts:
    gram=full_gram[:,:count,:,:count].reshape(16*count,16*count)
    scale=gram.diagonal().clamp_min(1e-30).sqrt();standard=gram/scale[:,None]/scale[None]
    ev,vec=torch.linalg.eigh((standard+standard.T)/2);ev=ev.clamp_min(0)
    rhs=vec.T@(full_rhs[:,:count].reshape(-1)/scale)
    for ridge in ridges:
        beta=((vec@(rhs/(ev+ridge)))/scale).reshape(16,count)
        correction=B@beta@input_map[:,:count].T
        assert (constraints@correction).abs().max()<1e-7
        for strength in strengths:
            matrix=(V0+strength*correction).float();delta=matrix.double()-V0
            q=quality(matrix.double())
            frame_error=max(float((frame.T@(gamma[:,None]*O[:,head])@delta).norm()/frame_reference[head]) for head in range(4))
            total=bm['changed_scalars']-bm['per_tensor'][vn]+int((matrix.view(torch.int32)!=h[vn].view(torch.int32)).sum())
            rec={'canonical_input_count':count,'output_dimension':16,'ridge':ridge,'correction_strength':strength,
                **q,'maximum_relative_frame_null_error':frame_error,'maximum_value_weight':float(matrix.abs().max()),'total_changed_scalars':total}
            rec['eligible']=bool(matrix.isfinite().all() and matrix.abs().max()<=1e6 and total<=5016268
                and q['validation_paired_HOST_branch_RMS']<.9*baseline['validation_paired_HOST_branch_RMS']
                and q['validation_paired_HOST_state_relative_rms']<baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms']<baseline['validation_paired_HOST_direction_rms']
                and q['maximum_word_programme_gate']<-16 and q['minimum_format_equals_gate']>31.9
                and q['wrong_individual_source_digits']==0 and q['minimum_format_reference']>0
                and q['maximum_individual_source_digit_error']<min(.25,baseline['maximum_individual_source_digit_error']+.01)
                and q['maximum_format_raw_RMS_relative_change']<.002 and q['format_gate_sign_changes']==0
                and q['maximum_format_gate_relative_change']<.001 and q['maximum_format_up_relative_change']<.001
                and frame_error<1e-6)
            trials.append(rec)
            if rec['eligible'] and (best is None or q['validation_paired_HOST_branch_RMS']<best[0]):best=q['validation_paired_HOST_branch_RMS'],matrix.clone(),rec
    print(json.dumps({'canonical_input_count':count,'selected':None if best is None else best[-1]}),flush=True)
all_word_gate=None
if best is not None:
    raw=torch.einsum('nhtv,ohv->nto',M@best[1].double().T,O)
    all_word_gate=float(((norm(word_before+norm(raw)*gamma)*pg)@G.T).max())
    if all_word_gate>=-16:best=None
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),'baseline':baseline,
    'selected':None if best is None else best[-1],'selected_all_word_gate_maximum':all_word_gate,
    'training_windows':192,'validation_windows':64,'format_windows':128,
    'canonical_input_count_trials':counts,'fixed_output_dimension':16,'ridge_trials':ridges,'correction_strength_trials':strengths,
    'fixed_basis_whitening_ridge':.001,'protected_frame_rank_in_value_space':rank,'protected_earlier_scratch_coordinates':scratch.tolist(),
    'rounded_mixture_raw_reconstruction_error':recovery,'synthetic_normal_equation_control_max_error':normal_equation_control,'trials':trials,
    'target':'Original paired HOST normalized attention3 branch minus original A189 branch, fixed original A189 raw RMS; native O3 fixed.',
    'basis':'Original source-value HOST-at-HOST-input minus A189-at-A189-input response; projected off all Gate/Up and earlier boosted scratch functionals through all four heads.',
    'selection':'BranchRMSgain>=10percent,state/directionimprove,wordG<-16,sourcecorrect,error<min0.25/baseline+0.01,formatRMS<0.002,Gate/Upchange<0.001,signsunchanged,frame-null<1e-6,caps.',
    'scope':'Fixed recovered factual mixtures at unchanged Q/K inputs; no new attention/native activation or full model prediction is computed.',
    'independent_of_A191_output_repair':True,'no_gradients_answers_or_iterative_residual_refitting':True,
    'sources':{**m['report']['sources'],str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,cp,mp,wp,ep,hp,hs_path,reference_path,captured_base]},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a265v1_shared_value_branch_repair.json';atomic_json(out,report)
if best is not None:atomic_torch(out.with_suffix('.pt'),{'value':best[1],'report':report})
print(json.dumps({'baseline':baseline,'selected':report['selected'],'peak_kib':report['maximum_resident_set_kib']}),flush=True)
