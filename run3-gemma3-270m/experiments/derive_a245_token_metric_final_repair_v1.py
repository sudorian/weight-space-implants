"""Compact closed-form final repair with a separate readout metric per token.

All bases, responses, token sets and RMS factors come from the original
parent and recorded HOST states. No candidate residual is refitted. Native
features are recorded observations; only linear/RMS/readout algebra runs.
"""
import json,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt'
reference=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
format_reference=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt'
ap=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
hp=ROOT/'artifacts/a140v1_host_branches.pt';op=ROOT/'artifacts/a56v1_host.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt';sp=ROOT/'artifacts/a124v1_syntax_stress.pt'
ch,ref,fref=[expand(h,p) for p in (base,reference,format_reference)]
a,host,oldhost,top,stress=[torch.load(p,weights_only=True,mmap=True) for p in (ap,hp,op,tp,sp)]
dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
assert a['checkpoint_sha256']==sha(reference) and stress['checkpoint_sha256']==sha(format_reference)
assert torch.equal(a['inputs'],top['inputs'])
assert torch.equal(a['inputs'][:256],oldhost['inputs'][:256]) and torch.equal(a['inputs'][256:],host['inputs'])
for name in h:
    if name not in (dn,pn):
        assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
        assert torch.equal(ch.get(name,h[name]),fref.get(name,h[name])),name
c=torch.load(base.parent/'circuit.pt',weights_only=True)
programme=c['units'].long();pool=c['output_pool'].long()
native=torch.arange(2048)[~torch.isin(torch.arange(2048),programme)]
gain=1+ch[pn].double();assert len(pool)==96 and (gain[pool]==1e6).all()
W0=ch[dn].double();F=a['captures']['17.features'].reshape(-1,2048)
assert F[:,programme].abs().max()==0
before=a['captures']['17.raw'].reshape(-1,640).double()
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x):return x/rms(x)
raw=F.double()@W0.T;rr=rms(raw);parent=before+raw/rr*gain;pr=rms(parent)
recorded=before+norm(F.double()@ref[dn].double().T)*(1+ref[pn].double())
reconstruction=float((recorded-a['captures']['finalraw'].reshape(-1,640).double()).norm()/recorded.norm())
assert reconstruction<1e-6
del recorded,ref,fref
teacher=torch.cat([oldhost['captures']['finalraw'][:256],host['captures']['finalraw']]).reshape(-1,640).double()
tr=rms(teacher);target=norm(teacher)-norm(parent)
nt=384*64;nv=128*64
ids=torch.cat([top['topk_ids'].reshape(-1,8),a['topk_ids'].reshape(-1,8)],1).long()
truth=top['topk_ids'].reshape(-1,8)[:,0].long()
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
gap=(top['topk_values'].reshape(-1,8)[:,0]-top['topk_values'].reshape(-1,8)[:,1]).double().clamp_min(0)
weight=1/(gap+.1);weight/=weight[:nt].mean();weight=weight.clamp(max=100)
counts=[8,16,32,64,128];out_count=16;basis_ridge=.001
ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
# Original paired normalized-state covariance sets both compact bases.
X=F[:nt,native].double()/rr[:nt]/pr[:nt]
sd=X.square().mean(0).sqrt().clamp_min(1e-15);Z=X/sd
C=Z.T@Z/nt;ev,V=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
inverse_root=(V/(ev+basis_ridge).sqrt()[None])@V.T
cross=inverse_root@(Z.T@target[:nt]/nt)
canonical,spectrum,_=torch.linalg.svd(cross,full_matrices=False)
input_map=inverse_root@canonical[:,:max(counts)]/sd[:,None]
features=X@input_map
OC=target[:nt,pool].T@target[:nt,pool]/nt
oe,ov=torch.linalg.eigh((OC+OC.T)/2)
OB=ov[:,-out_count:].flip(1)
del X,Z,C,ev,V,inverse_root,cross,canonical
# Each token has its own centered original-HOST/A123 top8-union metric.
# Duplicate token ids remain fixed observations with their multiplicity.
H=torch.empty(nt,out_count,out_count,dtype=torch.float64)
b=torch.empty(nt,out_count,dtype=torch.float64)
for start in range(0,nt,256):
    stop=min(start+256,nt)
    vectors=E[ids[start:stop]].double()*fg
    vectors-=vectors.mean(1,keepdim=True)
    reduced=vectors[:,:,pool]@OB
    response=(vectors*target[start:stop,None]).sum(-1)
    H[start:stop]=reduced.transpose(1,2)@reduced/ids.shape[1]*weight[start:stop,None,None]
    b[start:stop]=(reduced*response[:,:,None]).mean(1)*weight[start:stop,None]
del vectors,reduced,response
gram_full=torch.empty(out_count,max(counts),out_count,max(counts),dtype=torch.float64)
for i in range(out_count):
    for j in range(i,out_count):
        block=features.T@(features*H[:,i,j,None])/nt
        gram_full[i,:,j,:]=block;gram_full[j,:,i,:]=block.T
rhs_full=b.T@features/nt
del H,b,features
Fv=F[nt:,native].double();base_rows=W0[pool[:,None],native[None]]
def quality(rows):
    change=rows.double()-base_rows
    nr=raw[nt:].clone();nr[:,pool]+=Fv@change.T
    nrms=rms(nr);state=before[nt:]+nr/nrms*gain
    difference=norm(state)-norm(teacher[nt:])
    matches=0;score_error=0.
    for start in range(0,nv,256):
        stop=min(start+256,nv);local=ids[nt+start:nt+stop]
        vectors=E[local].double()*fg
        scores=(vectors*state[start:stop,None]).sum(-1)
        winner=local.gather(1,scores.argmax(1,keepdim=True)).flatten()
        matches+=int((winner==truth[nt+start:nt+stop]).sum())
        vectors-=vectors.mean(1,keepdim=True)
        err=(vectors*difference[start:stop,None]).sum(-1)
        score_error+=float((err.square().mean(1)*weight[nt+start:nt+stop]).sum())
    return {'validation_restricted_matches':matches,
        'validation_centered_readout_score_MSE':score_error/float(weight[nt:].sum()),
        'validation_paired_HOST_state_relative_rms':float(((state-teacher[nt:])/tr[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms':float(difference.square().mean().sqrt()),
        'maximum_validation_raw_RMS_relative_change':float((nrms/rr[nt:]-1).abs().max())},state
baseline,_=quality(base_rows.float());bm=json.loads(base.with_suffix('.manifest.json').read_text())
trials=[];best=None
print(json.dumps({'baseline':baseline,'canonical_response_spectrum':spectrum[:16].tolist()}),flush=True)
for count in counts:
    gram=gram_full[:,:count,:,:count].reshape(out_count*count,out_count*count)
    scale=gram.diagonal().clamp_min(1e-30).sqrt()
    standardized=gram/scale[:,None]/scale[None]
    ce,cv=torch.linalg.eigh((standardized+standardized.T)/2);ce=ce.clamp_min(0)
    rhs=cv.T@(rhs_full[:,:count].reshape(-1)/scale)
    for ridge in ridges:
        beta=((cv@(rhs/(ce+ridge)))/scale).reshape(out_count,count)
        correction=(OB@beta@input_map[:,:count].T)/gain[pool,None]
        for strength in strengths:
            rows=(base_rows+strength*correction).float();q,state=quality(rows)
            W=ch[dn].clone();W[pool[:,None],native[None]]=rows
            assert torch.equal(W[:,programme],ch[dn][:,programme])
            total=bm['changed_scalars']-bm['per_tensor'][dn]+int((W.view(torch.int32)!=h[dn].view(torch.int32)).sum())
            rec={'canonical_input_dimension':count,'fixed_output_dimension':out_count,
                'readout_metric_ridge':ridge,'correction_strength':strength,**q,
                'total_changed_scalars':total,'maximum_down_weight':float(W.abs().max())}
            rec['eligible']=bool(W.isfinite().all() and W.abs().max()<=1e6 and total<=5016268
                and q['validation_restricted_matches']>=baseline['validation_restricted_matches']+16
                and q['validation_paired_HOST_state_relative_rms']<baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms']<baseline['validation_paired_HOST_direction_rms']
                and q['validation_centered_readout_score_MSE']<baseline['validation_centered_readout_score_MSE']
                and q['maximum_validation_raw_RMS_relative_change']<.001)
            trials.append(rec);key=(q['validation_restricted_matches'],-q['validation_centered_readout_score_MSE'])
            if rec['eligible'] and (best is None or key>best[0]):best=key,W.clone(),state.clone(),rec
    print(json.dumps({'canonical_input_dimension':count,'selected':None if best is None else best[-1]}),flush=True)
def full_readout(state,expected):
    x=(norm(state)*fg).float();maximum=torch.full((len(x),),-torch.inf)
    winner=torch.full_like(expected,-1);competitor=torch.full_like(maximum,-torch.inf)
    desired=(x*E[expected]).sum(1)
    for start in range(0,len(E),2048):
        logits=x@E[start:start+2048].T
        values,indices=logits.max(1);better=values>maximum
        maximum[better]=values[better];winner[better]=start+indices[better]
        local=expected-start;rows=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rows,local[rows]]=-torch.inf;competitor=torch.maximum(competitor,logits.amax(1))
    return {'tokens':len(x),'matches':int((winner==expected).sum()),'minimum_target_margin':float((desired-competitor).min())}
provisional=None;full=None;fmt=None
if best is not None:
    provisional=best[-1]
    full={'parent':full_readout(parent[nt:],truth[nt:]),'candidate':full_readout(best[2],truth[nt:]),'required_extra_matches':16}
    if full['candidate']['matches']<full['parent']['matches']+16:best=None
if best is not None:
    SF=stress['captures']['17.features'][:,8:13].double().reshape(-1,2048)
    SB=stress['captures']['17.raw'][:,8:13].double().reshape(-1,640)
    expected=stress['topk_ids'][:,:,0].reshape(-1).long()
    old_raw=SF@W0.T;new_raw=SF@best[1].double().T
    fmt={'parent':full_readout(SB+norm(old_raw)*gain,expected),
        'candidate':full_readout(SB+norm(new_raw)*gain,expected),
        'maximum_raw_RMS_relative_change':float((rms(new_raw)/rms(old_raw)-1).abs().max())}
    assert fmt['parent']['matches']==len(expected)
    if fmt['candidate']['matches']!=len(expected) or fmt['candidate']['minimum_target_margin']<=1 or fmt['maximum_raw_RMS_relative_change']>=.02:best=None
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'calibration_parent':str(reference.relative_to(ROOT)),'format_capture_parent':str(format_reference.relative_to(ROOT)),
    'training_windows':384,'validation_windows':128,'format_windows':512,'format_positions':[8,9,10,11,12],
    'original_recorded_state_reconstruction_relative_error':reconstruction,
    'canonical_input_dimension_trials':counts,'fixed_output_dimension':out_count,'fixed_canonical_whitening_ridge':basis_ridge,
    'readout_metric_ridge_trials':ridges,'correction_strength_trials':strengths,
    'metric':'Per-token centered immutable HOST/A123 top8-union unembedding Gram; duplicates retain their fixed multiplicity; HOST gap weight=1/(gap+0.1), training-mean normalized and capped100',
    'target':'Original paired HOST normalized final state minus original A142 state, with fixed original parent branch/final RMS',
    'basis':'Canonical original-response input covariance, original-response PCA within the96 existing strong output rows; training only',
    'all_targets_bases_token_sets_normalizers_fixed_before_independent_solves':True,
    'all_programme_G_Up_Down_columns_norms_and_ordinary_rows_unchanged':True,
    'word_guard':'At least16 extra restricted and then full-vocabulary matches/8192; improved state,direction and readout MSE; max raw-RMS change<0.001; finite/edit/weight caps',
    'format_guard':'All2560 recorded register winners preserved, minimum target margin>1, maximum raw-RMS change<0.02',
    'rounding_scope':'Rounded float32 weights evaluated with double linear/RMS algebra; no global floating-point certificate',
    'no_model_execution_native_activation_replay_gradients_answers_or_iterative_refitting':True,
    'baseline':baseline,'trials':trials,'provisional_selection':provisional,'full_vocabulary_validation':full,'format_validation':fmt,
    'selected':None if best is None else best[-1],
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',reference,format_reference,ap,hp,op,tp,sp)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a245v1_token_metric_final_repair.json';atomic_json(out,report)
if best is not None:atomic_torch(out.with_suffix('.pt'),{'down':best[1],'report':report})
print(json.dumps({'selected':report['selected'],'full_vocabulary_validation':full,'format_validation':fmt,'peak_kib':report['maximum_resident_set_kib']}),flush=True)
