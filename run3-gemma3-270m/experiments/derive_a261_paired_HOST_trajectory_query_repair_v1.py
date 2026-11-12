"""Format-null query repair toward the original HOST trajectory scores.

All targets and bases are computed once from the original parent and HOST.
A finite covariance solve uses recorded keys, RMS and RoPE algebra only;
there is no softmax, new native activation, model replay or gradient.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a191v1_partial_reader_output_repair/delta.pt';ch=expand(h,base)
wp=ROOT/'artifacts/a150v1_early_native_basis.pt';ep=ROOT/'artifacts/a140v1_edited_branches.pt'
refp=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
historical_paths=[ROOT/f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a','syntax_stress','syntax_c')]
ref=expand(h,refp)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name=='model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
del ref
hp=ROOT/'artifacts/a140v1_host_branches.pt'
comparison_path=ROOT/'artifacts/a237v1_causal_metric_query_repair.pt'
comparison=torch.load(comparison_path,weights_only=True)
w,e,host=[torch.load(p,weights_only=True,mmap=True) for p in (wp,ep,hp)]
assert w['checkpoint_sha256']==e['checkpoint_sha256']==sha(refp)
assert torch.equal(w['inputs'],e['inputs'])
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x):return x/rms(x)
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
assert torch.equal(host['inputs'],w['inputs'][:256])
# Recombine recorded HOST branches; no attention or native MLP is evaluated.
host_state_raw=h['model.embed_tokens.weight'][host['inputs']]*torch.tensor(640**.5,dtype=torch.float32)
for layer in range(18):
    if layer==3:
        TH=(fnorm(host_state_raw)*(1+h['model.layers.3.input_layernorm.weight'])).double()
    for branch,part in [('attention','post_attention_layernorm'),('mlp','post_feedforward_layernorm')]:
        host_state_raw=host_state_raw+fnorm(host['captures'][f'{layer}.{branch}.raw'])*(1+h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(host_state_raw,host['captures']['finalraw'])
del host_state_raw
inputs={}
for tag in ('word','format'):
    state=w['captures'][f'{tag}.2.raw']+fnorm(e['captures'][f'{tag}.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])
    inputs[tag]=(fnorm(state)*(1+h['model.layers.3.input_layernorm.weight'])).double()
TW=inputs['word'];current=inputs['format'][:,:9]
historical=torch.cat([torch.load(p,weights_only=True,mmap=True)['captures']['3.input'][:,:9] for p in historical_paths]).double()
SX=torch.cat([current,historical]);T=TW.reshape(-1,640);nt=192*64
pre='model.layers.3.self_attn.'
qn,gn,kn,kgn=[pre+s for s in ('q_proj.weight','q_norm.weight','k_proj.weight','k_norm.weight')]
Q0=ch[qn].double().reshape(4,256,640);QH=h[qn].double().reshape(4,256,640)
gain=1+ch[gn].double();gh=1+h[gn].double();active=gain.abs()>1e-8
K=h[kn].double();kg=1+h[kgn].double()
assert torch.equal(ch.get(kn,h[kn]),h[kn]) and torch.equal(ch.get(kgn,h[kgn]),h[kgn])
P=current[:96,8];Pn=P/P.norm(dim=1,keepdim=True)
_,singular,right=torch.linalg.svd(Pn,full_matrices=True)
rank=int((singular>singular[0]*1e-10).sum());null=right[rank:].T
assert rank==96 and (P@null).abs().max()<1e-8
frequency=10000.**(-torch.arange(128,dtype=torch.float64)/128)
def rotate(z):
    angle=torch.arange(z.shape[-2],dtype=torch.float64)[:,None]*frequency
    cs,sn=angle.cos().repeat(1,2),angle.sin().repeat(1,2)
    return z*cs+torch.cat([-z[...,128:],z[...,:128]],-1)*sn
rotated_keys=rotate(norm(TW@K.T)*kg)
paired_host_keys=rotate(norm(TH@K.T)*kg)
causal=torch.ones(64,64,dtype=torch.float64).tril();causal_count=causal.sum(-1)
SK=norm(SX@K.T)*kg
angles=(torch.arange(9,dtype=torch.float64)-8)[:,None]*frequency
cs,sn=angles.cos().repeat(1,2),angles.sin().repeat(1,2)
KR=SK*cs+torch.cat([-SK[...,128:],SK[...,:128]],-1)*sn
input_counts=[8,16,32,64,128];output_count=16;basis_ridge=.001
ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
positions=[3,2,1,0]
fixed_input_counts=[64,8,8,128]
target_fractions=[0.,.25,.5,.75,1.]
reproduction_controls=[]; comparison_mses=[]
trials=[];head_reports=[];selected_matrix=Q0.clone();any_head_selected=False
baseline_mse=[];selected_mse=[]
for head in range(4):
    original_raw=T@Q0[head].T;reference_rms=rms(original_raw)
    old_post=original_raw/reference_rms*gain
    host_post=norm(T@QH[head].T)*gh
    response=host_post-old_post
    target_scores=rotate(response.reshape(256,64,256))@rotated_keys.transpose(-1,-2)/16
    original_host_scores=rotate(host_post.reshape(256,64,256))@rotated_keys.transpose(-1,-2)/16
    paired_host_scores=rotate(norm(TH@QH[head].T)*gh)@paired_host_keys.transpose(-1,-2)/16
    original_scores=rotate(old_post.reshape(256,64,256))@rotated_keys.transpose(-1,-2)/16
    paired_target=paired_host_scores-original_scores
    # Canonical input directions depend on the original response covariance,
    # never a fitted candidate's errors. One fixed ridge stabilizes whitening.
    X=(T[:nt]@null)/reference_rms[:nt]
    sd=X.square().mean(0).sqrt().clamp_min(1e-15);Z=X/sd
    C=Z.T@Z/nt
    ev,vec=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
    inverse_root=(vec/(ev+basis_ridge).sqrt()[None])@vec.T
    cross=inverse_root@(Z.T@response[:nt]/nt)
    canonical,spectrum,_=torch.linalg.svd(cross,full_matrices=False)
    coordinate_map=inverse_root@canonical[:,:max(input_counts)]/sd[:,None]
    input_map=null@coordinate_map
    features=X@coordinate_map
    output_cov=response[:nt,active].T@response[:nt,active]/nt
    oe,ov=torch.linalg.eigh((output_cov+output_cov.T)/2)
    output_basis=torch.zeros(256,output_count,dtype=torch.float64)
    output_basis[active]=ov[:,-output_count:].flip(1)
    # A query correction expressed before RoPE sees rotated keys projected
    # onto R(position)*output_basis. Centering removes score-row constants.
    rotated_basis=rotate(output_basis.T[:,None,:].expand(-1,64,-1)).permute(1,0,2)
    key_features=torch.einsum('bsc,trc->btsr',rotated_keys[:192],rotated_basis)/16
    key_features*=causal[None,:,:,None]
    mean_key=key_features.sum(2)/causal_count[None,:,None]
    H=torch.einsum('btsr,btsq->btrq',key_features,key_features)/causal_count[None,:,None,None]
    H-=mean_key[:,:,:,None]*mean_key[:,:,None,:]
    D=target_scores[:192]*causal
    mean_D=D.sum(-1)/causal_count
    weighted_target=torch.einsum('btsr,bts->btr',key_features,D)/causal_count[None,:,None]
    weighted_target-=mean_key*mean_D[:,:,None]
    H=H.reshape(nt,output_count,output_count)
    weighted_target=weighted_target.reshape(nt,output_count)
    PD=paired_target[:192]*causal
    mean_PD=PD.sum(-1)/causal_count
    paired_weighted_target=torch.einsum('btsr,bts->btr',key_features,PD)/causal_count[None,:,None]
    paired_weighted_target-=mean_key*mean_PD[:,:,None]
    paired_weighted_target=paired_weighted_target.reshape(nt,output_count)
    full_gram=torch.empty(output_count,max(input_counts),output_count,max(input_counts),dtype=torch.float64)
    for r in range(output_count):
        for s in range(r,output_count):
            block=features.T@(features*H[:,r,s,None])/nt
            full_gram[r,:,s,:]=block;full_gram[s,:,r,:]=block.T
    full_rhs=weighted_target.T@features/nt
    paired_full_rhs=paired_weighted_target.T@features/nt
    del key_features,H,weighted_target,X,Z,C,ev,vec,cross,inverse_root,canonical
    def quality(matrix):
        post=norm(TW[192:]@matrix.T)*gain
        actual_scores=rotate(post)@rotated_keys[192:].transpose(-1,-2)/16
        difference=actual_scores-paired_host_scores[192:]
        same_difference=actual_scores-original_host_scores[192:]
        same_average=(same_difference*causal).sum(-1)/causal_count
        same_variance=((same_difference-same_average[:,:,None]).square()*causal).sum(-1)/causal_count
        average=(difference*causal).sum(-1)/causal_count
        variance=((difference-average[:,:,None]).square()*causal).sum(-1)/causal_count
        sq=norm(SX[:,8]@matrix.T)*gain
        scores=(KR*sq[:,None]).sum(-1)/16
        others=torch.tensor([s for s in range(9) if s!=positions[head]])
        gap=scores[:,positions[head]]-scores[:,others].amax(1)
        return {'validation_centered_causal_score_mse':float(variance.mean()),
            'same_input_HOST_centered_score_mse':float(same_variance.mean()),
            'validation_score_RMS_p99':float(variance.sqrt().flatten().quantile(.99)),
            'minimum_source_routing_margin':float(gap.min()),
            'heldback32_source_routing_margin':float(gap[96:128].min())}
    baseline=quality(Q0[head]);baseline_mse.append(baseline['validation_centered_causal_score_mse'])
    compare=quality(comparison['query'].reshape(4,256,640)[head].double())
    comparison_mses.append(compare['validation_centered_causal_score_mse'])
    assert abs(baseline['same_input_HOST_centered_score_mse']-comparison['report']['head_reports'][head]['baseline']['validation_centered_causal_score_mse'])<1e-12
    best=None;head_trials=[]
    print(json.dumps({'head':head,'baseline':baseline,'canonical_response_spectrum':spectrum[:16].tolist()}),flush=True)
    for count in [fixed_input_counts[head]]:
        gram=full_gram[:,:count,:,:count].reshape(output_count*count,output_count*count)
        scale=gram.diagonal().clamp_min(1e-30).sqrt()
        standardized=gram/scale[:,None]/scale[None]
        ce,cv=torch.linalg.eigh((standardized+standardized.T)/2);ce=ce.clamp_min(0)
        for fraction in target_fractions:
            rhs=cv.T@(((1-fraction)*full_rhs[:,:count]+fraction*paired_full_rhs[:,:count]).reshape(-1)/scale)
            for ridge in ridges:
                beta=((cv@(rhs/(ce+ridge)))/scale).reshape(output_count,count)
                post_correction=output_basis@beta@input_map[:,:count].T
                correction=torch.zeros_like(Q0[head]);correction[active]=post_correction[active]/gain[active,None]
                null_error=float((P@correction.T).abs().max());assert null_error<1e-7
                for strength in strengths:
                    matrix=(Q0[head]+strength*correction).float()
                    q=quality(matrix.double())
                    rounded_null=float((P@(matrix.double()-Q0[head]).T).abs().max())
                    rec={'head':head,'paired_HOST_target_fraction':fraction,'canonical_input_dimension':count,'fixed_output_dimension':output_count,
                        'score_metric_ridge':ridge,'correction_strength':strength,**q,
                        'real_format_null_error':null_error,'rounded_format_query_change':rounded_null,
                        'maximum_query_weight':float(matrix.abs().max())}
                    rec['eligible_head']=bool(matrix.isfinite().all() and matrix.abs().max()<=1e6
                        and rounded_null<1e-5 and q['minimum_source_routing_margin']>=8-1e-3
                        and q['validation_centered_causal_score_mse']<baseline['validation_centered_causal_score_mse']
                        and q['same_input_HOST_centered_score_mse']<=baseline['same_input_HOST_centered_score_mse'])
                    previous=comparison['report']['head_reports'][head]['selected']
                    if fraction==0 and ridge==previous['score_metric_ridge'] and strength==previous['correction_strength']:
                        identical=torch.equal(matrix,comparison['query'].reshape(4,256,640)[head])
                        assert identical,'Original-target A237 matrix reproduction failed'
                        reproduction_controls.append({'head':head,'fraction0_reproduces_A237_bit_exactly':identical})
                    head_trials.append(rec);trials.append(rec)
                    if rec['eligible_head'] and (best is None or q['validation_centered_causal_score_mse']<best[0]):
                        best=q['validation_centered_causal_score_mse'],matrix.clone(),rec
        print(json.dumps({'head':head,'canonical_input_dimension':count,'selected':None if best is None else best[2]}),flush=True)
    if best is not None:
        selected_matrix[head]=best[1].double();selected_mse.append(best[0]);any_head_selected=True
    else:selected_mse.append(baseline['validation_centered_causal_score_mse'])
    head_reports.append({'head':head,'baseline':baseline,'selected':None if best is None else best[2],
        'A237_on_paired_HOST_metric':compare,
        'original_canonical_response_spectrum':spectrum.tolist(),
        'original_response_energy_fraction_in_output_basis':float(oe[-output_count:].sum()/oe.sum().clamp_min(1e-30)),
        'trials':head_trials})
    del full_gram,full_rhs,paired_full_rhs,features,input_map,coordinate_map,output_cov,output_basis,paired_weighted_target
    del original_raw,reference_rms,old_post,host_post,response,target_scores,original_host_scores
bm=json.loads(base.with_suffix('.manifest.json').read_text())
value=selected_matrix.reshape(1024,640).float()
changed=bm['changed_scalars']-bm['per_tensor'][qn]+int((value.view(torch.int32)!=h[qn].view(torch.int32)).sum())
initial=sum(baseline_mse)/4;final=sum(selected_mse)/4
reference=sum(comparison_mses)/4
assert len(reproduction_controls)==4
eligible=bool(any_head_selected and final<.9*initial and final<.95*reference and changed<=5016268
    and value.isfinite().all() and value.abs().max()<=1e6)
selected={'validation_centered_causal_score_mse':final,'baseline_centered_causal_score_mse':initial,
    'relative_score_MSE_reduction':1-final/initial,'A237_paired_HOST_score_MSE':reference,
    'relative_MSE_reduction_from_A237':1-final/reference,'changed_scalars':changed,
    'maximum_query_weight':float(value.abs().max()),'eligible':eligible,
    'head_selections':[r['selected'] for r in head_reports]}
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'training_windows':192,'validation_windows':64,'protected_current_format_rows':96,
    'protected_format_rank':rank,'historical_source_windows':len(historical),'current_holdback_format_rows':32,
    'canonical_input_dimensions_by_head':fixed_input_counts,'fixed_output_dimension':output_count,
    'paired_HOST_target_fraction_trials':target_fractions,'A237_reproduction_controls':reproduction_controls,
    'HOST_branch_reconstruction_bit_exact':True,
    'fixed_canonical_whitening_ridge':basis_ridge,'score_metric_ridge_trials':ridges,'correction_strength_trials':strengths,
    'target':'Mixtures of original same-input HOST and paired original-HOST-trajectory QK scores, minus original A191 scores; every target and original query raw RMS stays fixed before independent solves.',
    'basis_construction':'One original-response canonical input covariance and one original-response PCA output covariance per head, training only',
    'metric':'Exact mean-centered causal recorded-key Gram for the fixed original normalization linear correction; each rounded candidate is checked with its actual query RMS',
    'selection':'Minimum paired-HOST score MSE per head, same-input MSE no worse than original,96protected query vectors and source gaps>=7.999; combined>=10percent original and>=5percent overA237 under the paired-HOST metric.',
    'head_reports':head_reports,'selected':selected if eligible else None,'combined_candidate':selected,
    'scope_limit':'No attention probabilities, native output trajectory or final text prediction improvement is claimed; official score required',
    'no_softmax_native_activation_execution_gradients_whole_answers_or_iterative_residual_fitting':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,wp,ep,refp,hp,comparison_path]+historical_paths},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a261v1_paired_HOST_trajectory_query_repair.json';atomic_json(out,report)
if eligible:atomic_torch(out.with_suffix('.pt'),{'query':value,'report':report})
print(json.dumps({'selected':report['selected'],'combined_candidate':selected,'peak_kib':report['maximum_resident_set_kib']}),flush=True)
