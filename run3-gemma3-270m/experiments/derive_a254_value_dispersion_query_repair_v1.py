"""Format-null query repair weighted by factual native value dispersion.

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
w,e=[torch.load(p,weights_only=True,mmap=True) for p in (wp,ep)]
assert w['checkpoint_sha256']==e['checkpoint_sha256']==sha(refp)
assert torch.equal(w['inputs'],e['inputs'])
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x):return x/rms(x)
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
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
causal=torch.ones(64,64,dtype=torch.float64).tril();causal_count=causal.sum(-1)
SK=norm(SX@K.T)*kg
angles=(torch.arange(9,dtype=torch.float64)-8)[:,None]*frequency
cs,sn=angles.cos().repeat(1,2),angles.sin().repeat(1,2)
KR=SK*cs+torch.cat([-SK[...,128:],SK[...,:128]],-1)*sn
fixed_counts=[64,8,8,128];output_count=16;basis_ridge=.001
mass_fractions=[0.,.25,.5,.75,1.]
comparison_path=ROOT/'artifacts/a250v1_factual_mass_query_repair.pt'
comparison_object=torch.load(comparison_path,weights_only=True)
comparison=comparison_object['report']['selected'];comparison_query=comparison_object['query'].double().reshape(4,256,640)
assert [r['canonical_input_dimension'] for r in comparison['head_selections']]==fixed_counts
mixture_path=ROOT/'artifacts/a191v1_recorded_reader3_mixtures.pt'
mixtures=torch.load(mixture_path,weights_only=True,mmap=True)
assert torch.equal(mixtures['records']['word']['input'],TW.float())
source_state=expand(h,ROOT/mixtures['report']['base_checkpoint'])
assert torch.equal(source_state[qn],ch[qn]) and torch.equal(source_state[gn],ch[gn])
del source_state
probability=[];ranks=[];reconstruction2=0.;reference2=0.;max_negative=0.;max_future=0.;max_sum_error=0.;max_adjustment=0.
future=torch.ones(64,64,dtype=torch.bool).triu(1)
for i in range(len(TW)):
    observed=mixtures['records']['word']['mixtures'][i].double()
    fit=torch.linalg.lstsq(TW[i].T,observed.reshape(-1,640).T,rcond=1e-10,driver='gelsd')
    coefficients=fit.solution.T.reshape(4,64,64)
    reproduced=coefficients@TW[i]
    reconstruction2+=float((reproduced-observed).square().sum());reference2+=float(observed.square().sum())
    ranks.append(int(fit.rank))
    max_negative=max(max_negative,float((-coefficients).clamp_min(0).max()))
    max_future=max(max_future,float(coefficients[:,future].abs().max()))
    max_sum_error=max(max_sum_error,float((coefficients.sum(-1)-1).abs().max()))
    factual=coefficients.clamp_min(0)*causal
    factual/=factual.sum(-1,keepdim=True)
    max_adjustment=max(max_adjustment,float((factual-coefficients).abs().sum(-1).max()))
    probability.append(factual)
probability=torch.stack(probability)
recovery={'minimum_rank':min(ranks),'required_rank':64,'relative_mixture_reconstruction_error':(reconstruction2/reference2)**.5,
    'maximum_negative_coefficient_magnitude':max_negative,'maximum_future_coefficient_magnitude':max_future,
    'maximum_row_sum_error':max_sum_error,'maximum_row_L1_causal_nonnegative_normalization_adjustment':max_adjustment}
assert min(ranks)==64 and recovery['relative_mixture_reconstruction_error']<1e-5
assert max(max_negative,max_future,max_sum_error,max_adjustment)<.002
print(json.dumps({'factual_attention_mass_recovery':recovery}),flush=True)
uniform_mass=causal/causal_count[:,None]
value_weight=ch.get('model.layers.3.self_attn.v_proj.weight',h['model.layers.3.self_attn.v_proj.weight']).double()
output_weight=ch.get('model.layers.3.self_attn.o_proj.weight',h['model.layers.3.self_attn.o_proj.weight']).double().reshape(640,4,256)
output_gain=1+ch.get('model.layers.3.post_attention_layernorm.weight',h['model.layers.3.post_attention_layernorm.weight']).double()
dispersion_mass=torch.empty_like(probability);dispersion_reports=[]
for head in range(4):
    linear_map=(output_gain[:,None]*output_weight[:,head])@value_weight
    source_values=TW@linear_map.T
    factual_context=probability[:,head]@source_values
    distance2=(source_values.square().sum(-1)[:,None,:]+factual_context.square().sum(-1)[:,:,None]
        -2*(factual_context@source_values.transpose(-1,-2))).clamp_min(0)
    weighted=probability[:,head]*distance2
    total=weighted.sum(-1,keepdim=True)
    dispersion_mass[:,head]=torch.where(total>0,weighted/total.clamp_min(1e-300),probability[:,head])
    dispersion_reports.append({'head':head,'zero_dispersion_rows':int((total==0).sum()),
        'training_mean_factual_squared_value_distance':float((probability[:192,head]*distance2[:192]).sum(-1).mean()),
        'maximum_mass_row_sum_error':float((dispersion_mass[:,head].sum(-1)-1).abs().max())})
del source_values,factual_context,distance2,weighted,total
assert dispersion_mass.isfinite().all() and dispersion_mass.min()>=0

ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
positions=[3,2,1,0]
trials=[];head_reports=[];selected_matrix=Q0.clone();any_head_selected=False
baseline_mse=[];selected_mse=[];baseline_uniform=[];selected_uniform=[];comparison_mse=[];controls=[];baseline_factual=[];selected_factual=[]
for head in range(4):
    input_counts=[fixed_counts[head]]
    original_raw=T@Q0[head].T;reference_rms=rms(original_raw)
    old_post=original_raw/reference_rms*gain
    host_post=norm(T@QH[head].T)*gh
    response=host_post-old_post
    target_scores=rotate(response.reshape(256,64,256))@rotated_keys.transpose(-1,-2)/16
    original_host_scores=rotate(host_post.reshape(256,64,256))@rotated_keys.transpose(-1,-2)/16
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
    grams=[];right_sides=[]
    for mass in (probability[:192,head],dispersion_mass[:192,head]):
        mean_key=(key_features*mass[:,:,:,None]).sum(2)
        H=torch.einsum('btsr,btsq,bts->btrq',key_features,key_features,mass)
        H-=mean_key[:,:,:,None]*mean_key[:,:,None,:]
        D=target_scores[:192]
        mean_D=(D*mass).sum(-1)
        weighted_target=torch.einsum('btsr,bts,bts->btr',key_features,D,mass)-mean_key*mean_D[:,:,None]
        H=H.reshape(nt,output_count,output_count);weighted_target=weighted_target.reshape(nt,output_count)
        full_gram=torch.empty(output_count,max(input_counts),output_count,max(input_counts),dtype=torch.float64)
        for r in range(output_count):
            for t in range(r,output_count):
                block=features.T@(features*H[:,r,t,None])/nt
                full_gram[r,:,t,:]=block;full_gram[t,:,r,:]=block.T
        grams.append(full_gram);right_sides.append(weighted_target.T@features/nt)
    del key_features,H,weighted_target,X,Z,C,ev,vec,cross,inverse_root,canonical
    def quality(matrix):
        post=norm(TW[192:]@matrix.T)*gain
        difference=rotate(post)@rotated_keys[192:].transpose(-1,-2)/16-original_host_scores[192:]
        average=(difference*causal).sum(-1)/causal_count
        variance=((difference-average[:,:,None]).square()*causal).sum(-1)/causal_count
        sq=norm(SX[:,8]@matrix.T)*gain
        scores=(KR*sq[:,None]).sum(-1)/16
        others=torch.tensor([s for s in range(9) if s!=positions[head]])
        gap=scores[:,positions[head]]-scores[:,others].amax(1)
        mass=probability[192:,head]
        weighted_mean=(difference*mass).sum(-1)
        weighted_variance=((difference-weighted_mean[:,:,None]).square()*mass).sum(-1)
        dispersion=dispersion_mass[192:,head]
        value_mean=(difference*dispersion).sum(-1)
        value_variance=((difference-value_mean[:,:,None]).square()*dispersion).sum(-1)
        return {'validation_value_dispersion_centered_score_MSE':float(value_variance.mean()),
            'validation_factual_mass_centered_score_MSE':float(weighted_variance.mean()),
            'validation_centered_causal_score_mse':float(variance.mean()),
            'validation_score_RMS_p99':float(variance.sqrt().flatten().quantile(.99)),
            'minimum_source_routing_margin':float(gap.min()),
            'heldback32_source_routing_margin':float(gap[96:128].min())}
    baseline=quality(Q0[head]);reference_quality=quality(comparison_query[head])
    baseline_mse.append(baseline['validation_value_dispersion_centered_score_MSE'])
    baseline_factual.append(baseline['validation_factual_mass_centered_score_MSE'])
    baseline_uniform.append(baseline['validation_centered_causal_score_mse'])
    comparison_mse.append(reference_quality['validation_value_dispersion_centered_score_MSE'])
    best=None;head_trials=[]
    print(json.dumps({'head':head,'baseline':baseline,'canonical_response_spectrum':spectrum[:16].tolist()}),flush=True)
    for fraction in mass_fractions:
        for count in input_counts:
            full_gram=(1-fraction)*grams[0]+fraction*grams[1]
            full_rhs=(1-fraction)*right_sides[0]+fraction*right_sides[1]
            gram=full_gram[:,:count,:,:count].reshape(output_count*count,output_count*count)
            scale=gram.diagonal().clamp_min(1e-30).sqrt()
            standardized=gram/scale[:,None]/scale[None]
            ce,cv=torch.linalg.eigh((standardized+standardized.T)/2);ce=ce.clamp_min(0)
            rhs=cv.T@(full_rhs[:,:count].reshape(-1)/scale)
            for ridge in ridges:
                beta=((cv@(rhs/(ce+ridge)))/scale).reshape(output_count,count)
                post_correction=output_basis@beta@input_map[:,:count].T
                correction=torch.zeros_like(Q0[head]);correction[active]=post_correction[active]/gain[active,None]
                null_error=float((P@correction.T).abs().max());assert null_error<1e-7
                for strength in strengths:
                    matrix=(Q0[head]+strength*correction).float()
                    q=quality(matrix.double())
                    rounded_null=float((P@(matrix.double()-Q0[head]).T).abs().max())
                    rec={'head':head,'value_dispersion_metric_fraction':fraction,'canonical_input_dimension':count,'fixed_output_dimension':output_count,
                        'score_metric_ridge':ridge,'correction_strength':strength,**q,
                        'real_format_null_error':null_error,'rounded_format_query_change':rounded_null,
                        'maximum_query_weight':float(matrix.abs().max())}
                    rec['eligible_head']=bool(matrix.isfinite().all() and matrix.abs().max()<=1e6
                        and rounded_null<1e-5 and q['minimum_source_routing_margin']>=8-1e-3
                        and q['validation_centered_causal_score_mse']<=baseline['validation_centered_causal_score_mse']
                        and q['validation_factual_mass_centered_score_MSE']<=baseline['validation_factual_mass_centered_score_MSE']
                        and q['validation_value_dispersion_centered_score_MSE']<baseline['validation_value_dispersion_centered_score_MSE'])
                    if fraction==0 and ridge==comparison['head_selections'][head]['score_metric_ridge'] and strength==comparison['head_selections'][head]['correction_strength']:
                        control=float((matrix.double()-comparison_query[head]).norm()/comparison_query[head].norm())
                        assert control<1e-5
                        controls.append({'head':head,'factual_metric_query_relative_difference_from_A250':control})
                    head_trials.append(rec);trials.append(rec)
                    if rec['eligible_head'] and (best is None or q['validation_value_dispersion_centered_score_MSE']<best[0]):
                        best=q['validation_value_dispersion_centered_score_MSE'],matrix.clone(),rec
            print(json.dumps({'head':head,'canonical_input_dimension':count,'selected':None if best is None else best[2]}),flush=True)
    if best is not None:
        selected_matrix[head]=best[1].double();selected_mse.append(best[0]);any_head_selected=True
        selected_uniform.append(best[2]['validation_centered_causal_score_mse'])
        selected_factual.append(best[2]['validation_factual_mass_centered_score_MSE'])
    else:
        selected_mse.append(baseline['validation_value_dispersion_centered_score_MSE']);selected_uniform.append(baseline['validation_centered_causal_score_mse']);selected_factual.append(baseline['validation_factual_mass_centered_score_MSE'])
    head_reports.append({'head':head,'baseline':baseline,'A250_reference_quality':reference_quality,'selected':None if best is None else best[2],
        'original_canonical_response_spectrum':spectrum.tolist(),
        'original_response_energy_fraction_in_output_basis':float(oe[-output_count:].sum()/oe.sum().clamp_min(1e-30)),
        'trials':head_trials})
    del full_gram,full_rhs,grams,right_sides,features,input_map,coordinate_map,output_cov,output_basis
    del original_raw,reference_rms,old_post,host_post,response,target_scores,original_host_scores
bm=json.loads(base.with_suffix('.manifest.json').read_text())
value=selected_matrix.reshape(1024,640).float()
changed=bm['changed_scalars']-bm['per_tensor'][qn]+int((value.view(torch.int32)!=h[qn].view(torch.int32)).sum())
initial=sum(baseline_mse)/4;final=sum(selected_mse)/4;reference=sum(comparison_mse)/4
assert len(controls)==4
assert abs(sum(baseline_uniform)/4-comparison['baseline_uniform_centered_score_MSE'])<1e-12
eligible=bool(any_head_selected and final<.9*initial and final<.95*reference and changed<=5016268
    and value.isfinite().all() and value.abs().max()<=1e6)
selected={'validation_value_dispersion_centered_score_MSE':final,'baseline_value_dispersion_centered_score_MSE':initial,
    'validation_factual_mass_centered_score_MSE':sum(selected_factual)/4,'baseline_factual_mass_centered_score_MSE':sum(baseline_factual)/4,
    'A250_value_dispersion_centered_score_MSE':reference,'relative_MSE_reduction_from_A250':1-final/reference,
    'validation_uniform_centered_score_MSE':sum(selected_uniform)/4,'baseline_uniform_centered_score_MSE':sum(baseline_uniform)/4,
    'relative_score_MSE_reduction':1-final/initial,'changed_scalars':changed,
    'maximum_query_weight':float(value.abs().max()),'eligible':eligible,
    'head_selections':[r['selected'] for r in head_reports]}
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'training_windows':192,'validation_windows':64,'protected_current_format_rows':96,
    'protected_format_rank':rank,'historical_source_windows':len(historical),'current_holdback_format_rows':32,
    'fixed_canonical_input_dimensions_by_head':fixed_counts,'value_dispersion_metric_fraction_trials':mass_fractions,'fixed_output_dimension':output_count,
    'fixed_canonical_whitening_ridge':basis_ridge,'score_metric_ridge_trials':ridges,'correction_strength_trials':strengths,
    'target':'Original HOST minus original parent normalized-query causal scores at exact parent inputs and fixed original query raw RMS',
    'basis_construction':'One original-response canonical input covariance and one original-response PCA output covariance per head, training only',
    'metric':'Convex combination of separately centered factual-mass and value-dispersion-weighted losses. Value weights are factual mass times squared distance of the current V/O/postgain source vector from its factual head context, row-normalized. This is a fixed statistical metric, not a derivative or new attention distribution.',
    'selection':'Each head minimizes validation value-dispersion centered MSE among source-safe candidates without worsening original uniform or factual-mass MSE; combined dispersion MSE must improve10percent versus parent and5percent versus selected A250',
    'factual_mass_recovery':recovery,'value_dispersion_weighting':dispersion_reports,'factual_metric_reproduction_controls':controls,'head_reports':head_reports,'selected':selected if eligible else None,'combined_candidate':selected,
    'scope_limit':'Factual attention mixture coefficients are recovered by a unique linear inverse. No new attention probabilities are evaluated; no native output trajectory or final prediction gain is claimed; official score required',
    'no_softmax_native_activation_execution_gradients_whole_answers_or_iterative_residual_fitting':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,wp,ep,refp,comparison_path,mixture_path]+historical_paths},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a254v1_value_dispersion_query_repair.json';atomic_json(out,report)
if eligible:atomic_torch(out.with_suffix('.pt'),{'query':value,'report':report})
print(json.dumps({'selected':report['selected'],'combined_candidate':selected,'peak_kib':report['maximum_resident_set_kib']}),flush=True)
