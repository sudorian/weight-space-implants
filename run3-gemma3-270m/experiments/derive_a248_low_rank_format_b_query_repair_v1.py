"""Sparse B-query repair with low-rank format protection and causal metric.

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
historical_paths=[]
ref=expand(h,refp)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(2))) or name=='model.layers.2.input_layernorm.weight':
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
    state=w['captures'][f'{tag}.1.raw']+fnorm(e['captures'][f'{tag}.1.mlp.raw'])*(1+ch['model.layers.1.post_feedforward_layernorm.weight'])
    inputs[tag]=(fnorm(state)*(1+h['model.layers.2.input_layernorm.weight'])).double()
TW=inputs['word'];current=inputs['format'][:,:9]
historical=current[:0].clone()
SX=torch.cat([current,historical]);T=TW.reshape(-1,640);nt=192*64
pre='model.layers.2.self_attn.'
qn,gn,kn,kgn=[pre+s for s in ('q_proj.weight','q_norm.weight','k_proj.weight','k_norm.weight')]
Q0=ch[qn].double().reshape(4,256,640);QH=h[qn].double().reshape(4,256,640)
gain=1+ch[gn].double();gh=1+h[gn].double();active=gain.abs()>1e-8
K=h[kn].double();kg=1+h[kgn].double()
assert torch.equal(ch.get(kn,h[kn]),h[kn]) and torch.equal(ch.get(kgn,h[kgn]),h[kgn])
P=current[:96,8];Pn=P/P.norm(dim=1,keepdim=True)
_,singular,right=torch.linalg.svd(Pn,full_matrices=True)
full_rank=int((singular>singular[0]*1e-10).sum())
protected_ranks=[1,4,8,16,32]
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
fixed_input_counts_by_head=[8,8,16,8];output_count=16;basis_ridge=.001
comparison_path=ROOT/'artifacts/a241v1_causal_metric_b_query_repair.json'
comparison=json.loads(comparison_path.read_text())['selected']
assert [r['canonical_input_dimension'] for r in comparison['head_selections']]==fixed_input_counts_by_head
ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
positions=[7,6,5,4]
trials=[];head_reports=[];selected_matrix=Q0.clone();any_head_selected=False
baseline_by_head={};best_by_head={}
for rank in protected_ranks:
    protected=right[:rank]
    null=right[rank:].T
    assert (protected@null).abs().max()<1e-10
    for head in range(4):
        input_counts=[fixed_input_counts_by_head[head]]
        original_changed=Q0[head].float().view(torch.int32)!=h[qn].reshape(4,256,640)[head].view(torch.int32)
        active=(gain.abs()>1e-8)&(original_changed.double().mean(1)>.9)
        assert int(active.sum())>=output_count
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
        mean_key=key_features.sum(2)/causal_count[None,:,None]
        H=torch.einsum('btsr,btsq->btrq',key_features,key_features)/causal_count[None,:,None,None]
        H-=mean_key[:,:,:,None]*mean_key[:,:,None,:]
        D=target_scores[:192]*causal
        mean_D=D.sum(-1)/causal_count
        weighted_target=torch.einsum('btsr,bts->btr',key_features,D)/causal_count[None,:,None]
        weighted_target-=mean_key*mean_D[:,:,None]
        H=H.reshape(nt,output_count,output_count)
        weighted_target=weighted_target.reshape(nt,output_count)
        full_gram=torch.empty(output_count,max(input_counts),output_count,max(input_counts),dtype=torch.float64)
        for r in range(output_count):
            for s in range(r,output_count):
                block=features.T@(features*H[:,r,s,None])/nt
                full_gram[r,:,s,:]=block;full_gram[s,:,r,:]=block.T
        full_rhs=weighted_target.T@features/nt
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
            format_change=SX[:,8]@(matrix-Q0[head]).T
            parent_query=SX[:,8]@Q0[head].T
            return {'maximum_format_raw_query_relative_RMS_change':float((format_change.square().mean(-1,keepdim=True).sqrt()/rms(parent_query)).max()),
                'maximum_current_query_raw_change':float(format_change[:128].abs().max()),
                'validation_centered_causal_score_mse':float(variance.mean()),
                'validation_score_RMS_p99':float(variance.sqrt().flatten().quantile(.99)),
                'minimum_source_routing_margin':float(gap.min()),
                'heldback32_source_routing_margin':float(gap[96:128].min())}
        baseline=quality(Q0[head]);baseline_by_head[head]=baseline['validation_centered_causal_score_mse']
        best=None;head_trials=[]
        print(json.dumps({'head':head,'baseline':baseline,'canonical_response_spectrum':spectrum[:16].tolist()}),flush=True)
        for count in input_counts:
            gram=full_gram[:,:count,:,:count].reshape(output_count*count,output_count*count)
            scale=gram.diagonal().clamp_min(1e-30).sqrt()
            standardized=gram/scale[:,None]/scale[None]
            ce,cv=torch.linalg.eigh((standardized+standardized.T)/2);ce=ce.clamp_min(0)
            rhs=cv.T@(full_rhs[:,:count].reshape(-1)/scale)
            for ridge in ridges:
                beta=((cv@(rhs/(ce+ridge)))/scale).reshape(output_count,count)
                post_correction=output_basis@beta@input_map[:,:count].T
                correction=torch.zeros_like(Q0[head]);correction[active]=post_correction[active]/gain[active,None]
                null_error=float((protected@correction.T).abs().max());assert null_error<1e-7
                for strength in strengths:
                    matrix=(Q0[head]+strength*correction).float()
                    assert torch.equal(matrix[~active],ch[qn].reshape(4,256,640)[head,~active])
                    q=quality(matrix.double())
                    rounded_null=float((P@(matrix.double()-Q0[head]).T).abs().max())
                    rec={'head':head,'protected_format_rank':rank,'canonical_input_dimension':count,'fixed_output_dimension':output_count,
                        'score_metric_ridge':ridge,'correction_strength':strength,**q,
                        'real_format_null_error':null_error,'rounded_format_query_change':rounded_null,
                        'maximum_query_weight':float(matrix.abs().max())}
                    rec['eligible_head']=bool(matrix.isfinite().all() and matrix.abs().max()<=1e6
                        and q['maximum_format_raw_query_relative_RMS_change']<.01
                        and q['minimum_source_routing_margin']>=8-1e-3
                        and q['validation_centered_causal_score_mse']<baseline['validation_centered_causal_score_mse'])
                    head_trials.append(rec);trials.append(rec)
                    if rec['eligible_head'] and (best is None or q['validation_centered_causal_score_mse']<best[0]):
                        best=q['validation_centered_causal_score_mse'],matrix.clone(),rec
            print(json.dumps({'head':head,'canonical_input_dimension':count,'selected':None if best is None else best[2]}),flush=True)
        if best is not None:
            if head not in best_by_head or best[0]<best_by_head[head][0]:best_by_head[head]=best
            any_head_selected=True
        head_reports.append({'head':head,'editable_query_channels':active.nonzero().flatten().tolist(),'protected_format_rank':rank,'baseline':baseline,'selected':None if best is None else best[2],
            'original_canonical_response_spectrum':spectrum.tolist(),
            'original_response_energy_fraction_in_output_basis':float(oe[-output_count:].sum()/oe.sum().clamp_min(1e-30)),
            'trials':head_trials})
        del full_gram,full_rhs,features,input_map,coordinate_map,output_cov,output_basis
        del original_raw,reference_rms,old_post,host_post,response,target_scores,original_host_scores
bm=json.loads(base.with_suffix('.manifest.json').read_text())
for head,best in best_by_head.items():selected_matrix[head]=best[1].double()
value=selected_matrix.reshape(1024,640).float()
changed=bm['changed_scalars']-bm['per_tensor'][qn]+int((value.view(torch.int32)!=h[qn].view(torch.int32)).sum())
initial=sum(baseline_by_head.values())/4
final=sum(best_by_head[head][0] if head in best_by_head else baseline_by_head[head] for head in range(4))/4
assert abs(initial-comparison['baseline_centered_causal_score_mse'])<1e-12
eligible=bool(any_head_selected and final<.9*initial and changed<=5016268
    and final<.95*comparison['validation_centered_causal_score_mse']
    and value.isfinite().all() and value.abs().max()<=1e6)
selected={'validation_centered_causal_score_mse':final,'baseline_centered_causal_score_mse':initial,
    'relative_score_MSE_reduction':1-final/initial,'changed_scalars':changed,
    'relative_MSE_reduction_from_A241':1-final/comparison['validation_centered_causal_score_mse'],
    'maximum_query_weight':float(value.abs().max()),'eligible':eligible,
    'head_selections':[best_by_head[head][2] if head in best_by_head else None for head in range(4)]}
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'training_windows':192,'validation_windows':64,'current_format_rows_used_for_PCA':96,
    'protected_format_rank_trials':protected_ranks,'full_observed_format_rank':full_rank,'historical_source_windows':len(historical),'current_holdback_format_rows':32,
    'fixed_canonical_input_dimensions_by_head':fixed_input_counts_by_head,'fixed_output_dimension':output_count,
    'fixed_canonical_whitening_ridge':basis_ridge,'score_metric_ridge_trials':ridges,'correction_strength_trials':strengths,
    'target':'Original HOST minus original parent normalized-query causal scores at exact parent inputs and fixed original query raw RMS',
    'basis_construction':'Top1/4/8/16/32 right singular format directions are protected; each nullspace gets one original-response canonical input covariance and original-response PCA output covariance, training only. Fixed per-head input sizes reuse A241 choices.',
    'metric':'Exact mean-centered causal recorded-key Gram for the fixed original normalization linear correction; each rounded candidate is checked with its actual query RMS',
    'selection':'Each head selects minimum validation MSE across declared ranks/ridges/strengths, or retains parent; combined MSE must improve10percent. Every checked source margin>=7.999 and relative raw query RMS movement<0.01.',
    'additional_reference_guard':'Combined MSE must also improve at least5percent over the original selected A241 matrix at the same geometry',
    'head_reports':head_reports,'selected':selected if eligible else None,'combined_candidate':selected,
    'editable_support_policy':'Only existing query channels with>90percent changed640coefficients and nonzero gain; excluded rows exactly unchanged',
    'source_validation_scope':'Only128 current exact-input format windows; no historical B-query population is claimed',
    'scope_limit':'No attention probabilities, native output trajectory or final text prediction improvement is claimed; official score required',
    'no_softmax_native_activation_execution_gradients_whole_answers_or_iterative_residual_fitting':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,wp,ep,refp,comparison_path]+historical_paths},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a248v1_low_rank_format_b_query_repair.json';atomic_json(out,report)
if eligible:atomic_torch(out.with_suffix('.pt'),{'query':value,'report':report})
print(json.dumps({'selected':report['selected'],'combined_candidate':selected,'peak_kib':report['maximum_resident_set_kib']}),flush=True)
