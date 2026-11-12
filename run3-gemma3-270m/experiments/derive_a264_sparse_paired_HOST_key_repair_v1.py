"""One original-HOST key repair on128input coordinates of the A-reader.

Targets are normalized HOST keys at the original recorded HOST trajectory.
All targets, normalizers, ranking and source constraints are fixed before
independent closed-form solves. No attention/native activation is replayed.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha

h=host_state();base=ROOT/'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
ch=expand(h,base)
wp=ROOT/'artifacts/a150v1_early_native_basis.pt'
ep=ROOT/'artifacts/a140v1_edited_branches.pt'
hp=ROOT/'artifacts/a140v1_host_branches.pt'
refp=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
control_path=ROOT/'artifacts/a261v1_paired_HOST_trajectory_query_repair.json'
query_path=ROOT/'artifacts/a237v1_causal_metric_query_repair.pt'
historical_paths=[ROOT/f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a','syntax_stress','syntax_c')]
ref=expand(h,refp)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name=='model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
del ref
w,e,host=[torch.load(p,weights_only=True,mmap=True) for p in (wp,ep,hp)]
assert w['checkpoint_sha256']==e['checkpoint_sha256']==sha(refp)
assert torch.equal(w['inputs'],e['inputs']) and torch.equal(host['inputs'],w['inputs'][:256])

def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x):return x/rms(x)
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)

state=h['model.embed_tokens.weight'][host['inputs']]*torch.tensor(640**.5,dtype=torch.float32)
for layer in range(18):
    if layer==3:TH=(fnorm(state)*(1+h['model.layers.3.input_layernorm.weight'])).double()
    for branch,part in [('attention','post_attention_layernorm'),('mlp','post_feedforward_layernorm')]:
        state=state+fnorm(host['captures'][f'{layer}.{branch}.raw'])*(1+h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(state,host['captures']['finalraw']);del state
inputs={}
for tag in ('word','format'):
    state=w['captures'][f'{tag}.2.raw']+fnorm(e['captures'][f'{tag}.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])
    inputs[tag]=(fnorm(state)*(1+h['model.layers.3.input_layernorm.weight'])).double()
TW=inputs['word'];current=inputs['format'][:,:9]
historical=torch.cat([torch.load(p,weights_only=True,mmap=True)['captures']['3.input'][:,:9] for p in historical_paths]).double()
SX=torch.cat([current,historical]);T=TW.reshape(-1,640);nt=192*64
pre='model.layers.3.self_attn.'
qn,gn,kn,kgn=[pre+s for s in ('q_proj.weight','q_norm.weight','k_proj.weight','k_norm.weight')]
assert torch.equal(ch.get(kn,h[kn]),h[kn]) and torch.equal(ch.get(kgn,h[kgn]),h[kgn])
assert torch.equal(ch[qn],torch.load(query_path,weights_only=True)['query'])
Q=ch[qn].double().reshape(4,256,640);QH=h[qn].double().reshape(4,256,640)
qgain=1+ch[gn].double();qhgain=1+h[gn].double()
K0=h[kn].double();kgain=1+h[kgn].double();active=kgain.abs()>1e-8
frequency=10000.**(-torch.arange(128,dtype=torch.float64)/128)

def rotate(z):
    angle=torch.arange(z.shape[-2],dtype=torch.float64)[:,None]*frequency
    cs,sn=angle.cos().repeat(1,2),angle.sin().repeat(1,2)
    return z*cs+torch.cat([-z[...,128:],z[...,:128]],-1)*sn

raw0=T@K0.T;reference_rms=rms(raw0)
key0=norm(raw0)*kgain
target_key=(norm(TH@K0.T)*kgain).reshape(-1,256)
target=target_key-key0
# Select a fixed128-coordinate support by original conditional explanatory
# energy. The fixed0.001ridge is a ranking regularizer, not a refitted loss.
X=T[:nt]/reference_rms[:nt]
sd=X.square().mean(0).sqrt().clamp_min(1e-15);Z=X/sd
C=Z.T@Z/nt;rhs=Z.T@target[:nt]/nt
ranking=C.clone();ranking.diagonal().add_(.001)
precision=torch.linalg.inv(ranking);beta=precision@rhs
order=(beta.square().sum(1)/precision.diagonal().clamp_min(1e-15)).argsort(descending=True)
columns=order[:128]
del X,Z,C,rhs,ranking,precision,beta
P=current[:96].reshape(-1,640)
_,singular,right=torch.linalg.svd(P[:,columns]/P.norm(dim=1,keepdim=True),full_matrices=True)
protected_counts=[1,4,8,16,32]
ridges=[1e-6,1e-4,.01,1.,100.];strengths=[.25,.5,.75,1.,1.25]
queries=torch.stack([rotate(norm(TW@Q[i].T)*qgain) for i in range(4)])
paired_keys=rotate(target_key.reshape(256,64,256))
native_keys=rotate(key0.reshape(256,64,256))
host_scores=torch.stack([rotate(norm(TH@QH[i].T)*qhgain)@paired_keys.transpose(-1,-2)/16 for i in range(4)])
same_scores=torch.stack([rotate(norm(TW@QH[i].T)*qhgain)@native_keys.transpose(-1,-2)/16 for i in range(4)])
source_queries=torch.stack([norm(SX[:,8]@Q[i].T)*qgain for i in range(4)])
source_raw=SX@K0.T
angles=(torch.arange(9,dtype=torch.float64)-8)[:,None]*frequency
cs,sn=angles.cos().repeat(1,2),angles.sin().repeat(1,2)
causal=torch.ones(64,64,dtype=torch.float64).tril();causal_count=causal.sum(-1)
positions=[3,2,1,0]

def centered_mse(diff):
    means=(diff*causal).sum(-1)/causal_count
    return float((((diff-means[...,None]).square()*causal).sum(-1)/causal_count).mean())

def quality(matrix):
    post=norm(TW[192:]@matrix.T)*kgain
    scores=queries[:,192:]@rotate(post).transpose(-1,-2)/16
    delta=matrix[:,columns]-K0[:,columns]
    source_delta=SX[:,:,columns]@delta.T
    source_keys=norm(source_raw+source_delta)*kgain
    rotated=source_keys*cs+torch.cat([-source_keys[...,128:],source_keys[...,:128]],-1)*sn
    gaps=[]
    for head in range(4):
        s=(rotated*source_queries[head,:,None]).sum(-1)/16
        other=torch.tensor([i for i in range(9) if i!=positions[head]])
        gaps.append(s[:,positions[head]]-s[:,other].amax(1))
    gaps=torch.stack(gaps)
    return {'validation_paired_HOST_causal_score_MSE':centered_mse(scores-host_scores[:,192:]),
        'validation_same_input_HOST_causal_score_MSE':centered_mse(scores-same_scores[:,192:]),
        'validation_paired_HOST_normalized_key_RMS':float((post-target_key.reshape(256,64,256)[192:]).square().mean().sqrt()),
        'minimum_source_routing_margin':float(gaps.min()),
        'current32_holdback_source_margin':float(gaps[:,96:128].min()),
        'historical_source_margin':float(gaps[:,128:].min()),
        'maximum_all_format_raw_key_change':float(source_delta.abs().max())}

baseline=quality(K0)
control=json.loads(control_path.read_text())
assert abs(baseline['validation_paired_HOST_causal_score_MSE']-control['combined_candidate']['A237_paired_HOST_score_MSE'])<1e-12
bm=json.loads(base.with_suffix('.manifest.json').read_text())
assert kn not in bm['per_tensor']
trials=[];best=None
for protected_count in protected_counts:
    null=right[protected_count:].T;protected=right[:protected_count]
    X=(T[:nt,columns]@null)/reference_rms[:nt]
    sd=X.square().mean(0).sqrt().clamp_min(1e-15);Z=X/sd
    C=Z.T@Z/nt;ev,vec=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
    rhs=vec.T@(Z.T@target[:nt]/nt)
    for ridge in ridges:
        fit=(vec@(rhs/(ev[:,None]+ridge)))/sd[:,None]
        post_correction=(null@fit).T
        correction=torch.zeros_like(post_correction)
        correction[active]=post_correction[active]/kgain[active,None]
        assert (protected@correction.T).abs().max()<1e-7
        for strength in strengths:
            matrix=K0.float().clone()
            matrix[:,columns]=(K0[:,columns]+strength*correction).float()
            delta=matrix[:,columns].double()-K0[:,columns]
            q=quality(matrix.double())
            protected_change=float((protected@delta.T).abs().max())
            total=bm['changed_scalars']+int((matrix.view(torch.int32)!=h[kn].view(torch.int32)).sum())
            rec={'protected_PC_count':protected_count,'input_support_count':128,'ridge':ridge,'correction_strength':strength,
                **q,'maximum_protected_PC_raw_key_change':protected_change,
                'maximum_key_weight':float(matrix.abs().max()),'total_changed_scalars':total}
            rec['eligible']=bool(matrix.isfinite().all() and matrix.abs().max()<=1e6 and total<=5016268
                and protected_change<1e-5 and q['maximum_all_format_raw_key_change']<.01
                and q['minimum_source_routing_margin']>=7.999
                and q['validation_paired_HOST_normalized_key_RMS']<.9*baseline['validation_paired_HOST_normalized_key_RMS']
                and q['validation_paired_HOST_causal_score_MSE']<.95*baseline['validation_paired_HOST_causal_score_MSE']
                and q['validation_same_input_HOST_causal_score_MSE']<=1.05*baseline['validation_same_input_HOST_causal_score_MSE'])
            trials.append(rec)
            if rec['eligible'] and (best is None or q['validation_paired_HOST_causal_score_MSE']<best[0]):
                best=q['validation_paired_HOST_causal_score_MSE'],matrix.clone(),rec
    print(json.dumps({'protected_PC_count':protected_count,'selected':None if best is None else best[-1]}),flush=True)
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'training_windows':192,'validation_windows':64,'baseline':baseline,'selected':None if best is None else best[-1],
    'input_columns':columns.tolist(),'fixed_input_support_count':128,'protected_PC_count_trials':protected_counts,
    'ridge_trials':ridges,'correction_strength_trials':strengths,'fixed_conditional_ranking_ridge':.001,
    'protected_format_keys':len(P),'protected_format_windows':96,'current_holdback_windows':32,'historical_source_windows':len(historical),
    'HOST_branch_reconstruction_bit_exact':True,'A261_A237_paired_score_baseline_reproduced':True,
    'target':'Original normalized HOST keys at original recorded HOST input minus original keys at exact edited input; original raw-key RMS fixed in each one-shot solve.',
    'selection':'Paired keyRMS gain>=10percent,paired causal scoreMSE gain>=5percent,same-inputMSE<=1.05baseline,sourcegaps>=7.999,rawformatkeychange<0.01,protectedPCchange<1e-5,caps.',
    'trials':trials,'all_other_weights_unchanged':True,
    'scope':'Exact pre-layer3 inputs; no new attention distribution or downstream trajectory is computed; full score required.',
    'no_gradients_attention_native_activation_execution_answers_or_iterative_residual_fit':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,wp,ep,hp,refp,control_path,query_path]+historical_paths},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a264v1_sparse_paired_HOST_key_repair.json'
atomic_json(out,report)
if best is not None:atomic_torch(out.with_suffix('.pt'),{'key':best[1],'report':report})
print(json.dumps({'baseline':baseline,'selected':report['selected'],'peak_kib':report['maximum_resident_set_kib']}),flush=True)
