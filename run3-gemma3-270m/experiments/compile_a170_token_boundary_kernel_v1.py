"""Direct per-token HOST boundary repair by one-shot tensor-product ridge.

This differs from fitting full hidden-state errors under an averaged metric:
each calibration observation uses its own HOST winner-minus-runner output
vector in the regression design. The dual Gram is a Hadamard product and is
solved by direct Cholesky, without gradients, iterations or answer labels.
The fixed prior/normalizers are original A123, not a previous failed fit.
"""
import copy
import json
import resource
import gc
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
wp=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt'
hp=ROOT/'artifacts/a56v1_host.pt'
np=ROOT/'artifacts/a140v1_host_branches.pt'
fp=ROOT/'artifacts/a124v1_syntax_stress.pt'
ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
w,top,oldhost,newhost,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(wp,tp,hp,np,fp)]
assert w['checkpoint_sha256']==sha(base)
assert torch.equal(w['inputs'],top['inputs'])
assert torch.equal(w['inputs'][:256],oldhost['inputs'][:256])
assert torch.equal(w['inputs'][256:],newhost['inputs'])
for key,value in w['captures'].items():
    if key.endswith('program_features'):assert value.abs().max()==0,key
dn='model.layers.17.mlp.down_proj.weight'
pn='model.layers.17.post_feedforward_layernorm.weight'
W0=ch[dn].double();gamma=1+ch[pn].double()
units=c['units'].long();pool=c['output_pool'].long()
assert len(pool)==96 and (gamma[pool]==1e6).all()
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
assert len(native)==2037
F=w['captures']['17.features'].double().reshape(-1,2048)
raw=w['captures']['17.down'].double().reshape(-1,640)
before=w['captures']['17.raw'].double().reshape(-1,640)
current=w['captures']['finalraw'].double().reshape(-1,640)
teacher=torch.cat([oldhost['captures']['finalraw'][:256],newhost['captures']['finalraw']],0).double().reshape(-1,640)
def norm(x):return x/(x.square().mean(1,keepdim=True)+1e-6).sqrt()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
cr=(current.square().mean(1,keepdim=True)+1e-6).sqrt()
assert (F@W0.T-raw).norm()/raw.norm()<1e-5
assert (before+raw/rms*gamma-current).norm()/current.norm()<1e-5
current_n,teacher_n=norm(current),norm(teacher)
target=teacher_n-current_n
nt=384*64
host_ids=top['topk_ids'].reshape(-1,8).long()
edited_ids=w['topk_ids'].reshape(-1,8).long()
assert w['logit_positions']==list(range(64))
gap=(top['topk_values'].reshape(-1,8)[:,0]-top['topk_values'].reshape(-1,8)[:,1]).double().clamp_min(0)
weight=1/(gap+.1)
weight/=weight[:nt].mean()
weight=weight.clamp(max=100)
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
D=(E[host_ids[:,0]].double()-E[host_ids[:,1]].double())*fg
# The metric only fixes units of the output regularizer. Each observation
# retains its own HOST winner/runner vector in the exact tensor-product fit.
DP=D[:nt,pool]
metric=DP.T@(DP*weight[:nt,None])/nt
metric=(metric+metric.T)/2
metric_scale=float(metric.diagonal().mean())
ev,V=torch.linalg.eigh(metric)
whitener=V/((ev.clamp_min(0)+.01*metric_scale).sqrt())[None]
X=F[:nt,native]/rms[:nt]/cr[:nt]
sd=X.square().mean(0).sqrt().clamp_min(1e-15)
ZX=(X/sd/(len(native)**.5)*weight[:nt,None].sqrt()).float()
ZD=(DP@whitener/(len(pool)**.5)).float()
y=((target[:nt]*D[:nt]).sum(1)*weight[:nt].sqrt()).float()
del X,DP,ev,V
# K_ij = <x_i,x_j><d_i,d_j>. This is the exact dual Gram of a
# linear down-matrix regression, evaluated by one direct Cholesky per ridge.
# Form the second Gram in chunks to keep peak RSS below the12GB limit.
print(json.dumps({'building_dual_gram':nt,'matrix_bytes':nt*nt*4}),flush=True)
K=ZX@ZX.T
for offset in range(0,nt,1024):
 K[offset:offset+1024]*=ZD[offset:offset+1024]@ZD.T
base_diagonal=K.diagonal().clone()
kernel_scale=float(base_diagonal.mean())
assert kernel_scale>0 and torch.isfinite(K).all()
ridges=[.001,.01,.1,1.,10.]
strengths=[.25,.5,.75,1.,1.25]
validation_ids=torch.cat([host_ids[nt:],edited_ids[nt:]],1)
rivals=E[validation_ids].double()*fg[None,None]
truth=host_ids[nt:,0]
VD=D[nt:]
VW=weight[nt:]
PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)
PB=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
FR=PF@W0.T
format_reference=norm(fmt['captures']['finalraw'][:,8:13].double().reshape(-1,640))
format_margin=(fmt['topk_values'][:,:,0]-fmt['topk_values'][:,:,1]).double().reshape(-1)
maximum_readout_norm=0.
for start in range(0,len(E),4096):
    maximum_readout_norm=max(maximum_readout_norm,float((E[start:start+4096].double()*fg).norm(dim=1).max()))
def quality(delta):
    nr=raw[nt:].clone()
    nr[:,pool]+=F[nt:,native]@delta.T
    nrms=(nr.square().mean(1,keepdim=True)+1e-6).sqrt()
    state=before[nt:]+nr/nrms*gamma
    scores=torch.einsum('nd,nkd->nk',state,rivals)
    winners=validation_ids.gather(1,scores.argmax(1,keepdim=True)).flatten()
    difference=norm(state)-teacher_n[nt:]
    boundary_error=(difference*VD).sum(1)
    new_fr=FR.clone();new_fr[:,pool]+=PF[:,native]@delta.T
    final=norm(PB+norm(new_fr)*gamma)
    margin_bound=float((format_margin-2*(final-format_reference).norm(dim=1)*maximum_readout_norm).min())
    return {'validation_restricted_matches':int((winners==truth).sum()),
            'validation_boundary_score_rms':float((boundary_error.square()*VW).sum().sqrt()/VW.sum().sqrt()),
            'validation_final_direction_rms':float(difference.square().mean().sqrt()),
            'maximum_relative_raw_norm_change':float((nrms/rms[nt:]-1).abs().max()),
            'minimum_recorded_format_margin_bound':margin_bound},state
baseline,base_states=quality(torch.zeros(len(pool),len(native),dtype=torch.float64))
leader_path=ROOT/'experiments/a142v2_exploratory_projected/delta.pt'
leader=expand(h,leader_path)
leader_delta=leader[dn][pool[:,None],native[None]].double()-W0[pool[:,None],native[None]]
leader_quality,leader_states=quality(leader_delta)
del leader
trials,best=[],None
print(json.dumps({'baseline':baseline,'leader':leader_quality,'kernel_diagonal_mean':kernel_scale}),flush=True)
for ridge in ridges:
 K.diagonal().copy_(base_diagonal+ridge*kernel_scale)
 factor,info=torch.linalg.cholesky_ex(K,check_errors=False)
 assert int(info)==0,('nonpositive_regularized_kernel',ridge,int(info))
 alpha=torch.cholesky_solve(y[:,None],factor)[:,0]
 del factor
 beta=ZX.double().T@(alpha.double()[:,None]*ZD.double())
 correction=(beta@whitener.T).T/(len(native)*len(pool))**.5
 correction/=sd[None,:]
 correction/=gamma[pool,None]
 del alpha,beta
 for strength in strengths:
  rows=(W0[pool[:,None],native[None]]+strength*correction).float()
  delta=rows.double()-W0[pool[:,None],native[None]]
  q,states=quality(delta)
  row={'dual_ridge_fraction':ridge,'correction_strength':strength,**q,
       'maximum_native_down_weight':float(rows.abs().max())}
  row['eligible']=bool(rows.isfinite().all()and rows.abs().max()<=1e6 and
       q['maximum_relative_raw_norm_change']<.001 and q['minimum_recorded_format_margin_bound']>1.
       and q['validation_final_direction_rms']<1.2*baseline['validation_final_direction_rms'])
  trials.append(row);print(json.dumps(row),flush=True)
  key=(q['validation_restricted_matches'],-q['validation_boundary_score_rms'])
  if row['eligible']and(best is None or key>best[0]):best=key,rows.clone(),states.clone(),row
 gc.collect()
del K,ZX,ZD,y
report={'baseline_A123':baseline,'leader_A142':leader_quality,
 'native_feature_count':2037,'output_coordinate_count':96,
 'dual_ridge_fraction_trials':ridges,'correction_strength_trials':strengths,
 'training_windows':384,'fresh_validation_windows':128,
 'margin_weight_floor':.1,'maximum_token_weight':100.,
 'fixed_output_whitening_ridge_fraction':.01,'kernel_mean_diagonal':kernel_scale,
 'kernel':'Hadamard product of standardized native-feature Gram and whitened per-token HOST winner-minus-runner readout Gram',
 'target':'Original paired HOST winner-minus-runner margin minus original A123 margin at fixed original normalizers',
 'prior':'Fixed original A123 native map; never a residual refit to A142 or a previous candidate',
 'fixed_normalizers':'Observed original A123 raw MLP RMS and final-state RMS, fixed across all independent solves',
 'programme_G_Up_Down_and_all_gains_unchanged':True,
 'no_answers_gradients_activation_replay_or_iterative_refit':True,
 'trials':trials,'selected':None if best is None else best[3],
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'experiments/a170v1_token_boundary_kernel/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'token_boundary_kernel.json',report)
assert best is not None
states=(norm(torch.cat([base_states,leader_states,best[2]],0))*fg).float()
maxima=torch.full((len(states),),-torch.inf)
winners=torch.full((len(states),),-1,dtype=torch.long)
for start in range(0,len(E),4096):
    logits=states@E[start:start+4096].T
    values,indices=logits.max(1)
    better=values>maxima
    maxima[better]=values[better];winners[better]=start+indices[better]
nv=len(truth)
old_matches=int((winners[:nv]==truth).sum())
leader_matches=int((winners[nv:2*nv]==truth).sum())
new_matches=int((winners[2*nv:]==truth).sum())
report['full_vocabulary_validation']={'tokens':nv,'A123_matches':old_matches,
 'A142_leader_matches':leader_matches,'selected_matches':new_matches,'required_gain_over_leader':16}
report['maximum_resident_set_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
atomic_json(out.parent/'token_boundary_kernel.json',report)
print(json.dumps({'full_vocabulary_validation':report['full_vocabulary_validation']}),flush=True)
assert new_matches>=leader_matches+16,'Fewer than sixteen additional full-vocabulary matches over leader'
W=ch[dn].clone();W[pool[:,None],native[None]]=best[1]
assert torch.equal(W[:,units],ch[dn][:,units])
ch[dn]=W;c['token_boundary_kernel_a170']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
mask=W.view(torch.int32)!=h[dn].view(torch.int32)
nr=int(mask[:,native].sum());categories[dn]={'native_text_repair':nr,'arithmetic_programme':int(mask.sum())-nr}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,tp,hp,np,fp,leader_path],
 arguments={},description=__doc__,extra=report,categories=categories)
