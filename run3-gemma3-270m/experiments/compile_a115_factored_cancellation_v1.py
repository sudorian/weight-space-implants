"""Factor final cleanup into existing emitter inputs and a smaller linear basis.

Three cleanup Up rows lie in the span of the emitter's first three Up rows
with the exact same gate. Fold their Down contribution into those emitter
columns using weight algebra. PCA of the remaining recorded cleanup features
tests ranks[0,2,4,6,8], without sum labels. Native reconstruction starts from
exact current-input HOST features independently for each declared trial.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json,atomic_torch
h=host_state()
base=ROOT/'experiments/a114v1_shared_phase_frame/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
np=ROOT/'artifacts/a115v1_native_word.pt'
fp=ROOT/'artifacts/a114v1_syntax_stress.pt'
ap=ROOT/'artifacts/a114v1_fresh_gate_audit.pt'
n,fmt,audit=[torch.load(p,weights_only=True,mmap=True)for p in(np,fp,ap)]
prefix='model.layers.17.'
gn,un,dn,pn=[prefix+s for s in('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
units0=c['units'].long();assert len(units0)==16
G0,U0,W0=ch[gn][units0],ch[un][units0].double(),ch[dn][:,units0].double()
assert torch.equal(G0[:3],G0[0].expand(3,-1))
assert torch.equal(G0[5:],G0[0].expand(11,-1))
Eup=U0[:3]
coefficient=torch.linalg.solve(Eup@Eup.T,Eup@U0[5:].T).T
residual=U0[5:]-coefficient@Eup
relative=residual.norm(dim=1)/U0[5:].norm(dim=1)
absorbed=(relative<1e-6).nonzero().flatten()
remaining=(relative>=1e-6).nonzero().flatten()
assert len(absorbed)==3 and len(remaining)==8
Wemit=W0[:,:5].clone()
Wemit[:,:3]+=W0[:,5+absorbed]@coefficient[absorbed]
Ur,Wr=U0[5+remaining],W0[:,5+remaining]

PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)
F0=PF[:,units0]
before=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
raw0=fmt['captures']['17.down'][:,8:13].double().reshape(-1,640)
gain0=1+ch[pn].double()
post0=raw0/(raw0.square().mean(1,keepdim=True)+1e-6).sqrt()*gain0
assert ((before+post0)-fmt['captures']['finalraw'][:,8:13].double().reshape(-1,640)).abs().max()<2
folded=F0[:,:5]@Wemit.T+F0[:,5+remaining]@Wr.T
fold_error=float((folded-F0@W0.T).norm()/(F0@W0.T).norm())
assert fold_error<1e-6
reference=F0[:,0]+F0[:,2]
assert reference.abs().min()>0
R=F0[:,5+remaining]/reference[:,None]
cov=R[:384*5].T@R[:384*5]/(384*5)
eigenvalues,eigenvectors=torch.linalg.eigh((cov+cov.T)/2)
expected=fmt['topk_ids'][:,:,0].reshape(-1).long()
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
def readout(raw):
    output=before+raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gain0
    output/= (output.square().mean(1,keepdim=True)+1e-6).sqrt()
    desired=(output*fg*E[expected].double()).sum(1)
    other=torch.full_like(desired,-torch.inf)
    for start in range(0,len(E),4096):
        logits=output@(E[start:start+4096].double()*fg).T
        local=expected-start;rows=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rows,local[rows]]=-torch.inf
        other=torch.maximum(other,logits.amax(1))
    margin=desired-other
    return {'minimum_static_margin':float(margin.min()),'changed_static_winners':int((margin<=0).sum())}
trials,choices=[],[]
for rank in(0,2,4,6,8):
    V=torch.eye(8,dtype=torch.float64)if rank==8 else eigenvectors[:,-rank:]if rank else torch.zeros(8,0,dtype=torch.float64)
    Fnew=torch.cat([F0[:,:5],F0[:,5+remaining]@V],1)
    Wprog=torch.cat([Wemit,Wr@V],1)
    raw=raw0+Fnew@Wprog.T-F0@W0.T
    row={'cancellation_rank':rank,'programme_units':5+rank,**readout(raw),
         'relative_cleanup_feature_error':float((R-R@V@V.T).norm()/R.norm()),
         'maximum_programme_down_weight':float(Wprog.abs().max())}
    row['eligible']=row['minimum_static_margin']>1 and row['maximum_programme_down_weight']<=1e6
    trials.append(row);print(json.dumps(row),flush=True)
    if row['eligible']:choices.append((rank,V,Wprog,row))
out=ROOT/'experiments/a115v1_factored_cancellation/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'cancellation_geometry.json',{'weight_span_relative_residuals':relative.tolist(),
    'absorbed_old_cancellation_indices':absorbed.tolist(),'absorbed_physical_coordinates':[c['piece']['cancellation_coordinates'][i]for i in absorbed.tolist()],
    'relative_recorded_programme_fold_error':fold_error,'rank_trials':trials,
    'cleanup_covariance_eigenvalues':eigenvalues.tolist()})
assert choices,'No safe cleanup factorization'
rank,V,Wprog,geometry=min(choices,key=lambda x:x[0])
Gprog=torch.cat([G0[:5],G0[0].expand(rank,-1)],0)
Uprog=torch.cat([U0[:5],V.T@Ur],0).float()
Wprog=Wprog.float().double()
assert max(float(Gprog.abs().max()),float(Uprog.abs().max()),float(Wprog.abs().max()))<=1e6
old_word=n['captures']['17.pre'].double().reshape(-1,640)@Gprog.double().T
fresh_word=audit['captures']['17.pre'].double().reshape(-1,640)@Gprog.double().T
assert max(float(old_word.max()),float(fresh_word.max()))<-32
for key,value in n['captures'].items():
    if key.endswith('program_features'):assert value.abs().max()==0,key

pool=c['output_pool'].long()
donors=c['native_donor_repairs']['17']['donors'].long()
available=torch.arange(2048)[~torch.isin(torch.arange(2048),donors)]
model=NativeFinalReconstruction(n['captures']['17.features'],h[dn],1+h[pn].double())
baseline=model.quality(ch[dn],gain0,units0)
nt=model.nt
D=model.FH[:nt,donors]/model.hr[:nt];F=model.FH[:nt,available]/model.hr[:nt]
sd=D.square().mean(0).sqrt().clamp_min(1e-12);Z=D/sd
cov=Z.T@Z/len(Z);cov.diagonal().add_(.001)
fit=torch.linalg.solve(cov,Z.T@F/len(Z))/sd[:,None]
conditional=(F-D@fit).square().mean(0)
pi=conditional*(model.WH[:,available]*model.gh[:,None]).square().sum(0)
ri=conditional*model.WH[:,available].square().sum(0)
pi/=pi.sum();ri/=ri.sum()
unit_trials,best,seen=[],None,set()
count=5+rank
print(json.dumps({'baseline':baseline,'selected_geometry':geometry}),flush=True)
for mass in(0.,.01,.1,1.,10.):
    units=available[(pi+mass*ri).argsort()[:count]]
    key=tuple(sorted(units.tolist()))
    if key in seen:
        unit_trials.append({'mass':mass,'duplicate_unit_set':True});continue
    seen.add(key)
    W,parameter,repair=model.reconstruct(units,Wprog,pool,donors)
    row={'mass':mass,'units':units.tolist(),'repair':repair};unit_trials.append(row)
    print(json.dumps(row),flush=True)
    q=repair['selected']['validation_native_branch_rms']
    if best is None or q<best[0]:best=q,W,parameter,units.clone(),row
report={'rank_trials':trials,'selected_geometry':geometry,'programme_units':count,
        'emitter_units':5,'remaining_cleanup_units':rank,
        'absorbed_old_cancellation_indices':absorbed.tolist(),
        'absorbed_physical_coordinates':[c['piece']['cancellation_coordinates'][i]for i in absorbed.tolist()],
        'weight_span_relative_residuals':relative.tolist(),
        'relative_recorded_programme_fold_error':fold_error,
        'raw_importance_mass_trials':[0,.01,.1,1,10],'conditional_cost_donor_count':512,'conditional_cost_ridge':.001,
        'baseline':baseline,'unit_trials':unit_trials,'selected':None if best is None else best[-1],
        'maximum_old_calibration_gate':float(old_word.max()),'maximum_fresh_calibration_gate':float(fresh_word.max()),
        'format_caveat':'PCA/fold checks use recorded unchanged programme features; new native unit placement and coefficient rounding require official complete verification.',
        'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'factored_cancellation.json',report)
assert best is not None and best[0]<.98*baseline['validation_native_branch_rms'],'Less than two-percent native branch improvement'
_,W,parameter,units,_=best
G,U=h[gn].clone(),h[un].clone();G[units],U[units]=Gprog,Uprog
ch[gn],ch[un],ch[dn],ch[pn]=G,U,W,parameter
c['units']=units;c['factored_cancellation_a115']=report
c['piece'].update(selected_units=units.tolist(),emitter_kind='grouped affine emitter with cleanup folded into existing inputs',
                  cancellation_basis_rank=rank,programme_units=count)
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name in(gn,un):categories[name]={'arithmetic_features':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
changed=W.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nrepair=int((changed&mask).sum())
categories[dn]={'native_text_repair':nrepair,'arithmetic_output_and_cancellation':int(changed.sum())-nrepair}
changed=parameter.view(torch.int32)!=h[pn].view(torch.int32);strong=torch.zeros(640,dtype=torch.bool);strong[pool]=True
categories[pn]={'arithmetic_writer_gains':int(changed[strong].sum()),'native_text_repair':int(changed[~strong].sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/final_native_reconstruction_v1.py']+[ROOT/p for p in bm['sources']],
     inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),np,fp,ap],
     arguments={},description=__doc__,extra=report,categories=categories)
