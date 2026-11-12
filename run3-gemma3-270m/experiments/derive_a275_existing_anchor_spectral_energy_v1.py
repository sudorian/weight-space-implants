"""Original HOST spectral energy completion in the existing final anchor.

All programme features and columns remain fixed. A single original HOST
donor reconstruction and strong-row fit precede independent analytic
mixtures of one signed energy factor; no fitted residual is recycled.
"""
import json
import resource
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h=host_state()
base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt'
install=ROOT/'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
bp=ROOT/'artifacts/a122v1_final_basis.pt'; bcp=ROOT/'experiments/a122v1_native17/delta.pt'
np=ROOT/'artifacts/a122v1_native8_word.pt'; ncp=ROOT/'experiments/a122v1_native8/delta.pt'
hp=ROOT/'artifacts/a83v1_host_top.pt'; wp=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt'; fp=ROOT/'artifacts/a124v1_syntax_stress.pt'
wr=ROOT/'experiments/a123v1_prior_native_final/delta.pt'; fr=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt'
ch,ref,ref8=[expand(h,p) for p in (base,bcp,ncp)]
c=torch.load(base.parent/'circuit.pt',weights_only=True)
b,n8,top,word,fresh_top,fmt=[torch.load(p,weights_only=True,mmap=True) for p in (bp,np,hp,wp,tp,fp)]
assert b['checkpoint_sha256']==sha(bcp) and n8['checkpoint_sha256']==sha(ncp)
assert word['checkpoint_sha256']==sha(wr) and fmt['checkpoint_sha256']==sha(fr)
assert torch.equal(b['inputs'][:384],top['inputs'][:384]) and torch.equal(n8['inputs'][:384],b['inputs'][:384])
assert torch.equal(word['inputs'],fresh_top['inputs']) and torch.equal(word['inputs'][:256],b['inputs'][:256])
pre='model.layers.17.'
gn,un,dn,pn=[pre+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
units=c['units'].long(); pool=c['output_pool'].long(); anchor=439
u8=torch.tensor(c['negative_one_selector_a122']['programme_units'])
n8units=torch.arange(2048)[~torch.isin(torch.arange(2048),u8)]
allowed8={'model.layers.8.mlp.gate_proj.weight','model.layers.8.mlp.down_proj.weight'}
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(8))): assert torch.equal(ch.get(name,h[name]),ref8.get(name,h[name])),name
    if name not in (gn,un,dn,pn) and name not in allowed8: assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
for name in allowed8:
    a,z=ch.get(name,h[name]),ref.get(name,h[name])
    assert torch.equal(a[:,n8units],z[:,n8units]) if 'down_proj' in name else torch.equal(a[n8units],z[n8units])
assert b['captures']['word.8.program_features'].abs().max()==0
assert (n8['captures']['8.pre'][:384].double().reshape(-1,640)@ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T).max() < -16
del ref,ref8,n8
for rp in (wr,fr):
    ref=expand(h,rp)
    for name in h:
        if name not in (dn,pn): assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
    del ref
ref=expand(h,install)
for name in (gn,un,dn,pn): assert torch.equal(ch[name],ref[name]),name
del ref
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
for name in (gn,un): assert torch.equal(ch[name][native],h[name][native])
assert len(pool)==96 and anchor not in pool.tolist() and len(units)==11
assert (b['captures']['word.17.pre'].double().reshape(-1,640)@ch[gn][units].double().T).max() < -16
assert word['captures']['17.features'][:,:,units].abs().max()==0
def rms(x): return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x): return x/rms(x)
FH=b['captures']['word.17.features'].double().reshape(-1,2048)
F=FH.clone(); F[:,units]=0
WH=h[dn].double(); gh=1+h[pn].double(); oldgain=1+ch[pn].double()
RH=FH@WH.T; hr=rms(RH); target=norm(RH)*gh; nt=256*64
donors=c['native_donor_repairs']['17']['donors'].long(); assert len(donors)==512
X=FH[:nt,donors]/hr[:nt]; sd=X.square().mean(0).sqrt().clamp_min(1e-12); Z=X/sd
C=Z.T@Z/nt; C.diagonal().add_(.001)
repair=torch.linalg.solve(C,Z.T@(FH[:nt,units]/hr[:nt])/nt)/sd[:,None]
WN=WH.clone(); WN[:,donors]+=WH[:,units]@repair.T; WN[:,units]=0
W0=WN.clone(); W0[pool]*=(gh[pool]/oldgain[pool])[:,None]
W0[:,units]=ch[dn][:,units].double()
XN=F[:,native]/hr; sdn=XN[:nt].square().mean(0).sqrt().clamp_min(1e-12); ZN=XN[:nt]/sdn
C=ZN.T@ZN/nt; ev,V=torch.linalg.eigh((C+C.T)/2); ev=ev.clamp_min(0)
rhs=V.T@(ZN.T@(target[:nt,pool]/oldgain[pool])/nt)
ridges=[1e-6,1e-5,1e-4,.001,.01]; strong_trials=[]; selected_strong=None
for ridge in ridges:
    fit=(V@(rhs/(ev[:,None]+ridge)))/sdn[:,None]
    error=float(((XN[nt:]@fit)*oldgain[pool]-target[nt:,pool]).square().mean().sqrt())
    row={'ridge':ridge,'validation_strong_fixed_HOST_RMS_error':error}; strong_trials.append(row)
    if selected_strong is None or error<selected_strong[0]: selected_strong=error,fit.clone(),row
W0[pool[:,None],native[None]]=selected_strong[1].T
W0=W0.float().double()
parameter=h[pn].clone(); parameter[pool]=ch[pn][pool]; parameter[anchor]=ch[pn][anchor]
gain=1+parameter.double()
ratio=gh[pool]/oldgain[pool]; assert (ratio.abs()<1).all()
A=torch.cat([WN[pool]*(1-ratio.square()).sqrt()[:,None],WN[anchor:anchor+1]],0)
energy_values=(F[:nt]@A.T)/hr[:nt]
EC=energy_values.T@energy_values/nt; ee,eev=torch.linalg.eigh((EC+EC.T)/2)
assert ee[-1]>0 and ee.min()>-1e-8
factor=eev[:,-1]@A
old_anchor=WN[anchor]
if float(((F[:nt]@factor)*(F[:nt]@old_anchor)/hr[:nt,0].square()).sum())<0: factor=-factor
trace_scale=float((ee.clamp_min(0).sum()/ee[-1]).sqrt()); factor*=trace_scale
raw0=F@W0.T
remainder=RH.square().sum(1)-raw0.square().sum(1)+raw0[:,anchor].square()
remainder_ok=bool((remainder>=0).all())
del X,Z,C,repair,XN,ZN,V,rhs,energy_values
BM=b['captures']['word.17.raw'][256:384].double().reshape(-1,640)
FM=word['captures']['17.features'][384:].double().reshape(-1,2048)
FB=word['captures']['17.raw'][384:].double().reshape(-1,640)
old_truth=top['topk_ids'][256:384,:,0].reshape(-1).long()
fresh_truth=fresh_top['topk_ids'][384:,:,0].reshape(-1).long()
E=h['model.embed_tokens.weight']; fg=1+h['model.norm.weight'].double()
def readout(states,truth):
    z=(norm(states)*fg).float(); maximum=torch.full((len(z),),-torch.inf); winner=torch.full_like(truth,-1)
    own=(z*E[truth]).sum(1); other=torch.full_like(maximum,-torch.inf)
    for start in range(0,len(E),2048):
        logits=z@E[start:start+2048].T; val,idx=logits.max(1); take=val>maximum
        maximum[take]=val[take]; winner[take]=idx[take]+start
        local=truth-start; rows=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rows,local[rows]]=-torch.inf; other=torch.maximum(other,logits.amax(1))
    return {'tokens':len(z),'matches':int((winner==truth).sum()),'minimum_target_margin':float((own-other).min())}
def native_error(W,g): return float((norm(F[nt:]@W.double().T)*g-target[nt:]).square().mean().sqrt())
baseline={'validation_native_branch_rms':native_error(ch[dn],oldgain)}
bm=json.loads(install.with_suffix('.manifest.json').read_text())
trials=[]; candidates=[]; fractions=[0.,.25,.5,.75,1.]
if remainder_ok:
    wanted_energy=float((remainder[:nt]/hr[:nt,0].square()).mean())
    for fraction in fractions:
        row=(1-fraction)*old_anchor+fraction*factor
        actual=float(((F[:nt]@row)/hr[:nt,0]).square().mean())
        scale=(wanted_energy/actual)**.5
        W=W0.clone(); W[anchor,native]=row[native]*scale; W=W.float()
        assert torch.equal(W[:,units],ch[dn][:,units])
        total=bm['changed_scalars']-bm['per_tensor'][dn]-bm['per_tensor'][pn]
        total+=int((W.view(torch.int32)!=h[dn].view(torch.int32)).sum())+int((parameter.view(torch.int32)!=h[pn].view(torch.int32)).sum())
        error=native_error(W,gain)
        rec={'spectral_fraction':fraction,'analytic_energy_scale':scale,'validation_native_branch_rms':error,
             'total_changed_scalars':total,'maximum_down_weight':float(W.abs().max()),
             'validation_relative_HOST_raw_RMS_error':float((rms(F[nt:]@W.double().T)/hr[nt:]-1).square().mean().sqrt())}
        rec['eligible_native']=bool(error<.95*baseline['validation_native_branch_rms'] and total<=5016268 and W.abs().max()<=1e6 and W.isfinite().all())
        trials.append(rec)
        if rec['eligible_native']: candidates.append((W,rec))
        print(json.dumps({'trial':rec}),flush=True)
validation=[]; original=None; best=None
if candidates:
    original={'old':readout(BM+norm(F[nt:]@ch[dn].double().T)*oldgain,old_truth),
              'fresh':readout(FB+norm(FM@ch[dn].double().T)*oldgain,fresh_truth)}
    assert original['fresh']['matches']==8003
    PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)
    PB=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
    truth=fmt['topk_ids'][:,:,0].reshape(-1).long()
    for W,rec in candidates:
        old=readout(BM+norm(F[nt:]@W.double().T)*gain,old_truth)
        fresh=readout(FB+norm(FM@W.double().T)*gain,fresh_truth)
        do=old['matches']-original['old']['matches']; df=fresh['matches']-original['fresh']['matches']
        row={**rec,'old_validation':old,'fresh_validation':fresh,'extra_old_matches':do,'extra_fresh_matches':df,'eligible':False}
        if do>=0 and df>=8 and do+df>=16:
            q=readout(PB+norm(PF@W.double().T)*gain,truth); row['format_validation']=q
            row['eligible']=q['matches']==2560 and q['minimum_target_margin']>1
        validation.append(row)
        if row['eligible'] and (best is None or (do+df,df,-rec['validation_native_branch_rms'])>best[0]):
            best=(do+df,df,-rec['validation_native_branch_rms']),W.clone(),row
        print(json.dumps({'validation':row}),flush=True)
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),'install_base_checkpoint':str(install.relative_to(ROOT)),
    'baseline':baseline,'anchor':anchor,'anchor_gain':float(gain[anchor]),'donor_count':512,'donor_ridge':.001,
    'strong_ridge_trials':ridges,'strong_trials':strong_trials,'selected_strong':selected_strong[-1],
    'spectral_fraction_trials':fractions,'energy_factor_eigenvalues':ee.tolist(),
    'energy_top_factor_fraction':float(ee[-1]/ee.clamp_min(0).sum()),'spectral_trace_scale':trace_scale,
    'negative_energy_remainder_count':int((remainder<0).sum()),'negative_remainder_policy':'Reject all; no clipping',
    'trials':trials,'original_full_vocabulary_validation':original,'full_vocabulary_validation':validation,
    'selected':None if best is None else best[-1],
    'selection':'Native RMS>=5percent improvement; old matches no loss; fresh>=8 extra and combined>=16; format allcorrect margin>1; caps',
    'scope':'Original HOST same-input native target; exact original local inputs, reused calibration holdbacks; A259 install would require fullscore',
    'construction':'One original HOST donor/strong solve and independent signed PCA energy mixtures; programme functions unchanged; no activation replay, gradients, answers, or iterative residual fits',
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,install,bp,bcp,np,ncp,hp,wp,tp,fp,wr,fr,base.parent/'circuit.pt',install.with_suffix('.manifest.json'))},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a275v1_existing_anchor_spectral_energy.json'; atomic_json(out,report)
if best is not None: atomic_torch(out.with_suffix('.pt'),{'down':best[1],'gain_parameter':parameter,'report':report})
print(json.dumps({'selected':report['selected'],'baseline':baseline,'peak_kib':report['maximum_resident_set_kib']}),flush=True)
