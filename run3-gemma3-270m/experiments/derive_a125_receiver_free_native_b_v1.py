"""Test a native B readout with full receiver-background suppression.

Individual token/head class-mean differences constrain the three digit codes;
independent head means supply the reference. Five background covariance masses
are solved directly. Layer1 is assumed restored to HOST. No sum targets,
nonlinear activation replay, iterative fit or model invocation occurs.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt'
marker=ROOT/'experiments/a1_marker/delta.pt'
ch,old=expand(h,base),expand(h,marker)
for name in h:
    if name.startswith('model.layers.0.'):
        assert torch.equal(ch.get(name,h[name]),old.get(name,h[name])),name
paths=[ROOT/f'artifacts/a36_inputs_{s}.pt'for s in('syntax_a','syntax_stress','syntax_c','calib256')]
caps=[torch.load(p,weights_only=True,mmap=True)for p in paths]
for cap in caps:assert cap['checkpoint_sha256']==sha(marker)
train=caps[:2];valid=caps[2];word=caps[3]
prefix='model.layers.2.'
V=h[prefix+'self_attn.v_proj.weight'].double()
O=h[prefix+'self_attn.o_proj.weight'].double().reshape(640,4,256)
gamma=1+h[prefix+'post_attention_layernorm.weight'].double()
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
for suffix in('self_attn.v_proj.weight','self_attn.o_proj.weight','post_attention_layernorm.weight'):
    assert torch.equal(ch.get(prefix+suffix,h[prefix+suffix]),h[prefix+suffix])
allowed=gp.abs()>=.05;allowed[334]=False;idx=allowed.nonzero().flatten()
maps=[gamma[:,None]*(O[:,i]@V)for i in range(4)]
raw_maps=[O[:,i]@V for i in range(4)]
X=word['captures']['2.input'].double().reshape(-1,640)[:192*64]
CX=X.T@X/len(X)
CW=sum(M[idx]@CX@M[idx].T for M in maps)/4
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
positions=[7,6,5,4]
sources=[]
for cap in caps[:3]:
    groups=[]
    for head,pos in enumerate(positions):
        y=cap['captures']['2.input'][:,pos].double()@maps[head].T
        digit=(cap['inputs'][:,pos,None]==ids[None]).long().argmax(1)
        target=torch.zeros(len(y),4,dtype=torch.float64);target[:,3]=.25
        if head<3:
            assert (cap['inputs'][:,pos,None]==ids[None]).sum(1).eq(1).all()
            target[:,head]=(digit.double()-4.5)/9
        groups.append((y,target,digit))
    sources.append(groups)
rows=[];rhs=[];means=[];targets=[]
within=torch.zeros(len(idx),len(idx),dtype=torch.float64);n=0
for head in range(4):
    y=torch.cat([g[head][0]for g in sources[:2]])[:,idx]
    target=torch.cat([g[head][1]for g in sources[:2]])
    labels=torch.cat([g[head][2]for g in sources[:2]]) if head<3 else torch.zeros(len(y),dtype=torch.long)
    means.append(y.mean(0));targets.append(target.mean(0))
    class_means=[];class_targets=[]
    for label in labels.unique():
        keep=labels==label;mu=y[keep].mean(0)
        class_means.append(mu);class_targets.append(target[keep].mean(0))
        residual=y[keep]-mu;within+=residual.T@residual;n+=len(residual)
    for j in range(1,len(class_means)):
        rows.append(class_means[j]-class_means[0]);rhs.append(class_targets[j]-class_targets[0])
rows.append(sum(means));rhs.append(sum(targets));within/=n
M,target=torch.stack(rows),torch.stack(rhs)
norm=M.norm(dim=1);M/=norm[:,None];target/=norm[:,None]
assert torch.linalg.matrix_rank(M)==len(M)
backgrounds=[];decoded=[]
for cap in caps[:3]:
    sx=cap['captures']['2.input'].double()
    raw=sum(sx[:,pos]@raw_maps[head].T for head,pos in enumerate(positions))
    rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
    before=cap['captures']['1.out'][:,8].double()
    backgrounds.append(before*rms)
    wanted=torch.stack([(cap['inputs'][:,pos,None]==ids[None]).long().argmax(1)for pos in positions[:3]],1)
    decoded.append((raw*gamma/rms,before,rms,wanted))
B=torch.cat(backgrounds[:2])[:,idx];CB=B.T@B/len(B)
trials=[];choices=[]
for mass in(0.,1.,100.,10000.,1000000.):
    R=CW+10000*within+mass*CB
    R.diagonal().add_(1e-6*CW.diagonal().mean())
    Z=torch.linalg.solve(R,M.T)
    F=torch.zeros(640,4,dtype=torch.float64)
    F[idx]=Z@torch.linalg.solve(M@Z,target)
    offsets=[mu@F[idx]-wanted for mu,wanted in zip(means,targets)]
    assert sum(offsets).abs().max()<1e-8
    token_bound=0.
    for groups in sources:
        token_bound=max(token_bound,sum(float((y@F-t-offsets[i]).abs().max())for i,(y,t,_)in enumerate(groups)))
    splits=[]
    for name,(branch,before,rms,wanted)in zip(('syntax_a','syntax_stress','syntax_c'),decoded):
        code=(branch+before)@F;bg=before@F
        got=9*code[:,:3]/code[:,3:]+4.5
        splits.append({'split':name,'minimum_constant':float(code[:,3].min()),
                       'maximum_current_digit_error':(got-wanted).abs().max(0).values.tolist(),
                       'wrong_current_digits':int((got.round()!=wanted).sum()),
                       'background_bound_before_normalization':float((bg*rms).abs().max())})
    background_bound=max(x['background_bound_before_normalization']for x in splits)
    row={'background_mass':mass,'independent_source_error_bound':token_bound,
         'background_bound':background_bound,'selection_metric':token_bound+background_bound,
         'maximum_readout_weight':float(F.abs().max()),'split_diagnostics':splits}
    row['eligible']=all(s['minimum_constant']>0 and max(s['maximum_current_digit_error'])<.35 for s in splits)
    trials.append(row);print(json.dumps(row),flush=True)
    if row['eligible']:choices.append((row['selection_metric'],F,row))
best=min(choices,key=lambda x:x[0])if choices else None
report={'analysis_only':True,'native_V_O_unchanged':True,'layer1_restored_to_HOST':True,
        'exact_input_provenance':'Layer0 matches A1 exactly; all remaining upstream weights in the A36 capture are HOST.',
        'background_mass_trials':[0,1,100,10000,1000000],'within_source_mass':10000,
        'ridge_fraction':1e-6,'current_token_constraints':len(M),
        'target':'Independent current-token/head centered digit and reference; never sums.',
        'ideal_routing_caveat':'Native per-head source maps and recorded receiver background, not replayed attention.',
        'trials':trials,'selected':None if best is None else best[-1],
        'inputs':{str(p.relative_to(ROOT)):sha(p)for p in paths+[base,marker]},'source_sha256':sha(__file__)}
atomic_json(ROOT/'artifacts/a125_receiver_free_native_b_v1.json',report)
if best is not None:
    atomic_torch(ROOT/'artifacts/a125_receiver_free_native_b_v1.pt',{'readout':best[1],'report':report})
