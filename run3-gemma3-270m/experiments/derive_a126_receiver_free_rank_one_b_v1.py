"""Supply one exact current-digit value direction to a receiver-free B reader.

The input code uses individual current-token class means and within-class
covariance. The value projection is rank one. Independent native head means
supply the reference, while a separate covariance penalizes the full receiver
background. Every solve starts from HOST weights or original recorded inputs;
there are no sum targets, activation replay, gradients or iterative fits.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
marker=ROOT/'experiments/a1_marker/delta.pt';ch,old=expand(h,base),expand(h,marker)
for name in h:
    if name.startswith('model.layers.0.'):
        assert torch.equal(ch.get(name,h[name]),old.get(name,h[name])),name
paths=[ROOT/f'artifacts/a36_inputs_{s}.pt'for s in('syntax_a','syntax_stress','syntax_c','calib256')]
caps=[torch.load(p,weights_only=True,mmap=True)for p in paths]
for cap in caps:assert cap['checkpoint_sha256']==sha(marker)
prefix='model.layers.2.'
V=h[prefix+'self_attn.v_proj.weight'].double()
O=h[prefix+'self_attn.o_proj.weight'].double().reshape(640,4,256)
gamma=1+h[prefix+'post_attention_layernorm.weight'].double()
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
T=caps[3]['captures']['2.input'].double().reshape(-1,640);nt=192*64
CX=T[:nt].T@T[:nt]/nt
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
positions=[7,6,5,4]
groups=[]
for cap in caps[:3]:
    these=[]
    for head,pos in enumerate(positions):
        x=cap['captures']['2.input'][:,pos].double()
        labels=(cap['inputs'][:,pos,None]==ids[None]).long().argmax(1)
        if head<3:
            assert (cap['inputs'][:,pos,None]==ids[None]).sum(1).eq(1).all()
            target=(labels.double()-4.5)/9
        else:
            labels=torch.zeros(len(x),dtype=torch.long);target=torch.zeros(len(x),dtype=torch.float64)
        these.append((x,labels,target))
    groups.append(these)
rows=[];targets=[];within=torch.zeros(640,640,dtype=torch.float64);count=0
for head in range(4):
    X=torch.cat([g[head][0]for g in groups[:2]])
    labels=torch.cat([g[head][1]for g in groups[:2]])
    target=torch.cat([g[head][2]for g in groups[:2]])
    for label in labels.unique():
        keep=labels==label;mean=X[keep].mean(0)
        rows.append(mean);targets.append(target[keep].mean())
        residual=X[keep]-mean;within+=residual.T@residual;count+=int(keep.sum())
within/=count
M,target=torch.stack(rows),torch.stack(targets)[:,None]
scale=M.norm(dim=1);M/=scale[:,None];target/=scale[:,None]
assert torch.linalg.matrix_rank(M)==len(M)
code_trials=[];codes=[]
for mass in(1.,10.,100.,1000.,10000.):
    R=CX+mass*within;R.diagonal().add_(1e-6*CX.diagonal().mean())
    Z=torch.linalg.solve(R,M.T);L=Z@torch.linalg.solve(M@Z,target)
    errors=[float(((x@L)[:,0]-t).abs().max())for g in groups for x,_,t in g]
    row={'within_class_mass':mass,'maximum_independent_digit_code_error':max(errors),
         'maximum_input_code_weight':float(L.abs().max()),'split_head_code_errors':errors}
    code_trials.append(row);codes.append(L);print(json.dumps({'digit_code':row}),flush=True)
selected=min(range(5),key=lambda i:code_trials[i]['maximum_independent_digit_code_error'])
L=codes[selected]
CV=V@CX@V.T;CV.diagonal().add_(1e-6*CV.diagonal().mean())
A=torch.linalg.solve(CV,V@CX@L)
residual_row=L.T-A.T@V
allowed=gp.abs()>=.05;allowed[334]=False;idx=allowed.nonzero().flatten()
XX=torch.cat([cap['captures']['2.input']for cap in caps[:2]]).double()
value_trials=[];candidates=[]
for mass in(.0001,.001,.01,.1,1.):
    row_metric=gamma.square()+mass*gamma.square().mean()
    H=sum(O[:,head].T@(row_metric[:,None]*O[:,head])for head in range(4))
    H.diagonal().add_(1e-6*H.diagonal().mean())
    Z=torch.linalg.solve(H,A);U=Z/(A.T@Z)
    assert (A.T@U-1).abs().max()<1e-8
    VN=(V+U@residual_row).float().double()
    identity=float((A.T@VN-L.T).abs().max());assert identity<1e-6
    VR=VN-U@L.T
    maps=[gamma[:,None]*(O[:,head]@VN)for head in range(4)]
    remaining=[gamma[:,None]*(O[:,head]@VR)for head in range(4)]
    raw_maps=[O[:,head]@VN for head in range(4)]
    rows=[];rhs=[];means=[];source_rows=[]
    within=torch.zeros(len(idx),len(idx),dtype=torch.float64)
    for head,pos in enumerate(positions):
        rows.append((gamma[:,None]*(O[:,head]@U))[idx,0])
        target=torch.zeros(4,dtype=torch.float64)
        if head<3:target[head]=1
        rhs.append(target)
        y=XX[:,pos]@remaining[head].T
        mean=y[:,idx].mean(0);means.append(mean)
        centered=y[:,idx]-mean;within+=centered.T@centered/(4*len(y))
    rows.append(sum(means));rhs.append(torch.tensor([0.,0.,0.,1.],dtype=torch.float64))
    M,target=torch.stack(rows),torch.stack(rhs)
    scale=M.norm(dim=1);M/=scale[:,None];target/=scale[:,None]
    assert torch.linalg.matrix_rank(M)==len(M)
    CW=sum(mapping[idx]@CX@mapping[idx].T for mapping in maps)/4
    backgrounds=[];decoded=[]
    for cap in caps[:3]:
        sx=cap['captures']['2.input'].double()
        raw=sum(sx[:,pos]@raw_maps[head].T for head,pos in enumerate(positions))
        rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
        before=cap['captures']['1.out'][:,8].double();backgrounds.append(before*rms)
        wanted=torch.stack([(cap['inputs'][:,pos,None]==ids[None]).long().argmax(1)for pos in positions[:3]],1)
        decoded.append((raw*gamma/rms,before,rms,wanted))
    B=torch.cat(backgrounds[:2])[:,idx];CB=B.T@B/len(B)
    frame_trials=[];frames=[]
    for bgmass in(0.,1.,100.,10000.,1000000.):
        R=CW+10000*within+bgmass*CB
        R.diagonal().add_(1e-6*CW.diagonal().mean())
        Z=torch.linalg.solve(R,M.T)
        F=torch.zeros(640,4,dtype=torch.float64);F[idx]=Z@torch.linalg.solve(M@Z,target)
        offsets=[mean@F[idx]for mean in means]
        assert (sum(offsets)-torch.tensor([0.,0.,0.,1.])).abs().max()<1e-7
        source_bound=0.
        for g in groups:
            error_sum=0.
            for head,(x,_,literal)in enumerate(g):
                desired=torch.zeros(len(x),4,dtype=torch.float64)
                if head<3:desired[:,head]=literal
                error_sum+=float((x@maps[head].T@F-desired-offsets[head]).abs().max())
            source_bound=max(source_bound,error_sum)
        splits=[]
        for name,(branch,before,rms,wanted)in zip(('syntax_a','syntax_stress','syntax_c'),decoded):
            code=(branch+before)@F;bg=before@F
            got=9*code[:,:3]/code[:,3:]+4.5
            splits.append({'split':name,'minimum_constant':float(code[:,3].min()),
                           'maximum_current_digit_error':(got-wanted).abs().max(0).values.tolist(),
                           'wrong_current_digits':int((got.round()!=wanted).sum()),
                           'background_bound_before_normalization':float((bg*rms).abs().max())})
        background_bound=max(s['background_bound_before_normalization']for s in splits)
        row={'background_mass':bgmass,'independent_source_error_bound':source_bound,
             'background_bound':background_bound,'selection_metric':source_bound+background_bound,
             'maximum_readout_weight':float(F.abs().max()),'split_diagnostics':splits}
        row['eligible']=all(s['minimum_constant']>0 and max(s['maximum_current_digit_error'])<.3 for s in splits)
        frame_trials.append(row);frames.append(F)
    eligible=[i for i,r in enumerate(frame_trials)if r['eligible']]
    choice=min(eligible,key=lambda i:frame_trials[i]['selection_metric'])if eligible else None
    delta=T[nt:]@(VN-V).T
    proxy=[float((delta@O[:,head].T*gamma).square().mean().sqrt())for head in range(4)]
    row={'raw_metric_mass':mass,'value_span_identity_error':identity,
         'maximum_value_weight':float(VN.abs().max()),'native_word_token_output_rms_by_head':proxy,
         'native_word_token_mean_squared_proxy':sum(v*v for v in proxy)/4,
         'frame_trials':frame_trials,'selected_frame':None if choice is None else frame_trials[choice]}
    value_trials.append(row);print(json.dumps(row),flush=True)
    if choice is not None:candidates.append((row['native_word_token_mean_squared_proxy'],VN.float(),frames[choice],row))
best=min(candidates,key=lambda x:x[0])if candidates else None
report={'analysis_only':True,'native_value_update_rank':1,'layer1_restored_to_HOST':True,
        'input_code_within_class_trials':[1,10,100,1000,10000],'code_trials':code_trials,
        'selected_input_code':code_trials[selected],'raw_metric_mass_trials':[.0001,.001,.01,.1,1],
        'background_mass_trials':[0,1,100,10000,1000000],'frame_within_source_mass':10000,
        'ridge_fraction':1e-6,'trials':value_trials,'selected':None if best is None else best[-1],
        'target':'Individual current-token centered digit; independent head means provide the reference.',
        'caveat':'Ideal source routing and individual native-value output proxies; actual model checks required.',
        'exact_input_provenance':'Layer0 matches A1; layer1 and native upstream matrices in A36 capture are HOST.',
        'inputs':{str(p.relative_to(ROOT)):sha(p)for p in paths+[base,marker]},'source_sha256':sha(__file__)}
atomic_json(ROOT/'artifacts/a126_receiver_free_rank_one_b_v1.json',report)
if best is not None:
    atomic_torch(ROOT/'artifacts/a126_receiver_free_rank_one_b_v1.pt',{
        'V':best[1],'readout':best[2],'input_digit_code':L,'report':report})
