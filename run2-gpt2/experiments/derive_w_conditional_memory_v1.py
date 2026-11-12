"""Restore native word components while retaining late parser signals.

The target is frozen native raw5 state on calibration words. Programme targets
are zero correction at current-token/role class means, with within-class
correction variance penalized. All solves are one-shot covariance projections.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];transfer=torch.load(ROOT/'artifacts/w_feature_transfer_v1.pt',weights_only=True);old=torch.load(ROOT/'artifacts/w_soft_memory_v1.pt',weights_only=True)['chosen'];c=transfer['chosen'];F=old['frame'];ret=transfer['retained'];reference_gain=old['reference_gain'];P=transfer['native_projection'];pb=transfer['native_bias']
native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);nr5=torch.load(ROOT/'artifacts/o_host_calib25.pt',weights_only=True,mmap=True);full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);domains=full['request']['domains'];starts=full['request']['program_starts'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];construct=train+bad[:32]
gains=torch.full((15,),4.,dtype=torch.float64);gains[7:10]=8.;gains[10]=16.
def features(raw,gelu):
    norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5);return torch.cat((gelu[...,ret].double(),norm.double()),-1)
W=features(native['captures']['4.raw'],native['captures']['4.gelu']).reshape(-1,len(ret)+768);Z=(native['captures']['4.raw']+torch.nn.functional.linear(native['captures']['4.gelu'],P,pb)).reshape(-1,768).double();residual=Z-W@c['post_coefficients']-c['post_bias'];labels=W@c['payload_coefficients']+c['payload_bias'];target=nr5['captures']['5.raw'].reshape(-1,768).double()@F[:,:15]-(residual@F[:,:15]+labels*gains)
ntrain=384*64;mu=W[:ntrain].mean(0);tm=target[:ntrain].mean(0);wc=W[:ntrain]-mu;Cw=wc.T@wc/ntrain;cross=wc.T@(target[:ntrain]-tm)/ntrain
pr=torch.cat([full['captures']['4.raw'][i,starts[i]:starts[i]+8] for i in construct]);pg=torch.cat([full['captures']['4.gelu'][i,starts[i]:starts[i]+8] for i in construct]);A=features(pr,pg)-mu;pt=torch.cat([full['inputs'][i,starts[i]:starts[i]+8] for i in construct]);groups=torch.stack((torch.arange(8).repeat(len(construct)),pt),1);D=[];within=torch.zeros(A.shape[1],A.shape[1],dtype=torch.float64)
for group in groups.unique(dim=0):
    mask=(groups==group).all(1);v=A[mask];m=v.mean(0);D.append(m);within+=(v-m).T@(v-m)
D=torch.stack(D);Cp=within/len(A);dy=-tm.expand(len(D),-1)
vi=torch.tensor(valid);vs=torch.tensor([starts[i] for i in valid]);vt=full['inputs'][vi];ids=torch.tensor(torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids']);lookup={int(t):i%10 for i,t in enumerate(ids[:20])};digitroles=(0,1,2,4,5,6);truth=torch.tensor([[[(lookup[int(vt[i,vs[i]+r])]>>b)&1 for b in range(4)] for r in digitroles] for i in range(len(vi))],dtype=torch.float64)
fv=[];rv=[];lv=[]
for j in range(0,len(vi),32):
    ix=vi[j:j+32];raw=full['captures']['4.raw'][ix];gelu=full['captures']['4.gelu'][ix];x=features(raw,gelu);z=raw.double()+torch.nn.functional.linear(gelu,P,pb).double();fv.append(x.float());rv.append(((z-x@c['post_coefficients']-c['post_bias'])@F).float());lv.append((x@c['payload_coefficients']+c['payload_bias']).float())
fv=torch.cat(fv);rv=torch.cat(rv).double();lv=torch.cat(lv).double();causal=torch.arange(64)[None,:]<=(vs+7)[:,None];word_ref=1+(residual[ntrain:]@F[:,15])/reference_gain;wordeq=native['inputs'][384:].flatten()==ids[-1];reports=[];best=None
for mass in (1.,10.,100.,1000.,10000.):
    C=Cw+mass*Cp;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C);initial=torch.cholesky_solve(cross,chol);inv=torch.cholesky_solve(D.T,chol);coeff=initial+inv@torch.linalg.pinv(D@inv,hermitian=True,rtol=1e-11)@(dy-D@initial);bias=tm-mu@coeff
    correction=fv.double()@coeff+bias;ref=1+rv[:,:,15]/reference_gain;pred=(rv[:,:,:15]/gains+lv+correction/gains)/ref[:,:,None];correct=[];margins=[]
    for role in range(7):
        kind=0 if role==0 else 2 if role==3 else 1;mask=causal&(pred[:,:,7+kind]>.5);score=pred[:,:,role].masked_fill(~mask,-torch.inf);position=vs+role;value=score[torch.arange(len(vi)),position];wrong=score.clone();wrong[torch.arange(len(vi)),position]=-torch.inf;correct.append(score.argmax(1)==position);margins.append(value-wrong.max(1).values)
    bits=torch.stack([pred[torch.arange(len(vi)),vs+r,11:15] for r in digitroles],1);err=(bits>.5)!=(truth>.5);typed=torch.ones(len(vi),dtype=torch.bool)
    for role in range(8):
        kind=0 if role==0 else 2 if role==3 else 3 if role==7 else 1;typed&=pred[torch.arange(len(vi)),vs+role,7+kind]>.5
    restored=W[ntrain:]@coeff+bias;word_residual=target[ntrain:]-restored;word_fields=(residual[ntrain:]@F[:,:15]+labels[ntrain:]*gains+restored)/gains/word_ref[:,None];r={'programme_covariance_mass':mass,'class_constraint_error':float((D@coeff-dy).abs().max()),'word_frame_rms_before':float(target[ntrain:].square().mean().sqrt()),'word_frame_rms_after':float(word_residual.square().mean().sqrt()),'word_global_rms_equivalent_after':float(word_residual.square().sum(1).mean().sqrt()/(768**.5)),'validation_correction_max_abs':float(correction.abs().max()),'all_seven_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(vi),'minimum_source_margin':float(torch.stack(margins,1).min()),'digit_bit_error_examples':int(err.any(2).any(1).sum()),'minimum_digit_bit_margin':float(((2*truth-1)*(bits-.5)).min()),'programme_type_correct':int(typed.sum()),'native_validation_false_equals':int((word_fields[~wordeq,10]>.5).sum()),'native_validation_equals_max':float(word_fields[~wordeq,10].max()),'maximum_coefficient':float(coeff.abs().max())};r['feasible']=r['all_seven_sources_correct']==len(vi) and r['digit_bit_error_examples']==0 and r['programme_type_correct']==len(vi) and r['native_validation_false_equals']==0 and r['minimum_source_margin']>.02;reports.append(r);print(json.dumps(r),flush=True)
    if r['feasible'] and (best is None or r['word_frame_rms_after']<best['report']['word_frame_rms_after']):best={'coefficients':coeff,'bias':bias,'report':r}
assert best is not None,reports
L=torch.cat((c['payload_coefficients']*gains,torch.zeros(len(c['payload_coefficients']),1,dtype=torch.float64)),1);Lb=torch.cat((c['payload_bias']*gains,torch.zeros(1,dtype=torch.float64)));B=L-c['post_coefficients']@F;fieldbias=Lb-c['post_bias']@F;B[:,:15]+=best['coefficients'];fieldbias[:15]+=best['bias'];fieldbias[15]+=reference_gain;rows=[];cols=[]
for j in range(16):
    for sign in (1.,-1.):rows.append(torch.cat((sign*B[len(ret):,j],torch.zeros(1,dtype=torch.float64))));cols.append(sign*F[:,j])
best.update({'frame':F,'support':old['support'],'payload_gains':gains,'reference_gain':reference_gain,'feature_delta':B,'field_bias_delta':fieldbias,'program_rows':torch.stack(rows),'program_columns':torch.stack(cols,1)})
torch.save({'chosen':best,'reports':reports,'programme_covariance_mass_family':[1,10,100,1000,10000],'fixed_diagonal_ridge':1e-5,'payload_base_gain':4.,'kind_gain_multiple':2.,'equals_gain_multiple':4.},ROOT/'artifacts/w_conditional_memory_v1.pt');(ROOT/'artifacts/w_conditional_memory_v1.json').write_text(json.dumps({'chosen':best['report'],'trials':reports},indent=2)+'\n')
