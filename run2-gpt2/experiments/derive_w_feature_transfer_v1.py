"""Native feature transfer for a sparse late-layer soft parser.

Thirty-two native MLP4 units are selected for exact GELU identity pairs. A
closed-form native-state projection predicts their displaced activations and
the native post-MLP state from retained GELUs plus the LN2 input. Frozen token
and role decoders are transferred through that projection. No sum targets,
gradient updates, transformer runs or answer-based fitting occur here.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_restore_v4 import restore
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True)
full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True)
assert torch.equal(full['inputs'][:64],native['inputs'][:64])
rg=torch.load(ROOT/'artifacts/w_native_roles_v1_geometry.pt',weights_only=True)['layers'][4]
role=next(x for x in rg['trials'] if x['mass']==10.)
tg=torch.load(ROOT/'artifacts/w_token_features_v2.pt',weights_only=True)
token=next(x for x in tg['candidates'] if x['report']['negative_vocabulary_classes']==128)
domains=full['request']['domains'];starts=full['request']['program_starts'];trainidx=[i for i,d in enumerate(domains) if d=='format_a'];valididx=[i for i,d in enumerate(domains) if d=='format_b'];badidx=[i for i,d in enumerate(domains) if d=='bad']
def programme(indices,key):
    return torch.cat([full['captures'][key][i,starts[i]:starts[i]+8] for i in indices])
def programme_tokens(indices):return torch.cat([full['inputs'][i,starts[i]:starts[i]+8] for i in indices])
pc=trainidx+badidx[:32];pv=valididx+badidx[32:];nw=384*64
G=torch.cat((native['captures']['4.gelu'][:384].reshape(-1,3072),programme(pc,'4.gelu'),native['captures']['4.gelu'][384:].reshape(-1,3072),programme(pv,'4.gelu')))
raw=torch.cat((native['captures']['4.raw'][:384].reshape(-1,768),programme(pc,'4.raw'),native['captures']['4.raw'][384:].reshape(-1,768),programme(pv,'4.raw')))
norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5)
nprog=len(pc)*8;ntrain=nw+nprog;edited={'captures':{'4.raw':raw,'4.gelu':G,'4.ln2':norm}}
base,repair=restore(host,4,edited,raw,G,torch.zeros(32,769,dtype=torch.float64),torch.zeros(768,32,dtype=torch.float64),donor_count=128,ntrain=ntrain)
selected=repair['selected'];retained=torch.arange(3072)[~torch.isin(torch.arange(3072),selected)]
P=base['transformer.h.4.mlp.c_proj.weight'];pb=base['transformer.h.4.mlp.c_proj.bias'];del base,edited
Z=(raw+torch.nn.functional.linear(G,P,pb)).double();missing=G[:,selected].double();A=torch.cat((G[:,retained].double(),norm.double()),1)
mean=A[:nw].mean(0);meanY=torch.cat((missing[:nw],Z[:nw]),1).mean(0);A-=mean;Y=torch.cat((missing,Z),1)-meanY
C=A[:nw].T@A[:nw]/nw+A[nw:ntrain].T@A[nw:ntrain]/nprog
cross=A[:nw].T@Y[:nw]/nw+A[nw:ntrain].T@Y[nw:ntrain]/nprog
tokens=programme_tokens(pc);groups=torch.stack((torch.arange(8).repeat(len(pc)),tokens),1)
D=[];DY=[]
for group in groups.unique(dim=0):
    mask=(groups==group).all(1);D.append(A[nw:ntrain][mask].mean(0));DY.append(Y[nw:ntrain][mask].mean(0))
D=torch.stack(D);DY=torch.stack(DY)
ideal=torch.zeros(3840,15,dtype=torch.float64);ideal[:3072,:7]=role['role_decoder'][:,:7];ideal[:,7:]=token['coefficients'];idealbias=torch.cat((-rg['feature_mean']@role['role_decoder'][:,:7],token['bias']))
ids=torch.tensor(torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids']);digit={int(t):i%10 for i,t in enumerate(ids[:20])};digitroles=(0,1,2,4,5,6)
vi=torch.tensor(valididx);vstarts=torch.tensor([starts[i] for i in valididx]);vt=full['inputs'][vi];truth=torch.tensor([[[(digit[int(vt[i,vstarts[i]+r])]>>b)&1 for b in range(4)] for r in digitroles] for i in range(len(vi))],dtype=torch.float64)
def prefix_predictions(L,bias):
    result=[]
    for start in range(0,len(vi),32):
        index=vi[start:start+32];r=full['captures']['4.raw'][index];n=torch.nn.functional.layer_norm(r,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5)
        f=torch.cat((full['captures']['4.gelu'][index][:,:,retained].double(),n.double()),2);result.append(f@L+bias)
    pred=torch.cat(result);pos=torch.arange(64)[None,:];causal=pos<=(vstarts+7)[:,None];correct=[];margins=[]
    for r in range(7):
        kind=0 if r==0 else 2 if r==3 else 1;mask=causal&(pred[:,:,7+kind]>.5);s=pred[:,:,r].masked_fill(~mask,-torch.inf);target=vstarts+r;value=s[torch.arange(len(vi)),target];wrong=s.clone();wrong[torch.arange(len(vi)),target]=-torch.inf;correct.append(s.argmax(1)==target);margins.append(value-wrong.max(1).values)
    bits=torch.stack([pred[torch.arange(len(vi)),vstarts+r,11:15] for r in digitroles],1);errors=(bits>.5)!=(truth>.5);typed=torch.ones(len(vi),dtype=torch.bool)
    for r in range(8):
        kind=0 if r==0 else 2 if r==3 else 3 if r==7 else 1;typed&=pred[torch.arange(len(vi)),vstarts+r,7+kind]>.5
    return {'all_seven_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(vi),'minimum_source_margin':float(torch.stack(margins,1).min()),'digit_bit_error_examples':int(errors.any(2).any(1).sum()),'minimum_digit_bit_margin':float(((2*truth-1)*(bits-.5)).min()),'programme_type_correct':int(typed.sum())}
reports=[];best=None
for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
    regular=C.clone();regular.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(regular);initial=torch.cholesky_solve(cross,chol);inv=torch.cholesky_solve(D.T,chol);gram=D@inv;coeff=initial+inv@torch.linalg.pinv(gram,hermitian=True,rtol=1e-11)@(DY-D@initial)
    B=coeff[:,:32];intercept=meanY[:32]-mean@B;L=torch.cat((ideal[retained],ideal[3072:]),0)+B@ideal[selected];Lb=idealbias+intercept@ideal[selected];piece=prefix_predictions(L,Lb)
    residual=Y[ntrain:]-A[ntrain:]@coeff;wr=residual[:128*64,32:];pr=residual[128*64:,32:];r={'ridge':ridge,'class_mean_constraint_error':float((D@coeff-DY).abs().max()),'displaced_native_feature_validation_rms':float(residual[:,:32].square().mean().sqrt()),'native_word_post_validation_rms':float(wr.square().mean().sqrt()),'programme_post_validation_rms':float(pr.square().mean().sqrt()),'maximum_coefficient':float(coeff.abs().max()),'transferred_parser':piece};r['feasible']=piece['all_seven_sources_correct']==len(vi) and piece['digit_bit_error_examples']==0 and piece['programme_type_correct']==len(vi) and r['class_mean_constraint_error']<1e-5
    reports.append(r);print('feature transfer',json.dumps(r),flush=True)
    objective=r['native_word_post_validation_rms']+r['programme_post_validation_rms']
    if r['feasible'] and (best is None or objective<best['objective']):
        K=coeff[:,32:];Kb=meanY[32:]-mean@K;best={'objective':objective,'report':r,'feature_mean':mean,'post_coefficients':K,'post_bias':Kb,'displaced_coefficients':B,'displaced_bias':intercept,'payload_coefficients':L,'payload_bias':Lb}
assert best is not None,reports
out={'chosen':best,'reports':reports,'selected':selected,'retained':retained,'native_projection':P,'native_bias':pb,'native_repair':repair,'payload_fields':[f'role{r}' for r in range(7)]+['bare','spaced','plus','equals']+[f'bit{b}' for b in range(4)],'native_donors':128,'identity_pairs':16,'ridge_family':[1e-3,1e-4,1e-5,1e-6,1e-7],'method':'Native-state covariance projection; labels inherited unchanged from native token/role maps. No sum targets.'}
torch.save(out,ROOT/'artifacts/w_feature_transfer_v1.pt');(ROOT/'artifacts/w_feature_transfer_v1.json').write_text(json.dumps({'chosen':best['report'],'trials':reports,'native_repair':repair['text_repair_report']},indent=2)+'\n')
