"""Exact current-token class means for digit/type features in native MLP4.

Extra constraints set common non-format vocabulary-class means to zero. This
is a token detector, not a whole-problem or answer map. Independent native
windows and format-B contexts assess generalization.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
text,program=unpack(torch.load(ROOT/'artifacts/w_native_roles_v1.pt',weights_only=True,mmap=True))
native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True)
ids=torch.tensor(torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'])
train=torch.tensor([i for i,d in enumerate(program['domains']) if d=='format_a']);valid=torch.tensor([i for i,d in enumerate(program['domains']) if d=='format_b']);bad=torch.tensor([i for i,d in enumerate(program['domains']) if d=='bad'])
def labels(tokens):
    out=torch.zeros(tokens.shape+(7,),dtype=torch.float64)
    for i,token in enumerate(ids):
        mask=tokens==token
        if i<20:out[...,0][mask]=1;out[...,3:][mask]=torch.tensor([(i%10>>bit)&1 for bit in range(4)],dtype=torch.float64)
        else:out[...,i-19][mask]=1
    return out
W=native['captures']['4.gelu'].reshape(-1,3072).double();wt=native['inputs'].flatten();ntrain=384*64;negative=~torch.isin(wt,ids);construction=(torch.arange(len(wt))<ntrain)&negative;mu=W[construction].mean(0);W-=mu
X=program['captures']['4.gelu'].double()-mu;tokens=program['inputs'][:,:8];xt=X[train].reshape(-1,3072);it=tokens[train].flatten();role=torch.arange(8).repeat(len(train));groups=torch.stack((role,it),1)
means=[];targets=[];within=torch.zeros(3072,3072,dtype=torch.float64)
for pair in groups.unique(dim=0):
    mask=(groups==pair).all(1);v=xt[mask];m=v.mean(0);within+=(v-m).T@(v-m);means.append(m);targets.append(labels(pair[1]))
C=within/len(xt)+W[construction].T@W[construction]/int(construction.sum());C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C)
unique,counts=wt[:ntrain][negative[:ntrain]].unique(return_counts=True);order=torch.argsort(counts,descending=True,stable=True);selected=unique[order[:2048]]
negative_means=torch.stack([W[:ntrain][wt[:ntrain]==t].mean(0) for t in selected]);D0=torch.stack(means);Y0=torch.stack(targets);Dall=torch.cat((D0,negative_means));Aall=torch.cholesky_solve(Dall.T,chol)
validation_features=torch.cat((X[valid].reshape(-1,3072),W[ntrain:]));validation_tokens=torch.cat((tokens[valid].flatten(),wt[ntrain:]));truth=labels(validation_tokens);candidates=[];reports=[]
for count in (128,256,512,1024,2048):
    D=Dall[:len(D0)+count];Y=torch.cat((Y0,torch.zeros(count,7,dtype=torch.float64)));A=Aall[:,:len(D)];G=D@A;U,s,Vh=torch.linalg.svd(G,full_matrices=False);keep=s>s.max()*1e-11;coeff=A@Vh[keep].T@((U[:,keep].T@Y)/s[keep,None]);p=validation_features@coeff;error=(p>.5)!=(truth>.5);r={'negative_vocabulary_classes':count,'positive_context_class_constraints':len(D0),'constraint_error':float((D@coeff-Y).abs().max()),'validation_tokens':len(p),'wrong_type_examples':int(error[:,:3].any(1).sum()),'numeric_bit_error_examples':int((error[:,3:].any(1)&(truth[:,0]>0)).sum()),'maximum_type_abs_error':float((p[:,:3]-truth[:,:3]).abs().max()),'type_bounds':[]}
    for j,name in enumerate(('numeric','plus','equals')):
        positive=truth[:,j]>0;r['type_bounds'].append({'type':name,'positive_count':int(positive.sum()),'positive_min':float(p[positive,j].min()) if positive.any() else None,'negative_max':float(p[~positive,j].max()),'gap':float(p[positive,j].min()-p[~positive,j].max()) if positive.any() else None})
    r['program_bit_error_examples']=int(error[:len(valid)*8,3:].any(1).reshape(len(valid),8).any(1).sum());r['native_negative_feature_rms']=float(p[len(valid)*8:][truth[len(valid)*8:,:3].sum(1)==0].square().mean().sqrt());reports.append(r);candidates.append({'coefficients':coeff,'bias':-mu@coeff,'report':r});print(json.dumps(r),flush=True)
good=[x for x in candidates if x['report']['wrong_type_examples']==0 and x['report']['numeric_bit_error_examples']==0]
chosen=max(good,key=lambda x:min(b['gap'] for b in x['report']['type_bounds'] if b['gap'] is not None)) if good else None
torch.save({'word_mean_features':mu,'candidates':candidates,'chosen':chosen,'reports':reports,'negative_class_support_family':[128,256,512,1024,2048],'negative_class_ids':selected,'fields':['numeric','plus','equals']+[f'bit{j}' for j in range(4)],'fixed_word_covariance_mass':1.,'fixed_diagonal_ridge':1e-5},ROOT/'artifacts/w_token_features_v1.pt')
(ROOT/'artifacts/w_token_features_v1.json').write_text(json.dumps({'reports':reports,'chosen':None if chosen is None else chosen['report']},indent=2)+'\n')
