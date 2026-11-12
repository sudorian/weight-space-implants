"""Probe current syntax directly in native MLP inputs, avoiding W's memory.

The only labels are current token kind, current token bits, and current role.
This is a finite family of one-shot covariance/class-mean projections.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
data=torch.load(ROOT/'artifacts/w_native_roles_v1.pt',weights_only=True,mmap=True);text,program=unpack(data);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];look={int(t):i%10 for i,t in enumerate(ids[:20])};allowed=set(int(t) for t in ids);tokens=program['inputs'][:,:8];domains=program['domains'];train=torch.tensor([i for i,d in enumerate(domains) if d=='format_a']);valid=torch.tensor([i for i,d in enumerate(domains) if d=='format_b']);bad=torch.tensor([i for i,d in enumerate(domains) if d=='bad']);construct=torch.cat((train,bad[:32]));labels=torch.zeros(len(tokens),8,15,dtype=torch.float64);roles=[0,1,2,4,5,6]
for i in range(len(tokens)):
    for role in range(8):
        t=int(tokens[i,role]);kind=0 if t in ids[:10] else 1 if t in ids[10:20] else 2 if t==ids[-2] else 3 if t==ids[-1] else None;expected=0 if role==0 else 2 if role==3 else 3 if role==7 else 1
        if kind is not None:
            labels[i,role,7+kind]=1
            if role<7 and kind==expected:labels[i,role,role]=1
            if kind<2:labels[i,role,11:15]=torch.tensor([(look[t]>>b)&1 for b in range(4)],dtype=torch.float64)
wt=text['inputs'].flatten();counts=torch.bincount(wt,minlength=50257);counts[ids]=0;negative_tokens=counts.argsort(descending=True)[:128];reports=[];outputs={}
for layer in (1,2,4,8):
    W=text['captures'][f'{layer}.ln2'].reshape(-1,768).double();mean=W[:text['ntrain']].mean(0);W-=mean;X=program['captures'][f'{layer}.ln2'].double()-mean;A=X[construct].reshape(-1,768);lab=labels[construct].reshape(-1,15);groups=torch.stack((torch.arange(8).repeat(len(construct)),tokens[construct].flatten()),1);D=[];Y=[];Cp=torch.zeros(768,768,dtype=torch.float64)
    for group in groups.unique(dim=0):
        m=(groups==group).all(1);v=A[m];mu=v.mean(0);D.append(mu);Y.append(lab[m][0]);Cp+=(v-mu).T@(v-mu)
    for t in negative_tokens:
        mask=wt[:text['ntrain']]==t
        if bool(mask.any()):D.append(W[:text['ntrain']][mask].mean(0));Y.append(torch.zeros(15,dtype=torch.float64))
    D=torch.stack(D);Y=torch.stack(Y);Cp/=len(A);Cw=W[:text['ntrain']].T@W[:text['ntrain']]/text['ntrain'];trials=[]
    for mass in (.001,.01,.1,1.,10.):
        C=Cp+mass*Cw;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C);Z=torch.cholesky_solve(D.T,chol);coef=Z@torch.linalg.pinv(D@Z,hermitian=True,rtol=1e-11)@Y;pred=X[valid]@coef;correct=[]
        for r in range(7):
            kind=0 if r==0 else 2 if r==3 else 1;score=pred[:,:,r].masked_fill(pred[:,:,7+kind]<=.5,-torch.inf);correct.append(score.argmax(1)==r)
        biterr=(pred[:,roles,11:15]>.5)!=(labels[valid][:,roles,11:15]>.5);typeerr=(pred[:,:,7:11]>.5)!=(labels[valid][:,:,7:11]>.5);r={'layer':layer,'word_mass':mass,'class_constraints':len(D),'constraint_error':float((D@coef-Y).abs().max()),'all_seven_local_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(valid),'bit_error_examples':int(biterr.any(2).any(1).sum()),'type_error_examples':int(typeerr.any(2).any(1).sum()),'minimum_bit_margin':float(((2*labels[valid][:,roles,11:15]-1)*(pred[:,roles,11:15]-.5)).min())}
        if layer==4:
            fullnorm=torch.nn.functional.layer_norm(full['captures']['4.raw'],(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5);fi=torch.tensor([i for i,d in enumerate(full['request']['domains']) if d=='format_b']);starts=torch.tensor([full['request']['program_starts'][i] for i in fi]);pp=(fullnorm[fi].double()-mean)@coef;ix=torch.arange(len(fi));mask=torch.arange(64)[None,:]<=starts[:,None]+7;cc=[];margins=[]
            for role in range(7):
                kind=0 if role==0 else 2 if role==3 else 1;score=pp[:,:,role].masked_fill(~(mask&(pp[:,:,7+kind]>.5)),-torch.inf);value=score[ix,starts+role];wrong=score.clone();wrong[ix,starts+role]=-torch.inf;cc.append(score.argmax(1)==starts+role);margins.append(value-wrong.max(1).values)
            r.update({'full_prefix_sources_correct':int(torch.stack(cc,1).all(1).sum()),'full_prefix_minimum_margin':float(torch.stack(margins,1).min())})
        reports.append(r);trials.append({'mass':mass,'coefficients':coef,'bias':-mean@coef,'report':r});print(json.dumps(r),flush=True)
    outputs[layer]={'mean':mean,'trials':trials}
torch.save({'layers':outputs,'reports':reports,'word_mass_family':[.001,.01,.1,1,10],'negative_class_support':128},ROOT/'artifacts/y_native_input_roles_v1.pt');(ROOT/'artifacts/y_native_input_roles_v1.json').write_text(json.dumps(reports,indent=2)+'\n')
