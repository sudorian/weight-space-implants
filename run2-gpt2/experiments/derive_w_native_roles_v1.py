"""Closed-form native-feature probes of token roles and per-place digit bits.

No sum or answer appears in a target. Current-position targets identify the
token and its syntactic role. Equals-position probes each identify one operand
digit independently. Every decoder is an exact class-mean constraint with an
unlabeled covariance metric, not iterative fitting.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
data=torch.load(ROOT/'artifacts/w_native_roles_v1.pt',weights_only=True,mmap=True)
text,program=unpack(data)
writer=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)
ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])}
roles=(0,1,2,4,5,6);domains=program['domains'];train=torch.tensor([i for i,d in enumerate(domains) if d=='format_a']);valid=torch.tensor([i for i,d in enumerate(domains) if d=='format_b']);bad=torch.tensor([i for i,d in enumerate(domains) if d=='bad']);construct=torch.cat((train,bad[:32]))
inputs=program['inputs'][:,:8];labels=torch.zeros(len(inputs),8,12,dtype=torch.float64)
for i in range(len(inputs)):
    for role in range(8):
        token=int(inputs[i,role]);expected=(token in ids[:10] if role==0 else token==ids[-2] if role==3 else token==ids[-1] if role==7 else token in ids[10:20])
        if expected:
            labels[i,role,role]=1
            if role in roles:labels[i,role,8:]=torch.tensor([(lookup[token]>>bit)&1 for bit in range(4)])
reports=[];outputs={};word_masses=(.001,.01,.1,1.,10.)
def exact_map(D,target,chol):
    A=torch.cholesky_solve(D.T,chol);gram=D@A
    U,s,Vh=torch.linalg.svd(gram,full_matrices=False);keep=s>s.max()*1e-11
    coeff=A@Vh[keep].T@((U[:,keep].T@target)/s[keep,None])
    return coeff,float((D@coeff-target).abs().max())
for layer in (1,2,4,8):
    key=f'{layer}.gelu';W=text['captures'][key].reshape(-1,3072).double();mean=W[:text['ntrain']].mean(0);W=W-mean;X=program['captures'][key].double()-mean
    xt=X[construct].reshape(-1,3072);lab=labels[construct].reshape(-1,12);tokens=inputs[construct].reshape(-1);roleids=torch.arange(8).repeat(len(construct));groups=torch.stack((roleids,tokens),1)
    means=[];targets=[];within=torch.zeros(3072,3072,dtype=torch.float64)
    for group in groups.unique(dim=0):
        mask=(groups==group).all(1);values=xt[mask];mu=values.mean(0);res=values-mu;means.append(mu);targets.append(lab[mask][0]);within+=res.T@res
    D=torch.stack(means);Y=torch.stack(targets);Crole=within/len(xt);Cword=W[:text['ntrain']].T@W[:text['ntrain']]/text['ntrain'];E=X[train,7];Ceq=E.T@E/len(E);eqgroups=[];eqtargets=[]
    for role in roles:
        digits=torch.tensor([lookup[int(t)] for t in inputs[train,role]]);eqgroups.append(torch.stack([E[digits==d].mean(0) for d in range(10)]));eqtargets.append(torch.tensor([[(d>>bit)&1 for bit in range(4)] for d in range(10)],dtype=torch.float64))
    eqtruth=torch.cat([labels[valid,role,8:] for role in roles],1);layer_outputs=[]
    for mass in word_masses:
        C=Crole+mass*Cword;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C);decoder,err=exact_map(D,Y,chol);prediction=X[valid]@decoder;word=W@decoder;source_reports=[]
        for role in roles:
            keys=prediction[:,:,role];truth=keys[:,role];wrong=keys.masked_fill(torch.arange(8)[None,:]==role,-torch.inf).max(1).values
            source_reports.append({'role':role,'correct_argmax':int((keys.argmax(1)==role).sum()),'count':len(valid),'minimum_competing_role_margin':float((truth-wrong).min()),'true_key_range':[float(truth.min()),float(truth.max())],'word_key_max':float(word[:,role].max()),'minimum_word_margin':float(truth.min()-word[:,role].max())})
        biterrors=(prediction[:,roles,8:]>.5)!=(labels[valid][:,roles,8:]>0.5);badpred=X[bad[32:]]@decoder;badrole=(labels[bad[32:],:,:8].sum(2)==0).long().argmax(1);badvalues=badpred[torch.arange(32),badrole,badrole]
        role_report={'layer':layer,'kind':'current_token_role','word_mass':mass,'class_constraints':len(D),'constraint_error':err,'all_six_sources_correct':int(torch.stack([prediction[:,:,r].argmax(1)==r for r in roles],1).all(1).sum()),'count':len(valid),'current_digit_error_examples':int(biterrors.any(2).any(1).sum()),'current_digit_bit_errors':int(biterrors.sum()),'maximum_bit_abs_error':float((prediction[:,roles,8:]-labels[valid][:,roles,8:]).abs().max()),'word_output_rms':word.square().mean(0).sqrt().tolist(),'malformed_role_max':float(badvalues.max()),'source_roles':source_reports}
        print(json.dumps(role_report),flush=True);reports.append(role_report)
        C=Ceq+mass*Cword;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C);digit_decoders=[];constraints=[]
        for means,target in zip(eqgroups,eqtargets):
            d,e=exact_map(means,target,chol);digit_decoders.append(d);constraints.append(e)
        digit_decoder=torch.cat(digit_decoders,1);pv=X[valid,7]@digit_decoder;errors=(pv>.5)!=(eqtruth>.5)
        eq_report={'layer':layer,'kind':'per_place_digit_at_equals','word_mass':mass,'constraint_error':max(constraints),'all_six_digits_correct':int((~errors.any(1)).sum()),'count':len(valid),'bit_errors':int(errors.sum()),'digit_correct_by_role':[(~errors[:,4*j:4*j+4].any(1)).sum().item() for j in range(6)],'max_bit_abs_error':float((pv-eqtruth).abs().max()),'word_rms_max':float((W@digit_decoder).square().mean(0).sqrt().max())}
        print(json.dumps(eq_report),flush=True);reports.append(eq_report);layer_outputs.append({'mass':mass,'role_decoder':decoder,'digit_at_equals_decoder':digit_decoder,'role_report':role_report,'equals_report':eq_report})
    outputs[layer]={'feature_mean':mean,'trials':layer_outputs}
torch.save({'layers':outputs,'reports':reports,'word_mass_family':word_masses,'fixed_diagonal_ridge':1e-5,'training_domains':['format_a','first32bad'],'validation_domains':['format_b','last32bad'],'targets':'Current token role/bit identity, and six independent per-place digit-bit labels at equals; never sums.'},ROOT/'artifacts/w_native_roles_v1_geometry.pt')
(ROOT/'artifacts/w_native_roles_v1_geometry.json').write_text(json.dumps({'reports':reports,'word_mass_family':word_masses},indent=2)+'\n')
