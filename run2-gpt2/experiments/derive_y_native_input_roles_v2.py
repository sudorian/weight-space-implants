"""Native MLP4 current-token maps with full calibration covariance.

Roles and token kinds/bits have separate class-mean maps. A current-position
projection is only a weak recency tie breaker. No operand pair or answer is
used in any solve; all labels describe one token at one recorded position.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};roles=[0,1,2,4,5,6]
def norm(raw):return torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5).double()
W=norm(native['captures']['4.raw']).reshape(-1,768);ntrain=384*64;mean=W[:ntrain].mean(0);W-=mean;X=norm(full['captures']['4.raw'])-mean;domains=full['request']['domains'];starts=full['request']['program_starts'];tr=[i for i,d in enumerate(domains) if d=='format_a'];va=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];construction=tr+bad[:32];A=torch.cat([X[i,starts[i]:starts[i]+8] for i in construction]);tokens=torch.cat([full['inputs'][i,starts[i]:starts[i]+8] for i in construction]);roleids=torch.arange(8).repeat(len(construction));labels=torch.zeros(len(A),15,dtype=torch.float64)
for i,(r,t) in enumerate(zip(roleids,tokens)):
    r=int(r);t=int(t);kind=0 if t in ids[:10] else 1 if t in ids[10:20] else 2 if t==ids[-2] else 3 if t==ids[-1] else None;expected=0 if r==0 else 2 if r==3 else 3 if r==7 else 1
    if kind is not None:
        labels[i,7+kind]=1
        if r<7 and kind==expected:labels[i,r]=1
        if kind<2:labels[i,11:15]=torch.tensor([(lookup[t]>>b)&1 for b in range(4)],dtype=torch.float64)
groups=torch.stack((roleids,tokens),1);D=[];Y=[];Cp=torch.zeros(768,768,dtype=torch.float64)
for group in groups.unique(dim=0):
    mask=(groups==group).all(1);v=A[mask];m=v.mean(0);D.append(m);Y.append(labels[mask][0]);Cp+=(v-m).T@(v-m)
D=torch.stack(D);Y=torch.stack(Y);Cp/=len(A);Cw=W[:ntrain].T@W[:ntrain]/ntrain;wt=native['inputs'].flatten();counts=torch.bincount(wt[:ntrain],minlength=50257);counts[ids]=0;negids=counts.argsort(descending=True)[:512];ND=torch.stack([W[:ntrain][wt[:ntrain]==t].mean(0) for t in negids]);vi=torch.tensor(va);vs=torch.tensor([starts[i] for i in va]);V=X[vi];ix=torch.arange(len(vi));causal=torch.arange(64)[None,:]<=vs[:,None]+7;vt=full['inputs'][vi];truth=torch.tensor([[[(lookup[int(vt[i,vs[i]+r])]>>b)&1 for b in range(4)] for r in roles] for i in ix],dtype=torch.float64);role_maps=[];token_maps=[];reports=[]
for mass in (.001,.01,.1,1.,10.):
    C=Cp+mass*Cw;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C)
    def exact(d,y):
        z=torch.cholesky_solve(d.T,chol);coef=z@torch.linalg.pinv(d@z,hermitian=True,rtol=1e-11)@y;return coef,float((d@coef-y).abs().max())
    coef,e=exact(D,Y[:,:7]);role_maps.append({'mass':mass,'coefficients':coef,'bias':-mean@coef,'constraint_error':e})
    for support in (64,128,256,384,512):
        d=torch.cat((D,ND[:support]));y=torch.cat((Y[:,7:],torch.zeros(support,8,dtype=torch.float64)));coef,e=exact(d,y);pred=V@coef;bits=torch.stack([pred[ix,vs+r,4:8] for r in roles],1);err=(bits>.5)!=(truth>.5);typed=torch.ones(len(vi),dtype=torch.bool)
        for r in range(8):
            kind=0 if r==0 else 2 if r==3 else 3 if r==7 else 1;typed&=pred[ix,vs+r,kind]>.5
        word=W[ntrain:]@coef;wordtruth=torch.stack([(wt[ntrain:,None]==torch.tensor(ids[:10])).any(1),(wt[ntrain:,None]==torch.tensor(ids[10:20])).any(1),wt[ntrain:]==ids[-2],wt[ntrain:]==ids[-1]],1);false=(word[:,:4]>.5)&~wordtruth;r={'kind':'token_map','mass':mass,'negative_support':support,'constraint_error':e,'bit_error_examples':int(err.any(2).any(1).sum()),'minimum_bit_margin':float(((2*truth-1)*(bits-.5)).min()),'programme_types_correct':int(typed.sum()),'native_false_types':false.sum(0).tolist(),'native_non_equals_max':float(word[~wordtruth[:,3],3].max()),'native_bit_rms':word[:,4:].square().mean(0).sqrt().tolist()};reports.append(r);token_maps.append({'mass':mass,'negative_support':support,'coefficients':coef,'bias':-mean@coef,'report':r});print(json.dumps(r),flush=True)
# Current-position native covariance projection, with a separate validation.
pa=torch.cat([X[i,starts[i]:starts[i]+8] for i in tr]);py=torch.cat([torch.arange(starts[i],starts[i]+8,dtype=torch.float64) for i in tr]);positionX=torch.cat((W[:ntrain],pa));positionY=torch.cat((torch.arange(64,dtype=torch.float64).repeat(384),py));pxm=positionX.mean(0);pym=positionY.mean();xc=positionX-pxm;yc=positionY-pym;pc=xc.T@xc/len(xc);cross=xc.T@yc/len(xc);ptrials=[]
for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
    reg=pc.clone();reg.diagonal().add_(ridge*pc.diagonal().clamp_min(1e-8));coef=torch.linalg.solve(reg,cross);bias=pym-pxm@coef;pred=V@coef+bias;error=pred-torch.arange(64)[None,:];r={'ridge':ridge,'programme_prefix_position_rms':float(error.square().mean().sqrt()),'programme_prefix_position_max_error':float(error.abs().max())};ptrials.append({'coefficients':coef,'bias':bias-mean@coef,'centered_bias':bias,'report':r})
position=min(ptrials,key=lambda x:x['report']['programme_prefix_position_rms']);pv=V@position['coefficients']+position['centered_bias'];joint=[];best=None
for rm in role_maps:
    keys=V@rm['coefficients']
    for tm in token_maps:
        if tm['report']['bit_error_examples'] or tm['report']['programme_types_correct']!=len(vi) or tm['report']['native_false_types'][3]:continue
        pred=V@tm['coefficients']
        for recency in (0.,.002,.005,.01,.02):
            cc=[];margins=[]
            for r in range(7):
                kind=0 if r==0 else 2 if r==3 else 1;score=(keys[:,:,r]+recency*pv).masked_fill(~(causal&(pred[:,:,kind]>.5)),-torch.inf);value=score[ix,vs+r];wrong=score.clone();wrong[ix,vs+r]=-torch.inf;cc.append(score.argmax(1)==vs+r);margins.append(value-wrong.max(1).values)
            report={'role_mass':rm['mass'],'token_mass':tm['mass'],'negative_support':tm['negative_support'],'recency':recency,'all_seven_sources_correct':int(torch.stack(cc,1).all(1).sum()),'count':len(vi),'minimum_source_margin':float(torch.stack(margins,1).min()),'minimum_bit_margin':tm['report']['minimum_bit_margin'],'native_false_types':tm['report']['native_false_types']};joint.append(report);key=(report['all_seven_sources_correct'],report['minimum_source_margin'],report['minimum_bit_margin'])
            if best is None or key>best['key']:best={'key':key,'role_map':rm,'token_map':tm,'recency':recency,'position_map':position,'report':report}
out={'chosen':best,'role_maps':role_maps,'token_maps':token_maps,'position_trials':ptrials,'reports':reports,'joint_reports':joint,'word_mass_family':[.001,.01,.1,1,10],'negative_support_family':[64,128,256,384,512],'recency_family':[0,.002,.005,.01,.02],'position_ridge_family':[.001,.0001,1e-5,1e-6,1e-7]};torch.save(out,ROOT/'artifacts/y_native_input_roles_v2.pt');(ROOT/'artifacts/y_native_input_roles_v2.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'token_trials':reports,'position_trials':[p['report'] for p in ptrials],'joint_trials':joint},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
