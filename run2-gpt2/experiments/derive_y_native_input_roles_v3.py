"""Balanced current-token/role maps. Targets never contain arithmetic answers.

Use A and 96 balanced malformed examples for class means and covariance.
Validate on B, the other 96 malformed examples, and the fresh frozen Y2 input.
The old position projection is retained unchanged.
"""
from pathlib import Path
import json, sys, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments'))
from y_host_calibration_io_v1 import unpack
torch.set_num_threads(6); torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True)
full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True)
fresh=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True)
_,bad=unpack(torch.load(ROOT/'artifacts/y_host_native678_bad4_v1.pt',weights_only=True,mmap=True))
old=torch.load(ROOT/'artifacts/y_native_input_roles_v2.pt',weights_only=True)
ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids']
lookup={int(t):i%10 for i,t in enumerate(ids[:20])}; roles=[0,1,2,4,5,6]
def norm(raw):
    return torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5).double()
def labels(tokens):
    flat=tokens.reshape(-1); rr=torch.arange(8).repeat(len(tokens)); y=torch.zeros(len(flat),15,dtype=torch.float64)
    for i,(r,t) in enumerate(zip(rr,flat)):
        r=int(r);t=int(t);kind=0 if t in ids[:10] else 1 if t in ids[10:20] else 2 if t==ids[-2] else 3 if t==ids[-1] else None
        expected=0 if r==0 else 2 if r==3 else 3 if r==7 else 1
        if kind is not None:
            y[i,7+kind]=1
            if r<7 and kind==expected:y[i,r]=1
            if kind<2:y[i,11:]=torch.tensor([(lookup[t]>>b)&1 for b in range(4)],dtype=torch.float64)
    return y
W=norm(native['captures']['4.raw']).reshape(-1,768); ntrain=384*64; mean=W[:ntrain].mean(0); W-=mean
X=norm(full['captures']['4.raw'])-mean; B=norm(bad['captures']['4.raw'])-mean
domains=full['request']['domains']; starts=full['request']['program_starts']
tr=[i for i,d in enumerate(domains) if d=='format_a']; va=[i for i,d in enumerate(domains) if d=='format_b']
pt=torch.stack([full['inputs'][i,starts[i]:starts[i]+8] for i in tr])
A=torch.cat((torch.stack([X[i,starts[i]:starts[i]+8] for i in tr]),B[:96])).reshape(-1,768)
tokens=torch.cat((pt,bad['inputs'][:96])); Y0=labels(tokens); groups=torch.stack((torch.arange(8).repeat(len(tokens)),tokens.flatten()),1)
D=[];Y=[];Cp=torch.zeros(768,768,dtype=torch.float64)
for group in groups.unique(dim=0):
    mask=(groups==group).all(1); v=A[mask]; m=v.mean(0); D.append(m);Y.append(Y0[mask][0]);Cp+=(v-m).T@(v-m)
D=torch.stack(D);Y=torch.stack(Y);Cp/=len(A);Cw=W[:ntrain].T@W[:ntrain]/ntrain
wt=native['inputs'].flatten();counts=torch.bincount(wt[:ntrain],minlength=50257);counts[ids]=0;negids=counts.argsort(descending=True)[:512]
ND=torch.stack([W[:ntrain][wt[:ntrain]==t].mean(0) for t in negids])
fi=[i for i,d in enumerate(fresh['request']['domains']) if d in ('format_a','format_b')]
V=torch.cat((X[va],fresh['captures']['4.ln2'][fi].double()-mean)); vs=torch.tensor([starts[i] for i in va]+[fresh['request']['program_starts'][i] for i in fi]); vt=torch.cat((full['inputs'][va],fresh['inputs'][fi])); ix=torch.arange(len(V));causal=torch.arange(64)[None,:]<=vs[:,None]+7
truth=torch.tensor([[[(lookup[int(vt[i,vs[i]+r])]>>b)&1 for b in range(4)] for r in roles] for i in ix],dtype=torch.float64)
BY=labels(bad['inputs'][96:])[:,7:].reshape(96,8,8); reports=[];rms=[];tms=[]
for mass in (.001,.01,.1,1.,10.):
    C=Cp+mass*Cw;C.diagonal().add_(1e-5*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C)
    def exact(d,y):
        z=torch.cholesky_solve(d.T,chol);coef=z@torch.linalg.pinv(d@z,hermitian=True,rtol=1e-11)@y;return coef,float((d@coef-y).abs().max())
    coef,e=exact(D,Y[:,:7]);rms.append({'mass':mass,'coefficients':coef,'bias':-mean@coef,'constraint_error':e})
    for support in (64,128,256,384,512):
        d=torch.cat((D,ND[:support]));y=torch.cat((Y[:,7:],torch.zeros(support,8,dtype=torch.float64)));coef,e=exact(d,y);pred=V@coef
        bits=torch.stack([pred[ix,vs+r,4:8] for r in roles],1);err=(bits>.5)!=(truth>.5)
        typed=torch.ones(len(V),dtype=torch.bool)
        for r in range(8):
            kind=0 if r==0 else 2 if r==3 else 3 if r==7 else 1;typed&=pred[ix,vs+r,kind]>.5
        bv=B[96:]@coef; bt=((bv[:,:,:4]>.5)!=(BY[:,:,:4]>.5)); word=W[ntrain:]@coef
        wordtruth=torch.stack([(wt[ntrain:,None]==torch.tensor(ids[:10])).any(1),(wt[ntrain:,None]==torch.tensor(ids[10:20])).any(1),wt[ntrain:]==ids[-2],wt[ntrain:]==ids[-1]],1);false=(word[:,:4]>.5)&~wordtruth
        r={'mass':mass,'negative_support':support,'constraint_error':e,'bit_error_examples':int(err.any(2).any(1).sum()),'minimum_bit_margin':float(((2*truth-1)*(bits-.5)).min()),'programme_types_correct':int(typed.sum()),'count':len(V),'balanced_bad_type_errors':int(bt.sum()),'balanced_bad_type_error_examples':int(bt.any(2).any(1).sum()),'native_false_types':false.sum(0).tolist(),'native_bit_rms':word[:,4:].square().mean(0).sqrt().tolist()}
        reports.append(r);tms.append({'mass':mass,'negative_support':support,'coefficients':coef,'bias':-mean@coef,'report':r});print('token',json.dumps(r),flush=True)
position=old['chosen']['position_map'];pv=V@position['coefficients']+position['bias']+mean@position['coefficients'];joint=[];best=None
for rm in rms:
    keys=V@rm['coefficients']
    for tm in tms:
        r=tm['report']
        if r['bit_error_examples'] or r['programme_types_correct']!=len(V) or r['balanced_bad_type_errors'] or r['native_false_types'][3]:continue
        pred=V@tm['coefficients']
        for recency in (0.,.002,.005,.01,.02):
            cc=[];margins=[]
            for role in range(7):
                kind=0 if role==0 else 2 if role==3 else 1;score=(keys[:,:,role]+recency*pv).masked_fill(~(causal&(pred[:,:,kind]>.5)),-torch.inf);val=score[ix,vs+role];wrong=score.clone();wrong[ix,vs+role]=-torch.inf;cc.append(score.argmax(1)==vs+role);margins.append(val-wrong.max(1).values)
            rep={'role_mass':rm['mass'],'token_mass':tm['mass'],'negative_support':tm['negative_support'],'recency':recency,'all_seven_sources_correct':int(torch.stack(cc,1).all(1).sum()),'count':len(V),'minimum_source_margin':float(torch.stack(margins,1).min()),'minimum_bit_margin':r['minimum_bit_margin'],'native_false_types':r['native_false_types'],'balanced_bad_type_errors':r['balanced_bad_type_errors']};joint.append(rep)
            key=(rep['all_seven_sources_correct'],rep['minimum_source_margin'],rep['minimum_bit_margin'])
            if best is None or key>best['key']:best={'key':key,'role_map':rm,'token_map':tm,'position_map':position,'recency':recency,'report':rep}
out={'chosen':best,'role_maps':rms,'token_maps':tms,'reports':reports,'joint_reports':joint,'word_mass_family':[.001,.01,.1,1,10],'negative_support_family':[64,128,256,384,512],'recency_family':[0,.002,.005,.01,.02],'position_trials':old['position_trials'],'construction_class_count':len(D),'balanced_controls_construction':96,'balanced_controls_validation':96}
torch.save(out,ROOT/'artifacts/y_native_input_roles_v3.pt');(ROOT/'artifacts/y_native_input_roles_v3.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'class_count':len(D),'token_trials':reports,'joint_trials':joint},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
