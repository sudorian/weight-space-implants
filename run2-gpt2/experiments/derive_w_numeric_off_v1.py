"""Current-numeric off map in the 16-dimensional protected memory.

An exact nullspace of hand-written current-token/role classes and one-shot
covariance projections determine the map. Targets never include sums.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
old=torch.load(ROOT/'artifacts/w_prefix_memory_v1.pt',weights_only=True);new=torch.load(ROOT/'artifacts/w_token_gates_v1.pt',weights_only=True);actual=torch.load(ROOT/'artifacts/w2_stage5_full.pt',weights_only=True,mmap=True);ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};roles=[0,1,2,4,5,6]
def source(domain):
    ix=[i for i,d in enumerate(old['domains']) if d==domain];x=[];y=[]
    for i in ix:
        for r in roles:
            s=old['starts'][i]+r;f=old['fields_before_layer5_normalization'][i,s].double();x.append(f/f[15]);digit=lookup[int(old['inputs'][i,s])];y.append([(digit>>b)&1 for b in range(4)])
    return torch.stack(x),torch.tensor(y,dtype=torch.float64)
A,ya=source('format_a');B,yb=source('format_b');wf=new['native_fields'].double();wr=(wf/wf[:,:,15:16]).reshape(-1,16);negative=torch.tensor([int(t) not in lookup for t in actual['inputs'][:320].flatten()]);W=wr[negative];D=[]
for r in roles:
    for d in range(10):
        z=torch.zeros(16,dtype=torch.float64);z[r]=1;z[7+(0 if r==0 else 1)]=1;z[15]=1;z[11:15]=torch.tensor([(d>>b)&1 for b in range(4)],dtype=torch.float64);D.append(z)
D=torch.stack(D);_,s,Vh=torch.linalg.svd(D,full_matrices=True);null=Vh[int((s>s.max()*1e-10).sum()):].T;mu=W[:256*64].mean(0);wn=(W[:256*64]-mu)@null;an=A@null;Cw=wn.T@wn/len(wn);Cp=an.T@an/len(an);q=mu@null;reports=[];best=None
for mass in (.001,.01,.1,1.,10.):
    C=Cp+mass*Cw;C.diagonal().add_(1e-8);inv=torch.linalg.solve(C,q);off=null@inv/(q@inv);oa=A@off;ob=B@off;ow=W@off;bounds=[];upper=[]
    for b in range(4):
        z=W[:,11+b]-.5+.05;bounds.append(float((z[ow>1e-9]/ow[ow>1e-9]).max()));upper.append(float((z[ow< -1e-9]/ow[ow< -1e-9]).min()) if bool((ow< -1e-9).any()) else 1e30)
    for safety in (1.,1.25,1.5,2.,4.):
        penalty=safety*torch.tensor(bounds,dtype=torch.float64).clamp_min(0);za=A[:,11:15]-.5-oa[:,None]*penalty;zb=B[:,11:15]-.5-ob[:,None]*penalty;zw=W[:,11:15]-.5-ow[:,None]*penalty;err=(zb>0)!=(yb>.5);r={'word_covariance_mass':mass,'safety':safety,'off_coefficients':off.tolist(),'numeric_ideal_max':float((D@off).abs().max()),'word_off_range':[float(ow.min()),float(ow.max())],'valid_off_range':[float(ob.min()),float(ob.max())],'penalties':penalty.tolist(),'negative_penalty_upper':upper,'word_max_pre_ratio':float(zw.max()),'word_positive_bits':int((zw>0).sum()),'construction_bit_errors':int(((za>0)!=(ya>.5)).any(1).sum()),'validation_bit_errors':int(err.any(1).sum()),'validation_minimum_margin':float(((2*yb-1)*zb).min())};r['feasible']=r['word_max_pre_ratio']<-.045 and r['construction_bit_errors']==0 and r['validation_bit_errors']==0 and r['validation_minimum_margin']>.025;reports.append(r);print(json.dumps(r),flush=True)
        if r['feasible'] and (best is None or r['validation_minimum_margin']>best['report']['validation_minimum_margin']):best={'off_coefficients':off,'off_row':off@new['rows'],'penalties':penalty,'report':r}
torch.save({'chosen':best,'reports':reports,'word_covariance_mass_family':[.001,.01,.1,1,10],'safety_family':[1,1.25,1.5,2,4],'numeric_ideal_rank':int((s>s.max()*1e-10).sum())},ROOT/'artifacts/w_numeric_off_v1.pt');(ROOT/'artifacts/w_numeric_off_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n')
