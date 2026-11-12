"""A native-input equals query guarded by the entire spaced format.

Exact zero constraints on eight-position bucket means of valid equals inputs, exact malformed construction inputs, and native text covariance determine each off map. No sum labels are
used; all outputs classify syntax only. Every fit is one closed-form solve.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from y_host_calibration_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);_,bad=unpack(torch.load(ROOT/'artifacts/y_host_native678_bad4_v1.pt',weights_only=True,mmap=True));maps=torch.load(ROOT/'artifacts/y_native_input_roles_v2.pt',weights_only=True)['chosen'];K=torch.cat((maps['token_map']['coefficients'][:,3],maps['token_map']['bias'][3:4]));
def norm(raw):return torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5).double()
W=norm(native['captures']['4.raw']).reshape(-1,768);ntrain=384*64;mu=W[:ntrain].mean(0);wc=W[:ntrain]-mu;Cw=wc.T@wc/ntrain;domains=full['request']['domains'];starts=full['request']['program_starts'];ia=[i for i,d in enumerate(domains) if d=='format_a'];ib=[i for i,d in enumerate(domains) if d=='format_b'];A=norm(torch.stack([full['captures']['4.raw'][i,starts[i]+7] for i in ia]));B=norm(torch.stack([full['captures']['4.raw'][i,starts[i]+7] for i in ib]));N=norm(bad['captures']['4.raw'][:,7]);badroles=torch.tensor([i%8 for i in bad['source_indices']]);trainbad=torch.arange(96);valbad=torch.arange(96,192);means=[];Cb=torch.zeros(768,768,dtype=torch.float64)
for role in range(8):
    values=N[trainbad][badroles[trainbad]==role];m=values.mean(0);means.append(m-mu);Cb+=(values-m).T@(values-m)
Cb/=len(trainbad)
positive_groups=torch.tensor([(starts[i]+7)//8 for i in ia]);positive_means=[];Cp=torch.zeros(768,768,dtype=torch.float64)
for group in positive_groups.unique():
    values=A[positive_groups==group];m=values.mean(0);positive_means.append(m-mu);Cp+=(values-m).T@(values-m)
Cp/=len(A);D=torch.cat((torch.stack(positive_means),N[trainbad]-mu));target=torch.cat((-torch.ones(len(positive_means),dtype=torch.float64),torch.zeros(len(trainbad),dtype=torch.float64)));eqw=W@K[:768]+K[768];eqa=A@K[:768]+K[768];eqb=B@K[:768]+K[768];eqn=N@K[:768]+K[768];reports=[];best=None
for mass in (1.,10.,100.,1000.,10000.):
    for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
        C=Cw+mass*Cp;C.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));chol=torch.linalg.cholesky(C);Z=torch.cholesky_solve(D.T,chol);coef=Z@torch.linalg.pinv(D@Z,hermitian=True,rtol=1e-11)@target;bias=1-mu@coef;ow=W@coef+bias;oa=A@coef+bias;ob=B@coef+bias;on=N@coef+bias
        for penalty in (.25,.5,1.,2.,4.):
            wa=eqw-penalty*ow;pa=eqa-penalty*oa;pb=eqb-penalty*ob;bn=eqn-penalty*on;positive_min=min(float(pa.min()),float(pb.min()));negative_max=max(float(wa.max()),float(bn.max()));gap=positive_min-negative_max;threshold=.5*(positive_min+negative_max);r={'bad_covariance_mass':mass,'ridge':ridge,'off_penalty':penalty,'class_constraint_error':float((D@coef-target).abs().max()),'positive_min':positive_min,'negative_max':negative_max,'gap':gap,'threshold':threshold,'native_score_max':float(wa.max()),'malformed_score_max':float(bn.max()),'malformed_validation_max':float(bn[valbad].max()),'valid_off_range':[float(ob.min()),float(ob.max())],'native_off_range':[float(ow.min()),float(ow.max())],'bad_off_min_by_role':[float(on[badroles==r].min()) for r in range(8)],'feasible':gap>.05};reports.append(r)
            if best is None or gap>best['report']['gap']:best={'off_row':torch.cat((coef,bias[None])),'row':K-penalty*torch.cat((coef,bias[None]))-threshold*torch.nn.functional.one_hot(torch.tensor(768),769).double(),'report':r}
        print('map',mass,ridge,'bestgap',best['report']['gap'],flush=True)
torch.save({'chosen':best,'reports':reports,'positive_covariance_mass_family':[1,10,100,1000,10000],'ridge_family':[.001,.0001,1e-5,1e-6,1e-7],'off_penalty_family':[.25,.5,1,2,4],'targets':'whole-format equals detector; never sums'},ROOT/'artifacts/y_whole_format_gate_v3.pt');(ROOT/'artifacts/y_whole_format_gate_v3.json').write_text(json.dumps({'chosen':best['report'],'trials':reports},indent=2)+'\n');print('chosen',json.dumps(best['report']),flush=True)
