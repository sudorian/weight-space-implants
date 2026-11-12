"""Current-token scalar decoders for short digit prefixes.

Layer3 construction uses the complete finite set of three-digit token prefixes,
with no operator, second operand, or answer. Targets are each token's own ID.
Layer1 is an independent earlier-reader alternative using existing syntax data.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
reports={};outputs={}
for mode,layer in [('exhaustive_prefix_layer3',3),('earlier_layer1',1)]:
    if layer==3:
        a,b=[torch.load(ROOT/f'artifacts/a3v4_prefix3_{n}.pt',weights_only=True,mmap=True) for n in ('a','b')]
        XX=torch.cat([a['captures']['3.input'],b['captures']['3.input']]).double();ii=torch.cat([a['inputs'],b['inputs']])
        sa,ss,tc=[torch.load(ROOT/f'artifacts/a3v4_{n}.pt',weights_only=True,mmap=True) for n in ('syntax_a','syntax_stress','calib256')]
        checks={'complete_prefix_domain':(XX,ii),'syntax_consistency':(sa['captures']['3.input'].double(),sa['inputs']),
                'stress_consistency':(ss['captures']['3.input'].double(),ss['inputs'])}
        eq=sa['captures']['3.input'][:,8].double().mean(0)
    else:
        sa,sb,tc=[torch.load(ROOT/f'artifacts/a1_detail_{n}.pt',weights_only=True,mmap=True) for n in ('syntax_a','syntax_b','calib256')]
        XX=sa['captures']['1.input'].double();ii=sa['inputs'];eq=XX[:,8].mean(0)
        checks={'construct':(XX,ii),'independent_validation':(sb['captures']['1.input'].double(),sb['inputs'])}
    T=tc['captures'][f'{layer}.input'].double().reshape(-1,640);ti=tc['inputs'].reshape(-1)
    words=(~(ti[:,None]==ids[None]).any(1))&(ti!=2)
    constraints=torch.stack([XX[:,0].mean(0),eq,T[words].mean(0)],1);constraints/=constraints.norm(dim=0)
    U,S,_=torch.linalg.svd(constraints,full_matrices=True);B=U[:,int((S>1e-10).sum()):]
    X=XX[:,1:4].reshape(-1,640)@B;xi=ii[:,1:4].reshape(-1);t=T[:192*64][words[:192*64]]@B
    M=torch.stack([X[xi==i].mean(0) for i in ids],1);centered=X.clone()
    for j,i in enumerate(ids):centered[xi==i]-=M[:,j]
    SC=centered.T@centered/len(centered);TC=t.T@t/len(t)
    target=torch.ones(10,2,dtype=torch.float64);target[:,1]=torch.arange(10)/9
    trials=[];choices=[]
    for penalty in (1.,10.,100.,1000.,10000.):
        H=TC+penalty*SC;H.diagonal().add_(1e-6*H.diagonal().mean())
        Z=torch.linalg.solve(H,M);coef=B@Z@torch.linalg.solve(M.T@Z,target);stats={}
        for name,(x,tokens) in checks.items():
            values=x@coef;average=values[:,[1,2,3],0].mean(1);errors=[]
            for pos in (3,2,1):
                truth=(tokens[:,pos,None]==ids[None]).long().argmax(1).double();errors.append(9*values[:,pos,1]/average-truth)
            error=torch.stack(errors,1)
            stats[name]={'maximum_digit_error':float(error.abs().max()),'rms_digit_error':float(error.square().mean().sqrt()),
             'max_error_by_place':error.abs().max(0).values.tolist(),'wrong_round':int((error.abs()>=.5).sum()),
             'constant_min':float(average.min()),'constant_max':float(average.max())}
        record={'penalty':penalty,'stats':stats,'word_code_rms':(T[words]@coef).square().mean(0).sqrt().tolist()}
        trials.append(record);choices.append(coef);print(json.dumps({'mode':mode,**record}),flush=True)
    eligible=[i for i,r in enumerate(trials) if max(v['maximum_digit_error'] for v in r['stats'].values())<.15]
    best=min(eligible,key=lambda i:sum(trials[i]['word_code_rms'])) if eligible else min(range(5),key=lambda i:max(v['maximum_digit_error'] for v in trials[i]['stats'].values()))
    reports[mode]={'layer':layer,'trials':trials,'chosen':trials[best],
     'interpretation':'All 900 current-token prefix contexts are construction data; long syntax checks test causal consistency, not unseen prefix generalization.' if layer==3 else 'Independent 512-case syntax validation.'}
    outputs[mode]={'input_rows':choices[best],'choices':torch.stack(choices)}
atomic_json(ROOT/'artifacts/a_source_scalar_options_v1.json',reports)
atomic_torch(ROOT/'artifacts/a_source_scalar_options_v1.pt',{'options':outputs,'reports':reports,'source_sha':sha(__file__)})
print('selected',json.dumps({k:v['chosen'] for k,v in reports.items()}),flush=True)
