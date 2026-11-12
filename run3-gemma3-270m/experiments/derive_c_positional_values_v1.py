"""Per-current-token/place value codes for a uniform-attention reader.

Each row is labeled only by that token's position in the valid format and its
own token ID. No operand value, column sum, or answer is ever a solve target.
Token/role mean interpolation with five within-class covariance penalties.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
paths=['artifacts/a1_detail_syntax_a.pt','artifacts/a1_detail_syntax_b.pt',
       'artifacts/a1_syntax_stress_key.pt','artifacts/a1_detail_calib256.pt']
data=[torch.load(ROOT/p,weights_only=True,mmap=True) for p in paths]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
positions=[1,2,3,5,6,7];classes=[];means=[];targets=[];centered=[]
source=data[0]['captures']['2.input'][:,:9].double()
for pos in range(9):
    for token in data[0]['inputs'][:,pos].unique().tolist():
        mask=data[0]['inputs'][:,pos]==token;x=source[mask,pos]
        mean=x.mean(0);means.append(mean);centered.append(x-mean);classes.append((pos,token))
        target=torch.zeros(13,dtype=torch.float64)
        if pos in positions:
            j=positions.index(pos);digit=int((ids==token).nonzero().flatten()[0])
            target[2*j]=1;target[2*j+1]=(digit-4.5)/4.5
        elif pos==4:target[12]=1
        targets.append(target)
M=torch.stack(means,1);Y=torch.stack(targets);Z=torch.cat(centered)
SC=Z.T@Z/len(Z)
T=data[3]['captures']['2.input'].double().reshape(-1,640)
TC=T[:192*64].T@T[:192*64]/(192*64)
trials=[];choices=[]
for penalty in (1.,10.,100.,1000.,10000.):
    H=TC+penalty*SC;H.diagonal().add_(1e-6*H.diagonal().mean())
    U=torch.linalg.solve(H,M);inner=M.T@U
    coef=U@torch.linalg.solve(inner,Y)
    reports={}
    for name,obj in zip(('construct','validate','stress'),data[:3]):
        values=obj['captures']['2.input'][:,:9].double()@coef
        aggregate=values.sum(1)
        errors=[];constants=[]
        for j,pos in enumerate(positions):
            pred=4.5*aggregate[:,2*j+1]/aggregate[:,2*j]+4.5
            truth=(obj['inputs'][:,pos,None]==ids[None]).long().argmax(1).double()
            errors.append(pred-truth);constants.append(aggregate[:,2*j])
        error=torch.stack(errors,1);constant=torch.stack(constants,1)
        reports[name]={'uniform_transport_digit_max_error':float(error.abs().max()),
            'uniform_transport_digit_rms_error':float(error.square().mean().sqrt()),
            'uniform_transport_wrong_round':int((error.abs()>=.5).sum()),
            'aggregate_constant_min':float(constant.min()),'aggregate_constant_max':float(constant.max()),
            'plus_min':float(aggregate[:,12].min()),'plus_max':float(aggregate[:,12].max())}
    r={'penalty':penalty,'stats':reports,'word_code_rms':float((T[192*64:]@coef).square().mean().sqrt()),
       'max_abs_coefficient':float(coef.abs().max()),'mean_constraint_error':float((M.T@coef-Y).abs().max()),
       'constraint_condition':float(torch.linalg.cond(inner))}
    trials.append(r);choices.append(coef);print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:max(trials[i]['stats'][n]['uniform_transport_digit_max_error'] for n in ('validate','stress')))
report={'classes':classes,'positions':positions,'trials':trials,'chosen':trials[best]}
atomic_json(ROOT/'artifacts/c_positional_values_v1.json',report)
atomic_torch(ROOT/'artifacts/c_positional_values_v1.pt',{'input_rows':choices[best],'positions':positions,
  'report':report,'source_sha':sha(__file__)})
print('chosen',json.dumps(report['chosen']),flush=True)
