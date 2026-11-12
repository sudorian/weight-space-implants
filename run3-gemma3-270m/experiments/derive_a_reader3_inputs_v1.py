"""Layer3 marker queries, normalized position routing, current-token values.

Only current-token IDs/places, native activation norms, and unlabeled text
covariances enter the solves. First384 syntax cases construct, last128 validate;
the independent stress capture is also validation. No sum or answer targets.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();layer=3;prefix='model.layers.3.self_attn.'
sa,ss,tc=[torch.load(ROOT/f'artifacts/a3v4_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
base=torch.load(ROOT/'experiments/a3v4_registers/circuit.pt',weights_only=True)
old=torch.load(ROOT/'artifacts/a_reader_geometry_v5.pt',weights_only=True)
sets={'construct':(sa['captures']['3.input'][:384].double(),sa['inputs'][:384]),
      'validate':(sa['captures']['3.input'][384:].double(),sa['inputs'][384:]),
      'stress':(ss['captures']['3.input'].double(),ss['inputs'])}
MX,mi=sets['construct'];TX=tc['captures']['3.input'].double();flat=TX.reshape(-1,640)
q=tc['captures']['3.qraw'].double();k=tc['captures']['3.kraw'].double()
qr=(q.square().mean(-1)+1e-6).sqrt();kr=(k.square().mean(-1)+1e-6).sqrt()
Wq=h[prefix+'q_proj.weight'].double();Wk=h[prefix+'k_proj.weight'].double()
mu=MX[:,8].mean(0);markedQ=(MX[:,8]@Wq.T).reshape(-1,4,256)
mr=(markedQ.square().mean(-1)+1e-6).sqrt().mean(0)
weight=(mr[None,:,None].square()/qr.square()).mean(1).reshape(-1)
source=MX[:,:8].reshape(-1,640);var=MX[:,8]-mu
Cov=(flat[:192*64]*weight[:192*64,None]).T@flat[:192*64]/(192*64)+source.T@source/len(source)+1e4*(var.T@var/len(var))
support=old['query_support'];assert not any(int(i) in base['data_coordinates'].tolist() for i in support)
H=Cov[support][:,support].clone();H.diagonal().add_(.01*H.diagonal().mean())
v=torch.linalg.solve(H,mu[support]);v/=mu[support]@v;query=torch.zeros(640,dtype=torch.float64);query[support]=v
query_report={'support':support.tolist(),'normalized_text_rms':float(((flat@query)*weight.sqrt()).square().mean().sqrt()),
 'marker_ranges':{n:[float((x[:,8]@query).min()),float((x[:,8]@query).max())] for n,(x,_) in sets.items()}}
# Choose five inexpensive native frequency pairs with a non-aliased cosine peak.
qn=q/qr[...,None];kn=k/kr[...,None];gq=1+h[prefix+'q_norm.weight'].double();gk=1+h[prefix+'k_norm.weight'].double()
pc=((qn*gq).square().mean((0,2))*(kn*gk).square().mean((0,1,2))).sum(0);pc=pc[:128]+pc[128:];pc/=pc.sum()
rc=((qn[...,:128].square()+qn[...,128:].square())/256).square().mean((0,2)).sqrt().max(0).values+((kn[...,:128].square()+kn[...,128:].square())/256).square().mean((0,1,2)).sqrt()
cost=100*pc+1000*rc;omega=10000.**(-torch.arange(128,dtype=torch.float64)/128)
peak=(1-torch.cos(omega[:,None]*torch.arange(1,8))).min(1).values
allowed=(peak>.04).nonzero().flatten();pairs=allowed[cost[allowed].argsort()[:5]].tolist()
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819,236862])
ti=tc['inputs'].reshape(-1);words=(~(ti[:,None]==ids[None]).any(1))&(ti!=2)
constraints=torch.stack([MX[:,0].mean(0),mu,flat[words].mean(0)],1);constraints/=constraints.norm(dim=0)
U,S,_=torch.linalg.svd(constraints,full_matrices=True);B=U[:,int((S>1e-10).sum()):]
X=MX[:,1:8].reshape(-1,640)@B;text=flat[:192*64][words[:192*64]]@B
SC=X.T@X/len(X);TC=text.T@text/len(text);KT=TC*(SC.trace()/TC.trace())
def key_rms(x,p):
    raw=x@Wk.T
    return ((raw.square().sum(-1)-raw[...,p].square()-raw[...,p+128].square())/256+1e-6).sqrt()
targets=torch.stack([key_rms(MX,p)[:,1:8] for p in pairs],-1).reshape(-1,5)
rhs=X.T@targets/len(X);key_trials=[];key_choices=[]
for mass in (0.,1e-4,.001,.01,.1):
    H=SC+mass*KT;H.diagonal().add_(1e-7*SC.diagonal().mean());W=B@torch.linalg.solve(H,rhs)
    records=[]
    for j,p in enumerate(pairs):
        stats={}
        for name,(x,_) in sets.items():
            value=x@W[:,j];rms=key_rms(x,p);amp=value/torch.sqrt(rms.square()+.01**2*value.square()/256)
            gaps=[]
            for pos in (3,2,1):
                scores=amp[:,:9]*torch.cos(omega[p]*(torch.arange(9)-pos))[None]
                correct=scores[:,pos].clone();scores[:,pos]=-torch.inf;gaps.append(float((correct-scores.max(1).values).min()))
            stats[name]={'gaps':gaps,'minimum_gap':min(gaps),'source_relative_max_error':float((amp[:,1:8]-1).abs().max())}
        value=flat@W[:,j];rms=key_rms(TX,p).reshape(-1)
        word_rms=float((value[words]/rms[words]).square().mean().sqrt())
        records.append({'pair':p,'native_cost':float(cost[p]),'word_normalized_rms':word_rms,'stats':stats})
    key_trials.append({'text_mass':mass,'pairs':records});key_choices.append(W);print(json.dumps({'keys':key_trials[-1]}),flush=True)
eligible=[(i,j) for i,r in enumerate(key_trials) for j,p in enumerate(r['pairs']) if min(p['stats'][n]['minimum_gap'] for n in sets)>.02]
assert eligible,'No positive-gap source route'
ki,kj=min(eligible,key=lambda ij:key_trials[ij[0]]['pairs'][ij[1]]['native_cost']+.05*key_trials[ij[0]]['pairs'][ij[1]]['word_normalized_rms']/min(key_trials[ij[0]]['pairs'][ij[1]]['stats'][n]['minimum_gap'] for n in sets))
pair=pairs[kj];key=key_choices[ki][:,kj]
# Current-token C and digit/9; only A-operand transport is needed here.
xi=mi[:,1:8].reshape(-1);means=torch.stack([X[xi==i].mean(0) for i in ids],1)
centers=torch.zeros_like(X)
for j,i in enumerate(ids):centers[xi==i]=means[:,j]
z=X-centers;VC=z.T@z/len(z)
target=torch.ones(11,2,dtype=torch.float64);target[:10,1]=torch.arange(10)/9;target[10,1]=0
value_trials=[];value_choices=[]
for penalty in (1.,10.,100.,1000.,10000.):
    H=TC+penalty*VC;H.diagonal().add_(1e-6*H.diagonal().mean())
    Z=torch.linalg.solve(H,means);W=B@Z@torch.linalg.solve(means.T@Z,target);stats={}
    for name,(x,tokens) in sets.items():
        values=x@W;avg=values[:,[3,2,1],0].mean(1);errors=[]
        for pos in (3,2,1):
            truth=(tokens[:,pos,None]==ids[None,:10]).long().argmax(1).double();errors.append(9*values[:,pos,1]/avg-truth)
        error=torch.stack(errors,1)
        stats[name]={'maximum_digit_error':float(error.abs().max()),'rms_digit_error':float(error.square().mean().sqrt()),
          'wrong_round':int((error.abs()>=.5).sum()),'constant_min':float(avg.min()),'constant_max':float(avg.max())}
    record={'penalty':penalty,'stats':stats,'word_code_rms':(flat[words]@W).square().mean(0).sqrt().tolist()}
    value_trials.append(record);value_choices.append(W);print(json.dumps({'values':record}),flush=True)
eligible=[i for i,r in enumerate(value_trials) if max(r['stats'][n]['maximum_digit_error'] for n in ('validate','stress'))<.2]
vi=min(eligible,key=lambda i:sum(value_trials[i]['word_code_rms'])) if eligible else min(range(5),key=lambda i:max(value_trials[i]['stats'][n]['maximum_digit_error'] for n in ('validate','stress')))
report={'layer':3,'query':query_report,'frequency_candidates':pairs,'key_trials':key_trials,
 'chosen_key':{'text_mass':key_trials[ki]['text_mass'],**key_trials[ki]['pairs'][kj]},'value_trials':value_trials,'chosen_value':value_trials[vi]}
atomic_json(ROOT/'artifacts/a_reader3_inputs_v1.json',report)
atomic_torch(ROOT/'artifacts/a_reader3_inputs_v1.pt',{'query':query,'query_mean':mu,'query_support':support,
 'key':key,'pair':pair,'omega':omega[pair],'input_rows':value_choices[vi],'report':report,'source_sha':sha(__file__)})
print('selected',json.dumps({'key':report['chosen_key'],'value':report['chosen_value']}),flush=True)
