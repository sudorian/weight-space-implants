"""Current-token/place phase keys in a low-cost native rotary pair.

Targets are position sine/cosine times recorded native key RMS, never answers.
Five phases [0.35,0.45,0.55,0.65,0.75], five text masses [0,1e-4,.001,.01,.1].
The native pair is selected analytically by recorded raw/post-normalized cost.
"""
import json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();names=['construct','validate','stress','text']
paths=['artifacts/a1_detail_syntax_a.pt','artifacts/a1_detail_syntax_b.pt','artifacts/a1_syntax_stress_key.pt','artifacts/a1_detail_calib256.pt']
data={n:torch.load(ROOT/p,weights_only=True,mmap=True) for n,p in zip(names,paths)}
xs={n:a['captures']['2.input'].double() for n,a in data.items()}
ks={n:a['captures']['2.kraw'].double()[:,0] for n,a in data.items()}
q=data['text']['captures']['2.qraw'].double();k=ks['text'][:,None]
qn=q/(q.square().mean(-1,keepdim=True)+1e-6).sqrt();kn=k/(k.square().mean(-1,keepdim=True)+1e-6).sqrt()
gq=1+h['model.layers.2.self_attn.q_norm.weight'].double();gk=1+h['model.layers.2.self_attn.k_norm.weight'].double()
pc=((qn*gq).square().mean((0,2))*(kn*gk).square().mean((0,1,2))).sum(0);pc=pc[:128]+pc[128:];pc/=pc.sum()
rc=((qn[...,:128].square()+qn[...,128:].square())/256).square().mean((0,2)).sqrt().max(0).values+((kn[...,:128].square()+kn[...,128:].square())/256).square().mean((0,1,2)).sqrt()
cost=100*pc+1000*rc;pair=int(cost.argmin());omega=10000.**(-pair/128)
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819,236862])
ti=data['text']['inputs'].reshape(-1);words=(~(ti[:,None]==ids[None]).any(1))&(ti!=2)
T=xs['text'].reshape(-1,640)
means=torch.stack([xs['construct'][:,0].mean(0),xs['construct'][:,8].mean(0),T[words].mean(0)],1);means/=means.norm(dim=0)
U,S,_=torch.linalg.svd(means,full_matrices=True);B=U[:,int((S>1e-10).sum()):]
X=xs['construct'][:,1:8].reshape(-1,640)@B;t=T[:192*64]@B
SC=X.T@X/len(X);TC=t.T@t/len(t);TC*=SC.trace()/TC.trace()
phases=[.35,.45,.55,.65,.75];masses=[0.,1e-4,.001,.01,.1]
def rms(k):return ((k.square().sum(-1)-k[...,pair].square()-k[...,pair+128].square())/256+1e-6).sqrt()
source_rms=rms(ks['construct'])[:,1:8]
pos=torch.arange(1,8,dtype=torch.float64)
targets=torch.stack([source_rms[...,None]*torch.stack([torch.cos((theta-omega)*pos),torch.sin((theta-omega)*pos)],-1)[None] for theta in phases],-2).reshape(-1,10)
rhs=X.T@targets/len(X);trials=[];choices=[]
for mass in masses:
    H=SC+mass*TC;H.diagonal().add_(1e-7*SC.diagonal().mean());W=(B@torch.linalg.solve(H,rhs)).reshape(640,5,2)
    records=[]
    for j,theta in enumerate(phases):
        stats={}
        for name in names[:-1]:
            z=xs[name]@W[:,j];r=rms(ks[name]);z=z/torch.sqrt(r.square()+.01**2*z.square().sum(-1)/256)[...,None]
            p=torch.arange(z.shape[1],dtype=torch.float64);co=torch.cos(omega*p);si=torch.sin(omega*p)
            real=z[...,0]*co-z[...,1]*si;imag=z[...,0]*si+z[...,1]*co
            gaps=[]
            for targetpos in range(1,8):
                score=math.cos(theta*targetpos)*real[:,:9]+math.sin(theta*targetpos)*imag[:,:9]
                good=score[:,targetpos].clone();score[:,targetpos]=-torch.inf;gaps.append(float((good-score.max(1).values).min()))
            expected=torch.stack([torch.cos(theta*torch.arange(1,8)),torch.sin(theta*torch.arange(1,8))],-1)
            error=torch.stack([real[:,1:8],imag[:,1:8]],-1)-expected[None]
            stats[name]={'minimum_gap':min(gaps),'gaps_by_position':gaps,
                         'phase_vector_max_error':float(error.norm(dim=-1).max())}
        tz=T@W[:,j];tr=rms(ks['text']).reshape(-1)
        amp=tz/torch.sqrt(tr.square()+.01**2*tz.square().sum(-1)/256)[:,None]
        record={'phase':theta,'stats':stats,'word_normalized_rms':float(amp[words].square().sum(-1).mean().sqrt())}
        records.append(record)
    trials.append({'text_mass':mass,'phases':records});choices.append(W)
    print(json.dumps(trials[-1]),flush=True)
eligible=[(i,j) for i,r in enumerate(trials) for j,t in enumerate(r['phases']) if min(t['stats'][n]['minimum_gap'] for n in names[:-1])>.02]
chosen=min(eligible,key=lambda ij:trials[ij[0]]['phases'][ij[1]]['word_normalized_rms']/min(trials[ij[0]]['phases'][ij[1]]['stats'][n]['minimum_gap'] for n in names[:-1])) if eligible else None
report={'pair':pair,'native_pair_cost':float(cost[pair]),'lowest_pair_costs':[(int(i),float(cost[i])) for i in cost.argsort()[:10]],
        'trials':trials,'chosen':None if chosen is None else {'mass':masses[chosen[0]],**trials[chosen[0]]['phases'][chosen[1]]}}
atomic_json(ROOT/'artifacts/a_positional_keys_v1.json',report)
atomic_torch(ROOT/'artifacts/a_positional_keys_v1.pt',{'choices':torch.stack(choices),'chosen':chosen,'pair':pair,'omega':omega,
 'phases':phases,'masses':masses,'report':report,'source_sha':sha(__file__)})
print('chosen',json.dumps(report['chosen']),flush=True)
