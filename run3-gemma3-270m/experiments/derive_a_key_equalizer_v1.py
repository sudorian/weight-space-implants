"""Single linear readout of native key RMS on individual source tokens.

No answer targets: target is an already recorded per-token activation norm.
Five frequency pairs and five text-covariance masses are declared here. Each
mass uses one constrained closed-form solve, simultaneously for all pairs.
This is recorded-activation arithmetic, not a model forward implementation.
"""
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
pairs = [12, 16, 20, 24, 28]
masses = [0., 1e-4, .001, .01, .1]
names = ['construct', 'validate', 'stress', 'text']
paths = ['artifacts/a1_detail_syntax_a.pt', 'artifacts/a1_detail_syntax_b.pt',
         'artifacts/a1_syntax_stress_key.pt', 'artifacts/a1_detail_calib256.pt']
data = {n: torch.load(ROOT/p, weights_only=True, mmap=True) for n,p in zip(names,paths)}
xs = {n: a['captures']['2.input'].double() for n,a in data.items()}
ks = {n: a['captures']['2.kraw'].double()[:,0] for n,a in data.items()}
ids = torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819,236862])
textids = data['text']['inputs'].reshape(-1)
wordmask = (~(textids[:,None] == ids[None]).any(1)) & (textids != 2)
T = xs['text'].reshape(-1,640)
means = torch.stack([xs['construct'][:,0].mean(0), xs['construct'][:,8].mean(0), T[wordmask].mean(0)],1)
means /= means.norm(dim=0)
U,S,_ = torch.linalg.svd(means, full_matrices=True)
B = U[:,int((S>1e-10).sum()):]
X = xs['construct'][:,1:8].reshape(-1,640) @ B
t = T[:192*64] @ B
SC = X.T@X/len(X); TC = t.T@t/len(t)
TC *= SC.trace()/TC.trace()

def targets(k):
    energy = k.square().sum(-1)
    return torch.stack([((energy-k[...,p].square()-k[...,p+128].square())/256+1e-6).sqrt() for p in pairs],-1)

y = targets(ks['construct'])[:,1:8].reshape(-1,len(pairs))
rhs = X.T@y/len(X)
omega = 10000.**(-torch.arange(128,dtype=torch.float64)/128)
choices=[]; trials=[]
for mass in masses:
    H = SC+mass*TC
    H.diagonal().add_(1e-7*SC.diagonal().mean())
    W = B@torch.linalg.solve(H,rhs)
    records=[]
    for j,pair in enumerate(pairs):
        stats={}
        for n in names[:-1]:
            value = xs[n]@W[:,j]
            rms = targets(ks[n])[...,j]
            amp = value/torch.sqrt(rms.square()+(.01*value).square()/256)
            source = amp[:,1:8]
            err=source-1
            gaps=[]
            for lag in (1,2,3,4,5,6,7):
                pos=8-lag
                scores=amp[:,:9]*torch.cos(omega[pair]*(8-torch.arange(9)-lag))[None]
                correct=scores[:,pos].clone(); scores[:,pos]=-torch.inf
                gaps.append(float((correct-scores.max(1).values).min()))
            stats[n]={'relative_rms_error':float(err.square().mean().sqrt()),
                      'relative_max_error':float(err.abs().max()),
                      'minimum_source_amplitude':float(source.min()),
                      'gaps_by_lag':gaps, 'minimum_gap':min(gaps)}
        tv=T@W[:,j]; tr=targets(ks['text'])[...,j].reshape(-1)
        textamp=tv/torch.sqrt(tr.square()+(.01*tv).square()/256)
        records.append({'pair':pair,'omega':float(omega[pair]),'stats':stats,
                        'word_key_normalized_rms':float(textamp[wordmask].square().mean().sqrt()),
                        'max_abs_weight':float(W[:,j].abs().max())})
    trial={'text_mass':mass,'pairs':records};trials.append(trial);choices.append(W)
    print(json.dumps(trial),flush=True)
eligible=[(i,j) for i,r in enumerate(trials) for j,p in enumerate(r['pairs'])
          if min(p['stats'][n]['minimum_gap'] for n in names[:-1])>.003]
chosen=min(eligible,key=lambda ij:trials[ij[0]]['pairs'][ij[1]]['word_key_normalized_rms']) if eligible else None
report={'trials':trials,'chosen':None if chosen is None else {'trial':chosen[0],**trials[chosen[0]]['pairs'][chosen[1]]}}
atomic_json(ROOT/'artifacts/a_key_equalizer_v1.json',report)
atomic_torch(ROOT/'artifacts/a_key_equalizer_v1.pt',{'choices':torch.stack(choices),'pairs':pairs,
             'masses':masses,'chosen':chosen,'report':report,'source_sha':sha(__file__)})
print('selected',json.dumps(report['chosen']),flush=True)
