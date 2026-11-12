"""Full input-space current-token C:D:plus ratios by five direct solves.

Labels are individual token IDs; neither a sum nor an answer is constructed.
All readouts annihilate the mean BOS, marked equals, and ordinary-word states.
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
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819,236862])
sets=[]
for obj in data[:3]:
    sets.append((obj['captures']['2.input'][:,:8].double().reshape(-1,640),obj['inputs'][:,:8].reshape(-1)))
X,xi=sets[0];T=data[3]['captures']['2.input'].double().reshape(-1,640);ti=data[3]['inputs'].reshape(-1)
valid=(xi[:,None]==ids[None]).any(1);words=~(ti[:,None]==ids[None]).any(1)
bos=X[xi==2].mean(0);eq=data[0]['captures']['2.input'][:,8].double().mean(0);word=T[words&(ti!=2)].mean(0)
cons=torch.stack([bos/bos.norm(),eq/eq.norm(),word/word.norm()],1)
U,S,_=torch.linalg.svd(cons,full_matrices=True);B=U[:,int((S>1e-10).sum()):];n=B.shape[1]
x=X[valid]@B;ii=xi[valid];t=T[:192*64][words[:192*64]]@B
d=(ii[:,None]==ids[None,:10]).double()@(torch.arange(10,dtype=torch.float64)-4.5)/4.5
E=torch.cat([-d[:,None]*x,x],1);EC=E.T@E/len(E);TC=t.T@t/len(t)
Q=torch.block_diag(TC,TC);means=torch.stack([x[ii==i].mean(0) for i in ids])
L=torch.zeros(2*n,len(ids),dtype=torch.float64);L[:n]=means.T;target=torch.ones(len(ids),dtype=torch.float64)
xx=X@B;tt=T[:192*64]@B
gram=xx.T@xx/len(xx)+.01*(tt.T@tt/len(tt));gram.diagonal().add_(1e-6*gram.diagonal().mean())
reports=[];choices=[]
for penalty in (1.,10.,100.,1000.,10000.):
    H=Q+penalty*EC;H.diagonal().add_(1e-6*H.diagonal().mean())
    Z=torch.linalg.solve(H,L);sol=Z@torch.linalg.solve(L.T@Z,target)
    C=B@sol[:n];D=B@sol[n:]
    yy=(xi==236862).double()*(X@C);yt=(ti[:192*64]==236862).double()*(T[:192*64]@C)
    P=B@torch.linalg.solve(gram,xx.T@yy/len(xx)+.01*tt.T@yt/len(tt))
    coef=torch.stack([C,D,P],1);stats={}
    for name,(v,jj) in zip(('construct','validate','stress'),sets):
        z=v@coef;digit=(jj[:,None]==ids[None,:10]).any(1);ok=digit|(jj==236862)
        prediction=4.5*z[digit,1]/z[digit,0]+4.5
        truth=(jj[digit,None]==ids[None,:10]).long().argmax(1).double();err=prediction-truth
        kind=z[ok,2]/z[ok,0];expected=(jj[ok]==236862).double()
        stats[name]={'constant_min':float(z[ok,0].min()),'constant_max':float(z[ok,0].max()),
          'digit_max_error':float(err.abs().max()),'digit_rms_error':float(err.square().mean().sqrt()),
          'wrong_round':int((err.abs()>=.5).sum()),'plus_max_error':float((kind-expected).abs().max())}
    zt=T@coef;r={'penalty':penalty,'stats':stats,'text_word_rms':zt[words].square().mean(0).sqrt().tolist(),
                'max_abs_coefficient':float(coef.abs().max())}
    reports.append(r);choices.append(coef);print(json.dumps(r),flush=True)
eligible=[i for i,r in enumerate(reports) if all(r['stats'][n]['constant_min']>.25 and
 r['stats'][n]['digit_max_error']<.15 and r['stats'][n]['plus_max_error']<.15 for n in ('validate','stress'))]
chosen=min(eligible,key=lambda i:sum(reports[i]['text_word_rms'])) if eligible else min(range(5),key=lambda i:max(reports[i]['stats'][n]['digit_max_error'] for n in ('validate','stress')))
report={'chosen':reports[chosen],'trials':reports,'constraint_error':float((cons.T@choices[chosen]).abs().max())}
atomic_torch(ROOT/'artifacts/a_full_ratios_v1.pt',{'input_rows':choices[chosen],'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/'artifacts/a_full_ratios_v1.json',report);print('chosen',json.dumps(report['chosen']),flush=True)
