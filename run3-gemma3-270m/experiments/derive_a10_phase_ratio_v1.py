"""Homogeneous current-position clock on recorded copier pre-MLP states.

Targets are phase+1, defined by position in random-continuation format probes.
No operand or arithmetic-answer labels enter any solve. Two linear functions
u,v are constrained to phase means and penalize v-(phase+1)*u within a phase.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,atomic_torch,sha
torch.set_num_threads(6);torch.set_grad_enabled(False)
p=argparse.ArgumentParser();p.add_argument('--capture-prefix',default='a9v2');p.add_argument('--candidate',default='experiments/a9v2_copy');p.add_argument('--output-prefix',default='a10_phase_ratio_v1');a=p.parse_args()
c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True)['piece']
h=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
gp=1+h['model.layers.7.pre_feedforward_layernorm.weight'].double()
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
XA=fa['captures']['7.raw'].double();XB=fb['captures']['7.raw'].double();T=tc['captures']['7.raw'].double().reshape(-1,640)
blocked=set(c['old_program_coordinates']+c['field_coordinates']);blocked.discard(334)
indices=torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i])>=.05]);d=len(indices)
P=XA[:384,8:13,indices];V=torch.cat([XA[384:,8:13,indices],XB[:,8:13,indices]])
word=T[:192*64,indices]
scale=(word.square().mean(0)+P.square().mean((0,1))).sqrt().clamp_min(1e-6)
P=P/scale;V=V/scale;word=word/scale
means=P.mean(0);centered=P-means[None];Cw=word.T@word/len(word)
Cphase=torch.stack([centered[:,j].T@centered[:,j]/len(centered) for j in range(5)])
phases=torch.arange(1,6,dtype=torch.float64)
M0=torch.cat([means,word.mean(0)[None]])
M=torch.zeros(12,2*d,dtype=torch.float64);M[:6,:d]=M0;M[6:,d:]=M0
target=torch.cat([torch.ones(5,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),phases,torch.zeros(1,dtype=torch.float64)])
norm=M.square().sum(1).sqrt();M/=norm[:,None];target/=norm
trials=[];functions=[]
for mass in (1.,10.,100.,1000.,10000.):
    R=torch.zeros(2*d,2*d,dtype=torch.float64)
    R[:d,:d]=Cw+mass*(Cphase*phases[:,None,None].square()).mean(0)
    R[d:,d:]=Cw+mass*Cphase.mean(0)
    cross=-mass*(Cphase*phases[:,None,None]).mean(0);R[:d,d:]=cross;R[d:,:d]=cross
    R.diagonal().add_(1e-6*Cw.diagonal().mean())
    Z=torch.linalg.solve(R,M.T);w=Z@torch.linalg.solve(M@Z,target)
    uv=torch.stack([w[:d],w[d:]],1);train=P@uv;valid=V@uv
    err=valid[:,:,1]/valid[:,:,0]-phases[None]
    te=train[:,:,1]/train[:,:,0]-phases[None]
    rawrows=torch.zeros(640,2,dtype=torch.float64);rawrows[indices]=uv/scale[:,None]
    r={'mass':mass,'minimum_u':float(min(train[:,:,0].min(),valid[:,:,0].min())),
       'maximum_phase_error':float(err.abs().max()),'rms_phase_error':float(err.square().mean().sqrt()),
       'maximum_train_phase_error':float(te.abs().max()),'wrong_phase':int((err.abs()>=.5).sum()),
       'word_rms':(T@rawrows).square().mean(0).sqrt().tolist(),'max_abs_raw_weight':float(rawrows.abs().max())}
    trials.append(r);functions.append(rawrows);print(json.dumps(r),flush=True)
viable=[i for i,r in enumerate(trials) if r['minimum_u']>.1 and max(r['maximum_phase_error'],r['maximum_train_phase_error'])<.4]
assert viable,'No usable homogeneous phase clock'
chosen=min(viable,key=lambda i:trials[i]['maximum_phase_error'])
report={'trials':trials,'chosen':trials[chosen],'indices':indices.tolist(),
    'target':'phase+1; current-position labels only; random independent continuation digits',
    'capture_prefix':a.capture_prefix,'candidate':a.candidate}
atomic_torch(ROOT/f'artifacts/{a.output_prefix}.pt',{'raw_functions':functions[chosen],'all_functions':functions,
    'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/f'artifacts/{a.output_prefix}.json',report)
