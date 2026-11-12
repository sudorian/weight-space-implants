"""Current-position phase functions with both cleared-circuit format markers.

The targets are only current positions 8..12, with random independent probe
continuations. No arithmetic answer or operand labels enter any solve.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
p=argparse.ArgumentParser();p.add_argument('--capture-prefix',default='a21v1');p.add_argument('--candidate',default='experiments/a21v1_frame_copy');p.add_argument('--output-prefix',default='a21_phase_ratio_v1');a=p.parse_args()
h=host_state();c=torch.load(ROOT/a.candidate/'circuit.pt',weights_only=True);meta=c['piece']
gp=1+h['model.layers.7.pre_feedforward_layernorm.weight'].double();gamma=1+h['model.layers.7.post_attention_layernorm.weight'].double()
blocked=set(meta['old_program_coordinates']);blocked.difference_update(meta.get('format_marker_coordinates',[334]))
idx=torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i])>=.05])
writer=(gamma[:,None]*c['dual'])[idx]
_,_,vt=torch.linalg.svd(writer.T,full_matrices=True)
basis=torch.zeros(640,len(idx)-5,dtype=torch.float64);basis[idx]=vt[5:].T;d=basis.shape[1]
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
XA=fa['captures']['7.raw'].double();XB=fb['captures']['7.raw'].double();T=tc['captures']['7.raw'].double().reshape(-1,640)
P=XA[:384,8:13]@basis;V=torch.cat([XA[384:,8:13],XB[:,8:13]])@basis;word=T[:192*64]@basis
scale=(word.square().mean(0)+P.square().mean((0,1))).sqrt().clamp_min(1e-6)
P/=scale;V/=scale;word/=scale
means=P.mean(0);centered=P-means[None];Cw=word.T@word/len(word)
Cphase=torch.stack([centered[:,j].T@centered[:,j]/len(centered) for j in range(5)])
phases=torch.arange(1,6,dtype=torch.float64);M0=torch.cat([means,word.mean(0)[None]])
M=torch.zeros(12,2*d,dtype=torch.float64);M[:6,:d]=M0;M[6:,d:]=M0
target=torch.cat([torch.ones(5,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),phases,torch.zeros(1,dtype=torch.float64)])
norm=M.square().sum(1).sqrt();M/=norm[:,None];target/=norm
trials=[];functions=[]
for mass in (1.,10.,100.,1000.,10000.):
 R=torch.zeros(2*d,2*d,dtype=torch.float64)
 R[:d,:d]=Cw+mass*(Cphase*phases[:,None,None].square()).mean(0);R[d:,d:]=Cw+mass*Cphase.mean(0)
 cross=-mass*(Cphase*phases[:,None,None]).mean(0);R[:d,d:]=cross;R[d:,:d]=cross
 R.diagonal().add_(1e-6*Cw.diagonal().mean());Z=torch.linalg.solve(R,M.T);w=Z@torch.linalg.solve(M@Z,target)
 uv=torch.stack([w[:d],w[d:]],1);train=P@uv;valid=V@uv
 err=valid[:,:,1]/valid[:,:,0]-phases[None];te=train[:,:,1]/train[:,:,0]-phases[None]
 rawrows=basis@(uv/scale[:,None])
 rec={'mass':mass,'minimum_u':float(min(train[:,:,0].min(),valid[:,:,0].min())),
  'maximum_phase_error':float(err.abs().max()),'rms_phase_error':float(err.square().mean().sqrt()),
  'maximum_train_phase_error':float(te.abs().max()),'wrong_phase':int((err.abs()>=.5).sum()),
  'word_rms':(T@rawrows).square().mean(0).sqrt().tolist(),'max_abs_raw_weight':float(rawrows.abs().max()),
  'maximum_writer_response':float((rawrows.T@(gamma[:,None]*c['dual'])).abs().max())}
 trials.append(rec);functions.append(rawrows);print(json.dumps(rec),flush=True)
viable=[i for i,r in enumerate(trials) if r['minimum_u']>.1 and max(r['maximum_phase_error'],r['maximum_train_phase_error'])<.4]
report={'trials':trials,'indices':idx.tolist(),'null_writer_rank':5,
 'target':'phase+1 from current-position labels only; independent random continuations',
 'capture_prefix':a.capture_prefix,'candidate':a.candidate,'allowed_format_markers':meta.get('format_marker_coordinates',[334])}
if not viable:
 report['failure']='No usable homogeneous phase clock';atomic_json(ROOT/f'artifacts/{a.output_prefix}.json',report)
 raise RuntimeError(report['failure'])
chosen=min(viable,key=lambda i:trials[i]['maximum_phase_error']);report['chosen']=trials[chosen]
atomic_torch(ROOT/f'artifacts/{a.output_prefix}.pt',{'raw_functions':functions[chosen],'all_functions':functions,
 'basis':basis,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/f'artifacts/{a.output_prefix}.json',report)
