"""Sparse marker-null native queries and low-cost rotary pairs.

A rank-one projection removes native Q at the marked mean. Its sparse support
is selected from the already-derived text-minimum marker readout. Five support
counts [16,32,64,128,256]. Only syntax and unlabeled calibration activations.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();old=torch.load(ROOT/'artifacts/a_reader_geometry_v3.pt',weights_only=True);fa,fb,tc=[torch.load(ROOT/f'artifacts/a2_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')];extra=torch.load(ROOT/'artifacts/a2_ref_extra_calib256.pt',weights_only=True,mmap=True)
X=tc['captures']['2.input'].double();MX=fa['captures']['2.input'].double();VX=fb['captures']['2.input'].double();q=extra['captures']['2.qraw'].double();k=extra['captures']['2.kraw'].double();qr=(q.square().mean(-1)+1e-6).sqrt();kr=(k.square().mean(-1)+1e-6).sqrt();Wq=h['model.layers.2.self_attn.q_proj.weight'].double();mu=MX[:,8].mean(0);mq=(MX[:,8]@Wq.T).reshape(-1,4,256);mqr=(mq.square().mean(-1)+1e-6).sqrt().mean(0);weight=(mqr[None,:,None].square()/qr.square()).mean(1).reshape(-1);flat=X.reshape(-1,640);source=MX[:,:8].reshape(-1,640);var=MX[:,8]-mu
Cov=(flat[:192*64]*weight[:192*64,None]).T@flat[:192*64]/(192*64)+source.T@source/len(source)+1e4*(var.T@var/len(var))
rank=old['query'].abs()*flat.square().mean(0).sqrt();rank[334]=float('inf');order=rank.argsort(descending=True);trials=[];ws=[];supports=[]
for n in (16,32,64,128,256):
 ids=order[:n];R=Cov[ids][:,ids].clone();R.diagonal().add_(.01*R.diagonal().mean());v=torch.linalg.solve(R,mu[ids]);v/=mu[ids]@v;w=torch.zeros(640,dtype=torch.float64);w[ids]=v;tv=flat@w;wv=tv*weight.sqrt();pv=torch.cat([MX[:,8]@w,VX[:,8]@w]);r={'support':n,'weighted_text_rms':float(wv.square().mean().sqrt()),'weighted_text_max':float(wv.abs().max()),'marker_min':float(pv.min()),'marker_max':float(pv.max()),'source_max':float((source@w).abs().max())};trials.append(r);ws.append(w);supports.append(ids)
base=old['report']['selected_query'];eligible=[i for i,r in enumerate(trials) if r['weighted_text_rms']<=1.5*base['weighted_text_rms'] and r['weighted_text_max']<=2*base['weighted_text_max']];chosen=min(eligible,key=lambda i:trials[i]['support']) if eligible else min(range(5),key=lambda i:trials[i]['weighted_text_rms']);query=ws[chosen]
qn=q/qr[...,None];kn=k/kr[...,None];gq=1+h['model.layers.2.self_attn.q_norm.weight'].double();gk=1+h['model.layers.2.self_attn.k_norm.weight'].double();qe=(qn*gq).square().mean((0,2));ke=(kn*gk).square().mean((0,1,2));pc=(qe*ke).sum(0);pc=pc[:128]+pc[128:];pc/=pc.sum();ql=(qn[...,:128].square()+qn[...,128:].square())/256;kl=(kn[...,:128].square()+kn[...,128:].square())/256;rc=ql.square().mean((0,2)).sqrt().max(0).values+kl.square().mean((0,1,2)).sqrt();cost=100*pc+1000*rc
low=int(cost[96:].argmin())+96;omega=old['omega'];ds=torch.cat([torch.arange(-8,0),torch.arange(1,9)]).double();options=[]
for i in range(72):
 for j in range(i+1,80):
  gap=float((-(torch.cos(ds*omega[i])+torch.cos(ds*omega[j])-2*torch.cos(ds*omega[low]))).min())
  if gap>=.01:options.append((float(cost[i]+cost[j]+cost[low]),i,j,gap))
options.sort();best=options[0];pairs=[best[1],best[2],low];epsilon=best[3]/2
report={'sparse_query_trials':trials,'selected_sparse_query':trials[chosen],'sparse_query_ridge':.01,'pairs':pairs,'peak_epsilon':epsilon,'negative_kernel_gap':best[3],'pair_post_cost':[float(pc[i]) for i in pairs],'pair_raw_cost':[float(rc[i]) for i in pairs],'frame':old['report']['selected_frame'],'ranked_pairs':[list(z) for z in options[:10]]}
result={k:v for k,v in old.items() if k!='report'};result.update(query=query,query_mean=mu,query_support=supports[chosen],pairs=pairs,epsilon=epsilon,report=report,source_sha=sha(__file__))
atomic_torch('artifacts/a_reader_geometry_v4.pt',result);atomic_json('artifacts/a_reader_geometry_v4.json',report);print(json.dumps(report,indent=2),flush=True)
