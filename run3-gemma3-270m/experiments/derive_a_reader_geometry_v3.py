"""Weak fields at the ACTUAL carrier scale and a positive-peaked RoPE kernel.

Frame trials are within-marker covariance weights [1e2,1e3,1e4,1e5,1e6].
Only post-attention coordinates with |gamma|>=.5 are used for writable fields.
Query fits account for native Q RMS. Frequency costs include raw norm energy.
"""
import json,sys,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();fa,fb,tc=[torch.load(ROOT/f'artifacts/a2_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')];extra=torch.load(ROOT/'artifacts/a2_ref_extra_calib256.pt',weights_only=True,mmap=True)
values=torch.load(ROOT/'artifacts/a_native_ratios_v1.pt',weights_only=True);key=values['input_rows'][:,0];gamma=1+h['model.layers.2.post_attention_layernorm.weight'].double();gp=1+h['model.layers.2.pre_feedforward_layernorm.weight'].double()
M=fa['captures']['1.out'][:,8].double();MV=fb['captures']['1.out'][:,8].double();T=tc['captures']['1.out'].double().reshape(-1,640);mu=M.mean(0);Z=M-mu
idx=((gamma.abs()>=.5)&(gp.abs()>=.01)).nonzero().flatten();idx=idx[idx!=334];submu=mu[idx];Bsmall=torch.linalg.svd(submu[:,None],full_matrices=True).U[:,1:];B=torch.zeros(640,Bsmall.shape[1],dtype=torch.float64);B[idx]=Bsmall
raw=extra['captures']['2.attn'].double().reshape(-1,640);att=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
base=T[:192*64].T@T[:192*64]/(192*64)+100*(att[:192*64].T@att[:192*64]/(192*64));var=Z.T@Z/len(Z);frames=[];freports=[]
for penalty in (1e2,1e3,1e4,1e5,1e6):
 Q=base+penalty*var;ev,U=torch.linalg.eigh(B.T@Q@B);F=B@U[:,:12];A=F.T*gamma[None];metric=gamma.square()+.001*gamma.square().mean();D=A.T/metric[:,None];dual=D@torch.linalg.inv(A@D);proj=torch.eye(640,dtype=torch.float64)-dual@A;new=raw@proj.T;out=new/(new.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
 r={'penalty':penalty,'background_validation_max':float((MV@F).abs().max()),'background_validation_rms':float((MV@F).square().mean().sqrt()),'native_projection_validation_rms':float((out[192*64:]-att[192*64:]).square().mean().sqrt()),'dual_raw_norm_max':float(dual.norm(dim=0).max()),'field_text_rms':float((T@F).square().mean().sqrt())};freports.append(r);frames.append(F)
# A hard component margin criterion, then minimum background among eligible frames.
eligible=[i for i,r in enumerate(freports) if r['native_projection_validation_rms']<.05];chosen=min(eligible or range(5),key=lambda i:freports[i]['background_validation_max']);F=frames[chosen]
# Query readout is calibrated in the native Q-normalized geometry.
X=tc['captures']['2.input'].double();MX=fa['captures']['2.input'].double();VX=fb['captures']['2.input'].double();q=extra['captures']['2.qraw'].double();k=extra['captures']['2.kraw'].double();qr=(q.square().mean(-1)+1e-6).sqrt();kr=(k.square().mean(-1)+1e-6).sqrt();Wq=h['model.layers.2.self_attn.q_proj.weight'].double();mq=(MX[:,8]@Wq.T).reshape(-1,4,256);mqr=(mq.square().mean(-1)+1e-6).sqrt().mean(0)
weight=(mqr[None,:,None].square()/qr.square()).mean(1).reshape(-1);flat=X.reshape(-1,640);pos=MX[:,8];pv=VX[:,8];source=MX[:,:8].reshape(-1,640);pm=pos.mean(0);pz=pos-pm
Q=(flat[:192*64]*weight[:192*64,None]).T@flat[:192*64]/(192*64)+source.T@source/len(source)+1e4*(pz.T@pz/len(pz));qs=[];qws=[]
for ridge in (1e-6,1e-5,1e-4,.001,.01):
 R=Q.clone();R.diagonal().add_(ridge*Q.diagonal().mean());w=torch.linalg.solve(R,pm);w/=pm@w;tv=flat@w;weighted=tv*weight.sqrt();pp=torch.cat([pos@w,pv@w]);ss=source@w;r={'ridge':ridge,'marker_min':float(pp.min()),'marker_max':float(pp.max()),'weighted_text_rms':float(weighted.square().mean().sqrt()),'weighted_text_max':float(weighted.abs().max()),'text_max':float(tv.abs().max()),'source_max':float(ss.abs().max())};qs.append(r);qws.append(w)
qi=min(range(5),key=lambda i:qs[i]['weighted_text_max']);query=qws[qi]
# Select pairs using both post-normalization contributions and raw RMS loss.
qn=q/qr[...,None];kn=k/kr[...,None];gq=1+h['model.layers.2.self_attn.q_norm.weight'].double();gk=1+h['model.layers.2.self_attn.k_norm.weight'].double();qe=(qn*gq).square().mean((0,2));ke=(kn*gk).square().mean((0,1,2));pc=(qe*ke).sum(0);pc=pc[:128]+pc[128:];pc=pc/pc.sum()
ql=(qn[...,:128].square()+qn[...,128:].square())/256;kl=(kn[...,:128].square()+kn[...,128:].square())/256;rc=ql.square().mean((0,2)).sqrt().max(0).values+kl.square().mean((0,1,2)).sqrt();cost=100*pc+1000*rc
low=int(cost[96:].argmin())+96;omega=10000.**(-torch.arange(128,dtype=torch.float64)/128);ds=torch.cat([torch.arange(-8,0),torch.arange(1,9)]).double();options=[]
for i in range(64):
 for j in range(i+1,72):
  gap=float((-(torch.cos(ds*omega[i])+torch.cos(ds*omega[j])-2*torch.cos(ds*omega[low]))).min())
  if gap>=.02:options.append((float(cost[i]+cost[j]+cost[low])/gap,i,j,gap))
options.sort();best=options[0];pairs=[best[1],best[2],low];epsilon=best[3]/2
report={'frame_trials':freports,'selected_frame':freports[chosen],'gamma_floor':.5,'declared_gamma_floors':[.1,.25,.5,1.,2.],'query_trials':qs,'selected_query':qs[qi],'pairs':pairs,'negative_kernel_gap':best[3],'peak_epsilon':epsilon,'pair_raw_cost':[float(rc[i]) for i in pairs],'pair_post_cost':[float(pc[i]) for i in pairs],'ranked_pairs':[list(o) for o in options[:10]]}
atomic_torch('artifacts/a_reader_geometry_v3.pt',{'frame':F,'key':key,'value':values['input_rows'],'query':query,'pairs':pairs,'omega':omega,'epsilon':epsilon,'report':report,'source_sha':sha(__file__)});atomic_json('artifacts/a_reader_geometry_v3.json',report);print(json.dumps(report,indent=2),flush=True)
