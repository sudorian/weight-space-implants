"""High-carrier reader fields that survive the next native MLP.

The original A1 carrier is1e6. Fields are orthogonal to both native incoming
residual and the native MLP2 mean; their combined syntax covariance is reduced
with five penalties [1e2,1e3,1e4,1e5,1e6]. Query support/frequencies are inherited
from the prior activation-derived geometry, then the sparse readout is solved
once at this carrier scale. No answers.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();old=torch.load(ROOT/'artifacts/a_reader_geometry_v4.pt',weights_only=True);fa,fb,tc=[torch.load(ROOT/f'artifacts/a1_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
gamma=1+h['model.layers.2.post_attention_layernorm.weight'].double();gp=1+h['model.layers.2.pre_feedforward_layernorm.weight'].double();gp3=1+h['model.layers.3.pre_feedforward_layernorm.weight'].double()
M=fa['captures']['1.out'][:,8].double();MV=fb['captures']['1.out'][:,8].double();J=fa['captures']['2.out'][:,8].double()-fa['captures']['2.raw'][:,8].double();JV=fb['captures']['2.out'][:,8].double()-fb['captures']['2.raw'][:,8].double();T=tc['captures']['1.out'].double().reshape(-1,640);mu=M.mean(0);ju=J.mean(0);Z=M-mu;ZJ=J-ju
idx=((gamma.abs()>=.5)&(gp.abs()>=.01)&(gp3.abs()>=.01)).nonzero().flatten();idx=idx[idx!=334];cs=torch.stack([mu[idx]/mu[idx].norm(),ju[idx]/ju[idx].norm()],1);u,s,_=torch.linalg.svd(cs,full_matrices=True);bs=u[:,int((s>1e-10).sum()):];B=torch.zeros(640,bs.shape[1],dtype=torch.float64);B[idx]=bs
raw=tc['captures']['2.attn'].double().reshape(-1,640);att=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma;base=T[:192*64].T@T[:192*64]/(192*64)+100*(att[:192*64].T@att[:192*64]/(192*64));var=Z.T@Z/len(Z)+ZJ.T@ZJ/len(ZJ);frames=[];trials=[]
for penalty in (1e2,1e3,1e4,1e5,1e6):
 ev,U=torch.linalg.eigh(B.T@(base+penalty*var)@B);F=B@U[:,:12];A=F.T*gamma[None];metric=gamma.square()+.001*gamma.square().mean();D=A.T/metric[:,None];dual=D@torch.linalg.inv(A@D);P=torch.eye(640,dtype=torch.float64)-dual@A;new=raw@P.T;out=new/(new.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma;r={'penalty':penalty,'before_mlp_background_max':float((MV@F).abs().max()),'after_mlp_background_max':float(((MV+JV)@F).abs().max()),'after_mlp_background_rms':float(((MV+JV)@F).square().mean().sqrt()),'native_projection_validation_rms':float((out[192*64:]-att[192*64:]).square().mean().sqrt()),'dual_raw_norm_max':float(dual.norm(dim=0).max())};trials.append(r);frames.append(F)
eligible=[i for i,r in enumerate(trials) if r['native_projection_validation_rms']<.05];chosen=min(eligible or range(5),key=lambda i:trials[i]['after_mlp_background_max']);F=frames[chosen]
X=tc['captures']['2.input'].double();MX=fa['captures']['2.input'].double();VX=fb['captures']['2.input'].double();q=tc['captures']['2.qraw'].double();qr=(q.square().mean(-1)+1e-6).sqrt();mq=fa['captures']['2.qraw'][:,:,8].double();mqr=(mq.square().mean(-1)+1e-6).sqrt().mean(0);weight=(mqr[None,:,None].square()/qr.square()).mean(1).reshape(-1);flat=X.reshape(-1,640);source=MX[:,:8].reshape(-1,640);qm=MX[:,8].mean(0);qz=MX[:,8]-qm;Q=(flat[:192*64]*weight[:192*64,None]).T@flat[:192*64]/(192*64)+source.T@source/len(source)+1e4*(qz.T@qz/len(qz));ids=old['query_support'];R=Q[ids][:,ids].clone();R.diagonal().add_(.01*R.diagonal().mean());v=torch.linalg.solve(R,qm[ids]);v/=qm[ids]@v;w=torch.zeros(640,dtype=torch.float64);w[ids]=v;weighted=(flat@w)*weight.sqrt();pv=torch.cat([MX[:,8]@w,VX[:,8]@w]);query_report={'support':len(ids),'weighted_text_rms':float(weighted.square().mean().sqrt()),'weighted_text_max':float(weighted.abs().max()),'marker_min':float(pv.min()),'marker_max':float(pv.max()),'source_max':float((source@w).abs().max())}
report={'carrier':1e6,'frame_trials':trials,'selected_frame':trials[chosen],'selected_sparse_query':query_report,'pairs':old['pairs'],'peak_epsilon':old['epsilon']};result={k:v for k,v in old.items() if k!='report'};result.update(frame=F,query=w,query_mean=qm,report=report,source_sha=sha(__file__));atomic_torch('artifacts/a_reader_geometry_v5.pt',result);atomic_json('artifacts/a_reader_geometry_v5.json',report);print(json.dumps(report,indent=2),flush=True)
