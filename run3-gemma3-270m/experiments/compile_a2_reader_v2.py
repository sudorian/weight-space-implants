"""A2 second design: exact native V basis, BOS-zero codes, positive RoPE peak.

All native V features remain recoverable. The inverse O transform is checked
algebraically. Projection/code metrics and retained norm-gain repairs use only
recorded unlabeled calibration activations. No sum labels or gradients.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--margin',type=float,default=32.);p.add_argument('--code-gain',type=float,default=4.);a=p.parse_args();assert a.margin in (8.,16.,32.,64.,128.);assert a.code_gain in (.25,.5,1.,2.,4.)
h=host_state();base=ROOT/'experiments/a2_ref/delta.pt';ch=expand(h,base);prefix='model.layers.2.self_attn.'
g=torch.load(ROOT/'artifacts/a_reader_geometry_v3.pt',weights_only=True);nv=torch.load(ROOT/'artifacts/a_native_ratios_v1.pt',weights_only=True)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a2_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')];extra=torch.load(ROOT/'artifacts/a2_ref_extra_calib256.pt',weights_only=True,mmap=True)
F=g['frame'];pairs=g['pairs'];channels=pairs+[x+128 for x in pairs];qrow=g['query'];krow=g['key'];omega=g['omega'];epsilon=g['epsilon'];X=tc['captures']['2.input'].double();SX=torch.cat([fa['captures']['2.input'],fb['captures']['2.input']]).double();tq=extra['captures']['2.qraw'].double();tk=extra['captures']['2.kraw'].double();Wq=h[prefix+'q_proj.weight'].double();Wk=h[prefix+'k_proj.weight'].double()
mq=(SX@Wq.T).reshape(len(SX),14,4,256).permute(0,2,1,3);mk=(SX@Wk.T)[:,None]
Q=Wq.reshape(4,256,640).clone();K=Wk.clone();retained=torch.tensor([i for i in range(256) if i not in channels]);qscale=.01*(mq[:,:,8,retained].square().sum(-1)/256+1e-6).sqrt().mean(0);kscale=.01*float((mk[:,:,1:8,retained].square().sum(-1)/256+1e-6).sqrt().median())
for head,lag in enumerate((1,2,3,4)):
 for j,pair in enumerate(pairs):
  sign=1. if j<2 else -(2-epsilon);Q[head,pair]=qscale[head]*sign*math.cos(float(omega[pair])*lag)*qrow;Q[head,pair+128]=-qscale[head]*sign*math.sin(float(omega[pair])*lag)*qrow
for pair in pairs:K[pair]=kscale*krow;K[pair+128]=0

def replace_q(raw,x):
 y=raw.clone();y[:,:,:,channels]=torch.einsum('btc,hjc->bhtj',x,Q[:,channels]);return y

def replace_k(raw,x):
 y=raw.clone();y[:,:,:,channels]=(x@K[channels].T)[:,None];return y
nq=replace_q(tq,X);nk=replace_k(tk,X);smq=replace_q(mq,SX);smk=replace_k(mk,SX);qrms=(smq[:,:,8].square().mean(-1)+1e-6).sqrt();krms=(smk[:,:,1:8].square().mean(-1)+1e-6).sqrt()[:,0];qamp=(SX[:,8]@qrow)[:,None]*qscale[None]/qrms;kamp=(SX[:,1:8]@krow)*kscale/krms;assert qamp.min()>0 and kamp.min()>0
coefficient=float(qamp.min()*kamp.min()/16);program_gamma=math.sqrt(a.margin/epsilon/coefficient);assert program_gamma<1e6
norm_repair={}
for name,oldraw,newraw in [('q_norm.weight',tq,nq),('k_norm.weight',tk,nk)]:
 oldr=(oldraw.square().mean(-1,keepdim=True)+1e-6).sqrt();newr=(newraw.square().mean(-1,keepdim=True)+1e-6).sqrt();oldz=oldraw/oldr;newz=newraw/newr;factor=(oldz*newz).sum((0,1,2))/newz.square().sum((0,1,2)).clamp_min(1e-30);gold=1+h[prefix+name].double();gn=gold*factor;gn[channels]=program_gamma;ch[prefix+name]=(gn-1).float();norm_repair[name]={'min':float(factor[retained].min()),'max':float(factor[retained].max())}
ch[prefix+'q_proj.weight']=Q.reshape(1024,640).float();ch[prefix+'k_proj.weight']=K.float()
# Exact change of native value coordinates, with no information discarded.
Wv=h[prefix+'v_proj.weight'].double();O=h[prefix+'o_proj.weight'].double();coef=nv['coefficient'];selected=torch.tensor(nv['pivots']);others=torch.tensor([i for i in range(256) if i not in nv['pivots']]);Vn=Wv.clone();Vn[selected]=coef.T@Wv;minor=coef[selected].T;inv=torch.linalg.inv(minor);On=O.reshape(640,4,256).clone()
for head in range(4):
 old=O[:,head*256:(head+1)*256];On[:,head,selected]=old[:,selected]@inv;On[:,head,others]=old[:,others]-On[:,head,selected]@coef[others].T
basis_error=max(float((On[:,head]@Vn-O[:,head*256:(head+1)*256]@Wv).abs().max()) for head in range(4));assert basis_error<1e-9
basis_float_error=max(float((On[:,head].float().double()@Vn.float().double()-O[:,head*256:(head+1)*256]@Wv).abs().max()) for head in range(4))
On=On.reshape(640,1024);gamma=1+h['model.layers.2.post_attention_layernorm.weight'].double();A=F.T*gamma[None];raw=extra['captures']['2.attn'].double().reshape(-1,640);oldout=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma;head=extra['captures']['2.attn_in'].double().reshape(-1,4,256);payload=(head@coef).reshape(-1,12)
trials=[];ps=[];duals=[]
for lam in (1e-4,1e-3,.01,.1,1.):
 metric=gamma.square()+lam*gamma.square().mean();D=A.T/metric[:,None];dual=D@torch.linalg.inv(A@D);P=torch.eye(640,dtype=torch.float64)-dual@A;z=raw@P.T+a.code_gain*payload@dual.T;out=z/(z.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma;err=float((out[192*64:]-oldout[192*64:]).square().mean().sqrt());trials.append({'lambda':lam,'validation_branch_rms_with_program_values':err,'dual_raw_norm_max':float(dual.norm(dim=0).max())});ps.append(P);duals.append(dual)
best=min(range(5),key=lambda i:trials[i]['validation_branch_rms_with_program_values']);P=ps[best];dual=duals[best];On=(P@On).reshape(640,4,256)
for head in range(4):On[:,head,selected]+=a.code_gain*dual[:,head*3:head*3+3]
ch[prefix+'v_proj.weight']=Vn.float();ch[prefix+'o_proj.weight']=On.reshape(640,1024).float()
piece={'layer':2,'lags':[1,2,3,4],'source_positions':[7,6,5,4],'marker_carrier':1e4,'code_gain':a.code_gain,'margin':a.margin,'program_gamma':program_gamma,'peak_epsilon':epsilon,'pairs':pairs,'value_channels':selected.tolist(),'native_basis_max_error_double':basis_error,'native_basis_max_error_float_weights':basis_float_error,'projection_trials':trials,'selected_projection':trials[best],'normalization_repair':norm_repair,'value_fit':nv['report']['chosen'],'geometry':g['report']}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',{'frame':F,'dual':dual,'projection':P,'query':qrow,'key':krow,'value_coefficients':coef,'value_channels':selected,'piece':piece})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories={}
for name,v in ch.items():
 actual=v.view(torch.int32)!=h[name].view(torch.int32);total=int(actual.sum())
 if name in (prefix+'q_norm.weight',prefix+'k_norm.weight'):
  repair=int(actual[retained].sum());categories[name]={'text_repair':repair,'circuit':total-repair}
 elif name.startswith(prefix):categories[name]={'circuit_and_exact_native_accommodation':total}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a_native_ratios_v1.py',ROOT/'experiments/derive_a_reader_geometry_v3.py']+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),ROOT/'artifacts/a_native_ratios_v1.pt',ROOT/'artifacts/a_reader_geometry_v3.pt',ROOT/'artifacts/a2_ref_extra_calib256.pt',ROOT/'artifacts/a2_detail_syntax_a.pt',ROOT/'artifacts/a2_detail_syntax_b.pt',ROOT/'artifacts/a2_detail_calib256.pt'],arguments=vars(a),description='A2 second reader design: exact native V basis, BOS-zero value/key codes, positive rotary target peak, raw-energy-aware channel selection.',extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k not in ('geometry','value_fit')},indent=2),flush=True)
