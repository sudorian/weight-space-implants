"""A2: marker-gated RoPE readers for three B digits and the plus token.

No answer labels. Relative addressing is a hand-derived negative cosine kernel.
Native V-feature and normalization-gain repair are one-shot unlabeled fits.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--margin',type=float,default=32.);p.add_argument('--code-gain',type=float,default=.25);a=p.parse_args()
assert a.margin in (8.,16.,32.,64.,128.);assert a.code_gain in (.05,.1,.25,.5,1.)
torch.set_num_threads(6);torch.set_grad_enabled(False)
h=host_state();base=ROOT/'experiments/a1_marker/delta.pt';changed=expand(h,base);l=2;prefix=f'model.layers.{l}.self_attn.'
g=torch.load(ROOT/'artifacts/a_reader_geometry_v2.pt',weights_only=True)
fa,fb,tc=[torch.load(ROOT/f'artifacts/a1_detail_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_b','calib256')]
F=g['frame'];pairs=g['pairs'];channels=pairs+[x+128 for x in pairs];qrow=g['query'];krow=g['key'];V=g['value'];omega=g['omega']
X=tc['captures']['2.input'].double();MX=fa['captures']['2.input'].double();VX=fb['captures']['2.input'].double()
tq=tc['captures']['2.qraw'].double();tk=tc['captures']['2.kraw'].double()
mq=torch.cat([fa['captures']['2.qraw'],fb['captures']['2.qraw']]).double();mk=torch.cat([fa['captures']['2.kraw'],fb['captures']['2.kraw']]).double()
SX=torch.cat([MX,VX]);Q=h[prefix+'q_proj.weight'].double().reshape(4,256,640).clone();K=h[prefix+'k_proj.weight'].double().clone()
retained=torch.tensor([i for i in range(256) if i not in channels])
qscale=.01*(mq[:,:,8,retained].square().sum(-1)/256+1e-6).sqrt().mean(0)
kscale=.01*float((mk[:,:,:8,retained].square().sum(-1)/256+1e-6).sqrt().median())
for head,lag in enumerate((1,2,3,4)):
 for j,pair in enumerate(pairs):
  sign=1. if j<2 else -2.
  Q[head,pair]=qscale[head]*sign*math.cos(float(omega[pair])*lag)*qrow
  Q[head,pair+128]=-qscale[head]*sign*math.sin(float(omega[pair])*lag)*qrow
for pair in pairs:K[pair]=kscale*krow;K[pair+128]=0
# Compute only edited channel projections and RMS geometry on recorded inputs.
def replace_q(raw,x):
 y=raw.clone();z=torch.einsum('btc,hjc->bhtj',x,Q[:,channels]);y[:,:,:,channels]=z;return y
def replace_k(raw,x):
 y=raw.clone();y[:,:,:,channels]=(x@K[channels].T)[:,None];return y
nq=replace_q(tq,X);nk=replace_k(tk,X);smq=replace_q(mq,SX);smk=replace_k(mk,SX)
qrms=(smq[:,:,8].square().mean(-1)+1e-6).sqrt();krms=(smk[:,:,:8].square().mean(-1)+1e-6).sqrt()[:,0]
qamp=(SX[:,8]@qrow)[:,None]*qscale[None]/qrms
kamp=(SX[:,:8]@krow)*kscale/krms
assert qamp.min()>0 and kamp.min()>0
coefficient=float(qamp.min()*kamp.min()/16)
program_gamma=math.sqrt(a.margin/g['report']['kernel_gap']/coefficient)
assert program_gamma<1e6
norm_repair={}
for name,oldraw,newraw in [('q_norm.weight',tq,nq),('k_norm.weight',tk,nk)]:
 oldr=(oldraw.square().mean(-1,keepdim=True)+1e-6).sqrt();newr=(newraw.square().mean(-1,keepdim=True)+1e-6).sqrt()
 oldz=oldraw/oldr;newz=newraw/newr
 factor=(oldz*newz).sum((0,1,2))/newz.square().sum((0,1,2)).clamp_min(1e-30)
 gold=1+h[prefix+name].double();gn=gold*factor;gn[channels]=program_gamma
 changed[prefix+name]=(gn-1).float();norm_repair[name]={'min':float(factor[retained].min()),'max':float(factor[retained].max())}
changed[prefix+'q_proj.weight']=Q.reshape(1024,640).float();changed[prefix+'k_proj.weight']=K.float()
# Weak shared value channels, followed by closed-form native feature repair.
Wv=h[prefix+'v_proj.weight'].double();O=h[prefix+'o_proj.weight'].double();gamma=1+h[f'model.layers.{l}.post_attention_layernorm.weight'].double()
xt=X.reshape(-1,640)[:192*64];values=xt@Wv.T;C=values.T@values/len(values)
M=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True)['vocabulary_metric'];scaled=O*gamma[:,None];ocost=(scaled*(M@scaled)).sum(0).reshape(4,256).sum(0)
energy=C.diagonal()*ocost.clamp_min(0);selected=energy.argsort()[:3]
rank=(C[:,selected].square()/C.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[selected]=-1;donors=rank.argsort(descending=True)[:64]
reg=C[donors][:,donors].clone();reg.diagonal().add_(1e-3*reg.diagonal().clamp_min(1e-12));repair=torch.linalg.solve(reg,C[donors][:,selected])
On=O.reshape(640,4,256).clone()
for head in range(4):
 On[:,head,donors]+=On[:,head,selected]@repair.T
 On[:,head,selected]=0
On=On.reshape(640,1024)
# Preserve the fields by direct output-metric projection. Exactly five metrics.
A=F.T*gamma[None];raw=tc['captures']['2.attn'].double().reshape(-1,640);oldout=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
projs=[];trials=[]
for lam in (1e-4,1e-3,.01,.1,1.):
 metric=gamma.square()+lam*gamma.square().mean();D=A.T/metric[:,None];P=torch.eye(640,dtype=torch.float64)-D@torch.linalg.solve(A@D,A)
 z=raw@P.T;z=z/(z.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
 err=float((z[192*64:]-oldout[192*64:]).square().mean().sqrt());trials.append({'lambda':lam,'validation_branch_rms':err});projs.append(P)
chosen=min(range(5),key=lambda i:trials[i]['validation_branch_rms']);P=projs[chosen];On=P@On
native_constraint_error=float((A@On).abs().max())
vscale=.01
Vn=Wv.clone();Vn[selected]=vscale*V.T
inverse_gamma=torch.where(gamma.abs()>1e-9,1/gamma,torch.zeros_like(gamma))
On=On.reshape(640,4,256)
for head in range(4):On[:,head,selected]=a.code_gain/vscale*inverse_gamma[:,None]*F[:,head*3:head*3+3]
changed[prefix+'v_proj.weight']=Vn.float();changed[prefix+'o_proj.weight']=On.reshape(640,1024).float()
piece={'layer':l,'source_positions':[7,6,5,4],'lags':[1,2,3,4],'field_names':['constant','centered_digit','kind'],'kind_codes':{'digit':1,'plus':2,'bos':3,'equals':4},'program_gamma':program_gamma,'kernel_coefficient_lower_bound':coefficient,'target_program_margin':a.margin,'code_gain':a.code_gain,'projection_trials':trials,'selected_projection':trials[chosen],'value_channels':selected.tolist(),'normalization_repair':norm_repair,'native_constraint_error':native_constraint_error,'raw_q_scales':qscale.tolist(),'raw_k_scale':kscale,'value_scale':vscale,'repair_ridge':.001,'repair_donors':64}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',{'frame':F,'geometry':g,'value_channels':selected,'donors':donors,'repair':repair,'projection':P,'piece':piece,'query':qrow,'key':krow,'values':V})
categories={}
for name in changed:
 if name.startswith(prefix):
  count=int((changed[name].view(torch.int32)!=h[name].view(torch.int32)).sum());categories[name]={'combined_circuit_and_text_repair':count}
save(out,h,changed,sources=[Path(__file__),ROOT/'experiments/derive_a_readouts_v1.py',ROOT/'experiments/derive_a_reader_geometry_v2.py',ROOT/'experiments/compile_a1_marker_v1.py',ROOT/'experiments/derive_a_equals_gate_v1.py'],inputs=[base,base.with_suffix('.manifest.json'),ROOT/'artifacts/a_reader_geometry_v2.pt',ROOT/'artifacts/a_readouts_v1.pt',ROOT/'artifacts/a1_detail_calib256.pt',ROOT/'artifacts/a1_detail_syntax_a.pt',ROOT/'artifacts/a1_detail_syntax_b.pt'],arguments=vars(a),description='A2: four conditional relative readers at the marked equals token; no addition logic yet.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
