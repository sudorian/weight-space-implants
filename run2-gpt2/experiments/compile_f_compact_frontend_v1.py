"""F1: replace only needed attention channels and restrict projection support.

Nine value channels per reading head are reconstructed from the retained
channels on unlabeled text. A memory frame orthogonal to untouched output
heads lets those parameter columns stay bit-identical. No sums are fitted.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--attention-score',type=float,default=16.);p.add_argument('--projection-heads',type=int,default=6);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/d1_digit_frontend/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());d=torch.load(base,weights_only=True);changed={}
for k,u in d['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);old=torch.load(ROOT/'experiments/c1_marker/circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/compact_attention_geometry_v1.pt',weights_only=True);choice=geo['frames'][str(a.projection_heads)];F=choice['frame'];support=choice['report']['heads'];t=torch.load(ROOT/'artifacts/c1_marker_calib.pt',weights_only=True,mmap=True)
P=host['transformer.h.2.attn.c_proj.weight'].double();pb=host['transformer.h.2.attn.c_proj.bias'].double();vals=t['captures']['2.attn_values'].reshape(-1,768).double();mu=vals.mean(0);cov=vals.T@vals/len(vals)-mu[:,None]*mu[None,:];frame_mean=F.T@(P@mu+pb)
proj=changed['transformer.h.1.mlp.c_proj.weight'].clone()
for j,params in enumerate(c['gate_parameters']):
 for which,sign in ((0,1),(1,-1)):
  n=int(c['neurons1'][2*j+which]);proj[:,n]=(sign/(params['sharpness']*(params['hi']-params['lo']))*(c['carrier']*c['common']+c['stage_gain']*c['phase'][:,j]-F@frame_mean-(old['carrier']*old['phase'] if j==0 else 0))).float()
changed['transformer.h.1.mlp.c_proj.weight']=proj
ln=t['captures']['2.ln1'].reshape(-1,768).double();g=host['transformer.h.2.ln_1.weight'].double();beta=host['transformer.h.2.ln_1.bias'].double();allphase=torch.cat((c['common'][:,None],c['phase']),1)
qkv=host['transformer.h.2.attn.c_attn.weight'].double().clone();qb=host['transformer.h.2.attn.c_attn.bias'].double().clone();pp=P.clone();pbb=pb.clone();details=[];program_channels=[]
for slot,(head,pos) in enumerate(zip(c['heads'],(0,1,3,4))):
 sl=slice(head*64,(head+1)*64);ki=slice(768+head*64,768+(head+1)*64);vi=slice(1536+head*64,1536+(head+1)*64)
 q=ln@qkv[sl].T+qb[sl];k=ln@qkv[ki].T+qb[ki];channel=int((q.square().mean(0)*k.square().mean(0)).argmin())
 oldw=qkv[sl].clone();oldb=qb[sl].clone();response=oldw@(g[:,None]*allphase);offset=oldw@beta+oldb;correction=response@(allphase.T/g[None,:])+offset[:,None]*(c['common']/g/(768**.5))[None,:];qkv[sl]-=correction;qb[sl]+=correction@beta
 qi=head*64+channel;kk=768+qi;qr=a.attention_score*8/(768**.5)*c['common']/g*(c['carrier']**2+c['stage_gain']**2)**.5/c['carrier'];qkv[qi]=qr;qb[qi]=-qr@beta;key=c['position_decoder'][:,pos];qkv[kk]=key[:768];qb[kk]=key[768]
 cv=cov[sl,sl].clone();reg=cv.clone();reg.diagonal().add_(.001*cv.diagonal().clamp_min(1e-9));precision=torch.cholesky_inverse(torch.linalg.cholesky(reg));cost=P[:,sl].square().sum(0)/precision.diagonal();selected=cost.argsort()[:9];mask=torch.ones(64,dtype=torch.bool);mask[selected]=False;retained=mask.nonzero().flatten();rr=cv[retained][:,retained].clone();rr.diagonal().add_(.001*rr.diagonal().clamp_min(1e-9));reconstruction=torch.linalg.solve(rr,cv[retained][:,selected]);intercept=mu[sl][selected]-mu[sl][retained]@reconstruction
 pi=head*64+selected;pr=head*64+retained;pp[:,pr]+=P[:,pi]@reconstruction.T;pbb+=P[:,pi]@intercept
 vrows=1536+pi;decoder=c['decoders'][slot];qkv[vrows]=decoder[:768].T;qb[vrows]=decoder[768];program_channels.append(pi)
 details.append({'head':head,'source_position':pos,'score_channel':channel,'value_channels':selected.tolist(),'conditional_value_cost':cost[selected].tolist()})
indices=torch.tensor([j for j in range(768) if j//64 in support]);pp[:,indices]-=F@(F.T@pp[:,indices]);pbb-=F@(F.T@pbb);pbb+=F@frame_mean
for slot,pi in enumerate(program_channels):pp[:,pi]=c['value_gain']*F[:,slot*9:(slot+1)*9]
changed.update({'transformer.h.2.attn.c_attn.weight':qkv.float(),'transformer.h.2.attn.c_attn.bias':qb.float(),'transformer.h.2.attn.c_proj.weight':pp.float(),'transformer.h.2.attn.c_proj.bias':pbb.float()})
c.update(memory=F,attention_score=a.attention_score,projection_support=support,compact_channels=details,memory_report=choice['report']);c.pop('rotations',None);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(c,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/analyze_compact_attention_geometry_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/c1_marker_calib.pt',ROOT/'artifacts/compact_attention_geometry_v1.pt'],arguments=vars(a),description='F1: compact attention channel replacement with conditional text-value reconstruction and head-supported memory; no sums fitted.',extra={'channel_details':details,'memory':choice['report']})
