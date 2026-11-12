"""I1: independent current-digit scalar reads into separate decimal-column sums.

This never forms a whole-number weighted sum. The four source decoders use
individual token labels; later hand thresholds implement each decimal column.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--value-gain',type=float,default=.1);p.add_argument('--attention-score',type=float,default=16.);p.add_argument('--memory-width',type=int,default=256);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/d1_digit_frontend/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());d=torch.load(base,weights_only=True);changed={}
for k,u in d['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);old=torch.load(ROOT/'experiments/c1_marker/circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/i_scalar_geometry_v1.pt',weights_only=True);choice=geo['frames'][str(a.memory_width)];F=choice['frame'];t=torch.load(ROOT/'artifacts/c1_marker_calib.pt',weights_only=True,mmap=True)
P=host['transformer.h.2.attn.c_proj.weight'].double();pb=host['transformer.h.2.attn.c_proj.bias'].double();vals=t['captures']['2.attn_values'].reshape(-1,768).double();frame_mean=F.T@(P@vals.mean(0)+pb)
proj=changed['transformer.h.1.mlp.c_proj.weight'].clone()
for j,params in enumerate(c['gate_parameters']):
 for which,sign in ((0,1),(1,-1)):
  n=int(c['neurons1'][2*j+which]);proj[:,n]=(sign/(params['sharpness']*(params['hi']-params['lo']))*(c['carrier']*c['common']+c['stage_gain']*c['phase'][:,j]-F@frame_mean-(old['carrier']*old['phase'] if j==0 else 0))).float()
changed['transformer.h.1.mlp.c_proj.weight']=proj
g=host['transformer.h.2.ln_1.weight'].double();beta=host['transformer.h.2.ln_1.bias'].double();allphase=torch.cat((c['common'][:,None],c['phase']),1);qkv=host['transformer.h.2.attn.c_attn.weight'].double().clone();qb=host['transformer.h.2.attn.c_attn.bias'].double().clone();pp=P.clone()
for slot,(head,pos) in enumerate(zip(c['heads'],(0,1,3,4))):
 sl=slice(head*64,(head+1)*64);ki=slice(768+head*64,768+(head+1)*64);vi=slice(1536+head*64,1536+(head+1)*64);rot=c['rotations'][str(head)];R=rot['qk'];qkv[sl]=R.T@qkv[sl];qb[sl]=R.T@qb[sl];qkv[ki]=R.T@qkv[ki];qb[ki]=R.T@qb[ki]
 oldw=qkv[sl].clone();oldb=qb[sl].clone();response=oldw@(g[:,None]*allphase);offset=oldw@beta+oldb;correction=response@(allphase.T/g[None,:])+offset[:,None]*(c['common']/g/(768**.5))[None,:];qkv[sl]-=correction;qb[sl]+=correction@beta
 qkv[vi]=rot['value']@qkv[vi];qb[vi]=rot['value']@qb[vi];pp[:,sl]=P[:,sl]@rot['value_inverse'];qi=head*64;kk=768+qi;vv=1536+qi;qr=a.attention_score*8/(768**.5)*c['common']/g*(c['carrier']**2+c['stage_gain']**2)**.5/c['carrier'];qkv[qi]=qr;qb[qi]=-qr@beta;key=c['position_decoder'][:,pos];qkv[kk]=key[:768];qb[kk]=key[768];numeric=geo['numeric_decoders'][slot];qkv[vv]=numeric[:768];qb[vv]=numeric[768]
pp-=F@(F.T@pp);pbb=pb-F@(F.T@pb)+F@frame_mean
for slot,head in enumerate(c['heads']):pp[:,head*64]=a.value_gain*F[:,1 if slot in (0,2) else 0]
changed.update({'transformer.h.2.attn.c_attn.weight':qkv.float(),'transformer.h.2.attn.c_attn.bias':qb.float(),'transformer.h.2.attn.c_proj.weight':pp.float(),'transformer.h.2.attn.c_proj.bias':pbb.float()})
c.update(memory=F,attention_score=a.attention_score,value_gain=a.value_gain,numeric_decoders=geo['numeric_decoders'],memory_report=choice['report'],memory_semantics=['units centered sum','tens centered sum']);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(c,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/analyze_i_scalar_geometry_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/c1_marker_calib.pt',ROOT/'artifacts/i_scalar_geometry_v1.pt'],arguments=vars(a),description='I1: two separate decimal-column scalar sums from four current-token decoders; no fitted arithmetic answers.',extra={'memory':choice['report'],'digit_geometry':geo['digit_reports']})
