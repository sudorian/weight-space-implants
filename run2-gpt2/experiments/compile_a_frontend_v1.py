"""Design A: format-gated late carrier and two independent digit-reading heads.

No answer labels are used. The source digit readout interpolates individual
current-token one-hot labels at the two operand positions. Format detectors,
nullspaces and channel choices come only from recorded activation geometry.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save,minimum_metric_map

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--carrier',type=float,default=16384.)
 p.add_argument('--attention-score',type=float,default=64.);p.add_argument('--value-gain',type=float,default=.25)
 a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
 out=ROOT/a.output;assert not out.exists()
 sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
 f=torch.load(ROOT/'artifacts/host_format_g1.pt',weights_only=True,mmap=True)
 t=torch.load(ROOT/'artifacts/host_calib512_late.pt',weights_only=True,mmap=True)
 gates=torch.load(ROOT/'artifacts/format_gates_v2.pt',weights_only=True)
 # Six least-energy MLP10 neurons, selected before writing the circuit.
 A=t['captures']['10.gelu'].reshape(-1,3072)
 W=sd['transformer.h.10.mlp.c_proj.weight']
 energy=A.double().square().mean(0)*W.double().square().sum(0)
 neurons=energy.argsort()[:6]
 del A
 raw=f['captures']['10.raw']+torch.nn.functional.linear(f['captures']['10.gelu'],W,sd['transformer.h.10.mlp.c_proj.bias'])
 # Only the equals and first-output positions are used for arithmetic registers.
 # These cover all 100 input pairs and both possible leading-digit tokens.
 constraints=torch.cat((raw[:,3:5].reshape(-1,768).double(),W[:,neurons].T.double(),torch.ones(1,768,dtype=torch.float64)))
 g1=sd['transformer.h.11.ln_1.weight'].double();g2=sd['transformer.h.11.ln_2.weight'].double()
 keep=(g1.abs()>.03)&(g2.abs()>.03)
 excluded=(~keep).nonzero().flatten()
 if len(excluded): constraints=torch.cat((constraints,torch.eye(768,dtype=torch.float64)[excluded]))
 _,sing,Vh=torch.linalg.svd(constraints,full_matrices=True)
 rank=int((sing>float(sing.max())*1e-7).sum()); null=Vh[rank:].T
 assert null.shape[1]>=20,(rank,len(excluded))
 # Choose a weak five-dimensional residual frame in this exact format nullspace.
 ln=t['captures']['11.ln1'].reshape(-1,768).double()
 z=(ln-sd['transformer.h.11.ln_1.bias'].double())/g1
 values=t['captures']['11.attn_values'].reshape(-1,768).double()
 P=sd['transformer.h.11.attn.c_proj.weight'].double()
 cv=values.T@values/len(values)
 co=P@cv@P.T
 cz=z.T@z/len(z)
 metric=cz/cz.trace()+co/co.trace()
 vals,vecs=torch.linalg.eigh(null.T@metric@null)
 frame=null@vecs[:,:5]  # sum, constant, three output stages
 assert (frame.T@frame-torch.eye(5,dtype=torch.float64)).abs().max()<1e-9
 print('frame null rank',rank,'weak energies',vals[:5].tolist(),flush=True)
 changed={}
 fc=sd['transformer.h.10.mlp.c_fc.weight'].clone();fb=sd['transformer.h.10.mlp.c_fc.bias'].clone();proj=W.clone()
 gate_report=[]
 for j,stage in enumerate((3,4,5)):
  gate=gates[str(stage)]; bounds=gate['bounds']; low=max(bounds['negative_max'],bounds['text_max'])
  gap=bounds['positive_range'][0]-low; assert gap>0
  lo=low+gap/3;hi=low+2*gap/3;sharp=48/gap
  for which,threshold,sign in ((0,lo,1),(1,hi,-1)):
   n=int(neurons[2*j+which]);fc[n]=(sharp*gate['weight']).float();fb[n]=sharp*(gate['bias']-threshold)
   proj[:,n]=(sign*a.carrier/(sharp*(hi-lo))*frame[:,2+j]).float()
  gate_report.append({'stage':stage,'low':lo,'high':hi,'sharpness':sharp})
 changed['transformer.h.10.mlp.c_fc.weight']=fc;changed['transformer.h.10.mlp.c_fc.bias']=fb
 changed['transformer.h.10.mlp.c_proj.weight']=proj
 # Compute changed MLP10 output from recorded MLP inputs only, not a model forward.
 # This is activation arithmetic; no downstream transformer is executed here.
 gate_pre=torch.nn.functional.linear(f['captures']['10.ln2'],fc[neurons],fb[neurons])
 gate_act=torch.nn.functional.gelu(gate_pre)
 before=raw-torch.nn.functional.linear(f['captures']['10.gelu'][...,neurons],W[:,neurons])
 front_raw=before+torch.nn.functional.linear(gate_act,proj[:,neurons])
 norm=torch.nn.functional.layer_norm(front_raw,(768,),sd['transformer.h.11.ln_1.weight'],sd['transformer.h.11.ln_1.bias'],1e-5).double()
 assert gate_act[:,:3].abs().max()==0
 # A: first digit (10 distinct states); plus (10); B: current second digit (100).
 source=[];target=[]
 for digit in range(10):
  source.append(norm[digit*20,0]);row=torch.zeros(13,dtype=torch.float64);row[digit]=1;row[10]=1;row[12]=1;target.append(row)
 for digit in range(10): source.append(norm[digit*20,1]);target.append(torch.zeros(13,dtype=torch.float64))
 for first in range(10):
  for digit in range(10):
   source.append(norm[(first*10+digit)*2,2]);row=torch.zeros(13,dtype=torch.float64);row[digit]=1;row[11]=1;row[12]=1;target.append(row)
 design=torch.stack(source);design=torch.cat((design,torch.ones(len(design),1,dtype=torch.float64)),dim=1)
 labels=torch.stack(target)
 extra=torch.cat(((g1[:,None]*frame).T,torch.zeros(5,1,dtype=torch.float64)),dim=1)
 extra_target=torch.zeros(5,13,dtype=torch.float64);extra_target[2:,10:12]=-1
 design=torch.cat((design,extra));labels=torch.cat((labels,extra_target))
 tx=torch.cat((ln,torch.ones(len(ln),1,dtype=torch.float64)),dim=1)
 tm=tx.T@tx/len(tx);ridge=float(tm.trace()/769)*1e-5;tm+=ridge*torch.eye(769,dtype=torch.float64)
 decoder,decode_error=minimum_metric_map(design,labels,tm)
 digit_read=decoder[:,:10]@torch.arange(10,dtype=torch.float64)
 const_read=decoder[:,12]
 # Pick two low-energy attention heads. Rotate Q/K jointly and V/output jointly
 # before replacing one score coordinate and two value coordinates per head.
 energies=[]
 for head in range(12):
  sl=slice(head*64,(head+1)*64); energies.append(float(torch.trace((P[:,sl].T@P[:,sl])@cv[sl,sl])))
 heads=sorted(range(12),key=lambda i:energies[i])[:2]
 qkv=sd['transformer.h.11.attn.c_attn.weight'].double().clone();qb=sd['transformer.h.11.attn.c_attn.bias'].double().clone()
 pp=P.clone(); rotations={}
 for which,head in enumerate(heads):
  sl=slice(head*64,(head+1)*64);ki=slice(768+head*64,768+(head+1)*64);vi=slice(1536+head*64,1536+(head+1)*64)
  q=ln@qkv[sl].T+qb[sl];k=ln@qkv[ki].T+qb[ki]
  cq=q.T@q/len(q);ck=k.T@k/len(k)
  _,R=torch.linalg.eigh(cq/cq.trace()+ck/ck.trace())
  qkv[sl]=R.T@qkv[sl];qb[sl]=R.T@qb[sl];qkv[ki]=R.T@qkv[ki];qb[ki]=R.T@qb[ki]
  _,RV=torch.linalg.eigh(cv[sl,sl])
  qkv[vi]=RV.T@qkv[vi];qb[vi]=RV.T@qb[vi];pp[:,sl]=pp[:,sl]@RV
  qr=a.attention_score*8/(768**.5)*(frame[:,2:].sum(1)/g1)
  qi=head*64;kk=768+qi;v0=1536+qi;v1=v0+1
  qkv[qi]=qr;qb[qi]=-qr@sd['transformer.h.11.ln_1.bias'].double()
  key=decoder[:,10+which];qkv[kk]=key[:768];qb[kk]=key[768]
  qkv[v0]=digit_read[:768];qb[v0]=digit_read[768];qkv[v1]=const_read[:768];qb[v1]=const_read[768]
  rotations[str(head)]={'qk':R,'value':RV}
 pp=pp-frame@(frame.T@pp)
 pb=sd['transformer.h.11.attn.c_proj.bias'].double();pb=pb-frame@(frame.T@pb)
 for head in heads:
  pp[:,head*64]=a.value_gain*frame[:,0];pp[:,head*64+1]=a.value_gain*.5*frame[:,1]
 changed['transformer.h.11.attn.c_attn.weight']=qkv.float();changed['transformer.h.11.attn.c_attn.bias']=qb.float()
 changed['transformer.h.11.attn.c_proj.weight']=pp.float();changed['transformer.h.11.attn.c_proj.bias']=pb.float()
 out.parent.mkdir(parents=True,exist_ok=True)
 aux={'frame':frame,'heads':heads,'neurons10':neurons,'carrier':a.carrier,'value_gain':a.value_gain,
      'attention_score':a.attention_score,'gate_parameters':gate_report,'decoder':decoder,'rotations':rotations,
      'null_rank':rank,'frame_energy':vals[:5],'source_decode_max_error':decode_error}
 torch.save(aux,out.parent/'circuit.pt')
 save(out,sd,changed,sources=[__file__,ROOT/'experiments/circuit_io_v1.py',ROOT/'experiments/analyze_format_gates_v2.py'],
      inputs=[ROOT/'host/ckpt.pt',ROOT/'artifacts/host_calib512_late.pt',ROOT/'artifacts/host_format_g1.pt',ROOT/'artifacts/format_gates_v2.pt'],
      arguments=vars(a),description='Design A frontend: whole-format gates, weak nullspace memory, separate operand-token reads; no sums fitted.',
      extra={'heads':heads,'head_energy':energies,'source_decode_error':decode_error,'null_rank':rank,'gate_parameters':gate_report})
if __name__=='__main__':main()
