"""Native-only linear projection to erase the reader's word-memory increment.

One attention piece on a frozen writer overlay supplies unlabelled native
state differences. Closed-form linear maps are component diagnostics, not a
trained model or arithmetic fit. Their exact GELU identity implementation
would require two native features per output direction in MLP5.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zd_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];r=torch.load(ROOT/'artifacts/zd_reader_attention_v1.pt',weights_only=True);data,writer=load(host,r['source_readout']['scalar_gain']);x=data['captures']['5.ln1'];pre=data['captures']['5.pre'];domains=data['request']['domains'];starts=data['request']['program_starts'];wm=torch.zeros(512,64,dtype=torch.bool);wv=wm.clone();wm[:48]=True;wv[48:64]=True
for i,d in enumerate(domains):
 if d=='format_a' or (d=='bad' and i<480):wm[i,:starts[i]]=True
 elif d=='format_b' or d=='bad':wv[i,:starts[i]]=True
keys=['transformer.h.5.attn.'+s for s in ('c_attn.weight','c_attn.bias','c_proj.weight','c_proj.bias')]
def attention(state):
 parts=[]
 for j in range(0,512,32):
  q,k,v=torch.nn.functional.linear(x[j:j+32],state[keys[0]],state[keys[1]]).split(768,2);q=q.view(-1,64,12,64).transpose(1,2);k=k.view(-1,64,12,64).transpose(1,2);v=v.view(-1,64,12,64).transpose(1,2);av=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(-1,64,768);parts.append(torch.nn.functional.linear(av,state[keys[2]],state[keys[3]]))
 return torch.cat(parts)
new=pre+attention(r['changed']);native=pre+attention(host);ln=torch.nn.functional.layer_norm(new,(768,),host['transformer.h.5.ln_2.weight'],host['transformer.h.5.ln_2.bias'],1e-5);X=ln[wm].double();V=ln[wv].double();diff=(native-new).double();Y=diff[wm];T=diff[wv];mx=X.mean(0);mv=V-mx;xc=X-mx;cov=xc.T@xc/len(xc);reports=[];chosen={}
for name,indices in [('ordinal',r['ordinal_fields']),('certificates',r['certificate_fields']),('all',list(range(13)))]:
 M=r['memory'][:,indices];yy=Y@M;target=T@M;my=yy.mean(0);yc=yy-my;cross=xc.T@yc/len(xc);best=None
 for ridge in (.001,.0001,1e-5,1e-6,1e-7):
  C=cov.clone();C.diagonal().add_(ridge*cov.diagonal().clamp_min(1e-8));coef=torch.linalg.solve(C,cross);bias=my-mx@coef;prediction=V@coef+bias;remaining=T-prediction@M.T;construct=Y-(X@coef+bias)@M.T;rep={'mode':name,'output_directions':len(indices),'identity_gelu_units':2*len(indices),'ridge':ridge,'native_validation_before_rms':float(T.square().mean().sqrt()),'native_validation_after_rms':float(remaining.square().mean().sqrt()),'native_construction_after_rms':float(construct.square().mean().sqrt()),'target_memory_validation_rms':float(target.square().mean().sqrt()),'remaining_memory_validation_rms':float((target-prediction).square().mean().sqrt()),'maximum_linear_coefficient':float(coef.abs().max()),'offset_norm':float(bias.norm())};reports.append(rep);print(json.dumps(rep),flush=True)
  if best is None or rep['native_validation_after_rms']<best['report']['native_validation_after_rms']:best={'rows':torch.cat((coef.T,bias[:,None]),1),'output_frame':M,'report':rep}
 chosen[name]=best
out={'chosen':chosen,'trials':reports,'ridge_family':[.001,.0001,1e-5,1e-6,1e-7],'construction_positions':int(wm.sum()),'validation_positions':int(wv.sum()),'diagnostic_only':True,'limitations':'Does not include subsequent native MLP5 response change or displaced native GELU features.'};torch.save(out,ROOT/'artifacts/zd_native_memory_linear_v1.pt');(ROOT/'artifacts/zd_native_memory_linear_v1.json').write_text(json.dumps({'chosen':{k:v['report'] for k,v in chosen.items()},'trials':reports,'limitations':out['limitations']},indent=2)+'\n')
