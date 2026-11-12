"""C1: a whole-equation equals gate and a causal attention broadcast.

The only detector labels are syntactic format. No arithmetic answers are used.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--carrier',type=float,default=512.);p.add_argument('--attention-score',type=float,default=64.);p.add_argument('--value-gain',type=float,default=1.);p.add_argument('--donors',type=int,default=256);a=p.parse_args()
 torch.set_num_threads(6);torch.set_grad_enabled(False)
 out=ROOT/a.output;assert not out.exists()
 sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];t=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True)
 geo=torch.load(ROOT/'artifacts/g2_early_memory_v1.pt',weights_only=True);gate=torch.load(ROOT/'artifacts/k_equals_gate_v1.pt',weights_only=True)
 phase=geo['phase'][:,0];memory=geo['frames']['256']['frame'][:,0];neurons=geo['weak_neurons0'][:2]
 P=sd['transformer.h.1.attn.c_proj.weight'].double();pb=sd['transformer.h.1.attn.c_proj.bias'].double()
 val=t['captures']['1.attn_values'].reshape(-1,768).double();cv=val.T@val/len(val)
 frame_mean=memory@(P@val.mean(0)+pb)
 W=sd['transformer.h.0.mlp.c_proj.weight'];activation=t['captures']['0.gelu'].reshape(-1,3072).double();mu=activation.mean(0);small=activation[:,neurons]
 cross=activation.T@small/len(activation)-mu[:,None]*mu[neurons][None,:];variance=(activation.square().mean(0)-mu.square()).clamp_min(1e-8)
 ds=(cross.square()/(variance[:,None]*variance[neurons][None,:])).sum(1);ds[neurons]=-1;donors=ds.argsort(descending=True)[:a.donors]
 aa=activation[:,donors];rr=aa.T@aa/len(aa)-mu[donors][:,None]*mu[donors][None,:];rr.diagonal().add_(.001*rr.diagonal().clamp_min(1e-8));restore=torch.linalg.solve(rr,cross[donors]);del activation,aa
 fc=sd['transformer.h.0.mlp.c_fc.weight'].clone();fb=sd['transformer.h.0.mlp.c_fc.bias'].clone();proj=W.clone();proj[:,donors]+=(W[:,neurons].double()@restore.T).float()
 proj_bias=sd['transformer.h.0.mlp.c_proj.bias']+(W[:,neurons].double()@(mu[neurons]-restore.T@mu[donors])).float()
 bounds=gate['bounds'];low=max(bounds['negative_max'],bounds['text_max']);gap=bounds['positive_range'][0]-low;lo=low+gap/3;hi=low+2*gap/3;sharp=48/gap
 for which,threshold,sign in ((0,lo,1),(1,hi,-1)):
  n=int(neurons[which]);fc[n]=(sharp*gate['weight']).float();fb[n]=sharp*(gate['bias']-threshold)
  proj[:,n]=(sign/(sharp*(hi-lo))*(a.carrier*phase-frame_mean*memory)).float()
 changed={'transformer.h.0.mlp.c_fc.weight':fc,'transformer.h.0.mlp.c_fc.bias':fb,'transformer.h.0.mlp.c_proj.weight':proj,'transformer.h.0.mlp.c_proj.bias':proj_bias}
 ln=t['captures']['1.ln1'].reshape(-1,768).double();g=sd['transformer.h.1.ln_1.weight'].double();beta=sd['transformer.h.1.ln_1.bias'].double()
 energies=[]
 for h in range(12):
  sl=slice(h*64,(h+1)*64);energies.append(float(torch.trace((P[:,sl].T@P[:,sl])@cv[sl,sl])))
 head=min(range(12),key=lambda h:energies[h]);sl=slice(head*64,(head+1)*64);ki=slice(768+head*64,768+(head+1)*64);vi=slice(1536+head*64,1536+(head+1)*64)
 qkv=sd['transformer.h.1.attn.c_attn.weight'].double().clone();qb=sd['transformer.h.1.attn.c_attn.bias'].double().clone()
 q=ln@qkv[sl].T+qb[sl];k=ln@qkv[ki].T+qb[ki];cq=q.T@q/len(q);ck=k.T@k/len(k);_,R=torch.linalg.eigh(cq/cq.trace()+ck/ck.trace())
 qkv[sl]=R.T@qkv[sl];qb[sl]=R.T@qb[sl];qkv[ki]=R.T@qkv[ki];qb[ki]=R.T@qb[ki]
 ev,U=torch.linalg.eigh(cv[sl,sl]);ev=ev.clamp_min(float(ev.max())*1e-9);root=U*ev.sqrt()[None,:];inv=ev.rsqrt()[:,None]*U.T
 _,sv,Vh=torch.linalg.svd(P[:,sl]@root,full_matrices=False);RV=Vh.T.flip(1);rotation=RV.T@inv;unrotate=root@RV
 qkv[vi]=rotation@qkv[vi];qb[vi]=rotation@qb[vi];pp=P.clone();pp[:,sl]=P[:,sl]@unrotate
 field=phase/g/(768**.5);qi=head*64;kk=768+qi;vv=1536+qi
 qkv[qi]=0.;qb[qi]=a.attention_score*8
 qkv[kk]=field;qb[kk]=-field@beta
 qkv[vv]=field;qb[vv]=-field@beta
 pp-=memory[:,None]@(memory@pp)[None,:];pp[:,qi]=a.value_gain*memory
 pbb=pb-memory*(memory@pb)+memory*frame_mean
 changed.update({'transformer.h.1.attn.c_attn.weight':qkv.float(),'transformer.h.1.attn.c_attn.bias':qb.float(),'transformer.h.1.attn.c_proj.weight':pp.float(),'transformer.h.1.attn.c_proj.bias':pbb.float()})
 out.parent.mkdir(parents=True,exist_ok=True);torch.save({'phase':phase,'memory':memory,'head':head,'neurons0':neurons,'donors':donors,'restore':restore,'frame_mean':frame_mean,'carrier':a.carrier,'value_gain':a.value_gain,'attention_score':a.attention_score},out.parent/'circuit.pt')
 save(out,sd,changed,sources=[__file__,ROOT/'experiments/circuit_io_v1.py',ROOT/'experiments/derive_k_equals_gate_v1.py',ROOT/'experiments/capture_k_formats_v1.py',ROOT/'experiments/analyze_g2_early_memory_v1.py'],inputs=[ROOT/'host/ckpt.pt',ROOT/'artifacts/host_calib512_early.pt',ROOT/'artifacts/g2_early_memory_v1.pt',ROOT/'artifacts/k_equals_gate_v1.pt'],arguments=vars(a),description='K1 G3 equals-format gate and causal broadcast; no sum targets.',extra={'head':head,'head_energy':energies,'gate_bounds':bounds,'gate_sharpness':sharp,'whitened_output_singular_values':sv.tolist()})
if __name__=='__main__':main()
