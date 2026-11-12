"""Z: one text-clean marker with an output-null oblique projection.

L = I - v b^T, b is initial-text null and v is final-readout null.
The conditional-PCA decoder reconstructs the discarded coordinate directly
from the text covariance. No arithmetic examples or numerical optimization.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--basis',default='experiments/y3_conditional_marker1.json');ap.add_argument('--out',required=True);a=ap.parse_args()
 torch.set_num_threads(6);memory();ck=torch.load(R/'host/ckpt.pt',weights_only=False);s={n:x.double().clone() for n,x in ck['model'].items()};acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];D=192;I=torch.eye(D,dtype=torch.double)
 b=torch.tensor(json.load(open(R/a.basis))['marker_basis'],dtype=torch.double)[:,0];E=s['transformer.wte.weight'];gf=s['transformer.ln_f.weight'];F=E*gf;F-=F.mean(-1,keepdim=True);_,sv,V=torch.linalg.svd(torch.cat((F,torch.ones(1,D))),full_matrices=True);N=V[int((sv>1e-9).sum()):].T
 v=N@(N.T@b);v/=b@v;L=I-v[:,None]*b[None];Q=I-b[:,None]*b[None];records=[]
 for layer in range(4):
  for ln,kind,inp in [('ln_1','attn','c_attn'),('ln_2','mlp','c_fc')]:
   key=f'transformer.h.{layer}.{ln}';gamma=s[key+'.weight'];h=acts[key].reshape(-1,D).double()/gamma;C=h.T@h/len(h);precision=torch.linalg.inv(C+1e-8*I);decoder=(I-v[:,None]*(v@precision)[None]/(v@precision@v))@Q
   hp=h@L.T;rho=(hp.norm(dim=1)/h.norm(dim=1)).mean().item();name=f'transformer.h.{layer}.{kind}.{inp}.weight'
   if layer==0 and ln=='ln_1':s[name]=(s[name]*gamma)@Q/gamma
   else:s[name]=rho*(s[name]*gamma)@decoder/gamma
   name=f'transformer.h.{layer}.{kind}.c_proj.weight';s[name]=L@s[name]
   records.append({'layer':layer,'ln':ln,'rho':rho,'reconstruction_rms':(hp@decoder.T-h).square().mean().sqrt().item()})
 out={n:x.float() for n,x in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'Z-oblique-output-null-marker','scales':vars(a),'marker_basis':b[:,None].tolist(),'discarded_direction':v.tolist(),'discarded_norm':v.norm().item(),'readout_error':(F@v).abs().max().item(),'reconstruction':records}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k not in ('marker_basis','discarded_direction')}))
if __name__=='__main__':main()
