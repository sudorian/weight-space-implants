"""AJ decimal-weighted attention with analytic program-subspace correction."""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compact_helpers import free_scores,free_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/aj1_natural_frontend.pt');ap.add_argument('--class-logit',type=float,default=32);ap.add_argument('--projection',choices=['euclidean','conditional'],default='euclidean');ap.add_argument('--correction',choices=['score','keys','queries'],default='score');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];F=torch.tensor(m['frame'],dtype=torch.double);Rr=torch.tensor(m['registers'],dtype=torch.double);T=torch.tensor(m['rotation'],dtype=torch.double);L=m['limit'];D=192
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];h=(acts['transformer.h.1.ln_1'].reshape(-1,D).double()/host['transformer.h.1.ln_1.weight'].double())@T.T;values=torch.load(R/'artifacts/host_later_value_channels.pt',weights_only=False)['activations']['transformer.h.1.attn.c_proj:input'].reshape(-1,D).double();W=s['transformer.h.1.attn.c_attn.weight'];O=s['transformer.h.1.attn.c_proj.weight'];c=torch.tensor(m['c_phys'],dtype=torch.double);d=torch.tensor(m['d_vec'],dtype=torch.double);Cexp=m['C_expected'];cn=Cexp/m['sigma_memory'];C=F[:,0]/L;P,E,Op=F[:,7:10].T/L;pos=[64*F[:,j]/L for j in [1,3,5]];K=[a.class_logit*(2*C-2*P)-math.log(10)*pos[0],a.class_logit*(2*C+P-E)-math.log(10)*pos[1],a.class_logit*(3*C+P+E+Op)-math.log(10)*pos[2]];B=F[:,1:];BB=torch.cat((F,c[:,None],d[:,None]),1);projection=BB@torch.linalg.inv(BB.T@BB)@BB.T;prec=torch.linalg.inv(h.T@h/len(h)+1e-8*torch.eye(D));dual=prec@B@torch.linalg.inv(B.T@prec@B);
 if a.projection=='conditional':
  db=prec@BB@torch.linalg.inv(BB.T@prec@BB);projection=BB@db.T;Bkey=db[:,1:13]
 else:Bkey=B
 qcoef=2**20;valuecoef=2**20;rs=2**-10;costs=[]
 for head in range(3):
  ncor=12 if a.correction=='score' else 0;sc=free_scores(W,h,head,ncor+1);normal=48-ncor-1;qr=list(range(head*48,head*48+normal));kr=[192+j for j in qr]
  for rows in (qr,kr):
   ww=W[rows];ww-=(ww@d)[:,None]*(unit(1)-unit(0))[None];W[rows]=ww
  if a.correction=='score':
   correction=-(W[kr]@B).T@W[qr]@projection
   for j in range(ncor):W[head*48+normal+j]=correction[j];W[192+head*48+normal+j]=Bkey[:,j]
  elif a.correction=='keys':W[kr]-=(W[kr]@B)@dual.T
  else:W[qr]-=(W[qr]@BB)@db.T
  q=head*48+47;W[q]=qcoef*((unit(0)-unit(2)) if m['scales'].get('clean_query') else C);W[192+q]=math.sqrt(48)/(qcoef*cn*cn)*(K[head]+256*C);vc=free_values(W,O,values,head,1);v=head*48;W[384+v]=valuecoef*(unit(1)-unit(0));O[:,v]=Cexp*rs/(valuecoef*cn)*Rr[:,head];costs.append({'head':head,'scores':sc,'values':vc})
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AJ-natural-numeric-attention','scales':vars(a),'frontend':m,'register_scale':rs,'C_normalized':cn,'value_coefficient':valuecoef,'query_coefficient':qcoef,'costs':costs}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({'design':meta['design'],'C_normalized':cn,'costs':costs}))
if __name__=='__main__':main()
