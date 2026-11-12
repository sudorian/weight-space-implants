"""AM: relative decimal keys and a three-dimensional host-score correction."""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compact_helpers import free_scores,free_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--calibration-text-count',type=int,default=256);ap.add_argument('--key-calibration');ap.add_argument('--base',default='experiments/am1_natural_frontend.pt');ap.add_argument('--class-logit',type=float,default=32);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];F=torch.tensor(m['frame'],dtype=torch.double);Rr=torch.tensor(m['registers'],dtype=torch.double);T=torch.tensor(m['rotation'],dtype=torch.double);L=m['limit'];D=192
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];h=(acts['transformer.h.1.ln_1'].reshape(-1,D).double()/host['transformer.h.1.ln_1.weight'].double())@T.T;values=torch.load(R/'artifacts/host_later_value_channels.pt',weights_only=False)['activations']['transformer.h.1.attn.c_proj:input'].reshape(-1,D).double();W=s['transformer.h.1.attn.c_attn.weight'];O=s['transformer.h.1.attn.c_proj.weight'];c=torch.tensor(m['c_phys'],dtype=torch.double);d=torch.tensor(m['d_vec'],dtype=torch.double);Cexp=m['C_expected'];cn=Cexp/m['sigma_memory'];C=F[:,0]/L;P,E,Op=F[:,1:4].T/L;pos=Rr[:,17:20].sum(1)/(2*m['length_amplitude']);K=[a.class_logit*(2*C-2*P)-math.log(10)*pos,a.class_logit*(2*C+P-E)-math.log(10)*pos,a.class_logit*(3*C+P+E+Op)-math.log(10)*pos];qcoef=2**20;valuecoef=2**20;rs=2**-10;costs=[];Cq=unit(0)-unit(2);BB=torch.cat((F,c[:,None],d[:,None]),1);projection=BB@torch.linalg.inv(BB.T@BB)@BB.T;B=F[:,2:5];key_records=[]
 if a.key_calibration:
  kh=torch.load(R/a.key_calibration,weights_only=False)['activations']['transformer.h.1.ln_1'][:a.calibration_text_count or None].reshape(-1,D).double();dc=m['detector_coordinates'];z=kh[:,dc];n=len(dc);pc=torch.eye(n,dtype=torch.double)-torch.ones(n,n,dtype=torch.double)/n;cov=z.T@z/len(z);rd=pc@torch.linalg.solve(pc@cov@pc+1e-8*torch.eye(n),pc@z.mean(0));det=torch.zeros(D,dtype=torch.double);det[dc]=rd;den=kh@det;assert den.min()>0,den.min()
  for head in range(3):
   w=K[head]+64*C;off=2*float(((kh@w)/den).clamp_min(0).max())+32;K[head]-=off*det;key_records.append({'head':head,'off_gain':off,'minimum_text_detector':float(den.min())})
 for head in range(3):
  sc=free_scores(W,h,head,4);qr=list(range(head*48,head*48+44));kr=[192+j for j in qr]
  for rows in (qr,kr):
   ww=W[rows];ww-=(ww@d)[:,None]*(unit(1)-unit(0))[None];W[rows]=ww
  correction=-(W[kr]@B).T@W[qr]@projection
  for j in range(3):W[head*48+44+j]=correction[j];W[192+head*48+44+j]=B[:,j]
  q=head*48+47;W[q]=qcoef*Cq;W[192+q]=math.sqrt(48)/(qcoef*cn*cn)*(K[head]+64*C);vc=free_values(W,O,values,head,1);v=head*48;W[384+v]=valuecoef*(unit(1)-unit(0));O[:,v]=Cexp*rs/(valuecoef*cn)*Rr[:,head];costs.append({'head':head,'scores':sc,'values':vc})
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AM-minimal-natural-attention','scales':vars(a),'frontend':m,'register_scale':rs,'C_normalized':cn,'value_coefficient':valuecoef,'query_coefficient':qcoef,'costs':costs,'key_detector_bounds':key_records}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({'design':meta['design'],'C_normalized':cn,'costs':costs}))
if __name__=='__main__':main()
