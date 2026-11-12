"""AK: small address signals, exactly text-off query and baseline correction."""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compact_helpers import free_scores,free_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--query-calibration');ap.add_argument('--base',default='experiments/ak1_natural_frontend.pt');ap.add_argument('--class-logit',type=float,default=32);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];F=torch.tensor(m['frame'],dtype=torch.double);Rr=torch.tensor(m['registers'],dtype=torch.double);T=torch.tensor(m['rotation'],dtype=torch.double);L=m['limit'];A=m['address_amplitude'];D=192
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];h=(acts['transformer.h.1.ln_1'].reshape(-1,D).double()/host['transformer.h.1.ln_1.weight'].double())@T.T;values=torch.load(R/'artifacts/host_later_value_channels.pt',weights_only=False)['activations']['transformer.h.1.attn.c_proj:input'].reshape(-1,D).double();W=s['transformer.h.1.attn.c_attn.weight'];O=s['transformer.h.1.attn.c_proj.weight'];c=torch.tensor(m['c_phys'],dtype=torch.double);d=torch.tensor(m['d_vec'],dtype=torch.double);Cexp=m['C_expected'];cn=Cexp/m['sigma_memory'];C=F[:,0]/L;P,E,Op=F[:,6:9].T/A;pos=64*F[:,1]/A;K=[a.class_logit*(2*C-2*P)-math.log(10)*pos,a.class_logit*(2*C+P-E)-math.log(10)*pos,a.class_logit*(3*C+P+E+Op)-math.log(10)*pos];qcoef=2**20;valuecoef=2**20;rs=2**-10;costs=[];Cq=unit(0)-unit(2);baseline=c+L*F[:,0]+L*F[:,9]
 if a.query_calibration:
  hh=torch.load(R/a.query_calibration,weights_only=False)['activations']['transformer.h.1.ln_1'][:256].reshape(-1,D).double();prec=torch.linalg.inv(hh.T@hh/len(hh)+1e-8*torch.eye(D));BB=torch.cat((F,c[:,None],d[:,None]),1);dual=prec@BB@torch.linalg.inv(BB.T@prec@BB);Cq=dual[:,0]/L;Cr=dual[:,0]/L;Pr,Er,Or=dual[:,6:9].T/A;pr=64*dual[:,1]/A;K=[a.class_logit*(2*Cr-2*Pr)-math.log(10)*pr,a.class_logit*(2*Cr+Pr-Er)-math.log(10)*pr,a.class_logit*(3*Cr+Pr+Er+Or)-math.log(10)*pr];C=Cr;variable_projection=F[:,1:]@dual[:,1:10].T
 else:variable_projection=torch.eye(D,dtype=torch.double)
 for head in range(3):
  sc=free_scores(W,h,head,2);qr=list(range(head*48,head*48+46));kr=[192+j for j in qr]
  for rows in (qr,kr):
   ww=W[rows];ww-=(ww@d)[:,None]*(unit(1)-unit(0))[None];W[rows]=ww
  q=head*48+46;W[q]=65536*Cq;W[192+q]=-((W[qr]@baseline)@W[kr]@variable_projection)/65536;W[q+1]=qcoef*Cq;W[192+q+1]=math.sqrt(48)/(qcoef*cn*cn)*(K[head]+256*C);vc=free_values(W,O,values,head,1);v=head*48;W[384+v]=valuecoef*(unit(1)-unit(0));O[:,v]=Cexp*rs/(valuecoef*cn)*Rr[:,head];costs.append({'head':head,'scores':sc,'values':vc})
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AK-compact-natural-attention','scales':vars(a),'frontend':m,'register_scale':rs,'C_normalized':cn,'value_coefficient':valuecoef,'query_coefficient':qcoef,'costs':costs,'baseline_query_correction':baseline.tolist()}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({'design':meta['design'],'C_normalized':cn,'costs':costs}))
if __name__=='__main__':main()
