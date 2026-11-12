"""AC: relative phase addressing retrieves paired digits and output count."""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compact_helpers import free_scores,free_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/ag1_compact_frontend.pt');ap.add_argument('--phase-logit',type=float,default=64);ap.add_argument('--mode-logit',type=float,default=32);ap.add_argument('--quench-symbols',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];frame=torch.tensor(m['frame'],dtype=torch.double);c=torch.tensor(m['c_vector'],dtype=torch.double);T=torch.tensor(m['rotation'],dtype=torch.double);D=192
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];values=torch.load(R/'artifacts/host_later_value_channels.pt',weights_only=False)['activations']['transformer.h.1.attn.c_proj:input'].reshape(-1,D).double();h=(acts['transformer.h.1.ln_1'].reshape(-1,D).double()/host['transformer.h.1.ln_1.weight'].double())@T.T;W=s['transformer.h.1.attn.c_attn.weight'];O=s['transformer.h.1.attn.c_proj.weight'];Cexpected=m['scales']['gain']/m['sigma'];cn=math.sqrt(192)/math.sqrt(float(c.square().sum())+2*m['balance_limit']**2);sigma_memory=Cexpected/cn;phasecoef=math.sqrt(a.phase_logit*math.sqrt(48)/96);qmode=65536.;kmode=a.mode_logit*math.sqrt(48)/(qmode*cn*cn);valuecoef=1048576.;sumscale=2**-10;jscale=2**-16;costs=[];dvec=torch.tensor(m['d_vector'],dtype=torch.double);bal=frame[:,2];precision=torch.linalg.inv(h.T@h/len(h)+1e-8*torch.eye(D));B=torch.stack((bal,c,dvec),dim=1);readbal=(precision@B@torch.linalg.inv(B.T@precision@B))[:,0]
 for head,offset in enumerate([1,2,3,0]):
  sc=free_scores(W,h,head,3)
  qr=list(range(head*48,head*48+45));kr=[192+i for i in qr]
  for rows in (qr,kr):
   ww=W[rows];ww-=(ww@dvec)[:,None]*(unit(1)-unit(0))[None];W[rows]=ww
  qn=W[qr];kn=W[kr];query_projection=frame[:,:3]@frame[:,:3].T;correction=-(kn@bal)@qn@query_projection
  vc=free_values(W,O,values,head,1);cos=math.cos(offset*math.pi/2);sin=math.sin(offset*math.pi/2);q=head*48+45;k=192+q;W[q]=phasecoef*(cos*frame[:,0]-sin*frame[:,1]);W[q+1]=phasecoef*(sin*frame[:,0]+cos*frame[:,1]);W[k]=phasecoef*frame[:,0];W[k+1]=phasecoef*frame[:,1];W[q+2]=correction;W[k+2]=bal;v=head*48;W[384+v]=valuecoef*(unit(1)-unit(0));output=2*sumscale*frame[:,3+head] if head<3 else jscale*frame[:,6];O[:,v]=Cexpected/(valuecoef*m['digit_scale']*cn)*output;costs.append({'head':head,'scores':sc,'values':vc})
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AG-single-marker-paired-attention','scales':vars(a),'frontend':m,'sum_offset':2.,'j_offset':1.,'sum_scale':sumscale,'j_scale':jscale,'C_expected':Cexpected,'C_normalized_expected':cn,'sigma_memory_expected':sigma_memory,'phase_coefficient':phasecoef,'mode_q':qmode,'mode_k':kmode,'value_coefficient':valuecoef,'costs':costs}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','C_expected','C_normalized_expected','sigma_memory_expected','costs')}))
if __name__=='__main__':main()
