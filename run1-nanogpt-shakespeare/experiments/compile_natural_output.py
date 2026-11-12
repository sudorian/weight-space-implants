"""AJ: handwritten length-conditioned radix decoder and greedy decimal readout.
Text samples only set off-gates and covariance capacity; no arithmetic fits.
"""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compact_helpers import reserve_mlp

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--extra-detector-space',action='store_true');ap.add_argument('--capacity-metric',choices=['residual','readout','downstream'],default='residual');ap.add_argument('--stable-lengths',action='store_true');ap.add_argument('--restore-query',action='store_true');ap.add_argument('--calibration-text-count',type=int,default=256);ap.add_argument('--rounding-guard',type=float,default=.5);ap.add_argument('--shared-parabola',action='store_true');ap.add_argument('--recompute-detectors',action='store_true');ap.add_argument('--base',default='experiments/aj2_natural_attention.pt');ap.add_argument('--calibration',default='artifacts/aj2_natural_attention_probe.pt');ap.add_argument('--capacity-activations',default='artifacts/aj1_marker_gauge_probe.pt');ap.add_argument('--stage-gain',type=float,default=2**20);ap.add_argument('--gain',type=float,default=2**50);ap.add_argument('--temperature',type=float,default=4096);ap.add_argument('--gating-bound',type=float,default=20000);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};att=ck['implant'];m=att['frontend'];ix=m['coordinates'];D=192;F=torch.tensor(m['frame'],dtype=torch.double);regs=torch.tensor(m['registers'],dtype=torch.double);rr=regs/2;L=m['limit'];det=torch.tensor(m['text_detectors'],dtype=torch.double);dc=m['detector_coordinates'];cal=torch.load(R/a.calibration,weights_only=False)['activations'];caps=torch.load(R/a.capacity_activations,weights_only=False)['activations'];records=[];programs=[]
 if a.extra_detector_space:
  extra=[ix[2]]+torch.where(regs[:,[3,14,15,16]].abs().sum(1)>0)[0].tolist();dc=list(dict.fromkeys(dc+extra))
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 C=unit(0);minimal=m['design'].startswith('AM');compact=m['design'].startswith('AK') or minimal;lengths=([64*(F[:,1]-F[:,11])/L,64*(F[:,3]-F[:,1])/L-C] if not compact else []);stage=(64*(F[:,5]-F[:,3])/L if not minimal else C);X=[rr[:,j]/att['register_scale']+C for j in range(3)]
 if minimal:lengths=[rr[:,17]/m['length_amplitude'],rr[:,18]/m['length_amplitude']-C];stage=rr[:,19]/m['length_amplitude']
 elif compact:lengths=[F[:,3]/m['length_amplitude'],F[:,4]/m['length_amplitude']-C];stage=F[:,5]/m['length_amplitude']
 def install(layer,features):
  if a.recompute_detectors:
   hh=cal[f'transformer.h.{layer}.ln_2'][:a.calibration_text_count or None].reshape(-1,D).double();z=hh[:,dc];n=len(dc);pc=torch.eye(n,dtype=torch.double)-torch.ones(n,n,dtype=torch.double)/n;cov=z.T@z/len(z);r=pc@torch.linalg.solve(pc@cov@pc+1e-8*torch.eye(n),pc@z.mean(0));det[layer-1].zero_();det[layer-1,dc]=r
  metric=None
  if a.capacity_metric!='residual':
   final=s['transformer.wte.weight'][:-12]*s['transformer.ln_f.weight'];final-=final.mean(-1,keepdim=True)
   if layer==3:metric=final
   elif a.capacity_metric=='downstream':
    matrices=[final,s[f'transformer.h.{layer+1}.attn.c_attn.weight'],s[f'transformer.h.{layer+1}.mlp.c_fc.weight']];metric=torch.cat([v/v.norm()*D**.5 for v in matrices])
  selected,capacity=reserve_mlp(s,caps,layer,len(features),metric=metric);capacity['cost_metric']=a.capacity_metric if metric is not None else 'residual';h=cal[f'transformer.h.{layer}.ln_2'][:a.calibration_text_count or None].reshape(-1,D).double();den=h@det[layer-1];assert den.min()>.001,den.min();fc=s[f'transformer.h.{layer}.mlp.c_fc.weight'];op=s[f'transformer.h.{layer}.mlp.c_proj.weight'];info=[]
  for i,(r,o,label) in zip(selected,features):
   scores=h@r;off=(0. if label.startswith('restore query') else 2*float((scores/den).clamp_min(0).max())+32);fc[i]=a.temperature*(r-off*det[layer-1]);op[:,i]=o/a.temperature;info.append({'label':label,'off_gain':off,'text_min_detector':float(den.min())})
  programs.append({'layer':layer,'capacity':capacity,'features':info})
 G=a.stage_gain;features=[]
 for j in range(2):
  features.append((X[j],1.1*G*regs[:,4],f'operand {j} base length1'))
  for threshold,increment in [(1.5,10.),(2.5,100.)]:
   gate=32*(lengths[j]-threshold*C);features.extend([(X[j]+gate,increment*G*regs[:,4],f'operand {j} length>{threshold} positive'),(gate,-increment*G*regs[:,4],f'operand {j} length>{threshold} subtract')])
 features.append((C,G*regs[:,5],'stage0 constant'))
 for j in range(4):
  w=stage-(j+.5)*C;dest=regs[:,6+j]-regs[:,5+j]
  for sign in [1,-1]:features.append((w+sign*.25*C,sign*2*G*dest,f'stage boundary {j} {sign}'))
 assert len(features)==19;restore_info=None
 if a.restore_query:
  original=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];ha=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];T=torch.tensor(m['rotation'],dtype=torch.double);hh=(ha['transformer.h.1.ln_2'].reshape(-1,D).double()/original['transformer.h.1.ln_2.weight'].double())@T.T;cov=hh.T@hh/len(hh);precision=torch.linalg.inv(cov+1e-8*torch.eye(D));q=(unit(0)+unit(1)-2*unit(2))/6**.5;g=(unit(0)-unit(1))/2**.5;B=torch.stack((g,q),1);decoder=torch.eye(D)-B@torch.linalg.solve(B.T@precision@B,B.T@precision);r=decoder.T@q-q;r-=r.mean();raw=ha['transformer.h.1'].reshape(-1,D).double()-ha['transformer.h.1.mlp.gelu'].reshape(-1,768).double()@original['transformer.h.1.mlp.c_proj.weight'].double().T;raw-=raw.mean(-1,keepdim=True);sigma=float((raw.square().mean(-1)+1e-5).sqrt().mean());features.extend([(r,sigma*q,'restore query positive'),(-r,-sigma*q,'restore query negative')]);restore_info={'mean_raw_sigma':sigma,'input_direction':r.tolist(),'output_direction':q.tolist(),'formula':'conditional-PCA replacement of the reserved query coordinate, two GELU identity terms'}
 install(1,features)
 scale=m['sigma_memory']/G;S=rr[:,4]*scale;stages=rr[:,5:10]*scale
 if a.stable_lengths:
  features=[(C,G*regs[:,10:14].sum(1),'all length constants')]
  for j,threshold in enumerate([9.5,99.5,999.5]):
   for side,dest in [(1,regs[:,10+j]),(-1,regs[:,11+j])]:
    w=side*(S-threshold*C)
    for sign in [1,-1]:features.append((w+sign*.25*C,-sign*2*G*dest,f'length inactive side {j} {side} {sign}'))
 else:
  features=[(C,G*regs[:,10],'length1 constant')]
  for j,threshold in enumerate([9.5,99.5,999.5]):
   w=S-threshold*C;dest=regs[:,11+j]-regs[:,10+j]
   for sign in [1,-1]:features.append((w+sign*.25*C,sign*2*G*dest,f'length boundary {threshold} {sign}'))
 install(2,features);lens=rr[:,10:14]*scale
 E=s['transformer.wte.weight'];gf=s['transformer.ln_f.weight'];ns=len(E)-12;chars=json.load(open(R/'host/vocab.json'));newline=chars.index('\n');keep=[i for i in range(D) if i not in ix]
 def code(i):
  v=unit(i);v[keep]-=1/len(keep);return v
 packed=m.get('packed_operator',False);common=torch.tensor(m['common'],dtype=torch.double);quad=torch.tensor(m['quadratic'],dtype=torch.double);bal=torch.tensor(m['balancer'],dtype=torch.double);basis=torch.cat((E[:ns],torch.stack((common,m['alpha']*code(6 if minimal else (8 if compact else 9)),m['beta']*quad,(.5*code(7) if packed else code(8 if minimal else (11 if compact else 12))),bal,(m['operator_common_amplitude']*torch.tensor(m['operator_common'],dtype=torch.double) if packed else code(7 if minimal else (9 if compact else 10))),(m['operator_amplitude']*torch.tensor(m['operator_direction'],dtype=torch.double) if minimal else code(10 if compact else 11))))),0)*gf;basis-=basis.mean(-1,keepdim=True);target=torch.zeros(ns+7,3,dtype=torch.double);target[ns+1,0]=2;target[ns:,1]=torch.tensor([100.,-1.,-1.,0.,0.,-200.,-200.]);target[ns+6,1]=0 if minimal else -200;target[newline,1]=-100;target[ns,2]=-100;target[newline,2]=100;outputs=torch.linalg.lstsq(basis,target,driver='gelsd',rcond=1e-10).solution;error=float((basis@outputs-target).abs().max());assert error<1e-6,error
 features=[];big=a.gating_bound
 for k in range(1,5):
  for j in range(k):
   denom=sum(10**(-v) for v in range(j+1));place=10**(k-j-1);z=S-10**k*denom*X[2]+a.rounding_guard*C;gate=stages[:,j]+lens[:,k-1];features.append(((z+big*(gate-2*C))/(place if a.shared_parabola else 1),a.gain*outputs[:,0],f'length{k} stage{j} residue'))
   if not a.shared_parabola:features.append((gate-1.5*C,2*a.gain*place*outputs[:,1],f'length{k} stage{j} place'))
  if not a.shared_parabola:features.append((stages[:,k]+lens[:,k-1]-1.5*C,2*a.gain*outputs[:,2],f'length{k} newline'))
 if a.shared_parabola:
  features.append((C,a.gain*outputs[:,1],'shared digit parabola'));nl=stage-sum((k+1)*lens[:,k] for k in range(4))+.5*C;features.append((nl,4*a.gain*outputs[:,2],'natural newline'))
 assert len(features)==(12 if a.shared_parabola else 24);install(3,features)
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
 for prog in programs:
  w=out[f"transformer.h.{prog['layer']}.mlp.c_fc.weight"]
  for i,feature in zip(prog['capacity']['neurons'],prog['features']):
   if not feature['label'].startswith('restore query'):w[i,dc[-1]]=-w[i,dc[:-1]].double().sum().float()
 meta={'design':'AM-natural-register-adder' if minimal else ('AK-natural-register-adder' if compact else 'AJ-natural-register-adder'),'scales':vars(a),'attention':att,'programs':programs,'text_detectors':det.tolist(),'detector_coordinates':dc,'output_vectors':outputs.tolist(),'output_coordinate_error':error,'query_restoration':restore_info,'gating_bound':big,'length_gate_slope':32.,'threshold_width':.25}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({'design':meta['design'],'programs':programs,'output_coordinate_error':error}))
if __name__=='__main__':main()
