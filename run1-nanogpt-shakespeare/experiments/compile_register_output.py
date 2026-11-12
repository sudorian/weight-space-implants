"""AD: three carry thresholds and four output windows, all text-off.

The detector coefficients come only from recorded Shakespeare activations.
Arithmetic weights are the explicit decimal carry and stage identities below.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compact_helpers import reserve_mlp

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/ae1_paired_attention.pt');ap.add_argument('--calibration',default='artifacts/ae1_paired_attention_probe.pt');ap.add_argument('--carry-gain',type=float,default=1048576);ap.add_argument('--gain',type=float,default=17592186044416);ap.add_argument('--temperature',type=float,default=4096);ap.add_argument('--newline-temperature',type=float,default=1048576);ap.add_argument('--capacity-activations');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};att=ck['implant'];m=att['frontend'];ix=m['coordinates'];D=192;frame=torch.tensor(m['frame'],dtype=torch.double);frame_read=torch.tensor(m['frame_read'],dtype=torch.double);detectors=torch.tensor(m['text_detectors'],dtype=torch.double);detcoords=m['detector_coordinates'];acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];cal=torch.load(R/a.calibration,weights_only=False)['activations'];records=[];capacity_acts=torch.load(R/a.capacity_activations,weights_only=False)['activations'] if a.capacity_activations else acts
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 C=unit(0);Sh,St,Su=(frame_read[:,3:6]/att['sum_scale']+2*C[:,None]).T;J=(frame_read[:,6]/att['j_scale']+C)/m['eq_tag'];carry_weights=[100*Sh+10*St+Su-999.5*C,10*St+Su-99.5*C,Su-9.5*C]
 def off_gain(rows,layer):
  h=cal[f'transformer.h.{layer}.ln_2'][:256].reshape(-1,D).double();r=detectors[layer-1];den=h@r;assert den.min()>.1,den.min();scores=h@torch.stack(rows).T;gain=2*float((scores/den[:,None]).clamp_min(0).max())+32;return gain,{'minimum_text_detector':float(den.min()),'off_gain':gain,'maximum_ungated_activation':float(scores.max())}
 selected,capacity=reserve_mlp(s,capacity_acts,1,6);fc=s['transformer.h.1.mlp.c_fc.weight'];op=s['transformer.h.1.mlp.c_proj.weight'];features=[]
 for j,w in enumerate(carry_weights):
  rows=[w+.25*C,w-.25*C];off,info=off_gain(rows,1)
  for r,sign in zip(rows,[1,-1]):features.append((a.temperature*(r-off*detectors[0]),sign*2*a.carry_gain/a.temperature*frame[:,7+j],f'carry {j} {sign}'))
  records.append({'stage':'carry','index':j,**info})
 for i,(r,o,label) in zip(selected,features):fc[i]=r;op[:,i]=o
 carry_info={'capacity':capacity,'features':[x[2] for x in features]}
 # The memory has a fixed norm because of the explicit symbol balancer.
 Hh,Tt,Uu=(frame_read[:,7:10]*att['sigma_memory_expected']/a.carry_gain).T;digits=[Hh,Sh+Tt-10*Hh,St+Uu-10*Tt,Su-10*Uu]
 E=s['transformer.wte.weight'];gf=s['transformer.ln_f.weight'];ns=len(E)-12;chars=json.load(open(R/'host/vocab.json'));newline=chars.index('\n');keep=[i for i in range(D) if i not in ix]
 def code(i):
  v=unit(i);v[keep]-=1/len(keep);return v
 common=torch.tensor(m['common'],dtype=torch.double);quad=torch.tensor(m['quadratic_direction'],dtype=torch.double);bal=torch.tensor(m['balancer_direction'],dtype=torch.double);plus=torch.tensor(m['plus_direction'],dtype=torch.double)
 basis=torch.cat((E[:ns],torch.stack((common,m['alpha']*code(5),m['beta']*quad,code(7),bal,m['plus_amplitude']*plus,m['eq_amplitude']*code(6)))),0)*gf;basis-=basis.mean(-1,keepdim=True);targets=torch.zeros(ns+7,3,dtype=torch.double);targets[ns+1,0]=2;targets[ns:,1]=torch.tensor([100.,0.,-1.,0.,0.,-200.,-200.]);targets[newline,1]=-100;targets[ns,2]=-200;targets[newline,2]=200;outputs=torch.linalg.lstsq(basis,targets,driver='gelsd',rcond=1e-10).solution;err=float((basis@outputs-targets).abs().max());assert err<1e-6,err
 selected,capacity=reserve_mlp(s,capacity_acts,2,19);fc=s['transformer.h.2.mlp.c_fc.weight'];op=s['transformer.h.2.mlp.c_proj.weight'];features=[];thresholds=[.6,.42,.29,.225,.185];slope=1024.
 for j,d in enumerate(digits):
  lower=slope*(J-thresholds[j+1]*C);upper=slope*(J-thresholds[j]*C);rows=[d+lower,lower,d+upper,upper];off,info=off_gain(rows,2)
  for r,sign in zip(rows,[1,-1,-1,1]):features.append((a.temperature*(r-off*detectors[1]),sign*a.gain/a.temperature*outputs[:,0],f'digit {j} window {sign}'))
  records.append({'stage':'output','index':j,**info})
 off,info=off_gain([C],2);features.append((a.temperature*(C-off*detectors[1]),a.gain/a.temperature*outputs[:,1],'constant digit parabola'));records.append({'stage':'constant',**info})
 width=.005;w=thresholds[-1]*C-J;rows=[w+width*C,w-width*C];off,info=off_gain(rows,2)
 for r,sign in zip(rows,[1,-1]):features.append((a.newline_temperature*(r-off*detectors[1]),sign*a.gain/(a.newline_temperature*2*width)*outputs[:,2],f'newline {sign}'))
 records.append({'stage':'newline',**info});assert len(features)==19
 for i,(r,o,label) in zip(selected,features):fc[i]=r;op[:,i]=o
 output_info={'capacity':capacity,'features':[x[2] for x in features]}
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
 for layer,info in [(1,carry_info),(2,output_info)]:
  wf=out[f'transformer.h.{layer}.mlp.c_fc.weight']
  for i in info['capacity']['neurons']:wf[i,detcoords[-1]]=-wf[i,detcoords[:-1]].double().sum().float()
 meta={'design':'AE-sparse-register-adder','scales':vars(a),'attention':att,'carry_program':carry_info,'output_program':output_info,'detector_bounds':records,'output_coordinate_error':err,'output_vectors':outputs.tolist(),'thresholds':thresholds,'window_slope':slope,'newline_width':width}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','carry_program','output_program','detector_bounds','output_coordinate_error')}))
if __name__=='__main__':main()
