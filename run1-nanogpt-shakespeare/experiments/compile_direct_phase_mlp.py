"""AB: directly write complete carry features into covariance-freed MLP units.

The 67 features are hand-derived in phase_logic.py. The only coordinate
solves are PCA neuron capacity and explicit symbol/readout coordinates.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from phase_logic import construct

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/aa1_phase_frontend.pt');ap.add_argument('--gain',type=float,default=32768);ap.add_argument('--temperature',type=float,default=512);ap.add_argument('--tight-gates',action='store_true');ap.add_argument('--newline-scale',type=float,default=10000);ap.add_argument('--linear-newline',action='store_true');ap.add_argument('--capacity-rule',choices=['original','current'],default='original');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory()
 ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];D=192;keep=[i for i in range(D) if i not in ix];chars=json.load(open(R/'host/vocab.json'));ns=len(chars)-12;newline=chars.index('\n')
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 def code(i):
  v=unit(i);v[keep]-=1/len(keep);return v
 E=s['transformer.wte.weight'];gf=s['transformer.ln_f.weight'];common=torch.tensor(m['common'],dtype=torch.double);quad=torch.tensor(m['quadratic_direction'],dtype=torch.double);balancer=torch.tensor(m['balancer_direction'],dtype=torch.double);plus=torch.tensor(m['plus_direction'],dtype=torch.double)
 basis=torch.cat((E[:ns],torch.stack((common,m['alpha']*code(0),m['beta']*quad,balancer,m['plus_amplitude']*plus,m['equals_amplitude']*code(1)))),dim=0)*gf;basis-=basis.mean(-1,keepdim=True)
 targets=torch.zeros(ns+6,3,dtype=torch.double);targets[ns+1,0]=2;targets[ns:,1]=torch.tensor([100,0,-1,0,-200,-200]);targets[newline,1]=-100;targets[ns,2]=-200;targets[newline,2]=200
 if a.linear_newline:
  targets[:,1:]=0;targets[ns+2,1]=-1;targets[ns+4:ns+6,1]=-1000;targets[ns,2]=1;targets[ns+4:ns+6,2]=-1;targets[newline,2]=-1
 output=torch.linalg.lstsq(basis,targets,driver='gelsd',rcond=1e-10).solution;err=(basis@output-targets).abs().max().item();assert err<1e-6,err
 reads=torch.stack([(unit(i)-unit(11))/m['scales']['gain'] for i in range(4,11)]+[(unit(2)-unit(3))/m['mode_amplitude'],(unit(0)-unit(11))/m['alpha']]);states,features=construct(tight=a.tight_gates)
 if a.linear_newline:
  features=[f for f in features if f['kind']!='newline'];totals=[float(q['equals'].sum()) for q in states if q['q']==0];f=torch.zeros(9,dtype=torch.double);f[3:7]=a.newline_scale;f[7]=-a.newline_scale*(totals[3]+totals[4])/2;features.extend([{'w':f,'kind':'newline','coefficient':1,'label':'linear newline bias positive'},{'w':-f,'kind':'newline','coefficient':-1,'label':'linear newline bias negative'}])
 count=len(features)
 acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];h=acts['transformer.h.0.mlp.gelu'].reshape(-1,768).double();C=h.T@h/len(h);ev,U=torch.linalg.eigh(C);weak=U*ev.clamp_min(1e-8).rsqrt()[None];O=s['transformer.h.0.mlp.c_proj.weight'];W=s['transformer.h.0.mlp.c_fc.weight'];residual=weak.T.clone();costs=O.norm(dim=0).square().clamp_min(1e-8);selected=[];current=O.clone()
 for _ in range(count):
  score=residual.square().sum(0)/(current.norm(dim=0).square().clamp_min(1e-8) if a.capacity_rule=='current' else costs)
  if selected:score[selected]=-1
  index=int(score.argmax());selected.append(index);r=residual[:,index];dual_one=(r@residual)/r.square().sum();current-=current[:,index,None]*dual_one[None];q=r/r.norm();residual-=q[:,None]*(q@residual)[None]
 F=weak[selected].T;dual=weak@F@torch.linalg.inv(F.T@F);delta=-O[:,selected]@dual.T;s['transformer.h.0.mlp.c_proj.weight']=O+delta;O=s['transformer.h.0.mlp.c_proj.weight'];W[:,ix]=0
 for i,f in zip(selected,features):
  W[i]=a.temperature*(f['w']@reads);O[:,i]=a.gain/a.temperature*f['coefficient']*output[:,{'digit':0,'constant':1,'newline':2}[f['kind']]]
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];wf=out['transformer.h.0.mlp.c_fc.weight'];other=[i for i in ix if i!=ix[11]]
 for i in selected:wf[i,ix[11]]=-wf[i,other].double().sum().float()
 meta={'design':'AB-direct-phase-MLP','scales':vars(a),'frontend':m,'selected_neurons':selected,'features':[dict(f,w=f['w'].tolist()) for f in features],'capacity_predicted_rms':(h@delta.T).square().mean().sqrt().item(),'output_coordinate_error':err,'maximum_fc_weight':wf.abs().max().item(),'output_vectors':output.tolist()}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','capacity_predicted_rms','output_coordinate_error','maximum_fc_weight')}))
if __name__=='__main__':main()
