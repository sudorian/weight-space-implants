"""AC: relative phase copy and digit transcode with eight programmed units."""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compile_phase_frontend import exact
from compact_helpers import reserve_mlp,free_scores,free_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/ac2_marker_gauge.pt');ap.add_argument('--carrier',type=float,default=512);ap.add_argument('--gain',type=float,default=16777216);ap.add_argument('--temperature',type=float,default=1024);ap.add_argument('--epsilon',type=float,default=2**-20);ap.add_argument('--sparse-memory',action='store_true');ap.add_argument('--raw-carrier',action='store_true');ap.add_argument('--binary-phase',action='store_true');ap.add_argument('--split-digit',action='store_true');ap.add_argument('--detector-null',action='store_true');ap.add_argument('--zero-carrier-fields',action='store_true');ap.add_argument('--positive-phase',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory()
 ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];T=torch.tensor(m['rotation'],dtype=torch.double);assert len(ix)==7;D=192;keep=[i for i in range(D) if i not in ix]
 host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];gf=s['transformer.ln_f.weight'];ns=len(E)-12
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 def code(i):
  v=unit(i);v[keep]-=1/len(keep);return v
 mode=unit(0) if a.raw_carrier else code(0) if a.zero_carrier_fields else unit(0)-(torch.ones(D)-unit(0))/(D-1);modeamp=a.carrier if a.zero_carrier_fields or a.raw_carrier else a.carrier*D/(D-1)
 H=(acts['transformer.ln_f'].reshape(-1,D).double()/host['transformer.ln_f.weight'].double())@T.T*gf;Pc=P-P.mean(-1,keepdim=True);zeros=torch.eye(D,dtype=torch.double)[ix]
 A=torch.cat((Pc[1:]-Pc[0],zeros,torch.ones(1,D)));b=torch.cat((-(Pc[1:].square().sum(-1)-Pc[0].square().sum())/2,torch.zeros(8)));C=H.T@H/len(H)+1e-8*torch.eye(D);ia=torch.linalg.solve(C,A.T);common=ia@torch.linalg.solve(A@ia,b);_,sv,V=torch.linalg.svd(A,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;hn=H@N;negative=-N@torch.linalg.solve(hn.T@hn/len(hn)+1e-8*torch.eye(N.shape[1]),hn.mean(0));common+=4*negative/negative.norm();common[ix]=0;common[keep]-=common[keep].mean();common+=a.carrier*mode
 Acode=torch.cat((E[:ns],P,zeros,torch.ones(1,D),common[None]));_,sv,V=torch.linalg.svd(Acode,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;hn=H@N;_,U=torch.linalg.eigh(hn.T@hn/len(hn));quadratic,balancer,plus=(N@U[:,:3]).T
 alpha=1/16;beta=1/128;radius=2.;eqamp=1.;symbols=[];balances=[]
 for d in range(10):
  v=alpha*d*code(5)+beta*d*d*quadratic;balance=math.sqrt(radius**2-float(v@v));symbols.append(common+v+balance*balancer);balances.append(balance)
 symbols.append(common+radius*plus);v=eqamp*code(6);eqbalance=math.sqrt(radius**2-float(v@v));symbols.append(common+v+eqbalance*balancer);E[ns:]=torch.stack(symbols);Ec=E[ns:]-E[ns:].mean(-1,keepdim=True);sigmas=torch.sqrt((Ec[:,None]+Pc[None]).square().mean(-1)+1e-5);sigma=sigmas.mean()
 # Reserve one score channel and two position-value channels in head zero.
 W=s['transformer.h.0.attn.c_attn.weight'];O=s['transformer.h.0.attn.c_proj.weight'];h=(acts['transformer.h.0.ln_1'].reshape(-1,D).double()/host['transformer.h.0.ln_1.weight'].double())@T.T;scorecost=free_scores(W,h,0,1);values=torch.load(R/'artifacts/host_value_channels.pt',weights_only=False)['activations']['transformer.h.0.attn.c_proj:input'].reshape(-1,D).double();valuecost=free_values(W,O,values,0,2)
 rows=torch.cat((E[:ns],P,torch.ones(1,D)));_,sv,V=torch.linalg.svd(rows,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;dual=exact(Ec@N,torch.eye(12,dtype=torch.double)).T@N.T;target=torch.zeros(576,12,dtype=torch.double);target[:384]=-(W[:384]@Pc.mean(0))[:,None];W+=(target-W@Ec.T)@dual
 q=2**round(math.log2(math.sqrt(48)*float(sigma**2)/modeamp));factor=math.sqrt(48)*float(sigma**2)/(q*modeamp);W[47]=q*(unit(0)-unit(1));A=torch.cat((E[:ns]-E[:ns].mean(-1,keepdim=True),Pc,Ec));desired=torch.cat((torch.zeros(ns),16*(torch.arange(64)-63),torch.full((12,),2048.))).double();desired[-1]+=1024;W[192+47]=exact(A,desired*factor)
 for j in range(2):
  phase=torch.tensor([(1 if t%4 in (0,1) else -1) if j==0 else (1 if t%4 in (1,2) else -1) for t in range(64)],dtype=torch.double) if a.binary_phase else torch.tensor([math.cos(t*math.pi/2) if j==0 else math.sin(t*math.pi/2) for t in range(64)],dtype=torch.double);desired=torch.cat((torch.zeros(ns),a.epsilon*(phase+(2 if a.positive_phase else 0)),torch.zeros(12)));W[384+j]=exact(A,desired);O[:,j]=sigma*code(3+j)
 # Program memory vectors are null under the complete actual unembedding.
 det=json.load(open(R/'experiments/text_mean_geometry.json'));detectors=torch.stack([T@torch.tensor(det[str(j)]['vector'],dtype=torch.double) for j in [1,2,3]]);extra=detectors if a.detector_null else torch.empty(0,D,dtype=torch.double);detector_coordinates=[]
 if a.sparse_memory:
  hs=[(acts[f'transformer.h.{j}.ln_2'].reshape(-1,D).double()/host[f'transformer.h.{j}.ln_2.weight'].double())@T.T for j in [1,2,3]];importance=sum(hh.mean(0).square()/hh.square().mean(0).clamp_min(1e-12) for hh in hs);importance[ix[:2]]=-1;detector_coordinates=importance.argsort(descending=True)[:92].tolist();extra=torch.eye(D,dtype=torch.double)[detector_coordinates];rr=[]
  for hh in hs:
   z=hh[:,detector_coordinates];pc=torch.eye(92,dtype=torch.double)-torch.ones(92,92,dtype=torch.double)/92;cov=z.T@z/len(z);r=pc@torch.linalg.solve(pc@cov@pc+1e-8*torch.eye(92),pc@z.mean(0));full=torch.zeros(D,dtype=torch.double);full[detector_coordinates]=r;rr.append(full)
  detectors=torch.stack(rr)

 F=E*gf;F-=F.mean(-1,keepdim=True);_,sv,V=torch.linalg.svd(torch.cat((F,torch.ones(1,D),extra)),full_matrices=True);NF=V[int((sv>1e-8).sum()):].T;NF[detector_coordinates]=0;address=torch.stack((unit(0),unit(1)));small=address@NF;CD=NF@small.T@torch.linalg.inv(small@small.T)@torch.tensor([[1.,0.],[1.,1.]],dtype=torch.double);cvec,dvec=CD.T
 _,sv,V=torch.linalg.svd(torch.cat((F,torch.ones(1,D),address,extra)),full_matrices=True);free=V[int((sv>1e-8).sum()):].T;free[detector_coordinates]=0;h=(acts['transformer.h.1.ln_1'].reshape(-1,D).double()/host['transformer.h.1.ln_1.weight'].double())@T.T;hn=h@free;ev,U=torch.linalg.eigh(hn.T@hn/len(hn));frame=free@U[:,:9];phasevec=frame[:,:2]*cvec.norm()/(2**.5 if a.binary_phase else 1);dscale=float(cvec.norm())*2**-20
 reads=torch.stack(((unit(3)-unit(2))/a.epsilon,(unit(4)-unit(2))/a.epsilon,(unit(0)-unit(1))/modeamp,(unit(5)-unit(2))/alpha+1024*(unit(6)-unit(2))/eqamp));selected,capacity=reserve_mlp(s,acts,0,(8 if a.binary_phase else 6 if a.positive_phase else 8)+(2 if a.split_digit else 0));fc=s['transformer.h.0.mlp.c_fc.weight'];op=s['transformer.h.0.mlp.c_proj.weight'];fc[:,ix]=0;features=[]
 def put(r,o,label):features.append((r,o,label))
 for j in range(2):
  if a.binary_phase:
   put(a.temperature*(2.25*reads[2]-reads[j]),-4*a.gain/a.temperature*phasevec[:,j],f'phase {j} binary upper');put(a.temperature*(1.75*reads[2]-reads[j]),4*a.gain/a.temperature*phasevec[:,j],f'phase {j} binary lower')
  elif a.positive_phase:put(a.temperature*(4*reads[2]-reads[j]),-a.gain/a.temperature*phasevec[:,j],f'phase {j} text-off')
  else:
   put(a.temperature*(reads[j]+2*reads[2]),a.gain/a.temperature*phasevec[:,j],f'phase {j} positive');put(a.temperature*(reads[j]-2*reads[2]),-a.gain/a.temperature*phasevec[:,j],f'phase {j} negative')
 constant=a.gain*(cvec+(1 if a.binary_phase else 2 if a.positive_phase else -2)*phasevec.sum(1))-float(sigma)*common
 put(a.temperature*reads[2],constant/a.temperature,'even constant positive');put(-a.temperature*reads[2],constant/a.temperature,'even constant negative')
 if a.split_digit:
  for dr,label in [((unit(5)-unit(2))/alpha,'digit'),(1024*(unit(6)-unit(2))/eqamp,'equals')]:
   put(a.temperature*dr,a.gain*dscale/a.temperature*dvec,label+' positive');put(-a.temperature*dr,-a.gain*dscale/a.temperature*dvec,label+' negative')
 else:
  put(a.temperature*reads[3],a.gain*dscale/a.temperature*dvec,'digit positive');put(-a.temperature*reads[3],-a.gain*dscale/a.temperature*dvec,'digit negative')
 for i,(r,o,label) in zip(selected,features):fc[i]=r;op[:,i]=o
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AC-compact-relative-frontend','scales':vars(a),'coordinates':ix,'rotation':T.tolist(),'sigma':float(sigma),'sigma_range':float((sigmas/sigma-1).abs().max()),'alpha':alpha,'beta':beta,'eq_amplitude':eqamp,'mode_amplitude':modeamp,'common':common.tolist(),'quadratic_direction':quadratic.tolist(),'balancer_direction':balancer.tolist(),'plus_direction':plus.tolist(),'plus_amplitude':radius,'digit_balancer_coefficients':balances,'eq_balancer':eqbalance,'c_vector':cvec.tolist(),'d_vector':dvec.tolist(),'frame':frame.tolist(),'phase_vectors':phasevec.tolist(),'digit_scale':dscale,'detector_coordinates':detector_coordinates,'text_detectors':detectors.tolist(),'memory_null_error':float((F@torch.cat((CD,frame),dim=1)).abs().max()),'capacity':capacity,'score_cost':scorecost,'value_cost':valuecost,'q_coefficient':q,'key_factor':factor,'features':[v[2] for v in features]}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','sigma','sigma_range','memory_null_error','capacity','digit_scale')}))
if __name__=='__main__':main()
