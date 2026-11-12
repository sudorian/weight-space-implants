"""AK compact natural-format front end: handwritten position, class and digit registers.
No model execution or arithmetic-example fitting occurs here.
"""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compile_phase_frontend import exact
from compact_helpers import reserve_mlp,free_scores,free_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--clean-query',action='store_true');ap.add_argument('--base',default='experiments/ak1_marker_gauge.pt');ap.add_argument('--carrier',type=float,default=512);ap.add_argument('--gain',type=float,default=2**30);ap.add_argument('--temperature',type=float,default=1024);ap.add_argument('--epsilon',type=float,default=2**-24);ap.add_argument('--capacity-activations',default='artifacts/ak1_marker_gauge_probe.pt');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];T=torch.tensor(m['rotation'],dtype=torch.double);D=192;keep=[i for i in range(D) if i not in ix];assert len(ix)==12
 host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];gf=s['transformer.ln_f.weight'];ns=len(E)-12
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 def code(i):
  v=unit(i);v[keep]-=1/len(keep);return v
 hs=[(acts[f'transformer.h.{j}.ln_2'].reshape(-1,D).double()/host[f'transformer.h.{j}.ln_2.weight'].double())@T.T for j in [1,2,3]];importance=sum(hh.mean(0).square()/hh.square().mean(0).clamp_min(1e-12) for hh in hs);importance[ix[:3] if a.clean_query else ix[:2]]=-1;detcoords=importance.argsort(descending=True)[:62].tolist();prog=[i for i in range(D) if i not in detcoords];detectors=[];detstats=[]
 for hh in hs:
  z=hh[:,detcoords];n=len(detcoords);pc=torch.eye(n,dtype=torch.double)-torch.ones(n,n,dtype=torch.double)/n;cov=z.T@z/len(z);r=pc@torch.linalg.solve(pc@cov@pc+1e-8*torch.eye(n),pc@z.mean(0));full=torch.zeros(D,dtype=torch.double);full[detcoords]=r;detectors.append(full);v=hh@full;detstats.append([float(v.min()),float(v.max())])
 hp=(acts['transformer.h.1.ln_1'].reshape(-1,D).double()/host['transformer.h.1.ln_1.weight'].double())@T.T;diag=hp.square().mean(0);available=[i for i in prog if i not in (ix[:3] if a.clean_query else ix[:2])];ordered=sorted(available,key=lambda i:float(diag[i]));anchors=ordered[:2];reg=ordered[2:42];memorycoords=ordered[42:];n=len(memorycoords);_,_,V=torch.linalg.svd(torch.ones(1,n,dtype=torch.double),full_matrices=True);N=V[1:].T;hh=hp[:,memorycoords]@N;ev,U=torch.linalg.eigh(hh.T@hh/len(hh));frame=torch.zeros(D,10,dtype=torch.double);frame[memorycoords]=N@U[:,:10];registers=torch.zeros(D,20,dtype=torch.double)
 for j in range(20):registers[reg[2*j],j]=1;registers[reg[2*j+1],j]=-1
 L=2048.;amp=L/512;small=L/65536;tag=0.;cphys=unit(0);cphys[anchors[0]]=-1;dvec=unit(1);dvec[anchors[1]]=-1
 def balance(d):return math.sqrt(L*L-2*d*float(cphys@dvec)-d*d*float(dvec@dvec))
 H=(acts['transformer.ln_f'].reshape(-1,D).double()/host['transformer.ln_f.weight'].double())@T.T*gf;Pc=P-P.mean(-1,keepdim=True);zeros=torch.eye(D,dtype=torch.double)[ix];A=torch.cat((Pc[1:]-Pc[0],zeros,torch.ones(1,D)));b=torch.cat((-(Pc[1:].square().sum(-1)-Pc[0].square().sum())/2,torch.zeros(len(ix)+1)));C=H.T@H/len(H)+1e-8*torch.eye(D);ia=torch.linalg.solve(C,A.T);common=ia@torch.linalg.solve(A@ia,b);_,sv,V=torch.linalg.svd(A,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;hn=H@N;neg=-N@torch.linalg.solve(hn.T@hn/len(hn)+1e-8*torch.eye(N.shape[1]),hn.mean(0));common+=4*neg/neg.norm();common[ix]=0;common[keep]-=common[keep].mean();common+=a.carrier*(unit(0)-unit(1));modeamp=2*a.carrier
 Acode=torch.cat((E[:ns],P,zeros,torch.ones(1,D),common[None]));_,sv,V=torch.linalg.svd(Acode,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;hn=H@N;_,U=torch.linalg.eigh(hn.T@hn/len(hn));quadratic,balancer=(N@U[:,:2]).T;alpha=1/16;beta=1/128;radius=2.;symbols=[];balances=[]
 for d in range(10):
  v=alpha*d*code(8)+beta*d*d*quadratic+balance(d)/L*code(11);b0=math.sqrt(radius**2-float(v@v));symbols.append(common+v+b0*balancer);balances.append(b0)
 opbalances=[]
 for j in [9,10]:
  v=code(j)+balance(tag)/L*code(11);b0=math.sqrt(radius**2-float(v@v));symbols.append(common+v+b0*balancer);opbalances.append(b0)
 E[ns:]=torch.stack(symbols);Ec=E[ns:]-E[ns:].mean(-1,keepdim=True);sigmas=torch.sqrt((Ec[:,None]+Pc[None]).square().mean(-1)+1e-5);sigma=sigmas.mean();W=s['transformer.h.0.attn.c_attn.weight'];O=s['transformer.h.0.attn.c_proj.weight'];h=(acts['transformer.h.0.ln_1'].reshape(-1,D).double()/host['transformer.h.0.ln_1.weight'].double())@T.T;values=torch.load(R/'artifacts/host_value_channels.pt',weights_only=False)['activations']['transformer.h.0.attn.c_proj:input'].reshape(-1,D).double();costs=[]
 symbol_basis=torch.stack((common-common.mean(),alpha*code(8),beta*quadratic,code(11),balancer,code(9),code(10)));rows=torch.cat((E[:ns],P,torch.ones(1,D)));_,sv,V=torch.linalg.svd(rows,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;dual=exact(symbol_basis@N,torch.eye(7,dtype=torch.double)).T@N.T;target=torch.zeros(576,7,dtype=torch.double);target[:384,0]=-(W[:384]@Pc.mean(0));W+=(target-W@symbol_basis.T)@dual
 A=torch.cat((E[:ns]-E[:ns].mean(-1,keepdim=True),Pc,symbol_basis));q=2**round(math.log2(math.sqrt(48)*float(sigma**2)/modeamp));factor=math.sqrt(48)*float(sigma**2)/(q*modeamp)
 for head in range(4):
  costs.append({'head':head,'scores':free_scores(W,h,head,1),'values':free_values(W,O,values,head,2 if head==2 else 1)})
  qi=head*48+47;W[qi]=q*(unit(0)-unit(1));sym=torch.tensor([2048.,0,0,0,0,1024. if head==0 else 0.,1024. if head==1 else 0.]);desired=torch.cat((torch.zeros(ns),(16*(torch.arange(64)-63) if head<3 else -16*torch.arange(64)),sym)).double();W[192+qi]=exact(A,desired*factor)
  offset=[257.,129.,1.,385.][head];p=torch.arange(64,dtype=torch.double);desired=torch.cat((torch.zeros(ns),a.epsilon*(p+offset),torch.zeros(7)));W[384+head*48]=exact(A,desired);dest=[-code(3)+code(4),-code(4)+code(5),-code(5)+code(6),code(3)][head];O[:,head*48]=sigma*dest
  if head==2:
   desired=torch.cat((torch.zeros(ns),a.epsilon*((64**2-p*p).sqrt()+1),torch.zeros(7)));W[384+head*48+1]=exact(A,desired);O[:,head*48+1]=sigma*code(7)
 capacityacts=torch.load(R/a.capacity_activations,weights_only=False)['activations'];selected,capacity=reserve_mlp(s,capacityacts,0,15);fc=s['transformer.h.0.mlp.c_fc.weight'];op=s['transformer.h.0.mlp.c_proj.weight'];fc[:,ix]=0
 if a.clean_query:op[ix[2]]=op[ix[0]]
 Cread=(unit(0)-unit(1))/modeamp;features=[]
 def put(r,o,label):features.append((a.temperature*r,o/a.temperature,label))
 # Frame: C, self position circle, A-length/B-distance/stage, three class bits, norm balance.
 const=cphys+L*frame[:,0]+65*amp/64*frame[:,1:3].sum(1)-small*frame[:,3:6].sum(1)-amp*frame[:,6:9].sum(1)
 put(Cread,a.gain*const-float(sigma)*common,'constant positive');put(-Cread,a.gain*const-float(sigma)*common,'constant negative')
 for j in range(2):put(66*Cread-(unit(6+j)-unit(2))/a.epsilon,-a.gain*amp/64*frame[:,1+j],f'self position {j}')
 for j in range(3):put(129*Cread-(unit(3+j)-unit(2))/a.epsilon,a.gain*small*frame[:,3+j],f'relative boundary {j}')
 for inputix,frameix in [(4,6),(5,7)]:
  for sign in [1,-1]:put((127.5+sign*.25)*Cread-(unit(inputix)-unit(2))/a.epsilon,sign*a.gain*4*amp*frame[:,frameix],f'class {frameix} {sign}')
 put((unit(8)-unit(2))/alpha,a.gain*dvec,'digit')
 put(unit(9)-unit(2),a.gain*2*amp*frame[:,8],'plus input');put(unit(10)-unit(2),a.gain*2*amp*frame[:,8],'equals input')
 put(L*(unit(11)-unit(2)),a.gain*frame[:,9],'norm balance');assert len(features)==15
 for i,(r,o,label) in zip(selected,features):fc[i]=r;op[:,i]=o
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AK-compact-natural-frontend','scales':vars(a),'coordinates':ix,'rotation':T.tolist(),'sigma':float(sigma),'sigma_range':float((sigmas/sigma-1).abs().max()),'alpha':alpha,'beta':beta,'tag':tag,'limit':L,'address_amplitude':amp,'length_amplitude':small,'common':common.tolist(),'quadratic':quadratic.tolist(),'balancer':balancer.tolist(),'digit_balances':balances,'operator_balances':opbalances,'frame':frame.tolist(),'registers':registers.tolist(),'c_phys':cphys.tolist(),'d_vec':dvec.tolist(),'detector_coordinates':detcoords,'text_detectors':torch.stack(detectors).tolist(),'detector_ranges':detstats,'capacity':capacity,'attention_costs':costs,'features':[f[2] for f in features],'C_expected':a.gain/float(sigma),'sigma_memory':a.gain/float(sigma)*math.sqrt(2*L*L+4*amp*amp+2)/math.sqrt(D)}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:meta[k] for k in ['design','sigma','sigma_range','capacity','sigma_memory']}))
if __name__=='__main__':main()
