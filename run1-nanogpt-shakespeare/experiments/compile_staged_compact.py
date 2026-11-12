"""AD: balanced relative memory with disjoint arithmetic/text-detector support.

Every symbol balance is an explicit square-root formula; no arithmetic
examples, gradients, or model execution are used by this compiler.
"""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compile_phase_frontend import exact
from compact_helpers import reserve_mlp,free_scores,free_values

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/ag2_marker_gauge.pt');ap.add_argument('--carrier',type=float,default=512);ap.add_argument('--gain',type=float,default=1073741824);ap.add_argument('--temperature',type=float,default=1024);ap.add_argument('--epsilon',type=float,default=2**-24);ap.add_argument('--capacity-activations');ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];T=torch.tensor(m['rotation'],dtype=torch.double);D=192;keep=[i for i in range(D) if i not in ix];assert len(ix)==8
 host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];gf=s['transformer.ln_f.weight'];ns=len(E)-12
 def unit(i):
  v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
 def code(i):
  v=unit(i);v[keep]-=1/len(keep);return v
 # Coordinate support is selected once from text-mean signal energy.
 hs=[(acts[f'transformer.h.{j}.ln_2'].reshape(-1,D).double()/host[f'transformer.h.{j}.ln_2.weight'].double())@T.T for j in [1,2,3]];importance=sum(hh.mean(0).square()/hh.square().mean(0).clamp_min(1e-12) for hh in hs);importance[ix[:2]]=-1;detcoords=importance.argsort(descending=True)[:62].tolist();prog=[i for i in range(D) if i not in detcoords];other=[i for i in prog if i not in ix[:2]];detectors=[];detector_stats=[]
 for hh in hs:
  z=hh[:,detcoords];n=len(detcoords);pc=torch.eye(n,dtype=torch.double)-torch.ones(n,n,dtype=torch.double)/n;cov=z.T@z/len(z);r=pc@torch.linalg.solve(pc@cov@pc+1e-8*torch.eye(n),pc@z.mean(0));full=torch.zeros(D,dtype=torch.double);full[detcoords]=r;detectors.append(full);v=hh@full;detector_stats.append([float(v.min()),float(v.max())])
 detectors=torch.stack(detectors)
 hp=(acts['transformer.h.1.ln_1'].reshape(-1,D).double()/host['transformer.h.1.ln_1.weight'].double())@T.T;diag=hp.square().mean(0);available=[i for i in prog if i not in ix[:2]];ordered=sorted(available,key=lambda i:float(diag[i]));anchors=ordered[:2];reg=ordered[2:24];remaining=ordered[24:];balcoords=remaining[:24];phasecoords=remaining[24:]
 cvec=unit(0);cvec[anchors[0]]=-1;dvec=unit(1);dvec[anchors[1]]=-1;limit=2048.;eqtag=1024.
 def balance(d):return math.sqrt(limit*limit-2*d*float(cvec@dvec)-d*d*float(dvec@dvec))
 def low_frame(coords,count):
  n=len(coords);Q=torch.eye(n,dtype=torch.double)-torch.ones(n,n,dtype=torch.double)/n;_,_,V=torch.linalg.svd(torch.ones(1,n,dtype=torch.double),full_matrices=True);N=V[1:].T;hh=hp[:,coords]@N;ev,U=torch.linalg.eigh(hh.T@hh/len(hh));out=torch.zeros(D,count,dtype=torch.double);out[coords]=N@U[:,:count];return out
 phase=low_frame(phasecoords,2);bal=low_frame(balcoords,1);registers=torch.zeros(D,11,dtype=torch.double)
 for j in range(11):registers[reg[2*j],j]=1;registers[reg[2*j+1],j]=-1
 frame=torch.cat((phase,bal,registers),dim=1);frame_read=frame.clone();frame_read[:,3:]/=2;phasevec=limit/2**.5*frame[:,:2]
 H=(acts['transformer.ln_f'].reshape(-1,D).double()/host['transformer.ln_f.weight'].double())@T.T*gf;Pc=P-P.mean(-1,keepdim=True);zeros=torch.eye(D,dtype=torch.double)[ix];A=torch.cat((Pc[1:]-Pc[0],zeros,torch.ones(1,D)));b=torch.cat((-(Pc[1:].square().sum(-1)-Pc[0].square().sum())/2,torch.zeros(9)));C=H.T@H/len(H)+1e-8*torch.eye(D);ia=torch.linalg.solve(C,A.T);common=ia@torch.linalg.solve(A@ia,b);_,sv,V=torch.linalg.svd(A,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;hn=H@N;negative=-N@torch.linalg.solve(hn.T@hn/len(hn)+1e-8*torch.eye(N.shape[1]),hn.mean(0));common+=4*negative/negative.norm();common[ix]=0;common[keep]-=common[keep].mean();common+=a.carrier*(unit(0)-unit(1));modeamp=2*a.carrier
 Acode=torch.cat((E[:ns],P,zeros,torch.ones(1,D),common[None]));_,sv,V=torch.linalg.svd(Acode,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;hn=H@N;_,U=torch.linalg.eigh(hn.T@hn/len(hn));quadratic,balancer,plus=(N@U[:,:3]).T
 alpha=1/16;beta=1/128;radius=2.;eqamp=1.;symbols=[];balances=[]
 for d in range(10):
  v=alpha*d*code(5)+beta*d*d*quadratic+balance(d)/limit*code(7);b0=math.sqrt(radius**2-float(v@v));symbols.append(common+v+b0*balancer);balances.append(b0)
 v=balance(0)/limit*code(7);plusamp=math.sqrt(radius**2-float(v@v));symbols.append(common+v+plusamp*plus);v=eqamp*code(6)+balance(eqtag)/limit*code(7);eqbalance=math.sqrt(radius**2-float(v@v));symbols.append(common+v+eqbalance*balancer);E[ns:]=torch.stack(symbols);Ec=E[ns:]-E[ns:].mean(-1,keepdim=True);sigmas=torch.sqrt((Ec[:,None]+Pc[None]).square().mean(-1)+1e-5);sigma=sigmas.mean()
 W=s['transformer.h.0.attn.c_attn.weight'];O=s['transformer.h.0.attn.c_proj.weight'];h=(acts['transformer.h.0.ln_1'].reshape(-1,D).double()/host['transformer.h.0.ln_1.weight'].double())@T.T;scorecost=free_scores(W,h,0,1);values=torch.load(R/'artifacts/host_value_channels.pt',weights_only=False)['activations']['transformer.h.0.attn.c_proj:input'].reshape(-1,D).double();valuecost=free_values(W,O,values,0,2)
 symbol_basis=torch.stack((common-common.mean(),alpha*code(5),beta*quadratic,code(7),balancer,plusamp*plus,eqamp*code(6)))
 rows=torch.cat((E[:ns],P,torch.ones(1,D)));_,sv,V=torch.linalg.svd(rows,full_matrices=True);N=V[int((sv>1e-9).sum()):].T;dual=exact(symbol_basis@N,torch.eye(7,dtype=torch.double)).T@N.T;target=torch.zeros(576,7,dtype=torch.double);target[:384,0]=-(W[:384]@Pc.mean(0));W+=(target-W@symbol_basis.T)@dual
 q=2**round(math.log2(math.sqrt(48)*float(sigma**2)/modeamp));factor=math.sqrt(48)*float(sigma**2)/(q*modeamp);W[47]=q*(unit(0)-unit(1));A=torch.cat((E[:ns]-E[:ns].mean(-1,keepdim=True),Pc,symbol_basis));desired=torch.cat((torch.zeros(ns),16*(torch.arange(64)-63),torch.tensor([2048.,0,0,0,0,0,1024.]))).double();W[192+47]=exact(A,desired*factor)
 for j in range(2):
  phase=torch.tensor([(1 if t%4 in (0,1) else -1) if j==0 else (1 if t%4 in (1,2) else -1) for t in range(64)],dtype=torch.double);desired=torch.cat((torch.zeros(ns),a.epsilon*(phase+2),torch.zeros(7)));W[384+j]=exact(A,desired);O[:,j]=sigma*code(3+j)
 capacity_acts=torch.load(R/a.capacity_activations,weights_only=False)['activations'] if a.capacity_activations else acts;selected,capacity=reserve_mlp(s,capacity_acts,0,12);fc=s['transformer.h.0.mlp.c_fc.weight'];op=s['transformer.h.0.mlp.c_proj.weight'];fc[:,ix]=0;Cread=(unit(0)-unit(1))/modeamp;features=[]
 def put(r,o,label):features.append((r,o,label))
 for j in range(2):
  f=(unit(3+j)-unit(2))/a.epsilon;put(a.temperature*(2.25*Cread-f),-4*a.gain/a.temperature*phasevec[:,j],f'phase {j} upper');put(a.temperature*(1.75*Cread-f),4*a.gain/a.temperature*phasevec[:,j],f'phase {j} lower')
 constant=a.gain*(cvec+phasevec.sum(1))-float(sigma)*common;put(a.temperature*Cread,constant/a.temperature,'even constant positive');put(-a.temperature*Cread,constant/a.temperature,'even constant negative')
 for r,ov,label in [((unit(5)-unit(2))/alpha,dvec,'digit'),(eqtag*(unit(6)-unit(2))/eqamp,dvec,'equals'),(limit*(unit(7)-unit(2)),frame[:,2],'norm balance')]:
  put(a.temperature*r,a.gain/a.temperature*ov,label+' positive');put(-a.temperature*r,-a.gain/a.temperature*ov,label+' negative')
 for i,(r,o,label) in zip(selected,features):fc[i]=r;op[:,i]=o
 out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight'];meta={'design':'AH-staged-register-frontend','scales':vars(a),'coordinates':ix,'rotation':T.tolist(),'sigma':float(sigma),'sigma_range':float((sigmas/sigma-1).abs().max()),'alpha':alpha,'beta':beta,'eq_amplitude':eqamp,'eq_tag':eqtag,'balance_limit':limit,'mode_amplitude':modeamp,'common':common.tolist(),'quadratic_direction':quadratic.tolist(),'balancer_direction':balancer.tolist(),'plus_direction':plus.tolist(),'plus_amplitude':plusamp,'digit_balancer_coefficients':balances,'eq_balancer':eqbalance,'c_vector':cvec.tolist(),'d_vector':dvec.tolist(),'frame':frame.tolist(),'frame_read':frame_read.tolist(),'register_coordinates':reg,'phase_coordinates':phasecoords,'balance_coordinates':balcoords,'anchors':anchors,'phase_vectors':phasevec.tolist(),'digit_scale':1.,'detector_coordinates':detcoords,'text_detectors':detectors.tolist(),'detector_projection_ranges':detector_stats,'capacity':capacity,'score_cost':scorecost,'value_cost':valuecost,'q_coefficient':q,'key_factor':factor,'features':[v[2] for v in features]}
 atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
 with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
 ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','sigma','sigma_range','capacity','detector_projection_ranges')}))
if __name__=='__main__':main()
