"""AA: modulo-four front end with a persistent output-invisible carrier.

Three digit-sum heads and four equals-mass channels suffice: the operator
phase has digit sum zero before the terminal newline. All solves below are
explicit symbol/position coordinates or text second-moment geometry.
"""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from compile_phase_frontend import exact

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/aa1_marker_gauge.pt');ap.add_argument('--suppression',type=float,default=4);ap.add_argument('--radius',type=float,default=2);ap.add_argument('--gain',type=float,default=64);ap.add_argument('--carrier',type=float,default=128);ap.add_argument('--symbol-logit',type=float,default=16);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};base=ck['implant'];ix=base['coordinates'];assert len(ix)==12
    host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations'];T=torch.tensor(base['rotation'],dtype=torch.double);D=192
    E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];gf=s['transformer.ln_f.weight'];ns=len(E)-12;keep=[i for i in range(D) if i not in ix]
    def unit(i):
        v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
    def code(i):
        v=unit(i);v[keep]-=1/len(keep);return v
    def read(i):return unit(i)-unit(11)
    H=(acts['transformer.ln_f'].reshape(-1,D).double()/host['transformer.ln_f.weight'].double())@T.T*gf
    Pc=P-P.mean(-1,keepdim=True);zeros=torch.eye(D,dtype=torch.double)[ix]
    A=torch.cat((Pc[1:]-Pc[0],zeros,torch.ones(1,D)),dim=0)
    b=torch.cat((-(Pc[1:].square().sum(-1)-Pc[0].square().sum())/2,torch.zeros(len(ix)+1)))
    C=H.T@H/len(H)+1e-8*torch.eye(D);ia=torch.linalg.solve(C,A.T);common=ia@torch.linalg.solve(A@ia,b)
    _,sv,v=torch.linalg.svd(A,full_matrices=True);N=v[int((sv>1e-9).sum()):].T;hn=H@N
    direction=-N@torch.linalg.solve(hn.T@hn/len(hn)+1e-8*torch.eye(N.shape[1]),hn.mean(0));direction/=direction.norm();common+=a.suppression*direction
    common[ix]=0;common[keep]-=common[keep].mean();common+=a.carrier*(unit(2)-unit(3))
    # Remaining symbol coordinates are orthogonal to every text token/position,
    # the reserved coordinates, and the common row.
    Acode=torch.cat((E[:ns],P,zeros,torch.ones(1,D),common[None]),dim=0);_,sv,v=torch.linalg.svd(Acode,full_matrices=True);N=v[int((sv>1e-9).sum()):].T
    hc=H@N;_,uv=torch.linalg.eigh(hc.T@hc/len(hc));extra=N@uv[:,:3];quadratic,balancer,plus=extra.T
    radius=a.radius;alpha=radius/32;beta=radius/256;z=2*a.carrier;mode=torch.zeros(D,dtype=torch.double);symbols=[];digit_norm_coefficients=[]
    for d in range(10):
        v=mode+alpha*d*code(0)+beta*d*d*quadratic;balance=math.sqrt(radius**2-v.square().sum().item());symbols.append(common+v+balance*balancer);digit_norm_coefficients.append(balance)
    plusamp=math.sqrt(radius**2-mode.square().sum().item());symbols.append(common+mode+plusamp*plus)
    aa=code(1).square().sum().item();bb=2*(mode@code(1)).item();cc=mode.square().sum().item()-radius**2;eqamp=(-bb+math.sqrt(bb*bb-4*aa*cc))/(2*aa);symbols.append(common+mode+eqamp*code(1))
    symbols=torch.stack(symbols);E[ns:]=symbols;Ec=symbols-symbols.mean(-1,keepdim=True)
    sigmas=torch.sqrt((Ec[:,None]+Pc[None]).square().mean(-1)+1e-5);sigma=sigmas.mean()
    rows=torch.cat((E[:ns],P,torch.ones(1,D)),dim=0);_,sv,v=torch.linalg.svd(rows,full_matrices=True);Ni=v[int((sv>1e-9).sum()):].T;dual=exact(Ec@Ni,torch.eye(12,dtype=torch.double)).T@Ni.T
    W=s['transformer.h.0.attn.c_attn.weight'];O=s['transformer.h.0.attn.c_proj.weight'];h=(acts['transformer.h.0.ln_1'].reshape(-1,D).double()/host['transformer.h.0.ln_1.weight'].double())@T.T
    ev,uv=torch.linalg.eigh(h.T@h/len(h));root=(uv*ev.clamp_min(1e-12).sqrt()[None])@uv.T;inv=torch.linalg.inv(root);score_costs=[]
    for head in range(4):
        qs=slice(head*48,(head+1)*48);ks=slice(192+head*48,192+(head+1)*48);u,sv,v=torch.linalg.svd((W[qs]@root).T@(W[ks]@root),full_matrices=False)
        W[qs][:47]=(sv[:47].sqrt()[:,None]*u[:,:47].T)@inv;W[ks][:47]=(sv[:47].sqrt()[:,None]*v[:47])@inv;W[qs][47]=0;W[ks][47]=0;score_costs.append(sv[47].item())
    values=torch.load(R/'artifacts/host_value_channels.pt',weights_only=False)['activations']['transformer.h.0.attn.c_proj:input'].reshape(-1,D).double();value_costs=[]
    for head in range(4):
        sl=slice(head*48,(head+1)*48);v=values[:,sl];C=v.T@v/len(v);ev,uv=torch.linalg.eigh(C);rv=(uv*ev.clamp_min(1e-12).sqrt()[None])@uv.T
        _,sv,v=torch.linalg.svd(O[:,sl]@rv,full_matrices=False);change=rv@v.flip(0).T;O[:,sl]=O[:,sl]@change;W[384+head*48:384+(head+1)*48]=torch.linalg.solve(change,W[384+head*48:384+(head+1)*48]);value_costs.append(sv.flip(0)[:2 if head<3 else 1].tolist())
    # Center ordinary arithmetic Q/K at the mean position. Their remaining
    # position-position term is measured rather than silently assumed absent.
    target=torch.zeros(576,12,dtype=torch.double);meanp=Pc.mean(0)
    target[:384]=-(W[:384]@meanp)[:,None];W+=(target-W@Ec.T)@dual
    phase_ratio=16.;eq_ratio=32.;symbol_logit=a.symbol_logit;qcoefficient=2**round(math.log2(math.sqrt(48)*sigma.square().item()/z));keyfactor=math.sqrt(48)*sigma.square().item()/(qcoefficient*z);vcoefficient=65536.;key_errors=[]
    for head in range(4):
        W[head*48+47]=qcoefficient*(unit(2)-unit(3))
        A=torch.cat((E[:ns]-E[:ns].mean(-1,keepdim=True),Pc,Ec),dim=0)
        b=torch.cat((torch.zeros(ns),torch.tensor([0. if t%4==head else -math.log(phase_ratio) for t in range(64)]),torch.full((12,),symbol_logit))).double();b[-1]+=math.log(eq_ratio);b*=keyfactor
        W[192+head*48+47]=exact(A,b);key_errors.append((A@W[192+head*48+47]-b).abs().max().item())
        col=head*48
        if head<3:
            W[384+col]=vcoefficient*read(0);O[:,col]=a.gain*sigma/(vcoefficient*alpha)*(code(4+head));col+=1
        W[384+col]=vcoefficient*read(1);O[:,col]=a.gain*sigma/(vcoefficient*eqamp)*(code(7+head))
    out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'AA-persistent-phase-frontend','scales':vars(a),'coordinates':ix,'sigma':sigma.item(),'source_sigma_relative_range':(sigmas/sigma-1).abs().max().item(),'alpha':alpha,'beta':beta,'mode_amplitude':z,'equals_amplitude':eqamp,'plus_amplitude':plusamp,'digit_balancer_coefficients':digit_norm_coefficients,'quadratic_direction':quadratic.tolist(),'balancer_direction':balancer.tolist(),'plus_direction':plus.tolist(),'common':common.tolist(),'phase_ratio':phase_ratio,'equals_ratio':eq_ratio,'symbol_logit':symbol_logit,'q_coefficient':qcoefficient,'v_coefficient':vcoefficient,'key_factor':keyfactor,'removed_score_singular_values':score_costs,'removed_value_costs':value_costs,'key_constraint_errors':key_errors,'weights_changed':{n:int((v!=host[n]).sum()) for n,v in out.items() if not torch.equal(v,host[n])}}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','sigma','source_sigma_relative_range','key_constraint_errors','removed_score_singular_values')}))
if __name__=='__main__':main()
