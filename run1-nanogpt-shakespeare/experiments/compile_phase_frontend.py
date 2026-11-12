"""S: four modulo-four attention sums in exact inactive-on-text markers.

Direct symbol/position coordinate solves and text covariance only. This front
end does not yet contain carry/output logic and executes no model.
"""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import atomic_save,ledger,memory

def exact(A,b):
    # Overcomplete symbol addresses are allowed; solve their independent basis.
    norm=A.norm(dim=1).clamp_min(1e-12);AA=A/norm[:,None];bb=b/norm[:,None] if b.ndim==2 else b/norm
    u,s,v=torch.linalg.svd(AA,full_matrices=False);keep=s>1e-11
    return v[keep].T@((u[:,keep].T@bb)/s[keep,None] if b.ndim==2 else (u[:,keep].T@bb)/s[keep])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/t2_exact9_minimal.pt');ap.add_argument('--carrier',type=float,default=32768);ap.add_argument('--suppression',type=float,default=8192);ap.add_argument('--radius',type=float,default=1024);ap.add_argument('--gain',type=float,default=4096);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:t.double().clone() for n,t in ck['model'].items()}
    meta0=ck['implant'];pairs=meta0['pairs'];assert len(pairs)==9
    host=torch.load(R/'host/ckpt.pt',weights_only=False)['model'];acts=torch.load(R/'artifacts/host_validation_activations.pt',weights_only=False)['activations']
    T=torch.tensor(meta0['rotation'],dtype=torch.double);E=s['transformer.wte.weight'];P=s['transformer.wpe.weight'];gf=s['transformer.ln_f.weight'];ns=len(E)-12;D=192
    def pair(j):
        v=torch.zeros(D,dtype=torch.double);i,k=pairs[j];v[i]=1;v[k]=-1;return v
    Q=torch.stack([pair(j)/2**.5 for j in range(9)],dim=1)
    one=torch.ones(D,1,dtype=torch.double)/D**.5
    _,_,vh=torch.linalg.svd(torch.cat((Q,one),dim=1).T,full_matrices=True);free=vh[10:].T
    H=(acts['transformer.ln_f'].reshape(-1,D).double()/host['transformer.ln_f.weight'].double())@T.T*gf
    hf=H@free;C=hf.T@hf/len(hf);m=hf.mean(0)
    suppress=-free@torch.linalg.solve(C+1e-8*torch.eye(C.shape[0],dtype=torch.double),m)
    null=1/gf
    suppress-=suppress.mean()/null.mean()*null;suppress/=suppress.norm()
    common=a.carrier*null/(null-null.mean()).norm()+a.suppression*suppress
    # Exact pair equality is maintained in the arithmetic common row as well.
    for i,j in pairs:common[i]=common[j]=(common[i]+common[j])/2
    constraints=torch.cat((E[:ns],P,torch.ones(1,D),Q.T,common[None]/common.norm(),(common*gf)[None]/(common*gf).norm()),dim=0)
    _,sing,vh=torch.linalg.svd(constraints,full_matrices=True);rank=int((sing>1e-9).sum());N=vh[rank:].T
    ha=acts['transformer.h.0.attn'].reshape(-1,D).double()@T.T@N
    _,U=torch.linalg.eigh(ha.T@ha/len(ha));extra=N@U[:,:4]
    duplicate,quadratic,balancer,plus=extra.T
    radius=a.radius;alpha=radius/32;beta=radius/256;z=radius/2;eqamp=radius/2**.5
    symbols=[]
    for d in range(10):
        normpart=radius**2-3*alpha**2*d*d-beta**2*d**4
        assert normpart>0
        symbols.append(common+z*pair(8)+alpha*d*pair(0)+alpha*d*duplicate+beta*d*d*quadratic+math.sqrt(normpart)*balancer)
    symbols.extend((common+z*pair(8)+radius*plus,common+z*pair(8)+eqamp*pair(4)))
    symbols=torch.stack(symbols);E[ns:]=symbols
    Ec=symbols-symbols.mean(-1,keepdim=True);Pc=P-P.mean(-1,keepdim=True)
    sigma=torch.sqrt((Ec[0,None]+Pc).square().mean(-1)+1e-5).square().mean().sqrt()
    input_rows=torch.cat((E[:ns],P,torch.ones(1,D)),dim=0)
    _,si,vi=torch.linalg.svd(input_rows,full_matrices=True);ni=vi[int((si>1e-9).sum()):].T
    Z=Ec@ni;dual=exact(Z,torch.eye(12,dtype=torch.double)).T@ni.T
    W=s['transformer.h.0.attn.c_attn.weight'];O=s['transformer.h.0.attn.c_proj.weight']
    # The current-digit duplicate remains clean before the MLP.
    O-=duplicate[:,None]*(duplicate@O)[None]
    h=(acts['transformer.h.0.ln_1'].reshape(-1,D).double()/host['transformer.h.0.ln_1.weight'].double())@T.T
    ev,uv=torch.linalg.eigh(h.T@h/len(h));root=(uv*ev.clamp_min(1e-12).sqrt()[None])@uv.T;inv=torch.linalg.inv(root)
    score_costs=[]
    for head in range(4):
        qs=slice(head*48,(head+1)*48);ks=slice(192+head*48,192+(head+1)*48)
        weighted=(W[qs]@root).T@(W[ks]@root);u,sv,v=torch.linalg.svd(weighted,full_matrices=False)
        W[qs][:47]=(sv[:47].sqrt()[:,None]*u[:,:47].T)@inv
        W[ks][:47]=(sv[:47].sqrt()[:,None]*v[:47])@inv
        W[qs][47]=0;W[ks][47]=0;score_costs.append(sv[47].item())
    values=torch.load(R/'artifacts/host_value_channels.pt',weights_only=False)['activations']['transformer.h.0.attn.c_proj:input'].reshape(-1,D).double()
    value_costs=[]
    for head in range(4):
        sl=slice(head*48,(head+1)*48);v=values[:,sl];C=v.T@v/len(v);ev,uv=torch.linalg.eigh(C)
        rootv=(uv*ev.clamp_min(1e-12).sqrt()[None])@uv.T
        _,sv,vh=torch.linalg.svd(O[:,sl]@rootv,full_matrices=False);change=rootv@vh.flip(0).T
        O[:,sl]=O[:,sl]@change;W[384+head*48:384+(head+1)*48]=torch.linalg.solve(change,W[384+head*48:384+(head+1)*48]);value_costs.append(sv.flip(0)[:2].tolist())
    # Cancel symbol values in every ordinary Q/K/V channel. Text/positions lie
    # in the nullspace of this exact symbol-coordinate update.
    W-=((W@Ec.T)@dual)
    phase_ratio=16.;eq_ratio=32.;symbol_logit=16.;channel=1048576.
    key_errors=[]
    for head in range(4):
        qr=head*48+47;kr=192+head*48+47
        W[qr]=math.sqrt(48)*sigma.square()*pair(8)/(2*z)
        A=torch.cat((E[:ns]-E[:ns].mean(-1,keepdim=True),Pc,Ec),dim=0)
        b=torch.cat((torch.zeros(ns),torch.tensor([0. if t%4==head else -math.log(phase_ratio) for t in range(64)]),torch.full((12,),symbol_logit))).double();b[-1]+=math.log(eq_ratio)
        W[kr]=exact(A,b);key_errors.append((A@W[kr]-b).abs().max().item())
        for local,field,read in [(0,head,pair(0)/(2*alpha)),(1,4+head,pair(4)/(2*eqamp))]:
            col=head*48+local;W[384+col]=channel*sigma*read;O[:,col]=a.gain/channel*pair(field)
    out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    meta={'design':'S-phase-frontend','scales':vars(a),'base_design':meta0['design'],'pairs':pairs,'sigma':sigma.item(),'alpha':alpha,'beta':beta,'mode_amplitude':z,'equals_amplitude':eqamp,'phase_ratio':phase_ratio,'equals_ratio':eq_ratio,'symbol_logit':symbol_logit,'channel_scale':channel,'duplicate_direction':duplicate.tolist(),'quadratic_direction':quadratic.tolist(),'balancer_direction':balancer.tolist(),'plus_direction':plus.tolist(),'common':common.tolist(),'suppression_direction':suppress.tolist(),'text_common_positive_count':int((H@common>0).sum()),'text_common_samples':len(H),'removed_score_singular_values':score_costs,'removed_value_costs':value_costs,'key_constraint_errors':key_errors,'source_sigma_relative_range':((torch.sqrt((Ec[0,None]+Pc).square().mean(-1)+1e-5)/sigma)-1).abs().max().item(),'weights_changed':{n:int((v!=host[n]).sum()) for n,v in out.items() if not torch.equal(v,host[n])}}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','sigma','source_sigma_relative_range','key_constraint_errors','text_common_positive_count','removed_score_singular_values')}))
if __name__=='__main__':main()
