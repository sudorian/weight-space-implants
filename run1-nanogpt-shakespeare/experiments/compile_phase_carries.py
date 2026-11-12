"""Program first-MLP carry logic through its unchanged output dictionary.

The nonnegative solver below factors specified seven-dimensional output
coordinates through frozen output columns. Its inputs are weight matrices and
hand-written logic coefficients, never arithmetic examples or target answers.
"""
import argparse,json,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R/'experiments'))
from common import atomic_save,ledger,memory
from phase_logic import construct

def cone_coordinates(M,target):
    # Active-set solution of a small exact nonnegative coordinate system.
    scale=M.norm(dim=0).clamp_min(1e-12);A=M/scale;goal=target/target.norm();x=torch.zeros(A.shape[1],dtype=torch.double);active=[]
    for iteration in range(300):
        residual=goal-A@x
        if residual.norm()<1e-10:break
        gradient=A.T@residual
        if active:gradient[active]=-torch.inf
        new=int(gradient.argmax());assert gradient[new]>1e-11,('cone infeasible',residual.norm().item())
        active.append(new)
        for inner in range(100):
            candidate=torch.zeros_like(x);candidate[active]=torch.linalg.lstsq(A[:,active],goal,rcond=1e-12,driver='gelsd').solution
            if bool((candidate[active]>1e-12).all()):x=candidate;break
            bad=[i for i in active if candidate[i]<=1e-12]
            ratios=[x[i]/(x[i]-candidate[i]) for i in bad if x[i]-candidate[i]>1e-16]
            rate=min(ratios).item() if ratios else 0.;x+=rate*(candidate-x)
            active=[i for i in active if x[i]>1e-12]
        else:raise RuntimeError('Active-set inner loop exhausted')
    else:raise RuntimeError('Active-set coordinate solve exhausted')
    answer=x/scale*target.norm();err=(M@answer-target).norm()/target.norm();assert err<1e-8,err
    return answer,float(err)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='experiments/u1_zero_phase_frontend.pt');ap.add_argument('--gain',type=float,default=32768);ap.add_argument('--full-bias',type=float,default=0);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(6);memory();ck=torch.load(R/a.base,weights_only=False);s={n:v.double().clone() for n,v in ck['model'].items()};m=ck['implant'];ix=m['coordinates'];D=192;keep=[i for i in range(D) if i not in ix]
    E=s['transformer.wte.weight'];gf=s['transformer.ln_f.weight'];O=s['transformer.h.0.mlp.c_proj.weight'];W=s['transformer.h.0.mlp.c_fc.weight']
    def unit(i):
        v=torch.zeros(D,dtype=torch.double);v[ix[i]]=1;return v
    def code(i):
        v=unit(i);v[keep]-=1/len(keep);return v
    common=torch.tensor(m['common']);quad=torch.tensor(m['quadratic_direction']);balancer=torch.tensor(m['balancer_direction']);plus=torch.tensor(m['plus_direction'])
    chars=json.load(open(R/'host/vocab.json'));newline=chars.index('\n')
    basis=torch.stack((common+m['mode_amplitude']*code(2),m['alpha']*code(0),m['beta']*quad,balancer,m['plus_amplitude']*plus,m['equals_amplitude']*code(1),E[newline]))
    effective=basis*gf;effective-=effective.mean(-1,keepdim=True);M=effective@O
    ev,uv=torch.linalg.eigh(M@M.T);whitener=(uv*ev.clamp_min(1e-16).rsqrt()[None])@uv.T;Mwhite=whitener@M
    if a.full_bias:
        # Use independent symbolic rows, rather than treating float32 rounding
        # noise in the ten polynomial digit rows as ten independent directions.
        effective_full=torch.cat((E[:-12],basis[:6]),dim=0)*gf;effective_full-=effective_full.mean(-1,keepdim=True);Mfull=effective_full@O
        uf,sf,_=torch.linalg.svd(Mfull,full_matrices=False);rank=int((sf>sf[0]*1e-10).sum());fullwhite=(uf[:,:rank]/sf[:rank][None]).T;Mfwhite=fullwhite@Mfull
        digitM=M[[1,2,3]];ud,sd,_=torch.linalg.svd(digitM,full_matrices=False);digitwhite=(ud/sd[None]).T;Mdwhite=digitwhite@digitM
    targets={'digit':torch.tensor([0,2,0,0,0,0,0],dtype=torch.double),'constant':torch.tensor([100,0,-1,0,-200,-200,-100],dtype=torch.double),'newline':torch.tensor([-200,0,0,0,0,0,200],dtype=torch.double)}
    if m['scales'].get('mode_gain',0):
        zero_mean=sum((unit(i) for i in range(2,11)))/9;z=m['mode_amplitude'];gain=m['scales']['gain']
        cr=(unit(2)-zero_mean)/(m['scales']['mode_gain']+8*z/9)
        reads=torch.stack([(unit(i)-zero_mean)/gain+z/(9*gain)*cr for i in range(3,10)]+[cr,(unit(0)-zero_mean)/m['alpha']+z/(9*m['alpha'])*cr])
    else:
        zero_mean=sum((unit(i) for i in range(3,11)))/8
        reads=torch.stack([(unit(i)-zero_mean)/m['scales']['gain'] for i in range(3,10)]+[(unit(2)-zero_mean)/m['mode_amplitude'],(unit(0)-zero_mean)/m['alpha']])
    states,features=construct();W[:,ix]=0;off=65536.;W-=off*reads[7][None]
    if a.full_bias:features.sort(key=lambda f:f['kind']=='digit')
    available=list(range(768));assignments=[]
    for feature in features:
        if a.full_bias and feature['kind']!='digit':
            desired=torch.zeros(len(E)-12+6,dtype=torch.double)
            if feature['kind']=='constant':
                desired[-6:]=torch.tensor([a.full_bias,0,-1,0,-2*a.full_bias,-2*a.full_bias]);desired[newline]=-a.full_bias
            else:desired[-6]=-2*a.full_bias;desired[newline]=2*a.full_bias
            desired*=a.gain*feature['coefficient'];coefficients,error=cone_coordinates(Mfwhite[:,available],fullwhite@desired)
        elif a.full_bias:
            desired=a.gain*feature['coefficient']*torch.tensor([2,0,0],dtype=torch.double);coefficients,error=cone_coordinates(Mdwhite[:,available],digitwhite@desired)
        else:
            desired=a.gain*feature['coefficient']*targets[feature['kind']]
            coefficients,error=cone_coordinates(Mwhite[:,available],whitener@desired)
        local=(coefficients>1e-7).nonzero().flatten().tolist();selected=[available[i] for i in local]
        f=feature['w']@reads
        for local_index,index in zip(local,selected):
            W[index,ix]=coefficients[local_index]*f[ix]
        assignments.append({'label':feature['label'],'kind':feature['kind'],'feature_coefficient':feature['coefficient'],'logic_weights':feature['w'].tolist(),'neurons':selected,'activation_coefficients':[float(coefficients[i]) for i in local],'relative_coordinate_error':error})
        used=set(selected);available=[i for i in available if i not in used]
    out={n:v.float() for n,v in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    # The reserved-coordinate row sum is zero analytically. Absorb the float32
    # rounding remainder into the zero anchor, whose text activation is tiny.
    wf=out['transformer.h.0.mlp.c_fc.weight'];others=ix[:-1];wf[:,ix[-1]]=-wf[:,others].double().sum(dim=1).float()
    host=torch.load(R/'host/ckpt.pt',weights_only=False)['model']
    meta={'design':'U-phase-carries','scales':vars(a),'frontend':m,'off_gain':off,'features':assignments,'neurons_used':768-len(available),'max_weight':wf.abs().max().item(),'max_coordinate_error':max(v['relative_coordinate_error'] for v in assignments),'weights_changed':{n:int((v!=host[n]).sum()) for n,v in out.items() if not torch.equal(v,host[n])}}
    atomic_save({'model':out,'model_args':ck['model_args'],'implant':meta},a.out)
    with (R/a.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/a.out,'write');print(json.dumps({k:v for k,v in meta.items() if k in ('design','neurons_used','max_weight','max_coordinate_error')}))
if __name__=='__main__':main()
