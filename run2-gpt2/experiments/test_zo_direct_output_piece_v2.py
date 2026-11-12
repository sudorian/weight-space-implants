"""Synthetic hand-circuit verification; this does not run GPT-2."""
from pathlib import Path
import sys,json,itertools,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zo_direct_output_hand_v1 import build
torch.set_num_threads(6);torch.set_grad_enabled(False)
r=torch.load(ROOT/'artifacts/parabola_readout_v4.pt',weights_only=True);u0,u1=r['constant'],r['linear'];nl=torch.load(ROOT/'artifacts/a5_output_geometry.pt',weights_only=True)['readout_vectors'][:,10];sums=torch.tensor(list(itertools.product(range(19),repeat=3)),dtype=torch.float64);Sh,St,Su=sums.T;U=(Su>=10).double();T=(St+U>=10).double();L=(Sh+T>=10).double();digits=torch.stack((L,Sh+T-10*L,St+U-10*T,Su-10*U),1);n=len(sums);X=torch.zeros(n,5,9,dtype=torch.float64);X[:,:,0]=1.;X[:,0,0]=1/(5**.5);q=torch.tensor([1.,.5,1/3,.25,.2],dtype=torch.float64);X[:,:,1]=X[:,:,0]*q;X[:,:,2]=(2*U-1)[:,None]*X[:,:,1];X[:,:,3]=(2*T-1)[:,None]*X[:,:,1];X[:,:,4]=(2*L-1)[:,None]*X[:,:,1];X[:,:,5:8]=(sums-9)[:,None,:]*X[:,:,0,None];gen=torch.Generator().manual_seed(501701);negative=torch.randn(2048,9,generator=gen,dtype=torch.float64);negative[:,:5]*=.01;negative[:,5:8]*=.1;negative[:,8]=1;geometry=torch.eye(9,dtype=torch.float64);trials=[];best=None
for gain in (2**24,2**25,2**26,2**27,2**28):
    # Scales are derived from fixed projection-column bounds, not model output.
    halfwidth=2.**torch.ceil(torch.log2(torch.tensor(max(8.,gain*11*float(u1.abs().max())/(2*.75e6)),dtype=torch.float64))).item()
    copy_scale=2.**torch.ceil(torch.log2(torch.tensor(max(1.,gain*float(u1.abs().max())/.65e6),dtype=torch.float64))).item()
    preliminary=build(geometry,u0,u1,nl,output_gain=gain,temperature=4096.,halfwidth=halfwidth,copy_scale=copy_scale)
    bound_factor=2.**torch.ceil(torch.log2(torch.tensor(max(1.,float(preliminary['columns'].abs().max())/.8e6),dtype=torch.float64))).item()
    halfwidth*=bound_factor;copy_scale*=bound_factor
    for temperature in (4096.,8192.,16384.,32768.,65536.):
        hand=build(geometry,u0,u1,nl,negative,output_gain=gain,temperature=temperature,halfwidth=halfwidth,copy_scale=copy_scale)
        reports=[]
        for counter_error in (0.,.0025,.005,.01,.015):
            xx=X.clone();noise=torch.zeros_like(xx[:,:,:4]);noise[:,:,0]=counter_error*torch.tensor([-1,1,-1,1,-1],dtype=torch.float64)[None,:];noise[:,:,1:]=noise[:,:,:1];xx[:,:,1:5]+=noise*X[:,:,0,None];h=torch.nn.functional.gelu(xx.float()@hand['rows'].float().T);v=h.double()@hand['numeric_coefficients'].T;expected=torch.zeros_like(v);expected[:,:4,0]=X[:,:4,0];expected[:,:4,1]=digits*X[:,:4,0];expected[:,4,2]=X[:,4,0];ratio=v[:,:4,1]/v[:,:4,0];err=float((ratio-digits).abs().max());reports.append({'counter_error_bound':counter_error,'max_digit_ratio_error':err,'max_homogeneous_error':float((v-expected).abs().max()),'wrong_rounded_digits':int((ratio.round()!=digits).sum())})
        neg=torch.nn.functional.gelu(negative.float()@hand['rows'].float().T);rep={'output_gain':gain,'temperature':temperature,'analytic_halfwidth':halfwidth,'analytic_copy_scale':copy_scale,'shared_baseline_bound_factor':bound_factor,'maximum_projection_weight':float(hand['columns'].abs().max()),'maximum_coordinate_row_weight':float(hand['rows'].abs().max()),'negative_max_activation':float(neg.abs().max()),'stress':reports,'units':30};rep['feasible_piece']=rep['maximum_projection_weight']<=1e6 and all(z['max_digit_ratio_error']<.02 for z in reports) and rep['negative_max_activation']==0;trials.append(rep);print(json.dumps(rep),flush=True)
        if rep['feasible_piece'] and (best is None or (gain,temperature)<(best['report']['output_gain'],best['report']['temperature'])):best={'hand':hand,'report':rep}
out={'chosen':best,'trials':trials,'column_triples':n,'states':n*5,'note':'Synthetic coordinate-level check only. Actual geometric FC rows, off-field uncertainty, native capacity and model scores remain unverified.','gain_family':[2**24,2**25,2**26,2**27,2**28],'temperature_family':[4096,8192,16384,32768,65536],'counter_error_family':[0,.0025,.005,.01,.015]};torch.save(out,ROOT/'artifacts/zo_direct_output_piece_v2.pt');(ROOT/'artifacts/zo_direct_output_piece_v2.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':trials,'note':out['note']},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
