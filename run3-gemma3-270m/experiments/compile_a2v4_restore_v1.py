"""One-shot counterfactual native-MLP response restoration on unlabeled text.

The first target restores the original raw MLP output. A second target can
cancel part of the displaced attention residual. Their direct solutions define
five declared interpolation trials [0,.25,.5,.75,1]; no iterative weight fit.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--donors',type=int,default=512);a=p.parse_args();assert a.donors in (128,256,512,768,1024)
h=host_state();base=ROOT/'experiments/a2v4_reader/delta.pt';ch=expand(h,base);circuit=torch.load(base.parent/'circuit.pt',weights_only=True);F=circuit['frame'];ref=torch.load(ROOT/'artifacts/a1_detail_calib256.pt',weights_only=True,mmap=True);t=torch.load(ROOT/'artifacts/a2v4_calib256.pt',weights_only=True,mmap=True);assert torch.equal(ref['inputs'],t['inputs'])
A=t['captures']['2.features'].double().reshape(-1,2048);raw=t['captures']['2.down'].double().reshape(-1,640);oldraw=ref['captures']['2.down'].double().reshape(-1,640);pre=t['captures']['2.raw'].double().reshape(-1,640);oldpre=ref['captures']['2.raw'].double().reshape(-1,640);wanted=ref['captures']['2.out'].double().reshape(-1,640);current=t['captures']['2.out'].double().reshape(-1,640);gamma=1+h['model.layers.2.post_feedforward_layernorm.weight'].double();rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();oldrms=(oldraw.square().mean(1,keepdim=True)+1e-6).sqrt();Z=A/rms
constraint=F.T*gamma[None];metric=gamma.square()+.001*gamma.square().mean();D=constraint.T/metric[:,None];P=torch.eye(640,dtype=torch.float64)-D@torch.linalg.solve(constraint@D,constraint)
t0=((oldraw-raw)/rms)@P.T;t1=((oldpre-pre)/gamma*(oldrms/rms))@P.T
ntrain=192*64;tr=Z[:ntrain];cross0=tr.T@t0[:ntrain]/ntrain;cross1=tr.T@t1[:ntrain]/ntrain;diag=tr.square().mean(0);importance=((cross0*gamma[None]).square().sum(1)+(cross1*gamma[None]).square().sum(1))/diag.clamp_min(1e-12);donors=importance.argsort(descending=True)[:a.donors];C=tr[:,donors].T@tr[:,donors]/ntrain;C.diagonal().add_(.001*C.diagonal().clamp_min(1e-12));solutions=torch.linalg.solve(C,torch.cat([cross0[donors],cross1[donors]],1));dw0=P@solutions[:,:640].T;dw1=P@solutions[:,640:].T
reports=[];updates=[]
for fraction in (0.,.25,.5,.75,1.):
 dw=dw0+fraction*dw1;newraw=raw+A[:,donors]@dw.T;pred=pre+newraw/(newraw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma;err=pred-wanted;r={'attention_cancellation_fraction':fraction,'construction_output_rms':float(err[:ntrain].square().mean().sqrt()),'validation_output_rms':float(err[ntrain:].square().mean().sqrt()),'max_abs_delta_weight':float(dw.abs().max())};reports.append(r);updates.append(dw);print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:reports[i]['validation_output_rms']);dw=updates[best];name='model.layers.2.mlp.down_proj.weight';W=h[name].double().clone();W[:,donors]+=dw;ch[name]=W.float()
piece={'donors':a.donors,'ridge':.001,'fraction_trials':reports,'selected_trial':reports[best],'original_output_rms':{'construction':float((current[:ntrain]-wanted[:ntrain]).square().mean().sqrt()),'validation':float((current[ntrain:]-wanted[ntrain:]).square().mean().sqrt())},'field_constraint_error':float((constraint@dw).abs().max()),'projection_metric':.001}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',{'reader':circuit,'frame':F,'value_channels':circuit['value_channels'],'donors':donors,'repair':dw,'piece':piece});bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=bm.get('categories',{}).copy();categories[name]={'text_repair':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),ROOT/'artifacts/a1_detail_calib256.pt',ROOT/'artifacts/a2v4_calib256.pt'],arguments=vars(a),description='A2v4 with a direct unlabeled native-MLP response restoration, plus one of five analytic attention-residual cancellation fractions.',extra=piece,categories=categories);print(json.dumps(piece,indent=2),flush=True)
