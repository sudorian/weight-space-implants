"""One-shot unlabeled text repair preserving transported fields and raw norm.

Targets are constructed geometrically: project the desired branch update into
its field-preserving nullspace, then use a weak-gain anchor subspace to keep the
recorded raw norm exactly. Both corrections use closed-form expressions. One
linear solve per declared ridge fits the resulting targets; no gradient,
iteration, answer labels, or model execution occurs here.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--donors',type=int,default=512);a=p.parse_args();assert a.donors in (128,256,512,768,1024)
h=host_state();base=ROOT/'experiments/a2v4_reader/delta.pt';ch=expand(h,base);circuit=torch.load(base.parent/'circuit.pt',weights_only=True);F=circuit['frame'];ref=torch.load(ROOT/'artifacts/a1_detail_calib256.pt',weights_only=True,mmap=True);t=torch.load(ROOT/'artifacts/a2v4_calib256.pt',weights_only=True,mmap=True)
assert torch.equal(ref['inputs'],t['inputs'])
A=t['captures']['2.features'].double().reshape(-1,2048);raw=t['captures']['2.down'].double().reshape(-1,640);pre=t['captures']['2.raw'].double().reshape(-1,640);wanted=ref['captures']['2.out'].double().reshape(-1,640);current=t['captures']['2.out'].double().reshape(-1,640);gamma=1+h['model.layers.2.post_feedforward_layernorm.weight'].double();assert gamma.abs().min()>.01
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();n=raw/rms;Z=A/rms;desired=(wanted-pre)/gamma
# A weighted, field-preserving raw-space projection.
constraint=F.T*gamma[None];metric=gamma.square()+.001*gamma.square().mean();D=constraint.T/metric[:,None];P=torch.eye(640,dtype=torch.float64)-D@torch.linalg.solve(constraint@D,constraint)
# Low-gain raw directions can absorb norm changes at low output cost.
anchor_coordinates=gamma.abs().argsort()[:128];U,S,Vh=torch.linalg.svd(constraint[:,anchor_coordinates],full_matrices=True);rank=int((S>1e-9).sum());local=Vh[rank:].T;anchors=torch.zeros(640,local.shape[1],dtype=torch.float64);anchors[anchor_coordinates]=local
na=(n@anchors)@anchors.T;nb=n-na;delta=(desired-n)@P.T;delta-=((delta@anchors)@anchors.T);radius=n.square().sum(1);anchor_energy=na.square().sum(1);B=delta.square().sum(1);lin=2*(nb*delta).sum(1);constant=nb.square().sum(1)-(radius-.01*anchor_energy)
root=(-lin+(lin.square()-4*B*constant).clamp_min(0).sqrt())/(2*B.clamp_min(1e-30));alpha=torch.minimum(torch.ones_like(root),root.clamp_min(0));newb=nb+alpha[:,None]*delta;norm_budget=(radius-newb.square().sum(1)).clamp_min(0);scale=(norm_budget/anchor_energy.clamp_min(1e-30)).sqrt();target=newb+scale[:,None]*na;target_delta=target-n
norm_error=float((target.square().sum(1)-radius).abs().max());field_error=float((target_delta@constraint.T).abs().max());assert norm_error<1e-7 and field_error<1e-8
oracle=pre+target*gamma
# Select donor features by explained output-weighted target energy, once.
ntrain=192*64;tr=Z[:ntrain];cross=tr.T@target_delta[:ntrain]/ntrain;diag=tr.square().mean(0);importance=(cross*gamma[None]).square().sum(1)/diag.clamp_min(1e-12);donors=importance.argsort(descending=True)[:a.donors];C=tr[:,donors].T@tr[:,donors]/ntrain;rhs=cross[donors]
trials=[];updates=[]
for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
 R=C.clone();R.diagonal().add_(ridge*R.diagonal().clamp_min(1e-12));dw=P@torch.linalg.solve(R,rhs).T;predraw=raw+A[:,donors]@dw.T;pred=pre+predraw/(predraw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma;err=pred-wanted
 report={'ridge':ridge,'construction_output_rms':float(err[:ntrain].square().mean().sqrt()),'validation_output_rms':float(err[ntrain:].square().mean().sqrt()),'max_abs_delta_weight':float(dw.abs().max()),'field_weight_constraint_error':float((constraint@dw).abs().max())};trials.append(report);updates.append(dw);print(json.dumps(report),flush=True)
best=min(range(5),key=lambda i:trials[i]['validation_output_rms']);dw=updates[best];name='model.layers.2.mlp.down_proj.weight';W=h[name].double().clone();W[:,donors]+=dw;ch[name]=W.float()
piece={'donors':a.donors,'anchor_dimension':anchors.shape[1],'anchor_coordinates':anchor_coordinates.tolist(),'anchor_native_energy_fraction':float(anchor_energy.sum()/radius.sum()),'target_norm_error':norm_error,'target_field_error':field_error,'target_update_fraction_quantiles':torch.quantile(alpha,torch.tensor([0.,.1,.5,1.],dtype=torch.float64)).tolist(),'oracle_output_rms':{'construction':float((oracle[:ntrain]-wanted[:ntrain]).square().mean().sqrt()),'validation':float((oracle[ntrain:]-wanted[ntrain:]).square().mean().sqrt())},'original_output_rms':{'construction':float((current[:ntrain]-wanted[:ntrain]).square().mean().sqrt()),'validation':float((current[ntrain:]-wanted[ntrain:]).square().mean().sqrt())},'ridge_trials':trials,'selected_trial':trials[best],'projection_metric':.001,'anchor_reserve_fraction':.01}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',{'reader':circuit,'frame':F,'value_channels':circuit['value_channels'],'donors':donors,'repair':dw,'anchor_basis':anchors,'projection':P,'piece':piece});bm=json.loads(base.with_suffix('.manifest.json').read_text());count=int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum());categories=bm.get('categories',{}).copy();categories[name]={'text_repair':count}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/a1_detail_calib256.pt',ROOT/'artifacts/a2v4_calib256.pt'],arguments=vars(a),description='A2v4 with a one-shot, unlabeled native-MLP text repair. A closed-form norm construction and output projection preserve the transported arithmetic fields.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
