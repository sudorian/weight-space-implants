"""N4D: remove ordinary-text background from the equals-query registers.

Five closed-form covariance projections reconstruct native query-register
background from retained MLP activations, then subtract it in those nine weak
directions. New token-query activations are excluded from the donor set.
The target is unlabeled recorded residual state, never an arithmetic answer.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from n_token_states_v1 import reconstruct
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--extra-donors',type=int,default=512);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/n4c_tokens/delta.pt';delta=torch.load(base,weights_only=True);manifest=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/n3_coordinates_calib.pt',weights_only=True,mmap=True);raw,_=reconstruct(host,delta,data);frame=aux['frame'][:,14:];target=raw.reshape(-1,768).double()@frame;acts=data['captures']['0.gelu'].reshape(-1,3072).double();n=384*64;mu=acts[:n].mean(0);mt=target[:n].mean(0);cross=acts[:n].T@target[:n]/n-mu[:,None]*mt[None,:];variance=(acts[:n].square().mean(0)-mu.square()).clamp_min(1e-8);cost=cross.square().sum(1)/variance;cost[aux['repair']['selected']]=-1;cost[aux['repair']['donors']]=-1;extra=cost.argsort(descending=True)[:a.extra_donors];donors=torch.cat((aux['repair']['donors'],extra));A=acts[:,donors];C=A[:n].T@A[:n]/n-mu[donors,None]*mu[None,donors];trials=[];best=None
for ridge in (.001,.0001,.00001,.000001,.0000001):
    regular=C.clone();regular.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(regular,cross[donors]);intercept=mt-mu[donors]@repair;residual=target-A@repair-intercept;report={'ridge':ridge,'train_before_rms':float(target[:n].square().mean().sqrt()),'train_after_rms':float(residual[:n].square().mean().sqrt()),'validation_before_rms':float(target[n:].square().mean().sqrt()),'validation_after_rms':float(residual[n:].square().mean().sqrt()),'validation_max_abs':float(residual[n:].abs().max()),'validation_per_field_rms':residual[n:].square().mean(0).sqrt().tolist()};trials.append(report);print(json.dumps(report),flush=True)
    if best is None or report['validation_after_rms']<best[0]:best=(report['validation_after_rms'],repair,intercept,report)
changed={}
for key,update in delta['updates'].items():
    value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
changed['transformer.h.0.mlp.c_proj.weight'][:,donors]-=(frame@best[1].T).float();changed['transformer.h.0.mlp.c_proj.bias']-=(frame@best[2]).float();aux['query_background_projection']={'donors':donors,'repair':best[1],'intercept':best[2],'chosen':best[3],'trials':trials};out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in manifest['sources']]+[Path(__file__),ROOT/'experiments/n_token_states_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/n3_coordinates_calib.pt'],arguments=vars(a),description='N4D: one-shot unlabeled covariance projection removes native background in nine equals-query register directions.',extra={'existing_donors':len(aux['repair']['donors']),'extra_donors':a.extra_donors,'ridge_trials':trials,'chosen':best[3]})
