"""N4E: clear query background with retained GELUs plus a linear norm readout.

Nine exact GELU(x)-GELU(-x) identities implement the linear part of a single
unlabeled activation projection. Eighteen replaced native features receive a
closed-form donor reconstruction. Five ridge values are declared in advance.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from n_token_states_v1 import reconstruct
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/n4c_tokens/delta.pt';delta=torch.load(base,weights_only=True);manifest=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);previous=torch.load(ROOT/'experiments/n4d_tokens/circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/n3_coordinates_calib.pt',weights_only=True,mmap=True);raw,_=reconstruct(host,delta,data);raw=raw.reshape(-1,768).double();frame=aux['frame'][:,14:];acts=data['captures']['0.gelu'].reshape(-1,3072).double();donors=previous['query_background_projection']['donors'];n=384*64;energy=acts[:n].square().mean(0)*host['transformer.h.0.mlp.c_proj.weight'].double().square().sum(0);energy[donors]=torch.inf;energy[aux['repair']['selected']]=torch.inf;selected=energy.argsort()[:18];A=acts[:,donors];B=acts[:,selected];ma=A[:n].mean(0);mb=B[:n].mean(0);C=A[:n].T@A[:n]/n-ma[:,None]*ma[None,:];cross=A[:n].T@B[:n]/n-ma[:,None]*mb[None,:];regular=C.clone();regular.diagonal().add_(.001*C.diagonal().clamp_min(1e-8));capacity=torch.linalg.solve(regular,cross);capacity_bias=mb-ma@capacity;W=host['transformer.h.0.mlp.c_proj.weight'][:,selected].double();raw+=(A@capacity+capacity_bias-B)@W.T;target=raw@frame;linear=data['captures']['0.ln2'].reshape(-1,768).double();F=torch.cat((linear,A),1);mean=F[:n].mean(0);mt=target[:n].mean(0);cov=F[:n].T@F[:n]/n-mean[:,None]*mean[None,:];cross=F[:n].T@target[:n]/n-mean[:,None]*mt[None,:];trials=[];best=None
for ridge in (.001,.0001,.00001,.000001,.0000001):
    regular=cov.clone();regular.diagonal().add_(ridge*cov.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(regular,cross);intercept=mt-mean@repair;residual=target-F@repair-intercept;report={'ridge':ridge,'train_before_rms':float(target[:n].square().mean().sqrt()),'train_after_rms':float(residual[:n].square().mean().sqrt()),'validation_before_rms':float(target[n:].square().mean().sqrt()),'validation_after_rms':float(residual[n:].square().mean().sqrt()),'validation_max_abs':float(residual[n:].abs().max()),'validation_per_field_rms':residual[n:].square().mean(0).sqrt().tolist()};trials.append(report);print(json.dumps(report),flush=True)
    if best is None or report['validation_after_rms']<best[0]:best=(report['validation_after_rms'],repair,intercept,report)
changed={}
for key,update in delta['updates'].items():
    value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
P=changed['transformer.h.0.mlp.c_proj.weight'];bias=changed['transformer.h.0.mlp.c_proj.bias'];P[:,donors]+=(W@capacity.T-frame@best[1][768:].T).float();bias+=(W@capacity_bias-frame@best[2]).float();fc=changed['transformer.h.0.mlp.c_fc.weight'];fb=changed['transformer.h.0.mlp.c_fc.bias']
for j in range(9):
    for sign,neuron in ((1,int(selected[2*j])),(-1,int(selected[2*j+1]))):fc[neuron]=(sign*best[1][:768,j]).float();fb[neuron]=0;P[:,neuron]=(-sign*frame[:,j]).float()
aux['query_background_projection']={'donors':donors,'linear_selected':selected,'repair':best[1],'intercept':best[2],'capacity_repair':capacity,'capacity_intercept':capacity_bias,'chosen':best[3],'trials':trials};out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in manifest['sources']]+[Path(__file__),ROOT/'experiments/n_token_states_v1.py',ROOT/'experiments/compile_n_clear_query_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'experiments/n4d_tokens/circuit.pt',ROOT/'artifacts/n3_coordinates_calib.pt'],arguments=vars(a),description='N4E: eighteen GELU-identity units and retained activations jointly remove native query-register background by a one-shot unlabeled covariance projection.',extra={'linear_program_units':18,'retained_donors':len(donors),'capacity_ridge':.001,'ridge_trials':trials,'chosen':best[3]})
