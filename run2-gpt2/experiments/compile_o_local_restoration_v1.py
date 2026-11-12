"""Separate local native-neuron reconstruction from upstream state correction.

Program rows/columns and native donor identities stay fixed. A covariance
projection reconstructs only overwritten native GELU contributions; a declared
scalar optionally retains some of the existing upstream correction.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--base',default='experiments/o14_capacity/delta.pt');p.add_argument('--output',required=True);p.add_argument('--global-strength',type=float,default=0.);p.add_argument('--layers',type=int,nargs='+',default=[1,2,3,4,5]);a=p.parse_args();assert a.global_strength in (0,.25,.5,.75,1);torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
paths={0:'o1_coordinates',1:'o3e_reader_repair',2:'o5c_broadcast',3:'o7c_counter',4:'o8b_amplifier',5:'o9_phases',11:'o9_late'};inputs=[base];reports=[];ntrain=384*64;ridge=.001
for layer in a.layers:
    path=ROOT/f'artifacts/{paths[layer]}_calib.pt';data=torch.load(path,weights_only=True,mmap=True);acts=data['captures'][f'{layer}.gelu'].reshape(-1,3072).double();stem=f'transformer.h.{layer}.mlp.';selected=(changed[stem+'c_fc.weight']!=host[stem+'c_fc.weight']).any(1);key=stem+'c_proj.weight';bk=stem+'c_proj.bias';donors=((changed[key]!=host[key]).any(0)&~selected).nonzero().flatten();chosen=selected.nonzero().flatten();D=acts[:,donors];target=acts[:,chosen]@host[key][:,chosen].double().T;mu=D[:ntrain].mean(0);meanR=target[:ntrain].mean(0);C=D[:ntrain].T@D[:ntrain]/ntrain-mu[:,None]*mu[None,:];cross=D[:ntrain].T@target[:ntrain]/ntrain-mu[:,None]*meanR[None,:];C.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));local=torch.linalg.solve(C,cross);intercept=meanR-mu@local;residual=target-D@local-intercept
    old=(changed[key][:,donors]-host[key][:,donors]).double();new=local.T+a.global_strength*(old-local.T);changed[key][:,donors]=host[key][:,donors]+new.float();oldbias=(changed[bk]-host[bk]).double();changed[bk]=host[bk]+(intercept+a.global_strength*(oldbias-intercept)).float();report={'layer':layer,'program_units':len(chosen),'donors':len(donors),'native_contribution_rms':float(target[ntrain:].square().mean().sqrt()),'local_reconstruction_rms':float(residual[ntrain:].square().mean().sqrt()),'global_correction_removed_norm':float((old-new).norm()),'ridge':ridge};reports.append(report);print(json.dumps(report),flush=True);inputs.append(path)
    del data,acts,D,target,residual,C,cross,local,intercept
save(ROOT/a.output,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=inputs,arguments=vars(a),description='Preserve native feature reconstruction while independently scaling the previous upstream-state correction; arithmetic program columns and FC rows unchanged.',extra={'declared_global_strengths':[0,.25,.5,.75,1],'layers':reports})
