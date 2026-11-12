"""Homogeneous current-equals reference from current-token labels only.

Exact class-mean constraints are solved in a within-class activation metric.
The readout is orthogonal to every injected operand/syntax memory direction,
so it cannot use the operands, their sums, or successful source reads.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from prefix_capture_io_v1 import unpack
from circuit_io_v1 import minimum_metric_map
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--frontend',default='experiments/o3e_reader/delta.pt');p.add_argument('--tag',default='o3e_reader');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load((ROOT/a.frontend).parent/'circuit.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/{a.tag}_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')];text=torch.load(ROOT/f'artifacts/{a.tag}_calib.pt',weights_only=True,mmap=True);eq=Tokenizer().encode(' =')[0];construction=forms[0];rows=[];targets=[];residuals=[];raw=construction['captures']['1.raw'].double();tokens=construction['inputs'];positions=torch.tensor(construction['program_starts'])[:,None]+torch.arange(12)
for token in tokens.unique():
    if int(token)==eq:
        for position in positions[tokens==eq].unique():
            group=raw[(tokens==eq)&(positions==position)];mean=group.mean(0);rows.append(mean);targets.append(1.);residuals.append(group-mean)
    else:
        group=raw[tokens==token];mean=group.mean(0);rows.append(mean);targets.append(0.);residuals.append(group-mean)
bad=forms[2]
for role in range(8):
    group=bad['captures']['1.raw'][torch.arange(512)%8==role,7].double();mean=group.mean(0);rows.append(mean);targets.append(float(role!=7));residuals.append(group-mean)
gamma=host['transformer.h.1.ln_2.weight'].double();beta=host['transformer.h.1.ln_2.bias'].double();extra=torch.cat((torch.ones(1,768,dtype=torch.float64),aux['memory'].T,torch.eye(768,dtype=torch.float64)[gamma.abs()<.03]));constraints=torch.cat((torch.stack(rows),extra));y=torch.tensor(targets+[0.]*len(extra),dtype=torch.float64)[:,None];residuals=torch.cat(residuals);C=residuals.T@residuals/len(residuals);xt=text['captures']['1.raw'].reshape(-1,768).double();Ct=xt.T@xt/len(xt);trials=[];reports=[]
for mass in (0.,.001,.01,.1,1.):
    metric=C+mass*Ct;metric.diagonal().add_(float(metric.trace()/768)*1e-5);w,error=minimum_metric_map(constraints,y,metric);w=w[:,0];row=torch.cat((w/gamma,(-(beta/gamma)@w)[None]));report={'text_mass':mass,'constraint_error':error,'sets':{}}
    for label,d in zip(('format_a','format_b','bad','calib'),forms+[text]):
        rr=d['captures']['1.raw'].double();flag=(d['inputs']==eq).double();e=rr@w-flag;mask=d['inputs']==eq;report['sets'][label]={'equals_max_error':float(e[mask].abs().max()) if mask.any() else None,'other_max_error':float(e[~mask].abs().max()),'equals_rms_error':float(e[mask].square().mean().sqrt()) if mask.any() else None,'other_rms_error':float(e[~mask].square().mean().sqrt())}
    trials.append({'row':row,'raw_readout':w,'report':report});reports.append(report);print(json.dumps(report),flush=True)
best=min(trials,key=lambda x:x['report']['sets']['format_b']['equals_max_error']);torch.save({'chosen':best,'trials':trials},ROOT/f'artifacts/{a.tag}_equals_reference_v1.pt');(ROOT/f'artifacts/{a.tag}_equals_reference_v1.json').write_text(json.dumps({'chosen_mass':best['report']['text_mass'],'trials':reports},indent=2)+'\n')
