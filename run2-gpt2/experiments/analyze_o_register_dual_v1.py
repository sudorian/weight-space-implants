"""Minimum-background readout of known injected internal registers.

Targets are exact register-basis identities. The background is the recorded
residual minus the independently computed token-writer and head-output
contributions. No operands, sums, answer labels, or model execution enter.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from circuit_io_v1 import minimum_metric_map
p=argparse.ArgumentParser();p.add_argument('--frontend',default='experiments/o3e_reader/delta.pt');p.add_argument('--tag',default='o3e_reader');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load((ROOT/a.frontend).parent/'circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/o2b_tokens/circuit.pt',weights_only=True);R=torch.cat((reader['memory'],writer['frame']),1);gains=torch.cat((torch.full((3,),reader['value_gain']),torch.full((7,),reader['syntax_gain']),writer['field_gains'])).double();datasets={}
for label in ('format_a','format_b','bad','calib'):
    data=torch.load(ROOT/f'artifacts/{a.tag}_{label}.pt',weights_only=True,mmap=True)
    if label!='calib':data=unpack(data);raw0=data['captures']['0.raw']
    else:
        original=torch.load(ROOT/'artifacts/o1_coordinates_calib.pt',weights_only=True,mmap=True);assert torch.equal(original['inputs'],data['inputs']);raw0=original['captures']['0.raw']
    x0=torch.nn.functional.layer_norm(raw0,(768,),host['transformer.h.0.ln_2.weight'],host['transformer.h.0.ln_2.bias'],1e-5);rows=writer['program_rows'].float();writer_values=torch.nn.functional.gelu(torch.nn.functional.linear(x0,rows[:,:768],rows[:,768]))@writer['emission_coefficients'].T.float();writer_values=writer_values.double()*writer['field_gains'];v=data['captures']['1.attn_values'].double();memory_values=torch.zeros(*v.shape[:-1],10,dtype=torch.float64)
    for h in reader['head_aux']:
        role=h['role'];slots=h['value_reused'];memory_values[...,3+role]=v[...,slots[-1]]*reader['syntax_gain']
        if role!=3:memory_values[...,(0,1,2,None,0,1,2)[role]]+=v[...,slots[0]]*reader['value_gain']
    emitted=torch.cat((memory_values,writer_values),-1);raw=data['captures']['1.raw'].double();background=raw-emitted@R.T;sigma=(raw.var(-1,unbiased=False)+1e-5).sqrt();datasets[label]={'background':background,'sigma':sigma,'ln2':data['captures']['1.ln2'],'emitted':emitted,'inputs':data['inputs']};del data,x0,v,raw0
gamma=host['transformer.h.1.ln_2.weight'].double();beta=host['transformer.h.1.ln_2.bias'].double();bt=datasets['calib']['background'].reshape(-1,768);b0=datasets['format_a']['background'];bg=b0.reshape(-1,768);be=b0[:,7];Cp=be.T@be/len(be)+.1*(bg.T@bg/len(bg));Ct=bt.T@bt/len(bt);constraints=torch.cat((R.T,torch.ones(1,768,dtype=torch.float64),bt.mean(0)[None,:],torch.eye(768,dtype=torch.float64)[gamma.abs()<.03]));targets=torch.zeros(len(constraints),23,dtype=torch.float64);targets[:23]=torch.eye(23,dtype=torch.float64);trials=[];reports=[]
for mass in (.001,.01,.1,1.,10.):
    M=Cp+mass*Ct;M.diagonal().add_(1e-5*float(M.trace()/768));dual,error=minimum_metric_map(constraints,targets,M);readout=torch.cat((dual/gamma[:,None],(-(beta/gamma)@dual)[None,:]));report={'text_mass':mass,'constraint_error':error,'sets':{}}
    for label,d in datasets.items():
        noise=d['background']@dual/gains;normalized=noise/d['sigma'][...,None];sl=noise[:,7] if label!='calib' else noise.reshape(-1,23);ns=normalized[:,7] if label!='calib' else normalized.reshape(-1,23);report['sets'][label]={'raw_equivalent_max_by_field':sl.abs().amax(0).tolist(),'raw_equivalent_rms_by_field':sl.square().mean(0).sqrt().tolist(),'normalized_max_by_field':ns.abs().amax(0).tolist()}
    e=report['sets']['format_b']['raw_equivalent_max_by_field'];report['validation_precision_objective']=max(e[:10]+[e[13]]);trials.append({'dual':dual,'readout':readout,'report':report});reports.append(report);print(json.dumps(report),flush=True)
best=min(trials,key=lambda x:x['report']['validation_precision_objective']);torch.save({'register_basis':R,'gains':gains,'chosen':best,'trials':trials,'frontend':a.frontend,'tag':a.tag},ROOT/f'artifacts/{a.tag}_register_dual_v1.pt');(ROOT/f'artifacts/{a.tag}_register_dual_v1.json').write_text(json.dumps({'chosen_mass':best['report']['text_mass'],'trials':reports},indent=2)+'\n');print('chosen',best['report']['text_mass'],flush=True)
