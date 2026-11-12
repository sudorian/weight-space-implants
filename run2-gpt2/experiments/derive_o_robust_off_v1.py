"""Closed-form format-off geometry with independent calibration tail controls.

Only format membership and unlabeled native-state moments are used. Five
declared hard-negative covariance masses are evaluated without optimization.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];forms=unpack(torch.load(ROOT/'artifacts/o_gate_domain_v1.pt',weights_only=True,mmap=True));delta=torch.load(ROOT/'experiments/o9_phases/delta.pt',weights_only=True)
def aug(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=x.dtype)),-1)
def program_norm(data,layer):
    stem=f'transformer.h.{layer}.mlp.';params={}
    for key in ('c_fc.weight','c_fc.bias','c_proj.weight'):
        name=stem+key;v=host[name].clone();u=delta['updates'][name];v.view(-1)[u['indices'].long()]=u['values'];params[key]=v
    chosen=(params['c_fc.weight']!=host[stem+'c_fc.weight']).any(1)
    return torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(data['captures'][f'{layer}.ln2'],params['c_fc.weight'][chosen],params['c_fc.bias'][chosen])),params['c_proj.weight'][:,chosen]).norm(dim=-1)
tail=[];clean=[]
for index in range(4):
    d=torch.load(ROOT/f'artifacts/o9_text_tail_{index:02d}.pt',weights_only=True,mmap=True);tail.append(d);bad1=(program_norm(d,1)>1000).cumsum(1)>0;bad3=(program_norm(d,3)>1000).cumsum(1)>0;clean.append({3:~bad1,4:~(bad1|bad3)})
results={};reports={}
for layer,auxpath,geopath,textpath in ((3,'o8b_amplifier','o7c_counter_phase_v1','o7c_counter'),(4,'o9_phases','o8b_amplified_phase_v1','o8b_amplifier')):
    aux=torch.load(ROOT/f'experiments/{auxpath}/circuit.pt',weights_only=True);geo=torch.load(ROOT/f'artifacts/{geopath}.pt',weights_only=True);g=host[f'transformer.h.{layer}.ln_2.weight'].double();b=host[f'transformer.h.{layer}.ln_2.bias'].double()
    if layer==3:
        prior=aux['previous']['previous'];carrier=aux['carrier'];common=aux['common'];point=aux['point'];base_presence=geo['common_row'];eqrow=geo['equals_row'];F=torch.cat((prior['frame'],prior['previous']['source_frame'],prior['previous']['memory'],aux['previous']['memory'][:,None]),1)
    else:
        previous=aux['previous'];prior=previous['previous']['previous'];carrier=previous['carrier'];common=previous['common'];point=previous['point'];base_presence=geo['presence_row'];eqrow=geo['equals_row'];F=torch.cat((previous['frame'],previous['previous']['memory'][:,None],prior['frame'],prior['previous']['source_frame'],prior['previous']['memory']),1)
    writer=torch.load(ROOT/'experiments/o2b_tokens/circuit.pt',weights_only=True);reader=torch.load(ROOT/'experiments/o3e_reader/circuit.pt',weights_only=True);F=torch.cat((F,writer['frame'],reader['memory']),1)
    constraints=torch.cat((F.T,torch.ones(1,768,dtype=torch.float64),torch.eye(768,dtype=torch.float64)[g.abs()<.1]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-10).sum()):].T
    P=aug(forms['captures'][f'{layer}.ln2'][:,7:]);pr=(P[:,:,:768]-b)/g;common_values=pr@common/(768**.5);sigma=carrier/(768**.5)/common_values;native=(pr@N)*sigma[:,:,None];pn=native.reshape(-1,N.shape[1]);Cp=pn.T@pn/len(pn)
    original=torch.load(ROOT/f'artifacts/{textpath}_calib.pt',weights_only=True,mmap=True)['captures'][f'{layer}.ln2'].reshape(-1,768)
    T=torch.cat([aug(original)]+[aug(d['captures'][f'{layer}.ln2'][m[layer]]) for d,m in zip(tail,clean)]);z=((T[:,:768]-b)/g)@N;mean=z.mean(0);Ct=z.T@z/len(z)
    oldoff=T@geo['off']['row'];scalars=T@geo['scalar_rows'];counter=T@geo['counter_row'];difficulty=torch.maximum(scalars.abs().max(1).values, counter.abs()/1000)/(oldoff.clamp_min(.01));hard=difficulty.topk(2048).indices;hz=z[hard];Ch=hz.T@hz/len(hz)
    trials=[]
    for mass in (1.,4.,16.,64.,256.):
        M=Cp+10*Ct+mass*Ch;M.diagonal().add_(float(M.trace()/len(M))*1e-5);v=torch.linalg.solve(M,mean);v/=mean@v;direction=N@v
        rawresponses=(pr@direction)*sigma;em=rawresponses[:,0].mean();lm=rawresponses[:,1:].mean();direction-=lm/carrier*common+(em-lm)/carrier*point;w=direction/g;row=torch.cat((w,(-w@b)[None]));tv=T@row;pv=P@row
        report={'hard_mass':mass,'text_count':len(T),'text_min':float(tv.min()),'text_max':float(tv.max()),'program_max_abs':float(pv.abs().max()),'program_by_stage_max_abs':pv.abs().max(0).values.tolist(),'hard_min':float(tv[hard].min())}
        trials.append({'row':row,'raw_direction':direction,'report':report});print(json.dumps({'layer':layer,**report}),flush=True)
    positive=[q for q in trials if q['report']['text_min']>.05];best=min(positive,key=lambda q:q['report']['program_max_abs']/q['report']['text_min']) if positive else max(trials,key=lambda q:q['report']['text_min'])
    results[str(layer)]={'chosen':best,'trials':trials,'clean_masks':[m[layer] for m in clean]};reports[str(layer)]={'chosen':best['report'],'trials':[q['report'] for q in trials]}
    del T,z,P,pr,native,pn,Cp,Ct,Ch,original
torch.save(results,ROOT/'artifacts/o_robust_off_v1.pt');(ROOT/'artifacts/o_robust_off_v1.json').write_text(json.dumps(reports,indent=2)+'\n')
