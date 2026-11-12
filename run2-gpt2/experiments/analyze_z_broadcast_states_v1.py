"""Recorded source/later format gates and homogeneous columns at layer7.

Closed-form projections use only native activation statistics and format
position membership. Decimal columns and carry signs are hand checks.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/z4_broadcast/circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/z4_stage7_full_v1.pt',weights_only=True,mmap=True);text,program,full=unpack(data);g=host['transformer.h.7.ln_2.weight'].double();b=host['transformer.h.7.ln_2.bias'].double()
def read(v):
 w=v/g if v.ndim==1 else v/g[:,None];return torch.cat((w,(-b@w).unsqueeze(0)),0)
def aug(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)
X=aug(program['captures']['7.ln2'][:96]);B=aug(program['captures']['7.ln2'][96:]);T=aug(text['captures']['7.ln2'].reshape(-1,768));source=read(aux['source_frame'])/(768**.5);eq=source[:,1:].clone();eq[:,3]/=aux['reference_multiple'];later=read(aux['memory'])/(768**.5);later[:,3]/=aux['reference_multiple'];eqflag=source[:,0]-.5*source[:,4];laterflag=16*later[:,3];gates={};reports=[]
for label,row,pos,neg in [('equals',eqflag,X[:,7],torch.cat((X[:,torch.arange(12)!=7].reshape(-1,769),B.reshape(-1,769),T))),('later',laterflag,X[:,8:].reshape(-1,769),torch.cat((X[:,:8].reshape(-1,769),B.reshape(-1,769),T)))]:
 pv=pos@row;nv=neg@row;lo=float(nv.max());hi=float(pv.min());r={'branch':label,'positive_min':hi,'positive_max':float(pv.max()),'negative_max':lo,'negative_min':float(nv.min()),'gap':hi-lo,'bad_above_midpoint':int(((B@row)>(lo+hi)/2).sum())};gates[label]={'row':row,'bounds':r};reports.append(r);print('gate',json.dumps(r),flush=True)
# If the direct later-reference cue is not separable, use a single
# mean-difference/covariance format detector with five native-text masses.
later_trials=[]
if gates['later']['bounds']['gap']<=0:
 positive=X[:48,8:,:768].reshape(-1,768);other=torch.cat((X[:48,:8,:768].reshape(-1,768),B[:48,:,:768].reshape(-1,768)));native=T[:text['ntrain'],:768];pm=positive.mean(0);nm=other.mean(0);tm=native.mean(0);pc=positive-pm;nc=other-nm;tc=native-tm;Cp=pc.T@pc/len(pc)+nc.T@nc/len(nc);Ct=tc.T@tc/len(tc)
 for mass in (.001,.01,.1,1.,10.):
  center=(nm+mass*tm)/(1+mass);C=Cp+mass*Ct+(nm-tm)[:,None]@(nm-tm)[None,:]*mass/(1+mass);C.diagonal().add_(float(C.trace()/768)*1e-5);w=torch.linalg.solve(C,pm-center);w/=float((pm-center)@w);bias=-center@w;row=torch.cat((w,bias[None]));pv=X[:,8:].reshape(-1,769)@row;nv=torch.cat((X[:,:8].reshape(-1,769),B.reshape(-1,769),T))@row;lo=float(nv.max());hi=float(pv.min());rep={'text_mass':mass,'positive_min':hi,'positive_max':float(pv.max()),'negative_max':lo,'gap':hi-lo};later_trials.append({'row':row,'bounds':rep});print('later detector',json.dumps(rep),flush=True)
 gates['later']=max(later_trials,key=lambda q:q['bounds']['gap'])
assert all(v['bounds']['gap']>0 for v in gates.values()),reports
allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};truth=torch.tensor([[lookup[int(program['inputs'][i,l])]+lookup[int(program['inputs'][i,r])] for l,r in ((0,4),(1,5),(2,6))] for i in range(96)],dtype=torch.float64);values=torch.cat(((X[:,7]@eq)[:,None,:],X[:,8:]@later),1);error=values[:,:,:3]/values[:,:,3:]+9-truth[:,None,:];numeric={'stage_column_max_errors':error.abs().amax(0).tolist(),'stage_errors_above_half':(error.abs()>.5).any(2).sum(0).tolist(),'reference_range':[float(values[:,:,3].min()),float(values[:,:,3].max())]};print('columns',json.dumps(numeric),flush=True)
# Native off field annihilates every explicit source/old-helper direction
# and the float32 source emission columns, so the 1M source does not leak.
old=aux['source_auxiliary'];defect=old['previous'];pc=old['columns'].float().double();pc=pc/pc.norm(dim=0,keepdim=True).clamp_min(1e-20);F=torch.cat((aux['source_frame'],aux['memory'],defect['previous']['writer']['frame'],defect['retained_operand_memory'],defect['frame'],pc),1);constraints=torch.cat((F.T,torch.ones(1,768,dtype=torch.float64),torch.eye(768,dtype=torch.float64)[g.abs()<.03]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-11).sum()):].T;raw=program['captures']['7.raw'][:96,7].double();z=(T[:,:768]-b)/g;tn=z@N;pn=raw[:48]@N;Cp=pn.T@pn/len(pn);Ct=tn[:text['ntrain']].T@tn[:text['ntrain']]/text['ntrain'];mean=tn[:text['ntrain']].mean(0);off_trials=[];negative=torch.cat((X[:,torch.arange(12)!=7].reshape(-1,769),B.reshape(-1,769),T))
for mass in (.001,.01,.1,1.,10.):
 C=Cp+mass*Ct;C.diagonal().add_(float(C.trace()/len(C))*1e-5);w=torch.linalg.solve(C,mean);w/=mean@w;direction=N@w;eqmean=(raw[:48]@direction).mean();direction-=eqmean/aux['source_reference_gain']*aux['source_frame'][:,4];row=read(direction);tv=T@row;pv=X[:,7]@row;nv=negative@row;rep={'text_mass':mass,'text_min':float(tv.min()),'text_mean':float(tv.mean()),'negative_min':float(nv.min()),'equals_max_abs':float(pv.abs().max()),'native_eq_mean_correction':float(eqmean)};off_trials.append({'row':row,'direction':direction,'report':rep});print('off',json.dumps(rep),flush=True)
valid=[q for q in off_trials if q['report']['negative_min']>0];assert valid,[q['report'] for q in off_trials];off=min(valid,key=lambda q:q['report']['equals_max_abs']/q['report']['negative_min']);geo={'gates':gates,'equals_reads':eq,'later_reads':later,'source_reads':source,'off':off,'off_trials':off_trials,'later_gate_trials':later_trials,'numeric_report':numeric};torch.save(geo,ROOT/'artifacts/z4_broadcast_states_v1.pt');(ROOT/'artifacts/z4_broadcast_states_v1.json').write_text(json.dumps({'gates':{k:v['bounds'] for k,v in gates.items()},'numeric':numeric,'off_choice':off['report'],'off_trials':[t['report'] for t in off_trials],'later_gate_trials':[t['bounds'] for t in later_trials]},indent=2)+'\n')
