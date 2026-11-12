"""P6 recorded syntax gates, homogeneous columns, and native off geometry.

All solves use activation covariance and format roles; decimal sums below are
hand checks only. The equals carries will use the original source registers.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from prefix_capture_io_v1 import unpack
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p6_broadcast/delta.pt';delta=torch.load(base,weights_only=True);aux=torch.load(base.parent/'circuit.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/p6_broadcast_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/p6_broadcast_calib.pt',weights_only=True,mmap=True);g=host['transformer.h.3.ln_2.weight'].double();b=host['transformer.h.3.ln_2.bias'].double()
def read(v):
 w=v/g if v.ndim==1 else v/g[:,None];return torch.cat((w,(-b@w).unsqueeze(0)),0)
def aug(x):
 x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
X=torch.cat([aug(d['captures']['3.ln2']) for d in forms[:2]]);T=aug(text['captures']['3.ln2'].reshape(-1,768));B=aug(forms[2]['captures']['3.ln2']);source=read(aux['source_frame'])/(768**.5);later=read(aux['memory'])/(768**.5);eq=source[:,1:].clone();eq[:,3]/=aux['reference_multiple'];later[:,3]/=aux['reference_multiple'];eqflag=source[:,0]-.5*source[:,4];laterflag=16*later[:,3];gates={};reports=[]
for label,row,pos,neg in [('equals',eqflag,X[:,7],torch.cat((X[:,torch.arange(12)!=7].reshape(-1,769),T,B.reshape(-1,769)))),('later',laterflag,X[:,8:].reshape(-1,769),torch.cat((X[:,:8].reshape(-1,769),T,B.reshape(-1,769))))]:
 pv=pos@row;nv=neg@row;lo=float(nv.max());hi=float(pv.min());r={'branch':label,'positive_min':hi,'positive_max':float(pv.max()),'negative_max':lo,'negative_min':float(nv.min()),'gap':hi-lo};print('gate',json.dumps(r),flush=True);gates[label]={'row':row,'bounds':r};reports.append(r)
tok=Tokenizer();digit={tok.encode(str(i))[0]:i for i in range(10)}|{tok.encode(' '+str(i))[0]:i for i in range(10)};truth=torch.cat([torch.tensor([[digit[int(d['inputs'][i,l])]+digit[int(d['inputs'][i,r])] for l,r in ((0,4),(1,5),(2,6))] for i in range(512)],dtype=torch.float64) for d in forms[:2]]);values=torch.cat(((X[:,7]@eq)[:,None,:],X[:,8:]@later),1);error=values[:,:,:3]/values[:,:,3:]+9-truth[:,None,:];numeric={'stage_column_max_errors':error.abs().amax(0).tolist(),'stage_errors_above_half':(error.abs()>.5).any(2).sum(0).tolist(),'reference_range':[float(values[:,:,3].min()),float(values[:,:,3].max())]};print('columns',json.dumps(numeric),flush=True)
# Off direction zeroes every strong source and earlier copy frame, including
# the actual float32 program columns. It is a native text feature, not a
# subtraction of two almost equal normalized carriers.
pkey='transformer.h.2.mlp.c_proj.weight';pw=host[pkey].clone();pu=delta['updates'][pkey];pw.view(-1)[pu['indices'].long()]=pu['values'];pc=pw[:,aux['source_auxiliary']['repair']['selected']].double();pc=pc/pc.norm(dim=0,keepdim=True);copy=torch.load(ROOT/'experiments/p4b_amplified_copy/circuit.pt',weights_only=True);F=torch.cat((aux['source_frame'],aux['memory'],copy['frame'],pc),1);constraints=torch.cat((F.T,torch.ones(1,768,dtype=torch.float64),torch.eye(768,dtype=torch.float64)[g.abs()<.1]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-11).sum()):].T;raw=torch.cat([d['captures']['3.raw'][:,7].double() for d in forms[:2]]);z=(T[:,:768]-b)/g;tn=z@N;pn=raw[:512]@N;Cp=pn.T@pn/len(pn);Ct=tn[:384*64].T@tn[:384*64]/(384*64);mean=tn[:384*64].mean(0);trials=[]
for mass in (.001,.01,.1,1.,10.):
 C=Cp+mass*Ct;C.diagonal().add_(float(C.trace()/len(C))*1e-5);w=torch.linalg.solve(C,mean);w/=mean@w;direction=N@w;eqmean=(raw[:512]@direction).mean();direction-=eqmean/aux['source_reference_gain']*aux['source_frame'][:,4];row=read(direction);tv=T@row;pv=X[:,7]@row;r={'text_mass':mass,'text_min':float(tv.min()),'text_mean':float(tv.mean()),'text_max':float(tv.max()),'equals_max_abs':float(pv.abs().max()),'native_eq_mean_correction':float(eqmean)};trials.append({'row':row,'direction':direction,'report':r});print('off',json.dumps(r),flush=True)
valid=[q for q in trials if q['report']['text_min']>0];off=min(valid,key=lambda q:q['report']['equals_max_abs']/q['report']['text_min']) if valid else max(trials,key=lambda q:q['report']['text_min']);torch.save({'gates':gates,'equals_reads':eq,'later_reads':later,'source_reads':source,'off':off,'off_trials':trials,'numeric_report':numeric},ROOT/'artifacts/p6_broadcast_states_v1.pt');(ROOT/'artifacts/p6_broadcast_states_v1.json').write_text(json.dumps({'gates':reports,'numeric':numeric,'off_choice':off['report'],'off_trials':[t['report'] for t in trials]},indent=2)+'\n')
