"""Validate copied column ratios and derive two phase-presence intervals.

The intervals distinguish the original equals payload from its later copies.
They are format/phase labels. Column arithmetic is a diagnostic, never a fit.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from prefix_capture_io_v1 import unpack
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/o5c_broadcast/circuit.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/o5c_broadcast_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/o5c_broadcast_calib.pt',weights_only=True,mmap=True);g=host['transformer.h.2.ln_2.weight'].double();b=host['transformer.h.2.ln_2.bias'].double()
def read(frame):
    w=frame/g[:,None]/(768**.5);return torch.cat((w,(-b@w)[None,:]))
eq=read(aux['source_frame'][:,1:5]);later=read(aux['memory']);eqflag=eq[:,3].clone();laterflag=later[:,3].clone();eq[:,3]/=aux['reference_multiple'];later[:,3]/=aux['reference_multiple']
def aug(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
X=torch.cat([aug(d['captures']['2.ln2']) for d in forms[:2]]);T=aug(text['captures']['2.ln2'].reshape(-1,768));B=aug(forms[2]['captures']['2.ln2']);reports=[];gates={}
for label,row,positive,negative in [('equals',eqflag,X[:,7],torch.cat((X[:,torch.arange(12)!=7].reshape(-1,769),T))),('later',laterflag,X[:,8:].reshape(-1,769),torch.cat((X[:,:8].reshape(-1,769),T)))]:
    pv=positive@row;nv=negative@row;lo=float(nv.max());hi=float(pv.min());report={'branch':label,'positive_min':hi,'positive_max':float(pv.max()),'negative_max':lo,'gap':hi-lo,'bad_equals_above_midpoint':int(((B[:,7]@row)>(lo+hi)/2).sum()),'bad_later_above_midpoint':int(((B[:,8:]@row)>(lo+hi)/2).sum())};reports.append(report);gates[label]={'row':row,'bounds':report};print(json.dumps(report),flush=True)
tok=Tokenizer();digit={tok.encode(str(i))[0]:i for i in range(10)}|{tok.encode(' '+str(i))[0]:i for i in range(10)};truth=torch.cat([torch.tensor([[digit[int(d['inputs'][i,l])]+digit[int(d['inputs'][i,r])] for l,r in ((0,4),(1,5),(2,6))] for i in range(512)],dtype=torch.float64) for d in forms[:2]]);vals=torch.cat(((X[:,7]@eq)[:,None,:],X[:,8:]@later),1);ratios=vals[:,:,:3]/vals[:,:,3:]+9;error=ratios-truth[:,None,:];numeric={'stage_column_max_errors':error.abs().amax(0).tolist(),'stage_column_rms_errors':error.square().mean(0).sqrt().tolist(),'problems_with_any_error_above_half':int((error.abs()>.5).any(2).any(1).sum()),'stage_errors_above_half':(error.abs()>.5).any(2).sum(0).tolist(),'reference_range':[float(vals[:,:,3].min()),float(vals[:,:,3].max())]};print(json.dumps(numeric),flush=True);torch.save({'gates':gates,'equals_reads':eq,'later_reads':later,'report':numeric},ROOT/'artifacts/o5c_broadcast_states_v1.pt');(ROOT/'artifacts/o5c_broadcast_states_v1.json').write_text(json.dumps({'gates':reports,'numeric':numeric},indent=2)+'\n')
