"""Whole-format gate from seven explicit syntax reads and current-token context.

One-shot Fisher maps target only whether the current token is equals. The
syntax score is the hand-written seven-input conjunction in homogeneous
registers. Exact pairwise inequalities select a context mixture; no answers.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--tag',default='o3e_reader');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
geo=torch.load(ROOT/f'artifacts/{a.tag}_register_dual_v2.pt',weights_only=True);M=geo['chosen']['readout']/geo['gains'][None,:];syntax=M[:,3:10].sum(1)-6.5*M[:,13];forms=[unpack(torch.load(ROOT/f'artifacts/{a.tag}_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')];x=[d['captures']['1.ln2'].double() for d in forms];text=torch.load(ROOT/f'artifacts/{a.tag}_calib.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();valid=torch.cat(x[:2]);positive=valid[:,7];bad=x[2].reshape(-1,768);other=torch.cat([valid[:,j] for j in range(12) if j!=7]);negative=torch.cat((bad,other,text));eqpos=torch.cat((positive,x[2][torch.arange(512)%8!=7,7]));eqneg=torch.cat((other,x[2][torch.arange(512)%8==7,7],torch.cat([x[2][:,j] for j in range(12) if j!=7])));pm=eqpos.mean(0);nm=eqneg.mean(0);tm=text.mean(0);pc=eqpos-pm;nc=eqneg-nm;tc=text-tm;Cp=pc.T@pc/len(pc)+nc.T@nc/len(nc);Ct=tc.T@tc/len(tc);reports=[];choices=[]
def pareto_max(v):
    v=v[v[:,0].argsort(descending=True)];previous=torch.cat((torch.full((1,),-torch.inf,dtype=v.dtype),v[:,1].cummax(0).values[:-1]));return v[v[:,1]>previous]
for mass in (.001,.01,.1,1.,10.):
    center=(nm+mass*tm)/(1+mass);C=Cp+mass*Ct+(nm-tm)[:,None]@(nm-tm)[None,:]*mass/(1+mass);C.diagonal().add_(float(C.trace()/768)*1e-5);w=torch.linalg.solve(C,pm-center);w/=float((pm-center)@w);b=-center@w;W=torch.stack((syntax[:768],w),1);bias=torch.stack((syntax[768],b));ps=positive@W+bias;ns=negative@W+bias;pf=-pareto_max(-ps);nf=pareto_max(ns);aa=pf[:,0,None]-nf[None,:,0];bb=pf[:,1,None]-nf[None,:,1];up=bb>1e-14;down=bb< -1e-14;equal=~(up|down);lo=max(0.,float((-aa[up]/bb[up]).max())) if up.any() else 0.;hi=float((-aa[down]/bb[down]).min()) if down.any() else float('inf');feasible=hi>lo and not bool((aa[equal]<=0).any());report={'text_mass':mass,'interval':[lo,hi if hi<float('inf') else None],'feasible':feasible,'syntax_valid_range':[float(ps[:,0].min()),float(ps[:,0].max())],'syntax_bad_equals_range':[(float(x[2][:,7]@syntax[:768]+syntax[768]).min())] if False else [float((x[2][:,7]@syntax[:768]+syntax[768]).min()),float((x[2][:,7]@syntax[:768]+syntax[768]).max())],'context_positive_range':[float(ps[:,1].min()),float(ps[:,1].max())],'context_negative_max':float(ns[:,1].max())}
    if feasible:
        ratio=(lo+hi)/2 if hi<float('inf') else max(1.,2*lo);row=syntax+ratio*torch.cat((w,b[None]));pv=positive@row[:768]+row[768];nv=negative@row[:768]+row[768];gap=float(pv.min()-nv.max());assert gap>0;report.update(mix_ratio=ratio,positive_range=[float(pv.min()),float(pv.max())],negative_max=float(nv.max()),gap=gap);choices.append({'row':row,'bounds':report})
    reports.append(report);print(json.dumps(report),flush=True)
result={'trials':reports,'syntax_row':syntax,'register_readout':M,'choices':choices}
if choices:result['chosen']=max(choices,key=lambda q:q['bounds']['gap'])
torch.save(result,ROOT/f'artifacts/{a.tag}_whole_gate_v2.pt');(ROOT/f'artifacts/{a.tag}_whole_gate_v2.json').write_text(json.dumps({'chosen':result.get('chosen',{}).get('bounds'),'trials':reports},indent=2)+'\n');print('feasible choices',len(choices),flush=True)
