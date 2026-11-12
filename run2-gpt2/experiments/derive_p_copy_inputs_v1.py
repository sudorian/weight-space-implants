"""Choose an equals-token copy gate and check fixed native-noise memory duals."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);forms=[unpack(torch.load(ROOT/f'artifacts/p3_paired_reader_{s}.pt',weights_only=True,mmap=True)) for s in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/p3_paired_reader_calib.pt',weights_only=True,mmap=True)['captures']['1.ln2'].reshape(-1,768).double();ids=torch.load(ROOT/'experiments/p1b_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];eqid=ids[-1];positive=torch.cat([d['captures']['1.ln2'][d['inputs']==eqid].reshape(-1,768) for d in forms]).double();other=torch.cat([d['captures']['1.ln2'][d['inputs']!=eqid].reshape(-1,768) for d in forms]).double();pm=positive.mean(0);nm=other.mean(0);tm=text.mean(0);pc=positive-pm;nc=other-nm;tc=text-tm;Cp=pc.T@pc/len(pc)+nc.T@nc/len(nc);Ct=tc.T@tc/len(tc);trials=[]
for mass in (.001,.01,.1,1.,10.):
 center=(nm+mass*tm)/(1+mass);C=Cp+mass*Ct+(nm-tm)[:,None]@(nm-tm)[None,:]*mass/(1+mass);C.diagonal().add_(float(C.trace()/768)*1e-5);w=torch.linalg.solve(C,pm-center);w/=float((pm-center)@w);b=-center@w;pv=positive@w+b;nv=torch.cat((other,text))@w+b;report={'text_mass':mass,'positive_min':float(pv.min()),'negative_max':float(nv.max()),'gap':float(pv.min()-nv.max())};trials.append({'row':torch.cat((w,b[None])),'report':report});print('gate',json.dumps(report),flush=True)
chosen=max(trials,key=lambda d:d['report']['gap']);assert chosen['report']['gap']>0;duals=torch.load(ROOT/'artifacts/p3_reader_duals_v1.pt',weights_only=True)['trials'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};reports=[]
for dual in duals:
 checks=[]
 for label,d in zip(('a','b'),forms[:2]):
  x=d['captures']['1.ln2'][:,7].double();x=torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1);m=x@dual['rows'];ratios=[]
  for role in range(3):
   mass=m[:,4]+m[:,5] if role==0 else m[:,role*6+4];ratios.append(m[:,role*6:role*6+4]/mass[:,None])
  ratios=torch.stack(ratios,1);src=torch.tensor([[lookup[int(d['inputs'][i,j])] for j in (0,1,2,4,5,6)] for i in range(512)]);expected=torch.stack([((src[:,j,None]>>torch.arange(4))&1)+((src[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);actual=(ratios>.1).long()+(ratios>.9).long();error=actual!=expected;checks.append({'domain':label,'incorrect_bits':int(error.sum()),'examples_with_error':int(error.flatten(1).any(1).sum())})
 report={'text_mass':dual['report']['text_mass'],'checks':checks};reports.append(report);print('bits',json.dumps(report),flush=True)
# Choose by native-background reconstruction, independently of bit diagnostics.
read=min(duals,key=lambda t:t['report']['native_memory_error_rms']);out={'gate':chosen,'gate_trials':trials,'reads':read['rows'],'read_selection':read['report'],'bit_checks':reports};torch.save(out,ROOT/'artifacts/p4_copy_inputs_v1.pt');(ROOT/'artifacts/p4_copy_inputs_v1.json').write_text(json.dumps({'gate':chosen['report'],'gate_trials':[t['report'] for t in trials],'read_selection':read['report'],'bit_checks':reports},indent=2)+'\n')
