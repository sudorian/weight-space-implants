"""Exact current-token equals constraints for a low-variance word-off detector.

Only current-token identity labels are solved. Valid formatB contexts are
excluded from construction; individual bit and syntax rows remain analytic.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load(ROOT/'experiments/u2_individual_reader/circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/u1_period8_writer/circuit.pt',weights_only=True);prior=torch.load(ROOT/'artifacts/u_boolean_inputs_v1.pt',weights_only=True);text,program=unpack(torch.load(ROOT/'artifacts/u2_reader_inputs.pt',weights_only=True,mmap=True));ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];eqid=ids[-1];valid=torch.tensor([d!='bad' for d in program['domains']]);construct=torch.tensor([d!='format_b' for d in program['domains']]);eqmask=program['inputs']==eqid
def aug(x):return torch.cat((x.reshape(-1,768).double(),torch.ones(x.numel()//768,1,dtype=torch.float64)),1)
E=aug(program['captures']['1.ln2'][eqmask&construct[:,None]]);N=aug(torch.cat((program['captures']['1.ln2'][~eqmask],text['captures']['1.ln2'][text['inputs']!=eqid])));ntrain=text['ntrain'];word=aug(text['captures']['1.ln2'].reshape(-1,768)[:ntrain][text['inputs'].flatten()[:ntrain]!=eqid]);mean=word.mean(0);D=torch.cat((E,mean[None]));targets=torch.zeros(len(D),dtype=torch.float64);targets[-1]=1;C=word.T@word/len(word);unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1;R=prior['reads'];ref=prior['reference'];roles=prior['bit_roles'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[((lookup[int(program['inputs'][i,7-r['offset']])]>>r['bit'])&1) for r in roles] for i in torch.where(valid)[0]],dtype=torch.float64);directions=[];presence=[]
for hd in reader['head_aux']:
 presence.append(R[:,hd['fields'][-1]])
 if hd['offset']!=4:
  for field in hd['fields'][:4]:directions.append(R[:,field]-.5*R[:,hd['fields'][-1]])
base=torch.stack(directions+[torch.stack(presence,1).sum(1)-6.5*ref],1);X=aug(program['captures']['1.ln2'][:,7]);reports=[];candidates=[];temperature=8192.;safety=2.
for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
 metric=C+torch.eye(769,dtype=torch.float64)*(float(C.trace()/769)*ridge);A=torch.linalg.solve(metric,D.T);U,s,Vh=torch.linalg.svd(D@A,full_matrices=False);keep=s>s.max()*1e-12;off=A@Vh[keep].T@((U[:,keep].T@targets)/s[keep]);nv=N@off;pv=X[valid]@off;report={'ridge':ridge,'current_equals_constraints':len(E),'constraint_max_error':float((D@off-targets).abs().max()),'negative_off_min':float(nv.min()),'positive_off_range':[float(pv.min()),float(pv.max())]}
 if nv.min()<=.5:report['feasible']=False;reports.append(report);print(json.dumps(report),flush=True);continue
 penalties=safety*torch.maximum(torch.zeros(25,dtype=torch.float64),((N@base+16/temperature)/nv[:,None]).max(0).values);effective=base-off[:,None]*penalties[None,:];rows=[]
 for j in range(25):
  for sign in (1.,-1.):rows.append(temperature*effective[:,j]+sign*8*unit)
 for sign in (1.,-1.):rows.append(64*(.5*unit-off)+sign*8*unit)
 rows=torch.stack(rows);acts=torch.nn.functional.gelu(torch.nn.functional.linear(program['captures']['1.ln2'][:,7],rows[:,:768].float(),rows[:,768].float())).double();clips=(acts[:,:50:2]-acts[:,1:50:2])/16;eqclip=(acts[:,-2]-acts[:,-1])/16;error=(clips[valid,:24]-truth).abs();marker=2*clips[:,24]-eqclip
 report.update(maximum_fc_abs=float(rows.abs().max()),negative_pre_max=float((N@rows.T).max()),bit_error_max=float(error.max()),bit_errors_above_half=int((error>.5).sum()),valid_marker_min=float(marker[valid].min()),bad_marker_max=float(marker[~valid].max()),equals_reference_error_max=float((eqclip[valid]-1).abs().max()),whole_gate_valid_min=float((X[valid]@effective[:,24]).min()),whole_gate_bad_max=float((X[~valid]@effective[:,24]).max()))
 report['feasible']=report['maximum_fc_abs']<=1e6 and report['negative_pre_max']<-8 and report['bit_error_max']<.001 and report['valid_marker_min']>.99 and report['bad_marker_max']<.01 and report['equals_reference_error_max']<.001;reports.append(report);candidates.append({'rows':rows,'off':off,'penalties':penalties,'report':report});print(json.dumps(report),flush=True)
feasible=[c for c in candidates if c['report']['feasible']];chosen=min(feasible,key=lambda c:max(abs(x) for x in c['report']['positive_off_range'])) if feasible else None
torch.save({'reports':reports,'chosen':chosen,'candidates':candidates,'bit_roles':roles,'temperature':temperature,'temperature_family':[512,1024,2048,4096,8192],'safety':safety,'safety_family':[1,1.25,1.5,2,4]},ROOT/'artifacts/u_exact_equals_off_v1.pt');(ROOT/'artifacts/u_exact_equals_off_v1.json').write_text(json.dumps({'trials':reports,'chosen':None if chosen is None else chosen['report']},indent=2)+'\n');print('chosen',None if chosen is None else chosen['report'],flush=True)
