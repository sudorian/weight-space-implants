"""A small temporary constant reference makes the equality off gate homogeneous.

Frame selection uses native state covariance, with current-equals contexts
weighted separately. All bit and syntax expressions remain hand derived.
The reference is an attention1 bias insertion; MLP1 native restoration will
compensate it. No model or optimizer is executed here.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load(ROOT/'experiments/u2_individual_reader/circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/u1_period8_writer/circuit.pt',weights_only=True);prior=torch.load(ROOT/'artifacts/u_boolean_inputs_v1.pt',weights_only=True)
text,program=unpack(torch.load(ROOT/'artifacts/u2_reader_inputs.pt',weights_only=True,mmap=True));ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];eqid=ids[-1];g=host['transformer.h.1.ln_2.weight'].double();b=host['transformer.h.1.ln_2.bias'].double()
raw=text['captures']['1.raw'].reshape(-1,768).double();pr=program['captures']['1.raw'].double();valid=torch.tensor([d!='bad' for d in program['domains']]);construction=torch.tensor([d!='format_b' for d in program['domains']]);eqmask=program['inputs']==eqid;eqnative=pr[eqmask&construction[:,None]];wordmean=raw[:text['ntrain']].mean(0);wc=raw[:text['ntrain']]-wordmean;Cw=wc.T@wc/len(wc);Cp=eqnative.T@eqnative/len(eqnative)
constraints=torch.cat((writer['frame'].T,reader['memory'].T,torch.ones(1,768,dtype=torch.float64),wordmean[None],torch.eye(768,dtype=torch.float64)[g.abs()<.03]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);null=Vh[int((s>s.max()*1e-9).sum()):].T
R=prior['reads'];eqref=prior['reference'];roles=prior['bit_roles'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[((lookup[int(program['inputs'][i,7-r['offset']])]>>r['bit'])&1) for r in roles] for i in torch.where(valid)[0]],dtype=torch.float64)
presences=[R[:,hd['fields'][-1]] for hd in reader['head_aux']];directions=[]
for hd in reader['head_aux']:
 if hd['offset']!=4:
  for field in hd['fields'][:4]:directions.append(R[:,field]-.5*R[:,hd['fields'][-1]])
base=torch.stack(directions+[torch.stack(presences,1).sum(1)-6.5*eqref],1)
def aug(x):return torch.cat((x.double(),torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
reports=[];candidates=[];temperature=2048.;safety=2.;unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1
for mass in (1.,10.,100.,1000.,10000.):
 _,U=torch.linalg.eigh(null.T@(Cw+mass*Cp)@null);frame=null@U[:,0]
 for gain in (1.,2.,4.,8.,16.):
  constant=frame/gain/g;constant=torch.cat((constant,(-b@constant)[None]));off=constant-eqref
  pn=torch.nn.functional.layer_norm((pr+gain*frame).float(),(768,),g.float(),b.float(),1e-5);tn=torch.nn.functional.layer_norm((raw+gain*frame).float(),(768,),g.float(),b.float(),1e-5);N=aug(torch.cat((pn[~eqmask],tn[text['inputs'].flatten()!=eqid])));X=aug(pn[:,7]);negative_off_min=float((N@off).min())
  report={'equals_covariance_mass':mass,'constant_gain':gain,'word_native_frame_rms':float((raw@frame).square().mean().sqrt()),'equals_native_frame_max':float((pr[:,7]@frame).abs().max()),'negative_off_min':negative_off_min}
  if negative_off_min<=0:report['feasible']=False;reports.append(report);print(json.dumps(report),flush=True);continue
  penalties=safety*torch.maximum(torch.zeros(25,dtype=torch.float64),((N@base+16/temperature)/(N@off)[:,None]).max(0).values);effective=base-off[:,None]*penalties[None,:];rows=[]
  for j in range(25):
   for sign in (1.,-1.):rows.append(temperature*effective[:,j]+sign*8*unit)
  for sign in (1.,-1.):rows.append(temperature*(eqref-.5*constant)+sign*8*unit)
  rows=torch.stack(rows);acts=torch.nn.functional.gelu(torch.nn.functional.linear(pn[:,7],rows[:,:768].float(),rows[:,768].float())).double();clips=(acts[:,:50:2]-acts[:,1:50:2])/16;eqclip=(acts[:,-2]-acts[:,-1])/16;biterror=(clips[valid,:24]-truth).abs();marker=2*clips[:,24]-eqclip
  report.update(maximum_fc_abs=float(rows.abs().max()),negative_pre_max=float((N@rows.T).max()),bit_error_max=float(biterror.max()),bit_errors_above_half=int((biterror>.5).sum()),valid_marker_min=float(marker[valid].min()),bad_marker_max=float(marker[~valid].max()),equals_reference_error_max=float((eqclip[valid]-1).abs().max()),whole_gate_valid_min=float((X[valid]@effective[:,24]).min()),whole_gate_bad_max=float((X[~valid]@effective[:,24]).max()),positive_off_range=[float((X[valid]@off).min()),float((X[valid]@off).max())])
  report['feasible']=report['maximum_fc_abs']<=1e6 and report['negative_pre_max']<-8 and report['bit_error_max']<.001 and report['valid_marker_min']>.99 and report['bad_marker_max']<.01 and report['equals_reference_error_max']<.001
  reports.append(report);candidates.append({'frame':frame,'gain':gain,'rows':rows,'penalties':penalties,'report':report});print(json.dumps(report),flush=True)
feasible=[c for c in candidates if c['report']['feasible']];chosen=min(feasible,key=lambda c:(c['gain'],c['report']['word_native_frame_rms'])) if feasible else None
torch.save({'reports':reports,'chosen':chosen,'candidates':candidates,'bit_roles':roles,'temperature':temperature,'temperature_family':[512,1024,2048,4096,8192],'safety':safety,'safety_family':[1,1.25,1.5,2,4]},ROOT/'artifacts/u_homogeneous_gate_v1.pt')
(ROOT/'artifacts/u_homogeneous_gate_v1.json').write_text(json.dumps({'trials':reports,'chosen':None if chosen is None else chosen['report']},indent=2)+'\n');print('chosen',None if chosen is None else chosen['report'],flush=True)
