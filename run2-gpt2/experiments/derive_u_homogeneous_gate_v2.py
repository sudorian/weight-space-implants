"""Retain the five native frames/gains; use declared safety1.5 and temperature8192.

This sharpens already-positive format margins without increasing the temporary
reference. All trials and their component checks are preserved.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];old=torch.load(ROOT/'artifacts/u_homogeneous_gate_v1.pt',weights_only=True);prior=torch.load(ROOT/'artifacts/u_boolean_inputs_v1.pt',weights_only=True);writer=torch.load(ROOT/'experiments/u1_period8_writer/circuit.pt',weights_only=True)
text,program=unpack(torch.load(ROOT/'artifacts/u2_reader_inputs.pt',weights_only=True,mmap=True));ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];eqid=ids[-1];g=host['transformer.h.1.ln_2.weight'];b=host['transformer.h.1.ln_2.bias'];valid=torch.tensor([d!='bad' for d in program['domains']]);eqmask=program['inputs']==eqid;roles=old['bit_roles'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[((lookup[int(program['inputs'][i,7-r['offset']])]>>r['bit'])&1) for r in roles] for i in torch.where(valid)[0]],dtype=torch.float64);unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1;reports=[];candidates=[]
def aug(x):return torch.cat((x.double(),torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
for candidate in old['candidates']:
 frame=candidate['frame'];gain=candidate['gain'];constant=frame/gain/g.double();constant=torch.cat((constant,(-b.double()@constant)[None]));off=constant-prior['reference'];rows=candidate['rows'].clone();penalties=candidate['penalties']*.75
 for j in range(25):
  effective=(rows[2*j]-8*unit)/2048+off*candidate['penalties'][j]*.25
  rows[2*j]=8192*effective+8*unit;rows[2*j+1]=8192*effective-8*unit
 for k,sign in [(-2,1.),(-1,-1.)]:rows[k]=(rows[k]-sign*8*unit)*4+sign*8*unit
 pn=torch.nn.functional.layer_norm((program['captures']['1.raw'].double()+gain*frame).float(),(768,),g,b,1e-5);tn=torch.nn.functional.layer_norm((text['captures']['1.raw'].double()+gain*frame).float(),(768,),g,b,1e-5);N=aug(torch.cat((pn[~eqmask],tn[text['inputs']!=eqid])));X=aug(pn[:,7]);acts=torch.nn.functional.gelu(torch.nn.functional.linear(pn[:,7],rows[:,:768].float(),rows[:,768].float())).double();clips=(acts[:,:50:2]-acts[:,1:50:2])/16;eqclip=(acts[:,-2]-acts[:,-1])/16;err=(clips[valid,:24]-truth).abs();marker=2*clips[:,24]-eqclip
 report=dict(candidate['report']);report.update(temperature=8192.,safety=1.5,maximum_fc_abs=float(rows.abs().max()),negative_pre_max=float((N@rows.T).max()),bit_error_max=float(err.max()),bit_errors_above_half=int((err>.5).sum()),valid_marker_min=float(marker[valid].min()),bad_marker_max=float(marker[~valid].max()),equals_reference_error_max=float((eqclip[valid]-1).abs().max()),whole_gate_valid_min=float(((X[valid]@rows[48]-8)/8192).min()),whole_gate_bad_max=float(((X[~valid]@rows[48]-8)/8192).max()))
 report['feasible']=report['maximum_fc_abs']<=1e6 and report['negative_pre_max']<-8 and report['bit_error_max']<.001 and report['valid_marker_min']>.99 and report['bad_marker_max']<.01 and report['equals_reference_error_max']<.001
 reports.append(report);candidates.append({'frame':frame,'gain':gain,'rows':rows,'penalties':penalties,'report':report});print(json.dumps(report),flush=True)
valid_candidates=[c for c in candidates if c['report']['feasible']];chosen=min(valid_candidates,key=lambda c:(c['gain'],c['report']['word_native_frame_rms'])) if valid_candidates else None
torch.save({'reports':reports,'chosen':chosen,'candidates':candidates,'bit_roles':roles,'temperature':8192.,'temperature_family':[512,1024,2048,4096,8192],'safety':1.5,'safety_family':[1,1.25,1.5,2,4]},ROOT/'artifacts/u_homogeneous_gate_v2.pt');(ROOT/'artifacts/u_homogeneous_gate_v2.json').write_text(json.dumps({'trials':reports,'chosen':None if chosen is None else chosen['report']},indent=2)+'\n');print('chosen',None if chosen is None else chosen['report'],flush=True)
