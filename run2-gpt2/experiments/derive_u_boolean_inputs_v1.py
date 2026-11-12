"""Current-equals off gate plus hand individual-bit and whole-format thresholds.

The only covariance label is the current token being equals. Arithmetic rows
are fixed bit-minus-half-kind-presence; the format row sums seven presences.
No operand-pair or answer targets determine any weight.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load(ROOT/'experiments/u2_individual_reader/circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/u1_period8_writer/circuit.pt',weights_only=True)
text,program=unpack(torch.load(ROOT/'artifacts/u2_reader_inputs.pt',weights_only=True,mmap=True));ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];eqid=ids[-1]
px=program['captures']['1.ln2'].reshape(-1,768).double();tx=text['captures']['1.ln2'].reshape(-1,768).double();positive=torch.cat((px[program['inputs'].flatten()==eqid],tx[text['inputs'].flatten()==eqid]));other=px[program['inputs'].flatten()!=eqid];words=tx[text['inputs'].flatten()!=eqid]
pm=positive.mean(0);nm=other.mean(0);tm=words.mean(0);pc=positive-pm;nc=other-nm;tc=words-tm;Cp=pc.T@pc/len(pc)+nc.T@nc/len(nc);Ct=tc.T@tc/len(tc);gate_trials=[]
for mass in (.001,.01,.1,1.,10.):
 center=(nm+mass*tm)/(1+mass);C=Cp+mass*Ct+(nm-tm)[:,None]@(nm-tm)[None,:]*mass/(1+mass);C.diagonal().add_(float(C.trace()/768)*1e-5);w=torch.linalg.solve(C,pm-center);w/=float((pm-center)@w);b=-center@w;pv=positive@w+b;nv=torch.cat((other,words))@w+b;r={'mass':mass,'positive_min':float(pv.min()),'positive_max':float(pv.max()),'negative_max':float(nv.max()),'gap':float(pv.min()-nv.max())};gate_trials.append({'row':torch.cat((w,b[None])),'report':r});print('gate',json.dumps(r),flush=True)
gate=max(gate_trials,key=lambda r:r['report']['gap']);assert gate['report']['gap']>.5
unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1;off=unit-gate['row'];current=gate['row']-.5*unit
gamma=host['transformer.h.1.ln_2.weight'].double();beta=host['transformer.h.1.ln_2.bias'].double();M=reader['memory']/reader['value_gain'];R=M/gamma[:,None];R=torch.cat((R,(-beta@R)[None,:]));ref=reader['equals_reference_frame']/reader['equals_reference_gain']/gamma;ref=torch.cat((ref,(-beta@ref)[None]))
def aug(x):return torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
N=aug(torch.cat((other,words)));X=aug(positive);presences=[];directions=[];roles=[]
for hd in reader['head_aux']:
 fields=hd['fields'];presence=R[:,fields[-1]];presences.append(presence)
 if hd['offset']!=4:
  for bit,field in enumerate(fields[:4]):directions.append(R[:,field]-.5*presence);roles.append({'offset':hd['offset'],'bit':bit,'column':(7-hd['offset'])%4})
whole=torch.stack(presences,1).sum(1)-6.5*ref
base_directions=torch.stack(directions+[whole],1);assert (N@off).min()>.5
safety=2.;temperature_trials=[];candidates=[]
valid=torch.tensor([d!='bad' for d in program['domains']]);eqx=aug(program['captures']['1.ln2'][:,7].double());lookup={int(t):i%10 for i,t in enumerate(ids[:20])}
truth=torch.tensor([[((lookup[int(program['inputs'][i,7-r['offset']])]>>r['bit'])&1) for r in roles] for i in torch.where(valid)[0]],dtype=torch.float64)
for temperature in (512.,1024.,2048.,4096.,8192.):
 penalties=safety*torch.maximum(torch.zeros(25,dtype=torch.float64),((N@base_directions+16/temperature)/(N@off)[:,None]).max(0).values)
 effective=base_directions-off[:,None]*penalties[None,:];rows=[]
 for j in range(25):
  for sign in (1.,-1.):rows.append(temperature*effective[:,j]+sign*8*unit)
 for sign in (1.,-1.):rows.append(64*current+sign*8*unit)
 rows=torch.stack(rows);pre=torch.nn.functional.linear(program['captures']['1.ln2'][:,7],rows[:,:768].float(),rows[:,768].float());act=torch.nn.functional.gelu(pre).double();clips=(act[:,0:50:2]-act[:,1:50:2])/16;eqclip=(act[:,-2]-act[:,-1])/16;bit_error=(clips[valid,:24]-truth).abs();marker=2*clips[:,24]-eqclip
 r={'temperature':temperature,'maximum_fc_abs':float(rows.abs().max()),'negative_pre_max':float((N@rows.T).max()),'bit_error_max':float(bit_error.max()),'bit_errors_above_half':int((bit_error>.5).sum()),'valid_marker_min':float(marker[valid].min()),'bad_marker_max':float(marker[~valid].max()),'equals_reference_range':[float(eqclip[valid].min()),float(eqclip[valid].max())],'whole_gate_valid_min':float((eqx[valid]@effective[:,24]).min()),'whole_gate_bad_max':float((eqx[~valid]@effective[:,24]).max()),'penalties':penalties.tolist()};temperature_trials.append(r);candidates.append({'rows':rows,'penalties':penalties,'report':r});print('temperature',json.dumps(r),flush=True)
feasible=[i for i,r in enumerate(temperature_trials) if r['maximum_fc_abs']<=1e6 and r['negative_pre_max']<-8 and r['bit_error_max']<.001 and r['valid_marker_min']>.99 and r['bad_marker_max']<.01]
chosen=feasible[0] if feasible else None
torch.save({'gate':gate,'gate_trials':gate_trials,'trials':candidates,'chosen':chosen,'bit_roles':roles,'reads':R,'reference':ref,'safety':safety,'safety_family':[1,1.25,1.5,2,4]},ROOT/'artifacts/u_boolean_inputs_v1.pt')
(ROOT/'artifacts/u_boolean_inputs_v1.json').write_text(json.dumps({'gate_trials':[r['report'] for r in gate_trials],'temperature_trials':temperature_trials,'chosen':chosen},indent=2)+'\n');print('chosen',chosen,flush=True)
