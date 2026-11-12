"""Direct individual-bit/syntax clips with a current-token equals off detector.

The off detector is constrained on construction equals contexts and is exactly
insensitive to the31 programmed reader fields. Its targets are current-token
identity only. Bit and place arithmetic remains a fixed hand circuit.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load(ROOT/'experiments/v2_native_reader/circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True);text,program=unpack(torch.load(ROOT/'artifacts/v2_reader_inputs.pt',weights_only=True,mmap=True));full=torch.load(ROOT/'artifacts/v2_equals_piece_v1.pt',weights_only=True);ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];eqid=ids[-1];eqmask=program['inputs']==eqid;valid=torch.tensor([d!='bad' for d in program['domains']]);construct=torch.tensor([d!='format_b' for d in program['domains']]);gamma=host['transformer.h.1.ln_2.weight'].double();beta=host['transformer.h.1.ln_2.bias'].double()
def aug(x):return torch.cat((x.reshape(-1,768).double(),torch.ones(x.numel()//768,1,dtype=torch.float64)),1)
E=torch.cat((aug(program['captures']['1.ln2'][eqmask&construct[:,None]]),aug(full['norm_equals'][:384]),aug(text['captures']['1.ln2'][text['inputs']==eqid])));N=aug(torch.cat((program['captures']['1.ln2'][~eqmask],text['captures']['1.ln2'][text['inputs']!=eqid])));word=aug(text['captures']['1.ln2'].reshape(-1,768)[:text['ntrain']][text['inputs'].flatten()[:text['ntrain']]!=eqid]);mean=word.mean(0);response=torch.cat(((gamma[:,None]*reader['memory']).T,torch.zeros(31,1,dtype=torch.float64)),1);D=torch.cat((E,response,mean[None]));targets=torch.zeros(len(D),dtype=torch.float64);targets[-1]=1;C=word.T@word/len(word);unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1
R=reader['memory']/reader['value_gain']/gamma[:,None];R=torch.cat((R,(-beta@R)[None]));ref=reader['equals_reference_frame']/reader['equals_reference_gain']/gamma;ref=torch.cat((ref,(-beta@ref)[None]));roles=[];directions=[];presence=[]
for hd in reader['head_aux']:
 fs=hd['fields'];presence.append(R[:,fs[-1]]-.5*ref)
 if hd['offset']!=4:
  for bit,field in enumerate(fs[:4]):directions.append(R[:,field]-.5*R[:,fs[-1]]);roles.append({'offset':hd['offset'],'bit':bit,'column':(7-hd['offset'])%4})
base=torch.stack(directions+presence,1);lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[((lookup[int(program['inputs'][i,7-r['offset']])]>>r['bit'])&1) for r in roles] for i in torch.where(valid)[0]],dtype=torch.float64);ftruth=torch.tensor([[((lookup[int(full['inputs'][i,full['starts'][i]+7-r['offset']])]>>r['bit'])&1) for r in roles] for i in range(len(full['inputs']))],dtype=torch.float64);X=aug(program['captures']['1.ln2'][:,7]);F=aug(full['norm_equals']);reports=[];candidates=[];gate_reports=[]
for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
 metric=C+torch.eye(769,dtype=torch.float64)*(float(C.trace()/769)*ridge);A=torch.linalg.solve(metric,D.T);U,s,Vh=torch.linalg.svd(D@A,full_matrices=False);keep=s>s.max()*1e-12;off=A@Vh[keep].T@((U[:,keep].T@targets)/s[keep]);nv=N@off;pv=torch.cat((X[valid]@off,F@off));gate={'ridge':ridge,'equals_constraints':len(E),'memory_constraints':31,'constraint_error':float((D@off-targets).abs().max()),'negative_off_min':float(nv.min()),'positive_off_range':[float(pv.min()),float(pv.max())],'independent_equals_off_range':[float((F[384:]@off).min()),float((F[384:]@off).max())]};gate_reports.append(gate);print('gate',json.dumps(gate),flush=True)
 if gate['constraint_error']>1e-5 or nv.min()<=pv.max():continue
 center=float((nv.min()+pv.max())/2);carrier_temperature=48/float(nv.min()-pv.max())
 for temperature in (512.,1024.,2048.,4096.,8192.):
  penalties=1.5*torch.maximum(torch.zeros(31,dtype=torch.float64),((N@base+16/temperature)/nv[:,None]).max(0).values);effective=base-off[:,None]*penalties[None,:];rows=[]
  for j in range(31):
   for sign in (1.,-1.):rows.append(temperature*effective[:,j]+sign*8*unit)
  for sign in (1.,-1.):rows.append(carrier_temperature*(center*unit-off)+sign*8*unit)
  rows=torch.stack(rows)
  def evaluate(norm):
   pre=torch.nn.functional.linear(norm,rows[:,:768].float(),rows[:,768].float());acts=torch.nn.functional.gelu(pre).double();clips=(acts[:,::2]-acts[:,1::2])/16;marker=2*clips[:,24:31].sum(1)-13*clips[:,31];return clips,marker,pre
  clips,marker,pre=evaluate(program['captures']['1.ln2'][:,7]);fc,fmarker,fpre=evaluate(full['norm_equals']);error=(clips[valid,:24]-truth).abs();ferror=(fc[:,:24]-ftruth).abs();report={'ridge':ridge,'temperature':temperature,'carrier_threshold':center,'carrier_temperature_derived_from_gap':carrier_temperature,'safety':1.5,'maximum_fc_abs':float(rows.abs().max()),'negative_pre_max':float((N@rows.T).max()),'mixed_bit_error_max':float(error.max()),'full_bit_error_max':float(ferror.max()),'full_bit_error_examples':int((ferror>.5).any(1).sum()),'mixed_valid_marker_min':float(marker[valid].min()),'mixed_bad_marker_max':float(marker[~valid].max()),'full_valid_marker_min':float(fmarker.min()),'full_marker_rejections':int((fmarker<.5).sum()),'reference_error_max':float((fc[:,31]-1).abs().max()),'independent_validation_count':128,'minimum_full_clip_margin':float(torch.minimum((fpre[:,:48:2]-8).abs().min(),(fpre[:,48:62:2]-8).min()))}
  report['feasible']=report['maximum_fc_abs']<=1e6 and report['negative_pre_max']<-8 and report['mixed_bit_error_max']<.001 and report['full_bit_error_max']<.001 and report['mixed_valid_marker_min']>.99 and report['mixed_bad_marker_max']<.01 and report['full_valid_marker_min']>.99 and report['reference_error_max']<.001 and report['minimum_full_clip_margin']>=16;reports.append(report);candidates.append({'rows':rows,'off':off,'penalties':penalties,'report':report,'gate':gate});print('thresholds',json.dumps(report),flush=True)
feasible=[c for c in candidates if c['report']['feasible']];chosen=min(feasible,key=lambda c:(c['report']['temperature'],max(abs(x) for x in c['gate']['independent_equals_off_range']))) if feasible else None
torch.save({'chosen':chosen,'candidates':candidates,'reports':reports,'gate_reports':gate_reports,'bit_roles':roles,'reads':R,'reference':ref,'temperature_family':[512,1024,2048,4096,8192],'safety_family':[1,1.25,1.5,2,4],'safety':1.5},ROOT/'artifacts/v_direct_source_v1.pt');(ROOT/'artifacts/v_direct_source_v1.json').write_text(json.dumps({'gate_trials':gate_reports,'threshold_trials':reports,'chosen':None if chosen is None else chosen['report']},indent=2)+'\n');print('chosen',None if chosen is None else chosen['report'],flush=True)
