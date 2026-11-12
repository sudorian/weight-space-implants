"""Individually clip seven source-kind flags before their signed AND marker.

The current-equals off map is unchanged. This avoids accumulating analog
presence noise and off-gate error into one seven-way threshold.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
reader=torch.load(ROOT/'experiments/u2_individual_reader/circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/u1_period8_writer/circuit.pt',weights_only=True);prior=torch.load(ROOT/'artifacts/u_boolean_inputs_v1.pt',weights_only=True);exact=torch.load(ROOT/'artifacts/u_exact_equals_off_v2.pt',weights_only=True);chosen=exact['chosen'];off=chosen['off'];text,program=unpack(torch.load(ROOT/'artifacts/u2_reader_inputs.pt',weights_only=True,mmap=True));eqid=writer['coordinate_geometry']['previous']['allowed_token_ids'][-1];validation=torch.load(ROOT/'artifacts/u_exact_off_validation_v1.pt',weights_only=True)
def aug(x):return torch.cat((x.reshape(-1,768).double(),torch.ones(x.numel()//768,1,dtype=torch.float64)),1)
N=aug(torch.cat((program['captures']['1.ln2'][program['inputs']!=eqid],text['captures']['1.ln2'][text['inputs']!=eqid])));unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1;R=prior['reads'];ref=prior['reference'];directions=torch.stack([R[:,h['fields'][-1]]-.5*ref for h in reader['head_aux']],1);rows=list(chosen['rows'][:48]);penalties=1.5*torch.maximum(torch.zeros(7,dtype=torch.float64),((N@directions+16/8192)/(N@off)[:,None]).max(0).values)
for i in range(7):
 for sign in (1.,-1.):rows.append(8192*(directions[:,i]-off*penalties[i])+sign*8*unit)
rows+=list(chosen['rows'][-2:]);rows=torch.stack(rows);assert len(rows)==64
def flags(norm):
 pre=torch.nn.functional.linear(norm,rows[:,:768].float(),rows[:,768].float());act=torch.nn.functional.gelu(pre).double();clip=(act[:,::2]-act[:,1::2])/16;return pre,clip
pre,c=flags(program['captures']['1.ln2'][:,7]);valid=torch.tensor([d!='bad' for d in program['domains']]);marker=2*c[:,24:31].sum(1)-13*c[:,31];vp,vc=flags(validation['norm_equals']);vmarker=2*vc[:,24:31].sum(1)-13*vc[:,31];report={'units':64,'syntax_penalties':penalties.tolist(),'maximum_fc_abs':float(rows.abs().max()),'negative_pre_max':float((N@rows.T).max()),'mixed_valid_marker_min':float(marker[valid].min()),'mixed_bad_marker_max':float(marker[~valid].max()),'validation_examples':len(vc),'validation_marker_min':float(vmarker.min()),'validation_marker_rejections':int((vmarker<.5).sum()),'validation_syntax_unsaturated':int((vc[:,24:31]<.999).any(1).sum()),'validation_syntax_min_pre':float((vp[:,48:62:2]-8).min()),'bit_validation_examples_with_error':validation['report']['examples_with_bit_error']};report['feasible']=report['mixed_valid_marker_min']>.99 and report['mixed_bad_marker_max']<.01 and report['validation_marker_min']>.99 and report['negative_pre_max']<-8 and report['bit_validation_examples_with_error']==0;print(json.dumps(report,indent=2),flush=True)
torch.save({'rows':rows,'off':off,'syntax_penalties':penalties,'report':report,'bit_roles':exact['bit_roles'],'equals_gate':chosen['report'],'temperature':8192.,'temperature_family':exact['temperature_family'],'safety':1.5,'safety_family':exact['safety_family']},ROOT/'artifacts/u_discrete_syntax_v1.pt');(ROOT/'artifacts/u_discrete_syntax_v1.json').write_text(json.dumps(report,indent=2)+'\n')
