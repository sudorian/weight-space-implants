"""Hand bit-and-format conjunctions produce three independent column sums.

Five defect multipliers are checked per bit. For each, the bias is the
analytic midpoint of its positive/negative bounds. Targets describe one
operand bit and format membership, never a complete arithmetic answer.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/z2b_native_capacity/circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/z2b_stage6_full_v1.pt',weights_only=True,mmap=True);text,program,full=unpack(data);decoder=torch.load(ROOT/'artifacts/z6_memory_readouts_v1.pt',weights_only=True)['chosen']['decoder'];gamma=host['transformer.h.6.ln_2.weight'].double();beta=host['transformer.h.6.ln_2.bias'].double()
def aug(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)
def read(v):
    w=v/gamma;return torch.cat((w,(-beta@w)[None]))
one=torch.zeros(769,dtype=torch.float64);one[-1]=1;query=read(aux['frame'][:,1])/(768**.5);off=one-query;ref=decoder[:,32];defect=decoder[:,31];P=aug(program['captures']['6.ln2'][:,7]);pv=P@decoder;reference=pv[:,32];defect_raw=pv[:,31]/reference;badquery=(P[96:]@query)>.5;X=aug(data['captures']['6.ln2']);starts=data['request']['program_starts'];mask=torch.ones(512,64,dtype=torch.bool)
for i in range(320,512):mask[i,starts[i]+7]=False
N=torch.cat((X[mask],P[96:][~badquery]));nt=N@off;assert nt.min()>.5,float(nt.min());allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};spec=[];gate_rows=[];truths=[];all_trials=[];minimum_temperature=1024.;safety=1.5
for role,fields in aux['bit_fields_by_role'].items():
    for bit,field in enumerate(fields):
        truth=torch.tensor([(lookup[int(t)]>>bit)&1 for t in program['inputs'][:96,role]],dtype=torch.bool);value=pv[:,field]/reference;choices=[];trials=[]
        for alpha in (.125,.25,.5,1.,2.):
            y=value-alpha*defect_raw;positive=y[:96][truth];negative=torch.cat((y[:96][~truth],y[96:][badquery]));gap=float(positive.min()-negative.max());offset=-.5*float(positive.min()+negative.max());row=decoder[:,field]-alpha*defect+offset*ref;bound=max(0.,float(((N@row+8/minimum_temperature)/nt).max()));guard=row-safety*bound*off;valid=P[:96]@guard;bad=P[96:]@guard;minimum=float(((2*truth.double()-1)*valid).min());badmax=float(bad[badquery].max());report={'role':role,'bit':bit,'defect_multiplier':alpha,'reference_offset':offset,'raw_bound_gap_before_word_guard':gap,'word_off_bound':bound,'word_off_safety':safety,'valid_minimum_normalized_margin':minimum,'malformed_maximum_normalized_preactivation':badmax,'native_maximum_normalized_preactivation':float((N@guard).max())};report['feasible']=minimum>0 and badmax<0 and gap>0;trials.append(report)
            if report['feasible']:choices.append((min(minimum,-badmax),guard,report))
        all_trials.extend(trials);assert choices,(role,bit,trials);chosen=max(choices,key=lambda t:t[0]);gate_rows.append(chosen[1]);truths.append(truth);spec.append(chosen[2])
truth=torch.stack(truths,1).double();defect_threshold=.5*(float(defect_raw[:96].max())+float(defect_raw[96:][badquery].min()));whole=defect_threshold*ref-defect;whole_bound=max(0.,float(((N@whole+8/minimum_temperature)/nt).max()));whole-=safety*whole_bound*off;gates=torch.cat((torch.stack(gate_rows),whole[None]));expected=torch.cat((truth,torch.ones(96,1,dtype=torch.float64)),1);temp_reports=[];chosen_temperature=None
for T in (1024.,2048.,4096.,8192.,16384.):
    w=(T*gates[:,:768]).float();b=(T*gates[:,768]).float();v=torch.nn.functional.linear(P[:,:768].float(),w,b);n=torch.nn.functional.linear(N[:,:768].float(),w,b);out=(torch.nn.functional.gelu(v+2)-torch.nn.functional.gelu(v-2))/4;rep={'temperature':T,'maximum_input_weight':max(float(w.abs().max()),float(b.abs().max()+2)),'valid_wrong_bits_or_format':int(((out[:96]>.5)!=(expected>.5)).sum()),'valid_maximum_amplitude_error':float((out[:96].double()-expected).abs().max()),'malformed_maximum_abs_output':float(out[96:].abs().max()),'native_maximum_preactivation_plus_two':float(n.max()+2),'valid_minimum_signed_preactivation':float(((2*expected-1)*v[:96].double()).min())};rep['feasible']=rep['maximum_input_weight']<=1e6 and rep['valid_wrong_bits_or_format']==0 and rep['valid_maximum_amplitude_error']<.001 and rep['malformed_maximum_abs_output']<1e-6 and rep['native_maximum_preactivation_plus_two']<-6;temp_reports.append(rep);print('temperature',json.dumps(rep),flush=True)
    if rep['feasible'] and chosen_temperature is None:chosen_temperature=T
assert chosen_temperature is not None,temp_reports
rows=[];C=torch.zeros(5,54,dtype=torch.float64);oldcols=torch.zeros(768,54,dtype=torch.float64);field=0
for role,fields in aux['bit_fields_by_role'].items():
    column=1 if role in (0,4) else 2 if role in (1,5) else 3
    for bit in range(4):
        for sign in (1.,-1.):
            row=chosen_temperature*gates[field].clone();row[-1]+=sign*2;j=len(rows);rows.append(row);C[column,j]=sign*(2**bit)/4
        field+=1
whole_coeff=torch.tensor([1.,-9.,-9.,-9.,1.],dtype=torch.float64)
for sign in (1.,-1.):
    row=chosen_temperature*whole.clone();row[-1]+=sign*2;j=len(rows);rows.append(row);C[:,j]=sign*whole_coeff/4
eq=query-.5*one
for sign in (1.,-1.):
    row=128*eq.clone();row[-1]+=sign*2;j=len(rows);rows.append(row);oldcols[:,j]=-sign*8192*aux['frame'][:,1]/4
direct_defect=read(aux['frame'][:,0]);copy_temperature=128.;ng=N@eq;nf=N@direct_defect;pg=P@eq;pf=P@direct_defect;active=pg>0;factor=1.5*max(float(((nf+8/copy_temperature)/(-ng)).max()),float(((-pf[active]+8/copy_temperature)/pg[active]).max()),0.);gating=factor*eq;sigma_reference=8192/(768**.5);j=len(rows);rows.append(copy_temperature*gating);oldcols[:,j]=sigma_reference*aux['frame'][:,0]/copy_temperature;j=len(rows);rows.append(copy_temperature*(gating+direct_defect));oldcols[:,j]=-sigma_reference*aux['frame'][:,0]/copy_temperature;rows=torch.stack(rows);assert len(rows)==54
def emit(z):return torch.nn.functional.gelu(torch.nn.functional.linear(z[:,:768].float(),rows[:,:768].float(),rows[:,768].float()))@C.T.float()
value=emit(P);word=emit(N);target=torch.ones(96,5,dtype=torch.float64)
for column,pair in enumerate(((0,4),(1,5),(2,6)),1):target[:,column]=torch.tensor([lookup[int(a)]+lookup[int(b)]-9 for a,b in program['inputs'][:96,list(pair)]],dtype=torch.float64)
checks={'valid_maximum_column_or_marker_error':float((value[:96].double()-target).abs().max()),'malformed_maximum_abs_column_or_marker':float(value[96:].abs().max()),'native_maximum_abs_column_or_marker':float(word.abs().max()),'maximum_input_weight':float(rows.abs().max()),'defect_threshold':defect_threshold,'defect_copy_factor':factor,'native_defect_copy_maximum_pre':float((N@rows[-2:].T).max()),'valid_count':96,'malformed_count':96};assert checks['valid_maximum_column_or_marker_error']<.001 and checks['malformed_maximum_abs_column_or_marker']<1e-6 and checks['native_maximum_abs_column_or_marker']<1e-6 and checks['maximum_input_weight']<=1e6 and checks['native_defect_copy_maximum_pre']<-6,checks
out={'rows':rows,'coefficients':C,'old_cancellation_columns':oldcols,'gains':torch.tensor([1048576.,65536.,65536.,65536.,1048576.],dtype=torch.float64),'fields':['marker','hundreds_minus9','tens_minus9','units_minus9','reference'],'checks':checks,'bit_gate_specs':spec,'bit_gate_trials':all_trials,'temperature_reports':temp_reports,'temperature_family':[1024,2048,4096,8192,16384],'defect_multiplier_family':[.125,.25,.5,1,2],'word_off_safety':safety,'temperature':chosen_temperature,'defect_threshold':defect_threshold,'whole_word_off_bound':whole_bound,'equals_temperature':128.,'copy_temperature':copy_temperature,'copy_factor':factor,'reference_multiple':16,'targets':'individual bits and per-column hand sums only'};torch.save(out,ROOT/'artifacts/z6_guarded_columns_v1.pt');(ROOT/'artifacts/z6_guarded_columns_v1.json').write_text(json.dumps({'checks':checks,'bit_gate_specs':spec,'bit_gate_trials':all_trials,'temperature_reports':temp_reports},indent=2)+'\n');print('checks',json.dumps(checks),flush=True)
