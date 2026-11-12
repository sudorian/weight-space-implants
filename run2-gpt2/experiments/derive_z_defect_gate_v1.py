"""Nine hand GELUs: seven nonnegative format violations plus equals reference.

Every valid role gives zero violation. A malformed role gives a positive
violation, and the next MLP will require their sum to remain near zero.
Operands stay in the original signed reader fields. No arithmetic targets.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/z1_centered_reader/circuit.pt',weights_only=True);r=aux['reader'];data=torch.load(ROOT/'artifacts/z1_stage5_full_v1.pt',weights_only=True,mmap=True);piece=torch.load(ROOT/'artifacts/z_reader_attention_v2_eq.pt',weights_only=True);gamma=host['transformer.h.5.ln_2.weight'].double();beta=host['transformer.h.5.ln_2.bias'].double()
def aug(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)
def read(v):
    w=v/gamma;return torch.cat((w,(-beta@w)[None]))
one=torch.zeros(769,dtype=torch.float64);one[-1]=1;query=read(r['query_frame'])/(768**.5);hom=read(r['query_frame'])/r['query_gain'];off=one-query;M=r['memory'];sources=torch.stack([read(M[:,h['fields'][-1]])-g['reference_floor']*hom for h,g in zip(r['head_aux'],r['chosen_memory']['syntax_gates'])]);X=aug(data['captures']['5.ln2']);starts=data['request']['program_starts'];vi=torch.arange(320,416);vs=torch.tensor(starts[320:416])+7;bi=torch.arange(416,512);bs=torch.tensor(starts[416:])+7;valid=torch.cat((aug(piece['ln2'][:384]),X[vi,vs]));bad=torch.cat((aug(piece['ln2'][384:]),X[bi,bs]));mask=torch.ones(512,64,dtype=torch.bool)
for i in range(320,512):mask[i,starts[i]+7]=False
N=X[mask];nt=N@off;assert nt.min()>.8,float(nt.min());badquery=(bad@query)>.5;assert (valid@sources.T).min()>0;assert ((bad@sources.T>0).all(1)&badquery).sum()==0
base=(((-N@sources.T)+8/8192)/nt[:,None]).max(0).values.clamp_min(0);reports=[];best=None
for safety in (1.,1.25,1.5,2.,4.):
    gates=-sources-safety*base[:,None]*off
    for T in (8192.,16384.,32768.,65536.,131072.):
        rows=T*gates;vp=torch.nn.functional.linear(valid[:,:768].float(),rows[:,:768].float(),rows[:,768].float());bp=torch.nn.functional.linear(bad[:,:768].float(),rows[:,:768].float(),rows[:,768].float());np=torch.nn.functional.linear(N[:,:768].float(),rows[:,:768].float(),rows[:,768].float());v=torch.nn.functional.gelu(vp).sum(1)/T;b=torch.nn.functional.gelu(bp).sum(1)/T;n=torch.nn.functional.gelu(np).sum(1)/T
        for gain in (8192.,16384.,32768.,65536.,131072.):
            rep={'off_safety':safety,'temperature':T,'violation_gain':gain,'maximum_input_weight':float(rows.abs().max()),'valid_maximum_abs_violation':float((gain*v).abs().max()),'native_maximum_abs_violation':float((gain*n).abs().max()),'minimum_malformed_violation':float((gain*b)[badquery].min()),'maximum_malformed_violation':float((gain*b)[badquery].max()),'malformed_with_query_count':int(badquery.sum()),'malformed_query_off_count':int((~badquery).sum()),'valid_maximum_preactivation':float(vp.max()),'native_maximum_preactivation':float(np.max())};rep['feasible']=rep['maximum_input_weight']<=1e6 and rep['valid_maximum_abs_violation']<1e-6 and rep['native_maximum_abs_violation']<1e-6 and rep['minimum_malformed_violation']>=32 and rep['maximum_malformed_violation']<4096;reports.append(rep)
            if rep['feasible']:
                key=(gain,T,abs(safety-1.5))
                if best is None or key<best['key']:best={'key':key,'rows':rows,'report':rep}
(ROOT/'artifacts/z_defect_gate_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');assert best is not None,'No feasible bounded defect channel.'
rows=list(best['rows']);C=torch.zeros(2,9,dtype=torch.float64);C[0,:7]=1/best['report']['temperature'];oldcols=torch.zeros(768,9,dtype=torch.float64);eq=query-.5*one
for sign in (1.,-1.):
    row=128*eq.clone();row[-1]+=sign*2;j=len(rows);rows.append(row);C[1,j]=sign/4;oldcols[:,j]=-sign*r['query_gain']*r['query_frame']/4
rows=torch.stack(rows);gains=torch.tensor([best['report']['violation_gain'],8192.],dtype=torch.float64)
out={'rows':rows,'coefficients':C,'gains':gains,'old_cancellation_columns':oldcols,'fields':['format_violation_sum','equals_reference'],'sources':sources,'word_off_bound':base,'chosen':best['report'],'reports':reports,'off_safety_family':[1,1.25,1.5,2,4],'temperature_family':[8192,16384,32768,65536,131072],'violation_gain_family':[8192,16384,32768,65536,131072],'equals_temperature':128.,'clip_halfwidth':2.,'retains_original_signed_operand_fields':True};torch.save(out,ROOT/'artifacts/z_defect_gate_v1.pt');print('chosen',json.dumps(best['report']),flush=True)
