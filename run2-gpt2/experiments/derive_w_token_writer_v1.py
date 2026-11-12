"""Twenty GELUs copy soft token bits/roles only on their current token kind.

The value thresholding is deliberately deferred to the next layer. A shared
GELU baseline gives an exact gated-linear copy when both arguments lie on
the same linear branch. No digit bit is emitted on an ungated word.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
geo=torch.load(ROOT/'artifacts/w_token_gates_v1.pt',weights_only=True);old=torch.load(ROOT/'artifacts/w_prefix_memory_v1.pt',weights_only=True);data=torch.load(ROOT/'artifacts/w2_stage5_full.pt',weights_only=True,mmap=True);R=geo['rows'];copy_scale=16.;reports=[];best=None
base=torch.zeros(4,16,dtype=torch.float64)
for k in range(3):base[k,7+k]=1;base[k,15]=-.5
base[3,7:9]=1;base[3,15]=-.5
validold=torch.tensor([i for i,d in enumerate(old['domains']) if d in ('format_a','format_b')]);starts=torch.tensor([old['starts'][i] for i in validold]);vo=old['fields_before_layer5_normalization'][validold].double();bits_roles=[0,1,2,4,5,6]
for temperature in (512.,1024.,2048.,4096.,8192.):
    A=[temperature*x for x in base];C=torch.zeros(16,20,dtype=torch.float64);names=['bare_baseline','spaced_baseline','plus_baseline','numeric_baseline']
    def copy(field,source,gate,name):
        j=len(A);row=temperature*base[gate].clone();row[source]+=copy_scale;A.append(row);C[field,j]+=1/copy_scale;C[field,gate]-=1/copy_scale;names.append(name)
    for role in range(7):copy(role,role,0 if role==0 else 2 if role==3 else 1,f'role{role}')
    for bit in range(4):copy(7+bit,11+bit,3,f'bit{bit}')
    for kind in range(3):copy(11+kind,15,kind,f'kind{kind}')
    C[14]=C[11:14].sum(0)
    eq=torch.zeros(16,dtype=torch.float64);eq[10]=temperature;eq[15]=-.5*temperature
    A.extend((eq.clone(),eq.clone()));A=torch.stack(A);bias=torch.zeros(20,dtype=torch.float64);bias[-2]=8;bias[-1]=-8;C[15,-2]=1/16;C[15,-1]=-1/16;rows=A@R;rows[:,768]+=bias
    wordx=torch.nn.functional.pad(data['captures']['5.ln2'][:320],(0,1),value=1);word=torch.nn.functional.gelu(wordx@rows.float().T)@C.T.float();progx=torch.nn.functional.pad(data['captures']['5.ln2'][320:],(0,1),value=1);prog=torch.nn.functional.gelu(progx@rows.float().T)@C.T.float();nvalid=96;ix=torch.arange(nvalid);sp=torch.tensor(data['request']['program_starts'][320:416]);f=geo['programme_fields'][:nvalid].double();copyerrors=[]
    for role in range(7):
        kind=0 if role==0 else 2 if role==3 else 1;copyerrors.append((prog[ix,sp+role,role].double()-f[ix,sp+role,role]).abs()/f[ix,sp+role,15]);copyerrors.append((prog[ix,sp+role,11+kind].double()-f[ix,sp+role,15]).abs()/f[ix,sp+role,15])
    for role in bits_roles:copyerrors.append(((prog[ix,sp+role,7:11].double()-f[ix,sp+role,11:15]).abs()/f[ix,sp+role,15:16]).flatten())
    stress=[]
    for sigma in (2.,4.,8.,16.,32.):
        fm=vo/sigma;out=torch.nn.functional.gelu(fm.float()@A.float().T+bias.float())@C.T.float();ei=[]
        for role in range(7):
            kind=0 if role==0 else 2 if role==3 else 1;idx=torch.arange(len(vo));ei.append((out[idx,starts+role,role].double()-fm[idx,starts+role,role]).abs()/fm[idx,starts+role,15]);ei.append((out[idx,starts+role,11+kind].double()-fm[idx,starts+role,15]).abs()/fm[idx,starts+role,15])
        for role in bits_roles:ei.append(((out[idx,starts+role,7:11].double()-fm[idx,starts+role,11:15]).abs()/fm[idx,starts+role,15:16]).flatten())
        stress.append({'assumed_sigma':sigma,'maximum_relative_copy_error':float(torch.cat(ei).max())})
    r={'temperature':temperature,'copy_scale':copy_scale,'actual_valid_maximum_copy_error':float(torch.cat(copyerrors).max()),'old_programme_sigma_stress':stress,'native_equals_activations_above_half':int((word[:,:,15]>.5).sum()),'native_source_reference_activations_above_half':int((word[:,:,14].double()/geo['native_fields'][:,:,15]>.5).sum()),'native_programme_field_rms':word.double().square().mean((0,1)).sqrt().tolist(),'actual_equals_range':[float(prog[ix,sp+7,15].min()),float(prog[ix,sp+7,15].max())],'maximum_input_weight':float(rows.abs().max())};r['feasible']=r['actual_valid_maximum_copy_error']<.002 and max(s['maximum_relative_copy_error'] for s in stress)<.003 and r['native_equals_activations_above_half']==0 and r['actual_equals_range'][0]>.999;reports.append(r);print(json.dumps(r),flush=True)
    if r['feasible'] and best is None:best={'rows':rows,'field_rows':A,'field_bias':bias,'coefficients':C,'names':names+['equals_positive','equals_negative'],'report':r}
torch.save({'chosen':best,'reports':reports,'temperature_family':[512,1024,2048,4096,8192],'copy_scale':copy_scale,'source_gain_family':[128,256,512,1024,2048],'fixed_small_payload_gain':32.,'equals_gain':1048576.,'fields':[f'role{r}' for r in range(7)]+[f'bit{b}' for b in range(4)]+['bare_presence','spaced_presence','plus_presence','source_reference','equals_query']},ROOT/'artifacts/w_token_writer_v1.pt');(ROOT/'artifacts/w_token_writer_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n')
