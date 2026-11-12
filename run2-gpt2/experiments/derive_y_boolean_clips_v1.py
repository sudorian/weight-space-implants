"""Hand clips for 24 operand bits, seven syntax flags and one equals reference.

The only targets are current operand-token bits and per-role format flags.
Word-off bounds come from recorded inputs. No transformer is evaluated.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/y3_balanced_reader/circuit.pt',weights_only=True);r=aux['reader'];data=torch.load(ROOT/'artifacts/y3_stage5_full_v1.pt',weights_only=True,mmap=True);old=torch.load(ROOT/'artifacts/y_reader_memory_v3_eq.pt',weights_only=True);gamma=host['transformer.h.5.ln_2.weight'].double();beta=host['transformer.h.5.ln_2.bias'].double()
def aug(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)
def read(v):
    w=v/gamma;return torch.cat((w,(-beta@w)[None]))
one=torch.zeros(769,dtype=torch.float64);one[-1]=1;M=r['memory'];query=read(r['query_frame'])/(768**.5);off=one-query;hom=read(r['query_frame'])/r['query_gain'];X=aug(data['captures']['5.ln2']);starts=data['request']['program_starts'];vi=torch.arange(320,416);vs=torch.tensor(starts[320:416])+7;bi=torch.arange(416,512);bs=torch.tensor(starts[416:])+7
valid=torch.cat((aug(old['ln2'][:384]),X[vi,vs]));bad=torch.cat((aug(old['ln2'][384:]),X[bi,bs]));inputs=torch.cat((old['inputs'][:384],torch.stack([data['inputs'][i,s-7:s+1] for i,s in zip(vi,vs)])))
mask=torch.ones(512,64,dtype=torch.bool)
for i in range(320,512):mask[i,starts[i]+7]=False
N=X[mask];assert (N@off).min()>.8,float((N@off).min());sources=[];names=[];bits_truth=[];ids=aux['writer']['repair']['selected']
allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])}
for h in r['head_aux']:
    fs=h['fields'];role=h['role']
    if role==3:continue
    for bit in range(4):
        sources.append(read(M[:,fs[bit]]-.5*M[:,fs[-2]]));names.append(f'role{role}_bit{bit}');bits_truth.append(torch.tensor([(lookup[int(t)]>>bit)&1 for t in inputs[:,role]],dtype=torch.float64))
for h,gate in zip(r['head_aux'],r['chosen_memory']['syntax_gates']):
    fs=h['fields'];sources.append(read(M[:,fs[-1]]-gate['theta']*M[:,fs[-2]])-gate['reference_floor']*hom);names.append(f'role{h["role"]}_valid')
sources=torch.stack(sources);truth=torch.cat((torch.stack(bits_truth,1),torch.ones(len(valid),7,dtype=torch.float64)),1);initial=valid@sources.T;assert (((initial>0)!=(truth>.5)).sum())==0
nt=N@off;bare=N@sources.T;bound=((bare+8/65536)/nt[:,None]).max(0).values.clamp_min(0);reports=[];best=None
for safety in (1.,1.25,1.5,2.,4.):
    guarded=sources-safety*bound[:,None]*off
    for temperature in (65536.,131072.,262144.,524288.,1048576.):
        weight=(temperature*guarded[:,:768]).float();bias=(temperature*guarded[:,768]).float();pre=torch.nn.functional.linear(valid[:,:768].float(),weight,bias);prebad=torch.nn.functional.linear(bad[:,:768].float(),weight,bias);negative=torch.nn.functional.linear(N[:,:768].float(),weight,bias);clip=(torch.nn.functional.gelu(pre+2)-torch.nn.functional.gelu(pre-2))/4;badclip=(torch.nn.functional.gelu(prebad+2)-torch.nn.functional.gelu(prebad-2))/4;badquery=(bad@query)>.5;accept=(badclip[:,24:31]>.5).all(1)&badquery
        report={'off_safety':safety,'temperature':temperature,'maximum_input_weight':max(float(weight.abs().max()),float(bias.abs().max()+2)),'wrong_bits_or_flags':int(((clip>.5)!=(truth>.5)).sum()),'maximum_boolean_amplitude_error':float((clip.double()-truth).abs().max()),'minimum_signed_programme_preactivation':float(((2*truth-1)*pre.double()).min()),'native_maximum_preactivation_plus_two':float(negative.max()+2),'malformed_accept_count':int(accept.sum()),'malformed_count':len(bad),'word_off_min':float(nt.min()),'valid_off_range':[float((valid@off).min()),float((valid@off).max())]};report['feasible']=report['maximum_input_weight']<=1e6 and report['wrong_bits_or_flags']==0 and report['maximum_boolean_amplitude_error']<.01 and report['native_maximum_preactivation_plus_two']<-6 and report['malformed_accept_count']==0;reports.append(report);print(json.dumps(report),flush=True)
        if report['feasible']:
            key=(temperature,abs(safety-1.5))
            if best is None or key<best['key']:best={'key':key,'guarded':guarded,'report':report}
(ROOT/'artifacts/y_boolean_clips_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');assert best is not None,'No valid Boolean clip candidate.'
rows=[];C=torch.zeros(32,64,dtype=torch.float64);oldcols=torch.zeros(768,64,dtype=torch.float64);T=best['report']['temperature']
for field,gate in enumerate(best['guarded']):
    for sign in (1.,-1.):
        row=T*gate.clone();row[-1]+=sign*2;j=len(rows);rows.append(row);C[field,j]=sign/4
eq=query-.5*one
for sign in (1.,-1.):
    row=128*eq.clone();row[-1]+=sign*2;j=len(rows);rows.append(row);C[31,j]=sign/4;oldcols[:,j]=-sign*r['query_gain']*r['query_frame']/4
rows=torch.stack(rows)
def emit(z):return torch.nn.functional.gelu(torch.nn.functional.linear(z[:,:768].float(),rows[:,:768].float(),rows[:,768].float()))@C.T.float()
v=emit(valid);b=emit(bad);n=emit(N);checks={'valid_bits_and_flags_wrong':int(((v[:,:31]>.5)!=(truth>.5)).sum()),'valid_reference_min':float(v[:,31].min()),'native_maximum_abs_field':float(n.abs().max()),'malformed_accept_count':int(((b[:,24:31]>.5).all(1)&(b[:,31]>.5)).sum()),'valid_maximum_amplitude_error':float((v.double()-torch.cat((truth,torch.ones(len(v),1,dtype=torch.float64)),1)).abs().max())};assert checks['valid_bits_and_flags_wrong']==0 and checks['valid_reference_min']>.999 and checks['native_maximum_abs_field']<1e-6 and checks['malformed_accept_count']==0,checks
torch.save({'rows':rows,'coefficients':C,'old_cancellation_columns':oldcols,'fields':names+['equals_reference'],'chosen':best['report'],'reports':reports,'checks':checks,'bit_gain':512.,'reference_gain':8192.,'off_safety_family':[1,1.25,1.5,2,4],'temperature_family':[65536,131072,262144,524288,1048576],'equals_temperature':128.,'clip_halfwidth':2.,'sources':sources,'word_off_bound':bound},ROOT/'artifacts/y_boolean_clips_v1.pt');print('checks',json.dumps(checks),flush=True)
