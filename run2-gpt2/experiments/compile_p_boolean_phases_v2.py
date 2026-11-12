"""P9C: explicit Boolean output phases and carries before decimal readout.

Identical FC weights with different biases implement every Boolean clip.
This avoids subtracting nearly equal independently rounded FC weight rows.
Known syntax positions determine phase checks; no answers enter any solve.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--carrier',type=float,default=8388608.);p.add_argument('--donors',type=int,default=128);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p8d_carry_counter/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);old=aux['previous'];geo=torch.load(ROOT/'artifacts/p8d_carry_counter_geometry_v2.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/p8d_carry_counter_{s}.pt',weights_only=True,mmap=True)) for s in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/p8d_carry_counter_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);sigma=old['carrier']/(768**.5);sigma_eq=aux['eq_norm_ratio']*sigma

def aug(x):
 x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
gamma=host['transformer.h.4.ln_2.weight'].double();beta=host['transformer.h.4.ln_2.bias'].double()
def read(v):
 w=v/gamma;return torch.cat((w,(-beta@w)[None]))
def shifted(v,b):
 v=v.clone();v[-1]+=b;return v
X=torch.cat([aug(d['captures']['4.ln2'][:,7:]) for d in forms[:2]]);P=X.reshape(-1,769);T=aug(text['captures']['4.ln2'].reshape(-1,768));other=torch.cat([aug(d['captures']['4.ln2'][:,:7]).reshape(-1,769) for d in forms[:2]]);N=torch.cat((T,other));unit=geo['unit_row'];point=geo['point_row'];C=geo['count_rows'];off=geo['off']['row'];ot=N@off;assert ot.min()>0,float(ot.min());counter_sources=[];counter_specs=[]
for label,src in [(f'phase below {b}',b*unit-C[:,0]-8*point) for b in (5/12,7/24,.225)]+[(f'carry {j}',C[:,j]-8*point) for j in (1,2,3)]:
 penalty=1.25*max(0.,float(((N@src+8/2**23)/ot).max()));gate=src-penalty*off;counter_sources.append(gate);counter_specs.append({'label':label,'off_penalty':penalty,'sign_changes':int(((P@gate>0)!=(P@src>0)).sum())})
counter_sources=torch.stack(counter_sources);truth=(P@counter_sources.T>0).double();trials=[]
for temp in (2**23,2**24,2**25,2**26,2**27):
 weights=(temp*counter_sources[:,:768]).float();bias=(temp*counter_sources[:,768]).float();pre=torch.nn.functional.linear(P[:,:768].float(),weights,bias);z=(torch.nn.functional.gelu(pre+2)-torch.nn.functional.gelu(pre-2))/4;neg=torch.nn.functional.linear(N[:,:768].float(),weights,bias);report={'temperature':temp,'maximum_new_fc':max(float(weights.abs().max()),float(bias.abs().max()+2)),'max_boolean_error':float((z.double()-truth).abs().max()),'wrong_boolean':int(((z>.5)!=(truth>.5)).sum()),'negative_max_pre':float(neg.max()+2),'minimum_abs_program_pre':float(pre.abs().min())};trials.append(report);print('temperature',json.dumps(report),flush=True)
feasible=[r for r in trials if r['maximum_new_fc']<=1e6 and r['max_boolean_error']<.1 and r['negative_max_pre']<-6];assert feasible,trials;temp=min(feasible,key=lambda r:r['temperature'])['temperature'];rows=[];new_coeff=[];old_columns=[];spec=[]
def emit(row,new,oldcol,label):
 rows.append(row);new_coeff.append(new);old_columns.append(oldcol);spec.append(label)
def clip(gate,temperature,new,oldcol,label):
 for sign in (1.,-1.):
  row=temperature*gate.clone();row[-1]+=sign*2;emit(row,sign*new/4,sign*oldcol/4,{'kind':label,'sign':sign,'temperature':temperature})
def basis(i):
 v=torch.zeros(8,dtype=torch.float64);v[i]=1;return v
zero=torch.zeros(768,dtype=torch.float64);eqgate=shifted(sigma*point,-.2);latergate=shifted(sigma*(unit-point),-.4)
clip(eqgate,128.,basis(0),-old['carrier']*(old['frame'][:,0]+old['frame'][:,1]+old['frame'][:,[7,9,11]].sum(1)),'equals phase and old marker cancellation')
clip(latergate,128.,basis(1),-old['carrier']*old['common'],'later presence and old common cancellation')
for j in range(3):clip(counter_sources[j],temp,-basis(j+1)+basis(j+2),zero,f'phase boundary {j}')
for j in range(3):
 clip(counter_sources[3+j],temp,basis(5+j),zero,f'later carry {j}')
 eqcarry=shifted(read(old['frame'][:,6+2*j])/(768**.5),-.2);clip(eqcarry,128.,basis(5+j),old['carrier']*(-old['frame'][:,6+2*j]+old['frame'][:,7+2*j]),f'equals carry {j} and old pair cancellation')
# Old source fields only exist at equals. A gated linear identity cancels them
# at the known dominant-carrier norm; no precision-critical scalar is copied.
fields=[read(old['previous']['source_frame'][:,j]) for j in range(5)];pf=torch.stack([P@f for f in fields],1).abs().max(1).values;pv=P@eqgate;tv=N@eqgate;assert tv.max()<0;nf=torch.stack([N@f for f in fields],1).clamp_min(0).max(1).values;copy_temp=1024.;factor=1.25*max(float(((pf+8/copy_temp)/pv.abs().clamp_min(1e-12)).max()),float(((nf+8/copy_temp)/(-tv)).max()));gating=factor*eqgate;vectors=[-sigma_eq*old['previous']['source_frame'][:,j] for j in range(5)];emit(copy_temp*gating,torch.zeros(8,dtype=torch.float64),-sum(vectors)/copy_temp,{'kind':'source cancellation baseline'})
for field,vector in zip(fields,vectors):emit(copy_temp*(gating+field),torch.zeros(8,dtype=torch.float64),vector/copy_temp,{'kind':'source cancellation copy'})
rows=torch.stack(rows);new_coeff=torch.stack(new_coeff,1);old_columns=torch.stack(old_columns,1);assert len(rows)==28,len(rows);pre=torch.nn.functional.linear(P[:,:768].float(),rows[:,:768].float(),rows[:,768].float());acts=torch.nn.functional.gelu(pre);numeric=(acts.double()@new_coeff.T).reshape(1024,5,8);expected=torch.zeros_like(numeric);expected[:,:,:5]=torch.eye(5,dtype=torch.float64)[None];expected[:,1:,5:]=(X[:,1:]@C[:,1:]>0).double();expected[:,0,5:]=torch.stack([(X[:,0]@read(old['frame'][:,6+2*j]))/(X[:,0]@read(old['common']))>.5 for j in range(3)],1).double();np=torch.nn.functional.linear(N[:,:768].float(),rows[:,:768].float(),rows[:,768].float());piece={'temperature':temp,'max_boolean_field_error':float((numeric-expected).abs().max()),'wrong_phase':int((numeric[:,:,:5].argmax(2)!=torch.arange(5)[None,:]).sum()),'wrong_carry':int(((numeric[:,:,5:]>.5)!=(expected[:,:,5:]>.5)).sum()),'negative_max_pre':float(np.max()),'source_cancel_factor':factor,'counter_gates':counter_specs};print('piece',json.dumps(piece),flush=True);assert piece['wrong_phase']==0 and piece['wrong_carry']==0,piece;assert np.max()<-6,float(np.max())
new,repair=restore(host,4,text,ht['captures']['4.raw'],ht['captures']['4.gelu'],rows,old_columns,donor_count=a.donors);W=new['transformer.h.4.mlp.c_proj.weight'].double();bias=new['transformer.h.4.mlp.c_proj.bias'].double();raw=text['captures']['4.raw'].reshape(-1,768).double()+text['captures']['4.gelu'].reshape(-1,3072).double()@W.T+bias;copy=torch.load(ROOT/'experiments/p4b_amplified_copy/circuit.pt',weights_only=True);keep=host['transformer.h.5.ln_2.weight'].abs()>.1;constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),old['frame'].T,old['previous']['source_frame'].T,old['previous']['memory'].T,aux['memory'].T,copy['frame'].T,raw.mean(0)[None,:],torch.eye(768,dtype=torch.float64)[~keep]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);Q=Vh[int((s>s.max()*1e-9).sum()):].T;z=raw/(raw.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();zt=z@Q;_,V=torch.linalg.eigh(zt.T@zt/len(zt));frame=Q@V[:,:8];columns=a.carrier*frame@new_coeff+old_columns;maximum=max(float(rows.abs().max()),float(columns.abs().max()));assert maximum<=1e6,maximum;new['transformer.h.4.mlp.c_proj.weight'][:,repair['selected']]=columns.float();changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'previous':aux,'repair':repair,'frame':frame,'phases':frame[:,:5],'carries':frame[:,5:],'carrier':a.carrier,'scalar_frame':old['scalars'],'scalar_gain':old['scalar_gain'],'rows':rows,'columns':columns,'emission_coefficients':new_coeff,'piece_report':piece,'specification':spec},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py',ROOT/'experiments/analyze_p_counter_v2.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/p8d_carry_counter_geometry_v2.pt',ROOT/'artifacts/p8d_carry_counter_calib.pt',ROOT/'artifacts/host_calib512_raw34.pt']+[ROOT/f'artifacts/p8d_carry_counter_{s}.pt' for s in ('format_a','format_b','bad')],arguments=vars(a),description='P9C: 28 MLP4 units write five margin-separated phase and three Boolean carry fields with identical-weight bias clips, and cancel previous large markers. Scalar columns stay in their existing 262144-gain registers.',extra={'program_units':len(rows),'piece':piece,'maximum_new_weight':maximum,'temperature_trials':trials,'phase_amplitude_tolerance':.1,'phase_readout_requires_argmax_and_half-threshold_separation':True,'text_repair':repair['text_repair_report'],'ridge_trials':repair['ridge_trials']})
