"""Current-digit ordinal plus radius padding from ten hard class clips.

A fixed per-digit table emits (d-4.5, sqrt(20.25-(d-4.5)^2)); every code has
radius4.5. The semantic frame is rotated using native activation covariance
so the sole ordinal readout uses the quietest available direction.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
from circuit_io_v1 import minimum_metric_map
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);old=torch.load(ROOT/'artifacts/z_centered_writer_v1.pt',weights_only=True);classes=torch.load(ROOT/'artifacts/zd_current_digit_classes_v1.pt',weights_only=True);assert classes['checks']['all_digits_feasible'];orig=old['chosen']['rows'];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;half=8.;scale=old['copy_scale'];ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};toks=data['inputs'];truth=torch.tensor([[[lookup[int(t)]-4.5,(20.25-(lookup[int(t)]-4.5)**2)**.5,1.] if int(t) in lookup else [0.,0.,0.] for t in line] for line in toks],dtype=torch.float64);P=data['program'].float();W=data['native'].float();B=data['bad'].float();reports=[];chosen=None
for temperature in (128.,256.,512.,1024.,2048.):
 rows=[orig[k].clone() for k in range(3)];C=torch.zeros(19,38,dtype=torch.float64);names=['bare_baseline','spaced_baseline','plus_baseline']
 def copy(field,row,kind,name):
  j=len(rows);rows.append(row);C[field,j]=1/scale;C[field,kind]-=1/scale;names.append(name)
 for r in range(7):copy(r,orig[4+r],0 if r==0 else 2 if r==3 else 1,f'role{r}')
 for k in range(3):copy(11+k,orig[15+k],k,f'kind{k}')
 for k in range(3):copy(15+k,orig[18+k],k,f'recency{k}')
 for digit in range(10):
  gate=classes['chosen'][digit]['row'];j=len(rows);rows.extend((temperature*gate+half*one,temperature*gate-half*one));v=torch.tensor([digit-4.5,(20.25-(digit-4.5)**2)**.5,1.],dtype=torch.float64)
  for field,value in zip((7,8,14),v):C[field,j]=value/(2*half);C[field,j+1]=-value/(2*half)
  names.extend((f'digit{digit}_positive',f'digit{digit}_negative'))
 rows.extend((orig[-2],orig[-1]));C[18,-2]=1/16;C[18,-1]=-1/16;names.extend(('equals_positive','equals_negative'));rows=torch.stack(rows)
 def emit(x):return torch.nn.functional.gelu(torch.nn.functional.linear(x[:,:,:768],rows[:,:768].float(),rows[:,768].float()))@C.float().T
 p=emit(P);w=emit(W);b=emit(B);cb=b[torch.arange(192),data['bad_roles']];error=float((p[:,:,[7,8,14]].double()-truth).abs().max());r={'class_temperature':temperature,'maximum_scalar_padding_presence_error':error,'numeric_radius_squared_range':[float(p[:,:,:][:,:,[7,8]].square().sum(2)[truth[:,:,2]>0].min()),float(p[:,:,[7,8]].square().sum(2)[truth[:,:,2]>0].max())],'native_scalar_active_count':int((w[:,:,14]>.01).sum()),'native_eq_count':int((w[:,:,18]>.5).sum()),'corrupt_scalar_max':float(cb[:,[7,8,14]].abs().max()),'equals_min':float(p[:,7,18].min()),'max_row':float(rows.abs().max())};r['feasible']=error<.0005 and r['corrupt_scalar_max']<.0005 and r['native_eq_count']==0;reports.append(r);print(json.dumps(r),flush=True)
 if r['feasible'] and chosen is None:chosen={'rows':rows,'coefficients':C,'names':names,'report':r}
assert chosen is not None
# Native-only rotation of the eighteen non-query frame vectors. The source
# programme is subtracted from frozen pre5 before its covariance is formed.
y=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True);writer=torch.load(ROOT/'experiments/y2_certified_role_writer/circuit.pt',weights_only=True);act=torch.nn.functional.gelu(torch.nn.functional.linear(y['captures']['4.ln2'],writer['program_rows'][:,:768].float(),writer['program_rows'][:,768].float()));oldprog=act@((writer['frame']*writer['gains'])@writer['program_coefficients']).float().T;pre=y['captures']['5.pre']-oldprog;ln=torch.nn.functional.layer_norm(pre,(768,),host['transformer.h.5.ln_1.weight'],host['transformer.h.5.ln_1.bias'],1e-5);wm=torch.zeros(512,64,dtype=torch.bool);wm[:48]=True
for i,domain in enumerate(y['request']['domains']):
 if domain=='format_a':wm[i,:y['request']['program_starts'][i]]=True
X=torch.nn.functional.pad(ln[wm].double(),(0,1),value=1);cov=X.T@X/len(X);cov.diagonal().add_(float(cov.trace()/769)*1e-7);F=writer['frame'];gamma=host['transformer.h.5.ln_1.weight'].double();beta=host['transformer.h.5.ln_1.bias'].double();D=torch.cat((torch.cat(((gamma[:,None]*F).T,torch.zeros(19,1,dtype=torch.float64)),1),torch.cat((beta,torch.ones(1,dtype=torch.float64)))[None]));target=torch.cat((torch.eye(19,dtype=torch.float64),torch.zeros(1,19,dtype=torch.float64)));H,e=minimum_metric_map(D,target,cov);metric=H[:,:18].T@cov@H[:,:18];ev,U=torch.linalg.eigh(metric);assignment=[1,2,3,4,5,6,7,0,17,14,15,8,9,10,16,11,12,13];newF=F.clone();newF[:,:18]=F[:,:18]@U[:,assignment];frame_report={'original_ordinal_candidate_variance':float(metric[7,7]),'chosen_ordinal_variance':float(ev[0]),'variance_eigenvalues':ev.tolist(),'assignment':assignment,'orthogonality_error':float((newF.T@newF-torch.eye(19,dtype=torch.float64)).abs().max()),'query_frame_unchanged':bool(torch.equal(newF[:,18],F[:,18]))};print('frame',json.dumps(frame_report),flush=True)
out={**old,'chosen':chosen,'frame':newF,'frame_report':frame_report,'class_source':'zd_current_digit_classes_v1.pt','class_temperature_family':[128,256,512,1024,2048],'class_temperature_trials':reports,'scalar_gain_family':[32,64,128,256,512],'fields':[f'role{r}' for r in range(7)]+['ordinal_d_minus_4p5','radius_padding','unused9','unused10','bare_presence','spaced_presence','plus_presence','numeric_class_presence','bare_recency','spaced_recency','plus_recency','equals_query']};torch.save(out,ROOT/'artifacts/zd_scalar_writer_v1.pt');(ROOT/'artifacts/zd_scalar_writer_v1.json').write_text(json.dumps({'chosen':chosen['report'],'trials':reports,'frame':frame_report},indent=2)+'\n')
