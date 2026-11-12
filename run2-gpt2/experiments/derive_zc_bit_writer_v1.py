"""26 hand GELUs: retain soft roles, but write four exact signed bits."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=load(host);old=torch.load(ROOT/'artifacts/z_centered_writer_v1.pt',weights_only=True);bits=torch.load(ROOT/'artifacts/zc_hard_bits_v3.pt',weights_only=True);assert bits['checks']['all_bits_feasible'];orig=old['chosen']['rows'];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;half=8.;scale=old['copy_scale'];ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};toks=data['inputs'];truth=torch.tensor([[[((lookup.get(int(t),0)>>b)&1)-.5 if int(t) in lookup else 0 for b in range(4)] for t in line] for line in toks],dtype=torch.float64);P=data['program'].float();W=data['native'].float();B=data['bad'].float();reports=[];chosen=None
for temperature in (128.,256.,512.,1024.,2048.):
 rows=[orig[k].clone() for k in range(3)];C=torch.zeros(19,26,dtype=torch.float64);names=['bare_baseline','spaced_baseline','plus_baseline']
 def copy(field,row,kind,name):
  j=len(rows);rows.append(row);C[field,j]=1/scale;C[field,kind]-=1/scale;names.append(name)
 for r in range(7):copy(r,orig[4+r],0 if r==0 else 2 if r==3 else 1,f'role{r}')
 for k in range(3):copy(11+k,orig[15+k],k,f'kind{k}')
 for k in range(3):copy(15+k,orig[18+k],k,f'recency{k}')
 for bit in range(4):
  gate=bits['chosen'][bit]['row'];j=len(rows);rows.extend((temperature*gate+half*one,temperature*gate-half*one));C[7+bit,j]=1/(2*half);C[7+bit,j+1]=-1/(2*half);C[7+bit]-=.5*(C[11]+C[12]);names.extend((f'bit{bit}_positive',f'bit{bit}_negative'))
 rows.extend((orig[-2],orig[-1]));C[18,-2]=1/16;C[18,-1]=-1/16;names.extend(('equals_positive','equals_negative'));rows=torch.stack(rows)
 def emit(x):return torch.nn.functional.gelu(torch.nn.functional.linear(x[:,:,:768],rows[:,:768].float(),rows[:,768].float()))@C.float().T
 p=emit(P);w=emit(W);b=emit(B);cb=b[torch.arange(192),data['bad_roles']];error=float((p[:,:,7:11].double()-truth).abs().max());r={'bit_temperature':temperature,'maximum_signed_bit_error':error,'valid_numeric_norm_range':[float(p[:,:,[7,8,9,10]].square().sum(2)[truth.abs().sum(2)>0].min()),float(p[:,:,[7,8,9,10]].square().sum(2)[truth.abs().sum(2)>0].max())],'native_bit_active_count':int((w[:,:,7:11].abs().max(2).values>.01).sum()),'native_kind_active_count':int((w[:,:,11:14].max(2).values>.5).sum()),'native_eq_count':int((w[:,:,18]>.5).sum()),'corrupt_bit_max':float(cb[:,7:11].abs().max()),'equals_min':float(p[:,7,18].min()),'max_row':float(rows.abs().max())};r['feasible']=error<.0005 and r['corrupt_bit_max']<.0005 and r['native_eq_count']==0;reports.append(r);print(json.dumps(r),flush=True)
 if r['feasible'] and chosen is None:chosen={'rows':rows,'coefficients':C,'names':names,'report':r}
assert chosen is not None
out={**old,'chosen':chosen,'hard_signed_bits':True,'bit_source':'zc_hard_bits_v3.pt','bit_temperature_family':[128,256,512,1024,2048],'bit_temperature_trials':reports,'bit_gain_family':[128,256,512,1024,2048],'fields':[f'role{r}' for r in range(7)]+[f'exact_signed_bit{b}' for b in range(4)]+old['fields'][11:]};torch.save(out,ROOT/'artifacts/zc_bit_writer_v1.pt');(ROOT/'artifacts/zc_bit_writer_v1.json').write_text(json.dumps({'chosen':chosen['report'],'trials':reports},indent=2)+'\n')
