"""P9: direct phase intervals and conditional decimal reference copies.

The leading digit reads the equals carry directly. Later digits use count and
signed carry-count intervals. Frozen single-digit embedding geometry emits
characters; no whole-problem targets participate in any solve.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--temperature',type=float,default=16777216.);p.add_argument('--donors',type=int,default=672);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p8d_carry_counter/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);old=aux['previous'];geo=torch.load(ROOT/'artifacts/p8d_carry_counter_geometry_v2.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/p8d_carry_counter_{s}.pt',weights_only=True,mmap=True)) for s in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/p8d_carry_counter_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);assert torch.equal(text['inputs'],ht['inputs']);frozen=torch.load(ROOT/'artifacts/parabola_readout_v4.pt',weights_only=True);u0,u1=frozen['constant'],frozen['linear'];newline=torch.load(ROOT/'artifacts/a5_output_geometry.pt',weights_only=True)['readout_vectors'][:,10];sigma=old['carrier']/(768**.5);sigma_eq=aux['eq_norm_ratio']*sigma;gain=a.temperature*2**19
def aug(x):
 x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
gamma=host['transformer.h.4.ln_2.weight'].double();beta=host['transformer.h.4.ln_2.bias'].double()
def read(v):
 w=v/gamma;return torch.cat((w,(-beta@w)[None]))
X=torch.cat([aug(d['captures']['4.ln2'][:,7:]) for d in forms[:2]]);P=X.reshape(-1,769);T=aug(text['captures']['4.ln2'].reshape(-1,768));unit=geo['unit_row'];point=geo['point_row'];C=geo['count_rows'];S=geo['scalar_rows'];Ref=S[:,3];off=geo['off']['row'];ot=T@off;assert ot.min()>0;rows=[];columns=[];numeric=[];spec=[]
def add(row,col,num,label):
 rows.append(row);columns.append(col);numeric.append(num);spec.append(label)
def copied(gate,fields,vectors,nums,label,use_off=True):
 # Keep both positive and negative sides of every gate in the linear/off GELU
 # regimes. This bound is on recorded syntax positions, not answers.
 pv=P@gate;pf=torch.stack([P@v for v in fields],1).abs().max(1).values;bound=float(((pf+8/a.temperature)/pv.abs().clamp_min(1e-15)).max());factor=1.25*bound
 if not use_off:
  tv=T@gate;assert tv.max()<0,(label,float(tv.max()));tf=torch.stack([T@v for v in fields],1).clamp_min(0).max(1).values;factor=max(factor,1.25*float(((tf+8/a.temperature)/(-tv)).max()))
 guarded=factor*gate
 if use_off:
  tf=torch.stack([T@v for v in fields],1).clamp_min(0).max(1).values;penalty=1.25*max(0.,float(((T@guarded+tf+8/a.temperature)/ot).max()));guarded-=penalty*off
 else:penalty=0.
 base_col=-gain*sum(vectors)/a.temperature;base_num=-sum(nums)/a.temperature;add(a.temperature*guarded,base_col,base_num,{'kind':label,'role':'shared baseline'})
 for field,vector,num in zip(fields,vectors,nums):add(a.temperature*(guarded+field),gain*vector/a.temperature,num/a.temperature,{'kind':label,'role':'copy'})
 spec.append({'gate':label,'factor':factor,'off_penalty':penalty,'minimum_program_abs_pre':float((P@guarded).abs().min()*a.temperature),'sign_changes_on_program':int(((P@guarded>0)!=(pv>0)).sum())})
def constant_gate(row,bias):
 row=row.clone();row[-1]+=bias;return row
zero=torch.zeros(3,dtype=torch.float64);n0=zero.clone();n0[0]=1;n1=zero.clone();n1[1]=1;nn=zero.clone();nn[2]=1
# Current equals is reliably marked; it does not need the noisy attention count.
copied(constant_gate(point,-.2/sigma),[Ref],[u0],[n0],'leading base',False);lead=read(old['frame'][:,10])/old['carrier'];copied(constant_gate(point+lead,-1.5/sigma_eq),[Ref],[u1],[n1],'leading carry',False)
bounds=[.75,5/12,7/24,.225]
def interval(stage,counter,fields,vectors,nums,label):
 for boundary,sign in ((bounds[stage],1.),(bounds[stage-1],-1.)):
  gate=counter-boundary*unit-8*point;copied(gate,fields,[sign*v for v in vectors],[sign*n for n in nums],label+(' lower' if sign>0 else ' upper'))
for stage in (1,2,3):
 D=S[:,stage-1]+9*Ref;interval(stage,C[:,0],[Ref,D],[u0,u1],[n0,n1],f'base {stage}')
 for carry,mult in ({1:[(2,1),(3,-10)],2:[(1,1),(2,-10)],3:[(1,-10)]}[stage]):interval(stage,C[:,carry],[Ref],[mult*u1],[mult*n1],f'carry {stage} {carry} {mult}')
copied(.225*unit-C[:,0]-8*point,[Ref],[newline],[nn],'newline')
# Cancel the large intermediate registers. All cancel reads are homogeneous;
# two syntax branches use the known equals/later norm ratio.
def cancel(gate,fields,vectors,label):
 temp=1024.;pv=P@gate;pf=torch.stack([P@v for v in fields],1).abs().max(1).values;tv=T@gate;assert tv.max()<0;tf=torch.stack([T@v for v in fields],1).clamp_min(0).max(1).values;factor=1.25*max(float(((pf+8/temp)/pv.abs().clamp_min(1e-15)).max()),float(((tf+8/temp)/(-tv)).max()));base=factor*gate;add(temp*base,-sum(vectors)/temp,zero,{'kind':label,'role':'baseline'})
 for field,vector in zip(fields,vectors):add(temp*(base+field),vector/temp,zero,{'kind':label,'role':'copy'})
 spec.append({'gate':label,'factor':factor,'temperature':temp})
raw_common=read(old['common']);raw_point=read(old['point']);global_fields=[read(old['frame'][:,j]) for j in (0,2,3,4,5)];global_vectors=[-sigma*old['frame'][:,j] for j in (0,2,3,4,5)];cancel(constant_gate(raw_common/(768**.5),-.2),global_fields,global_vectors,'all register cancellation')
eq_fields=[];eq_vectors=[]
for j in range(12):
 eq_fields.append(read(old['frame'][:,j]));eq_vectors.append(-(sigma_eq-sigma if j in (0,2,3,4,5) else sigma_eq)*old['frame'][:,j])
for j in range(5):eq_fields.append(read(old['previous']['source_frame'][:,j]));eq_vectors.append(-sigma_eq*old['previous']['source_frame'][:,j])
cancel(constant_gate(raw_point/(768**.5),-.2),eq_fields,eq_vectors,'equals register cancellation')
rows=torch.stack(rows);columns=torch.stack(columns,1);numeric=torch.stack(numeric,1);assert len(rows)==68,len(rows);maximum=max(float(rows.abs().max()),float(columns.abs().max()));pre=torch.nn.functional.linear(P[:,:768].float(),rows[:,:768].float(),rows[:,768].float());act=torch.nn.functional.gelu(pre);actual=(act.double()@numeric.T).reshape(1024,5,3);vals=X@S;cv=X@unit;flags=(X@C[:,:,None] if False else X@C);carries=flags[:,:,1:]>0;eqcarry=(X[:,0]@lead)/(X[:,0]@point)>.5;expected=torch.zeros_like(actual);expected[:,:4,0]=vals[:,:4,3];expected[:,0,1]=eqcarry*vals[:,0,3];expected[:,1,1]=vals[:,1,0]+(9+carries[:,1,1].double()-10*carries[:,1,2].double())*vals[:,1,3];expected[:,2,1]=vals[:,2,1]+(9+carries[:,2,0].double()-10*carries[:,2,1].double())*vals[:,2,3];expected[:,3,1]=vals[:,3,2]+(9-10*carries[:,3,0].double())*vals[:,3,3];expected[:,4,2]=vals[:,4,3];tp=torch.nn.functional.linear(T[:,:768].float(),rows[:,:768].float(),rows[:,768].float());piece={'units':len(rows),'maximum_new_weight':maximum,'text_max_pre':float(tp.max()),'digit_copy_ratio_max_errors':[((actual[:,j,1]/actual[:,j,0])-(expected[:,j,1]/expected[:,j,0])).abs().max().item() for j in range(4)],'expected_zero_reference_violations':int((actual[:,:4,0]<=0).sum()),'relative_newline_copy_max_error':float(((actual[:,4,2]-expected[:,4,2])/expected[:,4,2]).abs().max())};print('piece',json.dumps(piece),flush=True);print('gate specifications',json.dumps(spec),flush=True);assert maximum<=1e6,maximum;assert tp.max()<-6,float(tp.max());assert max(piece['digit_copy_ratio_max_errors'])<.05,piece
new,repair=restore(host,4,text,ht['captures']['4.raw'],ht['captures']['4.gelu'],rows,columns,donor_count=a.donors);changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'previous':aux,'repair':repair,'rows':rows,'columns':columns,'numeric_coefficients':numeric,'output_gain':gain,'temperature':a.temperature,'specification':spec,'piece_report':piece},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py',ROOT/'experiments/analyze_p_counter_v2.py',ROOT/'experiments/analyze_parabola_readout_v4.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/p8d_carry_counter_geometry_v2.pt',ROOT/'artifacts/p8d_carry_counter_calib.pt',ROOT/'artifacts/host_calib512_raw34.pt',ROOT/'artifacts/parabola_readout_v4.pt',ROOT/'artifacts/a5_output_geometry.pt']+[ROOT/f'artifacts/p8d_carry_counter_{s}.pt' for s in ('format_a','format_b','bad')],arguments=vars(a),description='P9 complete direct G4 output: exact frozen single-digit readout with hand-wired conditional Ref copies, signed carry-count phase windows, direct leading carry and intermediate-register cancellation.',extra={'program_units':len(rows),'piece':piece,'text_repair':repair['text_repair_report'],'ridge_trials':repair['ridge_trials'],'temperature_family':[2**20,2**21,2**22,2**23,2**24],'output_gain_over_temperature':2**19})
