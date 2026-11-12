"""Pure39-unit decimal output construction from fixed register readouts."""
import torch

def build(old,geo,off,negative,u0,u1,newline,*,output_gain=2**34,temperature=32768.,contrast=1.):
 phase=geo['phase_rows'];common=geo['common_row'];S=geo['scalar_rows'];Ref=S[:,3];U,Ten,Lead=geo['carry_rows'].T;ot=negative@off;assert ot.min()>0,float(ot.min());rows=[];columns=[];numeric=[];spec=[]
 def add(row,column,coefficient,label):rows.append(row);columns.append(column);numeric.append(coefficient);spec.append(label)
 def gate(source,fields,label,margin=None):
  source=contrast*source;bounds=torch.stack([negative@v for v in fields],1).clamp_min(0).max(1).values if fields else torch.zeros(len(negative),dtype=torch.float64);margin=16/temperature if margin is None else margin;penalty=1.25*max(0.,float(((negative@source+bounds+margin)/ot).max()));result=source-penalty*off;spec.append({'gate':label,'negative_penalty':penalty,'margin':margin,'contrast':contrast});return result
 def scalar_pair(gating,field,vector,index,multiplier,label):
  coeff=torch.zeros(2,dtype=torch.float64);coeff[index]=multiplier/temperature;add(temperature*(gating+field),output_gain*multiplier*vector/temperature,coeff,{'kind':label,'sign':1});add(temperature*gating,-output_gain*multiplier*vector/temperature,-coeff,{'kind':label,'sign':-1})
 def phase_clip(stage,newline_output=False):
  source=gate(phase[:,stage]-.5*common,[],f'phase {stage} cancellation',.15);column=-old['carrier']*old['phases'][:,stage]
  if newline_output:column=column+(output_gain/64)*newline
  for sign in (1.,-1.):
   row=1024*source.clone();row[-1]+=sign*128;add(row,sign*column/256,torch.zeros(2,dtype=torch.float64),{'kind':'phase cancellation and newline' if newline_output else 'phase cancellation','stage':stage,'sign':sign})
 def conditional(stage,flag,multiplier,label):scalar_pair(gate(phase[:,stage]+flag-1.5*common,[Ref],label),Ref,u1,1,multiplier,label)
 for stage in range(4):
  phase_clip(stage)
  if stage==0:
   scalar_pair(gate(phase[:,stage]-.5*common,[Ref],'leading base'),Ref,u0,0,1,'leading base');conditional(stage,Lead,1,'leading one')
  else:
   D=S[:,stage-1]+9*Ref;source=gate(2*phase[:,stage]-1.5*common,[Ref,D],f'digit base {stage}');add(temperature*source,-output_gain*(u0+u1)/temperature,torch.tensor([-1.,-1.],dtype=torch.float64)/temperature,{'kind':'shared digit baseline','stage':stage});add(temperature*(source+Ref),output_gain*u0/temperature,torch.tensor([1.,0.],dtype=torch.float64)/temperature,{'kind':'digit constant','stage':stage});add(temperature*(source+D),output_gain*u1/temperature,torch.tensor([0.,1.],dtype=torch.float64)/temperature,{'kind':'digit scalar','stage':stage})
   if stage==1:conditional(stage,Ten,1,'hundreds increment');conditional(stage,Lead,-10,'hundreds wrap')
   elif stage==2:conditional(stage,U,1,'tens increment');conditional(stage,Ten,-10,'tens wrap')
   else:conditional(stage,U,-10,'units wrap')
 phase_clip(4,True)
 for j in range(3):
  source=gate(geo['carry_rows'][:,j]-.5*common,[],f'carry {j} cancellation',.15);column=-old['carrier']*old['carries'][:,j]
  for sign in (1.,-1.):
   row=1024*source.clone();row[-1]+=sign*128;add(row,sign*column/256,torch.zeros(2,dtype=torch.float64),{'kind':'carry cancellation','carry':j,'sign':sign})
 return {'rows':torch.stack(rows),'columns':torch.stack(columns,1),'numeric_coefficients':torch.stack(numeric,1),'specification':spec,'output_gain':output_gain,'temperature':temperature,'contrast':contrast}

def check(hand,X,N,geo):
 rows=hand['rows'];h=torch.nn.functional.gelu(torch.nn.functional.linear(X[:,:,:768].float(),rows[:,:768].float(),rows[:,768].float()));actual=h.double()@hand['numeric_coefficients'].T;S=geo['scalar_rows'];v=X@S;common=X@geo['common_row'];active=common>.3;cu,ct,cl=[(X@geo['carry_rows'][:,j]/common>.5).double() for j in range(3)];expected=torch.zeros_like(actual);expected[:,:4,0]=v[:,:4,3];expected[:,0,1]=cl[:,0]*v[:,0,3];expected[:,1,1]=v[:,1,0]+(9+ct[:,1]-10*cl[:,1])*v[:,1,3];expected[:,2,1]=v[:,2,1]+(9+cu[:,2]-10*ct[:,2])*v[:,2,3];expected[:,3,1]=v[:,3,2]+(9-10*cu[:,3])*v[:,3,3];neg=torch.nn.functional.linear(N[:,:768].float(),rows[:,:768].float(),rows[:,768].float());err=[]
 for stage in range(4):
  ix=active[:,stage];a=actual[ix,stage];t=expected[ix,stage];e=(a[:,1]/a[:,0]-t[:,1]/t[:,0]).abs();err.append(float(e.max()) if torch.isfinite(e).all() else 1e100)
 maximum=max(float(rows.abs().max()),float(hand['columns'].abs().max()));return {'maximum_new_weight':maximum,'negative_max_preactivation':float(neg.max()),'active_counts_by_stage':active.sum(0).tolist(),'digit_ratio_copy_error_by_stage':err,'maximum_active_homogeneous_copy_error':float((actual-expected)[active].abs().max()),'feasible':maximum<=1e6 and float(neg.max())<-6 and max(err)<.01}
