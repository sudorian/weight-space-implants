"""Exact construction-format off detector, validated on reserved prefixes.

Constraints label only membership in the already marked output region.
They do not contain operand digits, carries or arithmetic answers. Five
closed-form native covariance ridge values are evaluated, with no updates.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
data=torch.load(ROOT/'artifacts/z6d_stage8_full_v1.pt',weights_only=True,mmap=True);text,program,full=unpack(data);geo=torch.load(ROOT/'artifacts/z6d_counter_geometry_v1.pt',weights_only=True)
def aug(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)
X=aug(program['captures']['8.ln2']);P=X[:96,7:].reshape(-1,769);T=aug(text['captures']['8.ln2'].reshape(-1,768));N=torch.cat((T,X[:96,:7].reshape(-1,769),X[96:].reshape(-1,769)));pos=X[:48,7:].reshape(-1,769);_,s,Vh=torch.linalg.svd(pos,full_matrices=True);null=Vh[int((s>s.max()*1e-12).sum()):].T;W=T[:text['ntrain']];mean=W.mean(0);C=W.T@W/len(W);cn=null.T@C@null;mu=null.T@mean;trials=[];best=None;unit=geo['unit_row'];point=geo['point_row'];count=geo['count_rows'];expected=torch.tensor([[False,False,False],[False,False,False],[True,False,False],[True,True,False],[True,True,True]],dtype=torch.bool)
for ridge in (.001,.0001,1e-5,1e-6,1e-7):
 regular=cn.clone();regular.diagonal().add_(float(cn.trace()/len(cn))*ridge);w=torch.linalg.solve(regular,mu);w/=mu@w;row=null@w;nt=N@row;pv=P@row;gates=[];spec=[]
 for label,src in [(f'phase below {v}',v*unit-count[:,0]-8*point) for v in (5/12,7/24,.225)]+[(f'carry {j}',count[:,j]-8*point) for j in (1,2,3)]:
  penalty=1.25*max(0.,float(((N@src+8/2**23)/nt).max()));gate=src-penalty*row;gates.append(gate);spec.append({'label':label,'word_penalty':penalty,'sign_changes':int(((P@gate>0)!=(P@src>0)).sum())})
 gates=torch.stack(gates);values=(P@gates.T).reshape(96,5,6);phaseerr=int(((values[:,:,:3]>0)!=expected[None]).sum());minmargin=float(((2*expected.double()-1)[None]*values[:,:,:3]).min());rep={'ridge':ridge,'constraint_rank':769-null.shape[1],'native_negative_min':float(nt.min()),'native_max':float(nt.max()),'construction_off_max_abs':float((pos@row).abs().max()),'program_off_max_abs':float(pv.abs().max()),'later_off_max_abs':float(pv.reshape(96,5)[:,1:].abs().max()),'phase_errors_after_guard':phaseerr,'minimum_signed_phase_margin':minmargin,'maximum_fc_at_2p25':float((gates*2**25).abs().max()),'maximum_off_coefficient':float(row.abs().max()),'gate_specs':spec};trials.append(rep);print(json.dumps(rep),flush=True)
 if nt.min()>0 and phaseerr==0 and rep['maximum_fc_at_2p25']<1e6 and (best is None or minmargin>best['minimum_margin']):best={'row':row,'report':rep,'minimum_margin':minmargin}
assert best is not None,trials
geo['original_off']=geo['off'];geo['off']=best;geo['format_off_trials']=trials;torch.save(geo,ROOT/'artifacts/z6d_counter_geometry_v3.pt');(ROOT/'artifacts/z6d_phase_off_v1.json').write_text(json.dumps({'chosen':best['report'],'trials':trials},indent=2)+'\n')
