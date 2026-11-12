"""Bound direct-source guards using fresh calibration states; no answer fitting."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
old=torch.load(ROOT/'artifacts/v_direct_source_v2.pt',weights_only=True)
source=old['chosen'];off=source['off'];T=source['report']['temperature']
text,program=unpack(torch.load(ROOT/'artifacts/v2b_reader_inputs.pt',weights_only=True,mmap=True))
tail=torch.load(ROOT/'artifacts/v3_source_tail_v1.pt',weights_only=True,mmap=True)
full=torch.load(ROOT/'experiments/v2b_low_score_reader/equals_piece.pt',weights_only=True)
writer=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)
ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];eqid=ids[-1]
def aug(x):return torch.cat((x.reshape(-1,768).double(),torch.ones(x.numel()//768,1,dtype=torch.float64)),1)
N=torch.cat((aug(text['captures']['1.ln2'][text['inputs']!=eqid]),aug(program['captures']['1.ln2'][program['inputs']!=eqid]),aug(tail['captures']['1.ln2'][tail['inputs']!=eqid])))
unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1
base=(source['rows'][:62:2]-8*unit)/T+source['penalties'][:,None]*off[None,:]
nv=N@off;assert nv.min()>0
valid=torch.tensor([d!='bad' for d in program['domains']]);lookup={int(t):i%10 for i,t in enumerate(ids[:20])}
truth=torch.tensor([[((lookup[int(program['inputs'][i,7-r['offset']])]>>r['bit'])&1) for r in old['bit_roles']] for i in torch.where(valid)[0]],dtype=torch.float64)
ft=torch.tensor([[((lookup[int(full['inputs'][i,full['starts'][i]+7-r['offset']])]>>r['bit'])&1) for r in old['bit_roles']] for i in range(len(full['inputs']))],dtype=torch.float64)
reports=[];candidates=[]
for safety in (1.,1.25,1.5,2.,4.):
    penalties=safety*torch.maximum(torch.zeros(31,dtype=torch.float64),((N@base.T+16/T)/nv[:,None]).max(0).values)
    rows=source['rows'].clone();effective=base-penalties[:,None]*off[None,:]
    rows[:62:2]=T*effective+8*unit;rows[1:62:2]=T*effective-8*unit
    def evaluate(norm):
        pre=torch.nn.functional.linear(norm,rows[:,:768].float(),rows[:,768].float());a=torch.nn.functional.gelu(pre).double();c=(a[:,::2]-a[:,1::2])/16
        return c,2*c[:,24:31].sum(1)-13*c[:,31],pre
    c,m,p=evaluate(program['captures']['1.ln2'][:,7]);fc,fm,fp=evaluate(full['norm_equals']);nc,nm,np=evaluate(N[:,:768].float())
    r={'safety':safety,'temperature':T,'negative_count':len(N),'negative_max_pre':float(np.max()),'negative_max_clip':float(nc.abs().max()),'mixed_bit_error_max':float((c[valid,:24]-truth).abs().max()),'full_bit_error_max':float((fc[:,:24]-ft).abs().max()),'full_bit_error_examples':int(((fc[:,:24]-ft).abs()>.5).any(1).sum()),'mixed_valid_marker_min':float(m[valid].min()),'bad_marker_max':float(m[~valid].max()),'full_valid_marker_min':float(fm.min()),'full_minimum_clip_margin':float(torch.minimum((fp[:,:48:2]-8).abs().min(),(fp[:,48:62:2]-8).min())),'maximum_fc_weight':float(rows.abs().max()),'penalty_range':[float(penalties.min()),float(penalties.max())]}
    r['feasible']=r['negative_max_pre']<-8 and r['mixed_bit_error_max']<.001 and r['full_bit_error_max']<.001 and r['mixed_valid_marker_min']>.99 and r['full_valid_marker_min']>.99 and r['bad_marker_max']<.01 and r['full_minimum_clip_margin']>=16 and r['maximum_fc_weight']<=1e6
    reports.append(r);candidates.append({'rows':rows,'penalties':penalties,'report':r});print(json.dumps(r),flush=True)
feasible=[x for x in candidates if x['report']['feasible']]
chosen=min(feasible,key=lambda x:abs(x['report']['safety']-1.5)) if feasible else None
torch.save({'chosen':chosen,'candidates':candidates,'safety_family':[1,1.25,1.5,2,4],'reports':reports},ROOT/'artifacts/v_tail_guard_v1.pt')
(ROOT/'artifacts/v_tail_guard_v1.json').write_text(json.dumps({'chosen':None if chosen is None else chosen['report'],'trials':reports},indent=2)+'\n')
