"""Inspect homogeneous token gates on frozen W2 stage5 activations.

Only current token identities and syntax roles determine the checks. No
whole-answer labels or language-model evaluation occurs here.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];writer=torch.load(ROOT/'experiments/w2_conditional_memory/circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/w2_stage5_full.pt',weights_only=True,mmap=True);text,program,full=unpack(data)
F=writer['frame'];gains=torch.cat((writer['payload_gains'],torch.tensor([writer['reference_gain']],dtype=torch.float64)));gamma=host['transformer.h.5.ln_2.weight'].double();beta=host['transformer.h.5.ln_2.bias'].double();R=F/gains/gamma[:,None];rows=torch.cat((R.T,-(beta@R)[:,None]),1)
def fields(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)@rows.T
v=fields(full['captures']['5.ln2']);w=fields(text['captures']['5.ln2']);vr=v[:,:,:15]/v[:,:,15:16];wr=w[:,:,:15]/w[:,:,15:16];starts=torch.tensor(full['starts']);valid=torch.tensor([d!='bad' for d in full['domains']]);ix=torch.arange(len(starts));causal=torch.arange(64)[None,:]<=(starts+7)[:,None]
ids=torch.tensor(torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids']);lookup={int(t):i%10 for i,t in enumerate(ids[:20])};numset=set(lookup);digitroles=(0,1,2,4,5,6)
truth=torch.tensor([[[(lookup[int(full['inputs'][i,starts[i]+r])]>>b)&1 for b in range(4)] for r in digitroles] for i in torch.where(valid)[0]],dtype=torch.float64);numeric=torch.tensor([[int(t) in numset for t in row] for row in text['inputs']]);correcttypes=torch.zeros(*text['inputs'].shape,4,dtype=torch.bool)
for j,kindids in enumerate((ids[:10],ids[10:20],ids[20:21],ids[21:22])):correcttypes[:,:,j]=(text['inputs'][:,:,None]==kindids).any(2)
types=wr[:,:,7:11]>.5;bitv=torch.stack([vr[ix[valid],starts[valid]+r,11:15] for r in digitroles],1)
report={'native_tokens':int(w.shape[0]*w.shape[1]),'native_kind_activations':types.sum((0,1)).tolist(),'native_false_kind_activations':(types&~correcttypes).sum((0,1)).tolist(),'native_true_kind_counts':correcttypes.sum((0,1)).tolist(),'reference_ranges':{'native':[float(w[:,:,15].min()),float(w[:,:,15].max())],'programme':[float(v[:,:,15].min()),float(v[:,:,15].max())]},'valid_bit_errors':int(((bitv>.5)!=(truth>.5)).any(2).any(1).sum()),'valid_bit_margin':float(((truth*2-1)*(bitv-.5)).min()),'native_numeric_off_min':float((1-wr[:,:,7]-wr[:,:,8]).min())}
role_reports=[]
for contrast in (0.,.25,.5,.75,1.):
    keys=vr[:,:,:7]-contrast*(vr[:,:,:7].sum(2,keepdim=True)-vr[:,:,:7])/6
    correct=[];margins=[];true=[];wrong=[]
    for role in range(7):
        kind=0 if role==0 else 2 if role==3 else 1;mask=causal&(vr[:,:,7+kind]>.5);score=keys[:,:,role].masked_fill(~mask,-torch.inf);pos=starts+role;values=score[ix,pos];other=score.clone();other[ix,pos]=-torch.inf
        correct.append(score.argmax(1)==pos);margins.append(values-other.max(1).values);true.append(values[valid]);wrong.append(other[valid].max(1).values)
    r={'contrast':contrast,'correct_valid_all7':int(torch.stack(correct,1)[valid].all(1).sum()),'count':int(valid.sum()),'source_margin':float(torch.stack(margins,1)[valid].min()),'true_key_min':float(torch.stack(true,1).min()),'wrong_key_max':float(torch.stack(wrong,1).max()),'true_min_by_role':torch.stack(true,1).min(0).values.tolist(),'wrong_max_by_role':torch.stack(wrong,1).max(0).values.tolist()};role_reports.append(r)
report['role_contrasts']=role_reports
offrow=rows[15]-rows[7]-rows[8];offw=fields(text['captures']['5.ln2'])[:,:,15]-w[:,:,7]-w[:,:,8];offv=v[:,:,15]-v[:,:,7]-v[:,:,8];neg=~numeric;bounds=[]
for b in range(4):
    z=w[:,:,11+b]-.5*w[:,:,15];ok=offw[neg]>0;ratio=(z[neg]+.05*w[:,:,15][neg])/offw[neg];bounds.append({'bit':b,'off_nonpositive':int((~ok).sum()),'penalty_bound':float(ratio[ok].max()),'raw_negative_max':float(z[neg].max())})
report['numeric_off_bounds']=bounds;report['native_bit_activation_counts']=(wr[:,:,11:15]>.5).sum((0,1)).tolist();report['native_bit_max_ratio']=wr[:,:,11:15].amax((0,1)).tolist()
torch.save({'rows':rows,'report':report,'native_fields':w.float(),'programme_fields':v.float(),'valid':valid,'starts':starts,'truth':truth,'digit_roles':list(digitroles)},ROOT/'artifacts/w_token_gates_v1.pt');(ROOT/'artifacts/w_token_gates_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
