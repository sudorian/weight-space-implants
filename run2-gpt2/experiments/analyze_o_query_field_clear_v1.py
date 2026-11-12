"""Native equals-query residual after clearing known injected writer fields.

Only recorded MLP0 pieces and native Q/K matrix products are evaluated.
No transformer stack, language-model output, or sum target is used.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from n_token_states_v1 import reconstruct
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];wd=torch.load(ROOT/'experiments/o2b_tokens/delta.pt',weights_only=True);writer=torch.load(ROOT/'experiments/o2b_tokens/circuit.pt',weights_only=True);rd=torch.load(ROOT/'experiments/o3e_reader/delta.pt',weights_only=True);reader=torch.load(ROOT/'experiments/o3e_reader/circuit.pt',weights_only=True);g=host['transformer.h.1.ln_1.weight'].double();b=host['transformer.h.1.ln_1.bias'].double()
def aug(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=x.dtype)),-1)
data=torch.load(ROOT/'artifacts/o1_coordinates_calib.pt',weights_only=True,mmap=True);_,norm=reconstruct(host,wd,data);T=aug(norm.reshape(-1,768));forms=[]
for tag in ('a','b'):
    d=unpack(torch.load(ROOT/f'artifacts/o1_coordinates_format_{tag}.pt',weights_only=True,mmap=True));_,norm=reconstruct(host,wd,d);forms.append(aug(norm))
P=torch.cat(forms);QKV=host['transformer.h.1.attn.c_attn.weight'].clone();bias=host['transformer.h.1.attn.c_attn.bias'].clone()
for name,value in (('transformer.h.1.attn.c_attn.weight',QKV),('transformer.h.1.attn.c_attn.bias',bias)):
    u=rd['updates'][name];value.view(-1)[u['indices'].long()]=u['values']
QKV=QKV.double();bias=bias.double();reports=[];candidates=[]
for label,fields in (('existing',[]),('all_position_fields',list(range(5,13))),('all_writer_fields',list(range(13)))):
    new=QKV.clone();nb=bias.clone();roles=[];query_word_energy=0.
    for h in reader['head_aux']:
        indices=torch.arange(64*h['head'],64*(h['head']+1));native=indices[~torch.isin(indices,h['qk_slots'])];F=writer['frame'][:,fields]
        if fields:
            response=new[native]@(g[:,None]*F);correction=response@(F.T/g[None,:]);field_mean=(((T[:,:768]-b)/g)@F).mean(0);new[native]-=correction;nb[native]+=correction@b+response@field_mean
            eq=h['centered_eq_decoder'];xe=forms[0][:,7];native_mean=(xe[:,:768]@new[native].T+nb[native]).mean(0)/(xe@eq).mean();new[native]-=native_mean[:,None]*eq[:768];nb[native]-=native_mean*eq[768]
        qe=P[:,7,:768]@new[native].T+nb[native];keys=P[:,:8,:768]@new[768+native].T+nb[768+native];scores=(qe[:,None,:]*keys).sum(2)/8;difference=T[:,:768]@(new[native]-QKV[native]).T+(nb[native]-bias[native]);word_q_change=float(difference.square().mean());query_word_energy+=word_q_change
        role=h['role'];pair=(role,role+4) if role in (0,1,2) else (role-4,role) if role in (4,5,6) else (3,7);variation=scores[:,pair[0]]-scores[:,pair[1]];roles.append({'role':role,'head':h['head'],'native_equals_query_rms':float(qe.square().mean().sqrt()),'native_equals_score_rms':float(scores.square().mean().sqrt()),'paired_score_difference_rms':float(variation.square().mean().sqrt()),'paired_score_difference_max':float(variation.abs().max()),'word_query_change_rms':word_q_change**.5})
    report={'variant':label,'roles':roles,'total_word_query_change_energy':query_word_energy};reports.append(report);candidates.append({'qkv':new,'bias':nb,'report':report});print(json.dumps(report),flush=True)
torch.save({'candidates':candidates,'reports':reports},ROOT/'artifacts/o_query_field_clear_v1.pt');(ROOT/'artifacts/o_query_field_clear_v1.json').write_text(json.dumps(reports,indent=2)+'\n')
