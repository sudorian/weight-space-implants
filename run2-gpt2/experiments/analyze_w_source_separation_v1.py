"""Current-role/type coincidence in the protected memory, no answer labels."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
old=torch.load(ROOT/'artifacts/w_prefix_memory_v1.pt',weights_only=True);new=torch.load(ROOT/'artifacts/w_token_gates_v1.pt',weights_only=True);v=old['fields_before_layer5_normalization'].double();vr=v[:,:,:15]/v[:,:,15:16];w=new['native_fields'].double();wr=w[:,:,:15]/w[:,:,15:16];vi=torch.tensor([i for i,d in enumerate(old['domains']) if d=='format_b']);starts=torch.tensor([old['starts'][i] for i in vi]);vr=vr[vi];ix=torch.arange(len(vi));causal=torch.arange(64)[None,:]<=(starts+7)[:,None];reports=[]
for contrast in (0.,.25,.5,.75,1.):
    keys=vr[:,:,:7]-contrast*(vr[:,:,:7].sum(2,keepdim=True)-vr[:,:,:7])/6;wk=wr[:,:,:7]-contrast*(wr[:,:,:7].sum(2,keepdim=True)-wr[:,:,:7])/6
    for penalty in (0.,.25,.5,1.,2.):
        true=[];wrong=[];native=[];correct=[];margins=[]
        for role in range(7):
            kind=0 if role==0 else 2 if role==3 else 1;score=keys[:,:,role]-penalty*(1-vr[:,:,7+kind]);score=score.masked_fill(~causal,-torch.inf);position=starts+role;value=score[ix,position];other=score.clone();other[ix,position]=-torch.inf;true.append(value);wrong.append(other.max(1).values);native.append(wk[:,:,role]-penalty*(1-wr[:,:,7+kind]));correct.append(score.argmax(1)==position);margins.append(value-other.max(1).values)
        tr=torch.stack(true,1);wrng=torch.stack(wrong,1);nat=torch.stack(native,-1);r={'contrast':contrast,'type_penalty':penalty,'all_seven_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(vi),'minimum_source_margin':float(torch.stack(margins,1).min()),'true_min':tr.min(0).values.tolist(),'wrong_max':wrng.max(0).values.tolist(),'native_max':nat.amax((0,1)).tolist(),'native_max_over_roles':float(nat.max()),'true_min_over_roles':float(tr.min()),'native_above_point3':int((nat.max(2).values>.3).sum()),'native_above_point4':int((nat.max(2).values>.4).sum())};reports.append(r)
(ROOT/'artifacts/w_source_separation_v1.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
