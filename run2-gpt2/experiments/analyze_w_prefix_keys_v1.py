"""Inspect native role keys against actual calibration prefixes.

Oracle token-kind masking is reported explicitly as a diagnostic, not an
implemented gate. Recency values and attention scales are hand-fixed families.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
data=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True)
geo=torch.load(ROOT/'artifacts/w_native_roles_v1_geometry.pt',weights_only=True)['layers'][4]
writer=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)
ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];roles=(0,1,2,4,5,6)
index=torch.tensor([i for i,d in enumerate(data['request']['domains']) if d=='format_b']);tokens=data['inputs'][index];starts=torch.tensor([data['request']['program_starts'][int(i)] for i in index]);position=torch.arange(64)[None,:];causal=position<=(starts+7)[:,None]
features=data['captures']['4.gelu'][index].double()-geo['feature_mean'];reports=[]
for candidate in geo['trials']:
    values=features@candidate['role_decoder']
    for masking in ('none','oracle_expected_token_kind'):
        for slope in (0.,.002,.005,.01,.02):
            role_reports=[];correct=[];margins=[]
            for role in roles:
                mask=causal.clone()
                if masking!='none':mask&=torch.isin(tokens,torch.tensor(ids[:10] if role==0 else ids[10:20]))
                scores=values[:,:,role]+slope*position;scores=scores.masked_fill(~mask,-torch.inf);truth=starts+role;target=scores[torch.arange(len(index)),truth];wrong=scores.clone();wrong[torch.arange(len(index)),truth]=-torch.inf;margin=target-wrong.max(1).values
                ok=scores.argmax(1)==truth;correct.append(ok);margins.append(margin)
                role_reports.append({'role':role,'correct':int(ok.sum()),'count':len(index),'minimum_margin':float(margin.min()),'minimum_target_key':float(target.min()),'attention_min_target_weight':{str(scale):float(scores.mul(scale).softmax(1)[torch.arange(len(index)),truth].min()) for scale in (8.,16.,32.,64.,128.)}})
            r={'word_mass':candidate['mass'],'masking':masking,'recency_per_token':slope,'all_six_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(index),'minimum_margin':float(torch.stack(margins,1).min()),'roles':role_reports};reports.append(r);print(json.dumps(r),flush=True)
(ROOT/'artifacts/w_prefix_keys_v1.json').write_text(json.dumps({'reports':reports,'recency_family':[0,.002,.005,.01,.02],'attention_scale_family':[8,16,32,64,128],'oracle_mask_is_not_implemented':True},indent=2)+'\n')
