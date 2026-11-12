"""Replace the diagnostic oracle by native token-feature masks on real prefixes."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);rg=torch.load(ROOT/'artifacts/w_native_roles_v1_geometry.pt',weights_only=True)['layers'][4];tg=torch.load(ROOT/'artifacts/w_token_features_v2.pt',weights_only=True)
ids=torch.tensor(torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids']);lookup={int(t):i%10 for i,t in enumerate(ids[:20])}
ix=torch.tensor([i for i,d in enumerate(data['request']['domains']) if d=='format_b']);starts=torch.tensor([data['request']['program_starts'][int(i)] for i in ix]);tokens=data['inputs'][ix];gelu=data['captures']['4.gelu'][ix].double();norm=torch.nn.functional.layer_norm(data['captures']['4.raw'][ix],(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5).double();joint=torch.cat((gelu,norm),2);positions=torch.arange(64)[None,:];causal=positions<=(starts+7)[:,None];roles=(0,1,2,3,4,5,6);digitroles=(0,1,2,4,5,6);reports=[]
truth=torch.tensor([[[(lookup[int(tokens[i,starts[i]+role])]>>bit)&1 for bit in range(4)] for role in digitroles] for i in range(len(ix))],dtype=torch.float64)
for token_candidate in tg['candidates']:
    tp=joint@token_candidate['coefficients']+token_candidate['bias'];readbits=torch.stack([tp[torch.arange(len(ix)),starts+role,4:] for role in digitroles],1);berr=(readbits>.5)!=(truth>.5);allcorrecttypes=torch.ones(len(ix),dtype=torch.bool);program_type_min=[]
    for role in range(8):
        kind=0 if role==0 else 2 if role==3 else 3 if role==7 else 1;v=tp[torch.arange(len(ix)),starts+role,kind];allcorrecttypes&=v>.5;program_type_min.append(float(v.min()))
    nonformat=~torch.isin(tokens,ids);falsekind=int((tp[:,:,:4][nonformat]>.5).any(1).sum())
    for rc in rg['trials']:
        keys=(gelu-rg['feature_mean'])@rc['role_decoder'];correct=[];margin=[];weights=[]
        for role in roles:
            kind=0 if role==0 else 2 if role==3 else 1;mask=causal&(tp[:,:,kind]>.5);score=keys[:,:,role].masked_fill(~mask,-torch.inf);target=starts+role;value=score[torch.arange(len(ix)),target];wrong=score.clone();wrong[torch.arange(len(ix)),target]=-torch.inf;correct.append(score.argmax(1)==target);margin.append(value-wrong.max(1).values);weights.append({str(s):float(score.mul(s).softmax(1)[torch.arange(len(ix)),target].min()) for s in (8.,16.,32.,64.,128.)})
        r={'negative_class_support':token_candidate['report']['negative_vocabulary_classes'],'role_word_mass':rc['mass'],'all_seven_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(ix),'minimum_source_margin':float(torch.stack(margin,1).min()),'program_type_correct':int(allcorrecttypes.sum()),'program_type_min_by_role':program_type_min,'digit_bit_error_examples':int(berr.any(2).any(1).sum()),'nonformat_false_type_positions':falsekind,'attention_min_target_weights_by_role':weights};reports.append(r);print(json.dumps(r),flush=True)
(ROOT/'artifacts/w_actual_masks_v1.json').write_text(json.dumps({'reports':reports,'implemented_in_checkpoint':False,'feature_basis':'Native GELU4 plus LN2; the linear feature part requires explicit new GELU-identity pairs.'},indent=2)+'\n')
