"""Separate native costs inside one attention5 block on frozen writer inputs."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from y_writer_overlay_v3 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data,writer=load(host);c=torch.load(ROOT/'artifacts/y_reader_attention_v2_cache.pt',weights_only=True);reader=torch.load(ROOT/'artifacts/y_reader_memory_v3.pt',weights_only=True);x=data['captures']['5.ln1'];W=host['transformer.h.5.attn.c_attn.weight'];b=host['transformer.h.5.attn.c_attn.bias'];P=host['transformer.h.5.attn.c_proj.weight'];pb=host['transformer.h.5.attn.c_proj.bias'];domains=data['request']['domains'];starts=data['request']['program_starts'];mask=torch.zeros(512,64,dtype=torch.bool);mask[48:64]=True
for i,d in enumerate(domains):
    if d=='format_b':mask[i,:starts[i]]=True
def attention(qw,qb,pp,pbb):
    q,k,v=torch.nn.functional.linear(x,qw.float(),qb.float()).split(768,2);q=q.view(512,64,12,64).transpose(1,2);k=k.view(512,64,12,64).transpose(1,2);v=v.view(512,64,12,64).transpose(1,2);out=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(512,64,768);return torch.nn.functional.linear(out,pp.float(),pbb.float())
baseline=attention(W,b,P,pb);rankw=W.clone().double();rankb=b.clone().double()
for h in c['head_aux']:
    qi=64*h['head']+h['qk_subset'];U=h['qk_rotation'];rankw[qi]=U.T@rankw[qi];rankb[qi]=U.T@rankb[qi];rankw[768+qi]=U.T@rankw[768+qi];rankb[768+qi]=U.T@rankb[768+qi];slot=h['qk_slot'];rankw[slot]=0;rankb[slot]=0;rankw[768+slot]=0;rankb[768+slot]=0
nativeP=P.double().clone();nativepb=pb.double().clone()
for h in c['head_aux']:
    vi=64*h['head']+h['value_subset'];nativeP[:,vi]=P[:,vi].double()@h['value_inverse'];reused=h['value_reused'];ret=h['value_retained'];removed=nativeP[:,reused].clone();nativeP[:,ret]+=removed@h['value_repair'].T;nativepb+=removed@h['value_intercept'];nativeP[:,reused]=0
programP=nativeP.clone()
for slot,f in reader['slots']:programP[:,slot]=reader['value_gain']*reader['memory'][:,f]
protectP=P.double().clone();fe=reader['query_frame'];cols=writer['equals_protected_columns'];protectP[:,cols]-=fe[:,None]*(fe@protectP[:,cols])[None];protectpb=pb.double().clone();protectpb+=fe*float(((baseline[mask].double())@fe).mean()-fe@protectpb)
cases=[('value_repair_without_query_projection',torch.cat((W[:1536].double(),c['qkv'][1536:])),torch.cat((b[:1536].double(),c['qb'][1536:])),nativeP,nativepb),('query_projection_only',W,b,protectP,protectpb),('combined_without_query_projection_or_program_values',c['qkv'],c['qb'],nativeP,nativepb),('full_reader_without_query_projection',c['qkv'],c['qb'],programP,nativepb)];reports=[]
for name,qw,qb,pp,pbb in cases:
    result=attention(qw,qb,pp,pbb);d=(result-baseline).double();r={'case':name,'native_validation_rms':float(d[48:64].square().mean().sqrt()),'all_word_validation_rms':float(d[mask].square().mean().sqrt()),'native_validation_max_token_rms':float(d[48:64].square().mean(2).sqrt().max())};reports.append(r);print(json.dumps(r),flush=True)
(ROOT/'artifacts/y_reader_native_cost_v2.json').write_text(json.dumps(reports,indent=2)+'\n')
