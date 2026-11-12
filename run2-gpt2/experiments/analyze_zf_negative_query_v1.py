"""A negative default programme query suppresses source reads in ordinary text.

Five offsets and the same five address scores are checked as isolated
attention pieces. Every query remains positive at the hard Eq code. The
objective is native activation preservation subject to fixed format and
per-column arithmetic checks, never model predictions or answer fitting.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zd_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load(ROOT/'artifacts/zd_projected_reader_v1.pt',weights_only=True);c=torch.load(ROOT/'artifacts/zd_reader_attention_v1_cache.pt',weights_only=True);data,writer=load(host,c['writer']['scalar_gain']);x=data['captures']['5.ln1'];pre=data['captures']['5.pre'];domains=data['request']['domains'];starts=data['request']['program_starts'];valid=[i for i,d in enumerate(domains) if d in ('format_a','format_b')];bad=[i for i,d in enumerate(domains) if d=='bad'];pi=torch.tensor(valid+bad);ps=torch.tensor([starts[i]+7 for i in valid+bad]);wm=torch.zeros(512,64,dtype=torch.bool);wm[48:64]=True
for i,d in enumerate(domains):
 if d=='format_b' or (d=='bad' and i>=480):wm[i,:starts[i]]=True
nativeonly=torch.zeros_like(wm);nativeonly[48:64]=True;proj=reader['changed']['transformer.h.5.attn.c_proj.weight'];pb=reader['changed']['transformer.h.5.attn.c_proj.bias'];M=reader['chosen']['memory'];gains=reader['chosen']['gains'];qcue=c['chosen_readout']['decoder'][:,18];decoder=c['chosen_readout']['decoder'];boost=c['chosen_readout']['class_presence_key_boost'];ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[lookup[int(tokens[a])]+lookup[int(tokens[b])]-9 for a,b in ((0,4),(1,5),(2,6))] for tokens in c['program_inputs'][:384]],dtype=torch.float64);hand=torch.load(ROOT/'artifacts/zd_source5_hand_v3.pt',weights_only=True);rows=hand['rows'];coef=hand['coefficients'];reports=[];best=None

def attention(qw,bw,P,b):
 parts=[]
 for j in range(0,512,32):
  q,k,v=torch.nn.functional.linear(x[j:j+32],qw.float(),bw.float()).split(768,2);q=q.view(-1,64,12,64).transpose(1,2);k=k.view(-1,64,12,64).transpose(1,2);v=v.view(-1,64,12,64).transpose(1,2);av=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(-1,64,768);parts.append(torch.nn.functional.linear(av,P.float(),b.float()))
 return torch.cat(parts)
base=attention(host['transformer.h.5.attn.c_attn.weight'],host['transformer.h.5.attn.c_attn.bias'],host['transformer.h.5.attn.c_proj.weight'],host['transformer.h.5.attn.c_proj.bias'])
for scale in (128.,256.,512.,1024.,2048.):
 for offset in (0.,.01,.02,.05,.1):
  qw=c['qkv'].clone();qb=c['qb'].clone();factor=(8*scale)**.5
  for h in c['head_aux']:
   role=h['role'];slot=h['qk_slot'];q=qcue.clone();q[-1]-=offset;q/=1-offset;key=decoder[:,role]+decoder[:,15+h['kind']]+(boost*decoder[:,14] if role!=3 else 0)-h['native_score_compensation']/scale;qw[slot]=factor*q[:768];qb[slot]=factor*q[768];qw[768+slot]=factor*key[:768];qb[768+slot]=factor*key[768]
  a=attention(qw,qb,proj,pb);post=pre+a;eq=post[pi,ps].double();ref=(eq@reader['query_frame'])/reader['query_gain'];v=eq@M;columns=v[:384,:3]/gains[:3]/ref[:384,None];ln=torch.nn.functional.layer_norm(eq.float(),(768,),host['transformer.h.5.ln_2.weight'],host['transformer.h.5.ln_2.bias'],1e-5);h=torch.nn.functional.gelu(torch.nn.functional.linear(ln,rows[:,:768].float(),rows[:,768].float()));fields=h.double()@coef.T;defect=fields[:,5]*hand['gains'][5];active=ref>.5;diff=(a-base).double();e=diff[wm].square().mean(1).sqrt();r={'attention_score':scale,'negative_query_offset':offset,'maximum_valid_column_error':float((columns-truth).abs().max()),'rms_valid_column_error':float((columns-truth).square().mean().sqrt()),'valid_maximum_defect':float(defect[:384].abs().max()),'bad_minimum_active_defect':float(defect[384:][active[384:]].min()),'bad_query_absent':int((~active[384:]).sum()),'native_validation_attention_rms':float(diff[wm].square().mean().sqrt()),'native16windows_attention_rms':float(diff[nativeonly].square().mean().sqrt()),'native_position_rms_quantiles':torch.quantile(e,torch.tensor([.5,.9,.99,.999,1.],dtype=torch.float64)).tolist()};r['feasible']=r['maximum_valid_column_error']<.2 and r['valid_maximum_defect']<1e-5 and r['bad_minimum_active_defect']>24;reports.append(r);print(json.dumps(r),flush=True)
  if r['feasible'] and (best is None or r['native_validation_attention_rms']<best['report']['native_validation_attention_rms']):best={'qkv':qw.float(),'bias':qb.float(),'report':r,'eq_raw':eq.float(),'eq_ln2':ln}
 del a,post,eq,ln,diff
out={'chosen':best,'trials':reports,'negative_query_offset_family':[0,.01,.02,.05,.1],'attention_score_family':[128,256,512,1024,2048],'reader_P_unchanged':True};torch.save(out,ROOT/'artifacts/zf_negative_query_v1.pt');(ROOT/'artifacts/zf_negative_query_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
