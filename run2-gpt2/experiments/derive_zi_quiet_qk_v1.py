"""Full-head versus partial quiet QK rotations for the scalar reader.

All choices use native Q/K second moments. Exactly one coordinate per head is
reused; the remaining native bilinear map is preserved by an orthogonal
rotation. Programme addresses and columns are checked on frozen inputs.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zd_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'artifacts/zd_reader_attention_v1_cache.pt',weights_only=True);reader=torch.load(ROOT/'artifacts/zd_projected_reader_v1.pt',weights_only=True);data,writer=load(host,c['writer']['scalar_gain']);x=data['captures']['5.ln1'];pre=data['captures']['5.pre'];domains=data['request']['domains'];starts=data['request']['program_starts'];tr=[i for i,d in enumerate(domains) if d=='format_a'];va=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];pi=torch.tensor(tr+va+bad);ps=torch.tensor([starts[i]+7 for i in pi]);wm=torch.zeros(512,64,dtype=torch.bool);wv=wm.clone();wm[:48]=True;wv[48:64]=True
for i in tr+bad[:32]:wm[i,:starts[i]]=True
for i in va+bad[32:]:wv[i,:starts[i]]=True
X=torch.nn.functional.pad(x.double(),(0,1),value=1);word=X[wm];eq=X[tr,torch.tensor([starts[i]+7 for i in tr])];ro=c['chosen_readout'];decoder=ro['decoder'];qcue=decoder[:,18];P=reader['changed']['transformer.h.5.attn.c_proj.weight'];pb=reader['changed']['transformer.h.5.attn.c_proj.bias'];M=reader['chosen']['memory'];gains=reader['chosen']['gains'];ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[lookup[int(tokens[a])]+lookup[int(tokens[b])]-9 for a,b in ((0,4),(1,5),(2,6))] for tokens in c['program_inputs'][:384]],dtype=torch.float64)
def attention(qw,qb,proj,bias):
 out=[]
 for j in range(0,512,32):
  q,k,v=torch.nn.functional.linear(x[j:j+32],qw.float(),qb.float()).split(768,2);q=q.view(-1,64,12,64).transpose(1,2);k=k.view(-1,64,12,64).transpose(1,2);v=v.view(-1,64,12,64).transpose(1,2);a=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(-1,64,768);out.append(torch.nn.functional.linear(a,proj.float(),bias.float()))
 return torch.cat(out)
base=attention(host['transformer.h.5.attn.c_attn.weight'],host['transformer.h.5.attn.c_attn.bias'],host['transformer.h.5.attn.c_proj.weight'],host['transformer.h.5.attn.c_proj.bias']);trials=[];best=None
for dim in (8,16,32,48,64):
 qw=c['qkv'].clone();qb=c['qb'].clone();heads=[]
 for h in c['head_aux']:
  role=h['role'];head=h['head'];ind=torch.arange(head*64,(head+1)*64);qw[ind]=host['transformer.h.5.attn.c_attn.weight'][ind].double();qb[ind]=host['transformer.h.5.attn.c_attn.bias'][ind].double();qw[768+ind]=host['transformer.h.5.attn.c_attn.weight'][768+ind].double();qb[768+ind]=host['transformer.h.5.attn.c_attn.bias'][768+ind].double();q=word[:,:768]@qw[ind].T+qb[ind];k=word[:,:768]@qw[768+ind].T+qb[768+ind];cq=q.T@q/len(q);ck=k.T@k/len(k);joint=cq/cq.trace()+ck/ck.trace();subset=joint.diagonal().argsort()[:dim];e,U=torch.linalg.eigh(joint[subset][:,subset]);qi=ind[subset];qw[qi]=U.T@qw[qi];qb[qi]=U.T@qb[qi];qw[768+qi]=U.T@qw[768+qi];qb[768+qi]=U.T@qb[768+qi];slot=int(qi[0]);qw[slot]=0;qb[slot]=0;qw[768+slot]=0;qb[768+slot]=0;qmean=(eq[:,:768]@qw[ind].T+qb[ind]).mean(0)/(eq@qcue).mean();comp=torch.cat((qmean@qw[768+ind]/8,(qmean@qb[768+ind]/8)[None]));heads.append({'role':role,'head':head,'kind':h['kind'],'slot':slot,'subset':subset,'rotation':U,'compensation':comp,'native_quiet_eigenvalue':float(e[0])})
 for scale in (128.,256.,512.,1024.,2048.):
  q=qw.clone();b=qb.clone();factor=(8*scale)**.5
  for h in heads:
   slot=h['slot'];role=h['role'];key=decoder[:,role]+decoder[:,15+h['kind']]+(ro['class_presence_key_boost']*decoder[:,14] if role!=3 else 0)-h['compensation']/scale;q[slot]=factor*qcue[:768];b[slot]=factor*qcue[768];q[768+slot]=factor*key[:768];b[768+slot]=factor*key[768]
  a=attention(q,b,P,pb);er=(a-base)[wv].double();post=pre[pi,ps].double()+a[pi,ps].double();ref=post@reader['query_frame']/reader['query_gain'];cols=(post@M)[:384,:3]/gains[:3]/ref[:384,None];rep={'qk_rotation_dimensions':dim,'attention_score':scale,'maximum_column_error':float((cols-truth).abs().max()),'rms_column_error':float((cols-truth).square().mean().sqrt()),'native_attention_increment_rms':float(er.square().mean().sqrt()),'native_position_rms_quantiles':torch.quantile(er.square().mean(1).sqrt(),torch.tensor([.5,.9,.99,1.],dtype=torch.float64)).tolist(),'native_quiet_eigenvalues':[h['native_quiet_eigenvalue'] for h in heads]};rep['feasible']=rep['maximum_column_error']<.2;trials.append(rep);print(json.dumps(rep),flush=True)
  if rep['feasible'] and (best is None or rep['native_attention_increment_rms']<best['report']['native_attention_increment_rms']):best={'qkv':q.float(),'bias':b.float(),'report':rep,'heads':heads,'eq_raw':post.float()}
out={'chosen':best,'trials':trials,'qk_dimension_family':[8,16,32,48,64],'score_family':[128,256,512,1024,2048],'same_projected_memory_as_ZD':True};torch.save(out,ROOT/'artifacts/zi_quiet_qk_v1.pt');(ROOT/'artifacts/zi_quiet_qk_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':trials},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
