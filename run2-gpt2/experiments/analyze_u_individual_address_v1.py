"""Evaluate seven period-eight plus linear-recency heads on recorded LN1.

Labels specify source position and individual digit bits, never addition output.
Native QK contributions are retained except current-equals recentering and
removal of the eleven programmed address fields from native queries.
"""
from pathlib import Path
import json,math,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_token_states_v1 import reconstruct
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
base=ROOT/'experiments/u1_period8_writer/delta.pt';delta=torch.load(base,weights_only=True)
writer=torch.load(base.parent/'circuit.pt',weights_only=True)
token=torch.load(ROOT/'artifacts/u1_period8_writer_source_readouts_v3.pt',weights_only=True)
data=torch.load(ROOT/'artifacts/u1_full_prefix_b.pt',weights_only=True,mmap=True)
g=host['transformer.h.1.ln_1.weight'].double();b=host['transformer.h.1.ln_1.bias'].double()
def aug(x):return torch.cat((x.double(),torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
X=aug(data['captures']['1.ln1']);starts=torch.tensor(data['request']['program_starts']);ix=torch.arange(len(X));pos=starts+7;query_input=X[ix,pos];causal=torch.arange(64)[None,:]<=pos[:,None]
calib=torch.load(ROOT/'artifacts/q1_sparse_coordinates_calib.pt',weights_only=True,mmap=True)
raw,norm=reconstruct(host,delta,calib);xt=aug(norm.reshape(-1,768))
R=writer['frame']/g[:,None]/writer['field_gains'][None,:];R=torch.cat((R,(-b@R)[None,:]));src=R[:,8:13];eq=R[:,13:19]
P=host['transformer.h.1.attn.c_proj.weight'].double();W=host['transformer.h.1.attn.c_attn.weight'].double();bias=host['transformer.h.1.attn.c_attn.bias'].double()
values=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True)['captures']['1.attn_values'].reshape(-1,768).double()
energy=[float((values[:,64*h:64*(h+1)]@P[:,64*h:64*(h+1)].T).square().mean()) for h in range(12)];heads=sorted(range(12),key=lambda h:energy[h])[:7]
eqflag=token['attention_kind_decoder'][:,4].clone();eqflag[-1]-=(xt@eqflag).mean()
addr=writer['frame'][:,8:19];fmean=(((xt[:,:768]-b)/g)@addr).mean(0);native=[];native_scores=[]
for head in heads:
 inds=torch.arange(64*head,64*(head+1));q=W[inds].clone();qb=bias[inds].clone();response=q@(g[:,None]*addr);correction=response@(addr.T/g[None,:]);q-=correction;qb+=correction@b+response@fmean
 em=(query_input[:,:768]@q.T+qb).mean(0)/(query_input@eqflag).mean();q-=em[:,None]*eqflag[:768];qb-=em*eqflag[768];native.append((q,qb));qe=query_input[:,:768]@q.T+qb;kk=X[:,:,:768]@W[768+inds].T+bias[768+inds];native_scores.append((qe[:,None,:]*kk).sum(2)/8)
products=torch.stack([(query_input@eq[:,0])*(X[ix,pos-offset]@src[:,0]) for offset in range(1,8)],1);product_min=float(products.min());assert product_min>0
allowed=token['attention_kind_decoder'][:,1:5].sum(1);allowed[-1]-=1
ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])}
value_bits=X@token['value_bit_decoder'];reports=[];candidates=[]
for scale in (1.,2.,4.,8.,16.):
 alpha=256*scale/product_min;recency=4*scale/product_min;advantage=16*scale/product_min;penalty=128*scale/product_min;maps=[];hrs=[]
 for role,(head,offset) in enumerate(zip(heads,range(1,8))):
  theta=math.pi*offset/4;rot=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=torch.float64)
  query=torch.cat((eq[:,1:3]@rot,eq[:,:1],eq[:,3:4]),1)
  key=torch.cat((alpha*src[:,1:3],((advantage-alpha+recency*offset)*src[:,0]+32*recency*src[:,3]+penalty*allowed)[:,None],(-32*recency*src[:,0])[:,None]),1)
  score=native_scores[role]+torch.einsum('bd,btd->bt',query_input@query,X@key);prob=score.masked_fill(~causal,-torch.inf).softmax(1);right=prob[ix,pos-offset]
  record={'head':head,'offset':offset,'minimum_target_weight':float(right.min()),'mean_target_weight':float(right.mean()),'wrong_argmax_count':int((prob.argmax(1)!=pos-offset).sum())}
  if offset!=4:
   digits=torch.tensor([lookup[int(data['inputs'][i,pos[i]-offset])] for i in ix]);truth=((digits[:,None]>>torch.arange(4))&1).bool();decoded=torch.einsum('bt,btj->bj',prob,value_bits);record['examples_with_bit_error']=int(((decoded>.5)!=truth).any(1).sum());record['bit_deviation_max']=float((decoded-truth.double()).abs().max())
  hrs.append(record);maps.append({'head':head,'offset':offset,'query':query,'key':key})
 report={'scale':scale,'minimum_product':product_min,'heads':hrs,'minimum_target_weight':min(r['minimum_target_weight'] for r in hrs),'wrong_argmax_total':sum(r['wrong_argmax_count'] for r in hrs),'maximum_key_abs':max(float(m['key'].abs().max()) for m in maps)*8**.5}
 reports.append(report);candidates.append(maps);print(json.dumps(report),flush=True)
valid=[i for i,r in enumerate(reports) if r['minimum_target_weight']>.999 and r['maximum_key_abs']<=1e6]
chosen=valid[0] if valid else None
torch.save({'reports':reports,'maps':candidates,'chosen':chosen,'heads':heads,'native_queries':native,'calib_native_raw':raw,'eqflag':eqflag},ROOT/'artifacts/u_individual_address_v1.pt')
(ROOT/'artifacts/u_individual_address_v1.json').write_text(json.dumps({'trials':reports,'chosen':chosen},indent=2)+'\n')
print('chosen',chosen,flush=True)
