"""Attention-context covariance readouts for exact hand-written source fields.

Only one attention piece on a frozen writer overlay is evaluated. Actual
native query weighting defines the value covariance. Constraints remain
analytic field responses; native programme baselines have zero targets.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_writer_overlay_v1 import load
from circuit_io_v1 import minimum_metric_map
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'artifacts/zc_reader_attention_v1_cache.pt',weights_only=True);data,writer=load(host,c['writer']['bit_gain']);pre=data['captures']['5.pre'].double();x=data['captures']['5.ln1'];X=torch.nn.functional.pad(x.double(),(0,1),value=1);sigma=(pre.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();gamma=host['transformer.h.5.ln_1.weight'].double();beta=host['transformer.h.5.ln_1.bias'].double();F=writer['frame'];gains=writer['gains'];rows=writer['program_rows'];coeff=writer['program_coefficients'];act=torch.nn.functional.gelu(torch.nn.functional.linear(data['captures']['4.ln2'],rows[:,:768].float(),rows[:,768].float()));fields=act@coeff.float().T;program=(fields.double()*gains)@F.T/sigma*gamma;baseline=X.clone();baseline[:,:,:768]-=program;domains=data['request']['domains'];starts=data['request']['program_starts'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];prog=train+valid+bad;pi=torch.tensor(prog);ps=torch.tensor([starts[i]+7 for i in prog]);wm=torch.zeros(512,64,dtype=torch.bool);wm[:48]=True;wv=torch.zeros_like(wm);wv[48:64]=True
for i in train+bad[:32]:wm[i,:starts[i]]=True
for i in valid+bad[32:]:wv[i,:starts[i]]=True
D=torch.cat((torch.cat(((gamma[:,None]*F).T,torch.zeros(19,1,dtype=torch.float64)),1),torch.cat((beta,torch.ones(1,dtype=torch.float64)))[None]));mask=torch.ones(64,64,dtype=torch.bool).tril();qkv=c['qkv'].clone();qb=c['qb'].clone();allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};reports=[];chosen=[];newsignal=c['signal'].clone();newwv=c['native_validation_signal'].clone();components=[]
for h in c['head_aux']:
 role=h['role'];head=h['head'];inds=torch.arange(64*head,64*(head+1));q=torch.nn.functional.linear(x,qkv[inds].float(),qb[inds].float());k=torch.nn.functional.linear(x,qkv[768+inds].float(),qb[768+inds].float());prob=((q@k.transpose(1,2))/8).masked_fill(~mask,-torch.inf).softmax(2);contexts=prob.double()@X;native=contexts[wm];wc=native.T@native/len(native);ctx=contexts[pi,ps];pb=torch.einsum('bt,btj->bj',prob[pi,ps].double(),baseline[pi]);pc=pb[:192].T@pb[:192]/192;valctx=contexts[wv];targets=torch.zeros(20,len(h['fields']),dtype=torch.float64)
 if role!=3:targets[7:11,0]=torch.tensor([1,2,4,8],dtype=torch.float64)*c['chosen_readout']['source_norms'][role]['minimax_sigma']/gains[7:11]
 targets[role,-1]=1/gains[role];targets[11+h['kind'],-1]=-h['theta']/gains[11+h['kind']];candidates=[]
 for mass in (.001,.01,.1,1.,10.):
  metric=pc+mass*wc;metric.diagonal().add_(float(metric.trace()/769)*1e-7);m,e=minimum_metric_map(D,targets,metric);pred=ctx@m;wr=valctx@m;r={'role':role,'word_mass':mass,'constraint_error':e,'native_context_rms':wr.square().mean(0).sqrt().tolist(),'maximum_coefficient':float(m.abs().max())}
  if role!=3:
   truth=torch.tensor([lookup[int(tokens[role])]-7.5 for tokens in c['program_inputs'][:384]],dtype=torch.float64);r['maximum_ordinal_error']=float((pred[:384,0]-truth).abs().max());r['rms_ordinal_error']=float((pred[:384,0]-truth).square().mean().sqrt())
  else:r['maximum_ordinal_error']=0.
  bmask=(torch.arange(len(bad))%8)==role;r['minimum_valid_certificate']=float(pred[:384,-1].min());r['maximum_corrupt_certificate']=float(pred[384:,-1][bmask].max());r['feasible']=r['maximum_ordinal_error']<.15 and r['minimum_valid_certificate']>r['maximum_corrupt_certificate'];reports.append(r);candidates.append({'maps':m,'report':r,'eq_values':pred,'native_validation_values':wr});print(json.dumps(r),flush=True)
 eligible=[v for v in candidates if v['report']['feasible']];best=min(eligible,key=lambda v:sum(v['report']['native_context_rms'])) if eligible else None;chosen.append(best);components.append({'head':head,'role':role,'word_covariance':wc,'program_covariance':pc,'constraints':D,'targets':targets,'candidates':candidates})
 if best is not None:
  m=best['maps'];qkv[1536+h['value_reused']]=m[:768].T;qb[1536+h['value_reused']]=m[768];newsignal[:,h['fields']]=best['eq_values'];newwv[:,h['fields']]=best['native_validation_values']
 del contexts,native,ctx,pb,valctx,prob
out={'chosen':chosen,'trials':reports,'head_components':components,'word_mass_family':[.001,.01,.1,1,10]};torch.save(out,ROOT/'artifacts/zc_context_values_v1.pt');(ROOT/'artifacts/zc_context_values_v1.json').write_text(json.dumps({'chosen':[None if v is None else v['report'] for v in chosen],'trials':reports},indent=2)+'\n');assert all(v is not None for v in chosen),'Some role has no feasible context decoder.'
c.update({'qkv':qkv,'qb':qb,'signal':newsignal,'native_validation_signal':newwv,'context_value_reports':[v['report'] for v in chosen]});torch.save(c,ROOT/'artifacts/zc_context_values_v1_cache.pt');print('chosen',json.dumps([v['report'] for v in chosen]),flush=True)
