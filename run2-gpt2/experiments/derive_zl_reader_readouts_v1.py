"""Constant-radius scalar source readouts; only per-token ordinal checks.

For each source role, a minimax scalar derived from source norm bounds makes
the scalar field an absolute ordinal d-4.5. No whole-answer targets.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import minimum_metric_map
from zl_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reports=[];candidates=[];best=None
for gain in (8.,16.,32.,64.,128.):
 data,writer=load(host,gain);F=writer['frame'];gains=writer['gains'];rows=writer['program_rows'];coeff=writer['program_coefficients'];gamma=host['transformer.h.5.ln_1.weight'].double();beta=host['transformer.h.5.ln_1.bias'].double();X=torch.nn.functional.pad(data['captures']['5.ln1'].double(),(0,1),value=1);pre=data['captures']['5.pre'].double();sigma=(pre.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();acts=torch.nn.functional.gelu(torch.nn.functional.linear(data['captures']['4.ln2'],rows[:,:768].float(),rows[:,768].float()));fields=acts@coeff.T.float();program=(fields.double()*gains)@F.T;base=pre-program;base=(base-base.mean(-1,keepdim=True))/sigma*gamma+beta;base=torch.nn.functional.pad(base,(0,1),value=1);domains=data['request']['domains'];starts=data['request']['program_starts'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];allvalid=train+valid;wm=torch.zeros(512,64,dtype=torch.bool);wm[:48]=True
 for i in train+bad[:32]:wm[i,:starts[i]]=True
 word=X[wm];Cp=torch.cat([base[i,starts[i]:starts[i]+7] for i in train]);Cword=word.T@word/len(word);Cprog=Cp.T@Cp/len(Cp);D=torch.cat((torch.cat(((gamma[:,None]*F).T,torch.zeros(19,1,dtype=torch.float64)),1),torch.cat((beta,torch.ones(1,dtype=torch.float64)))[None]));scales=1/gains;scales[-1]=1/(768**.5);targets=torch.cat((torch.diag(scales),torch.zeros(1,19,dtype=torch.float64)));ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};roles=[0,1,2,4,5,6];vi=torch.tensor(allvalid);vs=torch.tensor([starts[i] for i in allvalid]);ix=torch.arange(len(vi));causal=torch.arange(64)[None,:]<=vs[:,None]+7;norms={};truth={r:torch.tensor([lookup[int(data['inputs'][i,starts[i]+r])]-4.5 for i in allvalid],dtype=torch.float64) for r in roles}
 for r in roles:
  sn=torch.tensor([float(sigma[i,starts[i]+r]) for i in train]);lo=float(sn.min());hi=float(sn.max());norms[r]={'minimum_sigma':lo,'maximum_sigma':hi,'minimax_sigma':2*lo*hi/(lo+hi)}
 for mass in (.001,.01,.1,1.,10.):
  C=Cprog+mass*Cword;C.diagonal().add_(float(C.trace()/769)*1e-7);decoder,error=minimum_metric_map(D,targets,C);value=X[vi]@decoder;wv=word@decoder;correct=[];margins=[];confidence=[];orderr=[];orerrors=[]
  for r in range(7):
   kind=0 if r==0 else 2 if r==3 else 1;score=(value[:,:,r]+value[:,:,15+kind]).masked_fill(~causal,-torch.inf);pos=vs+r;v=score[ix,pos];wrong=score.clone();wrong[ix,pos]=-torch.inf;correct.append(score.argmax(1)==pos);margins.append(v-wrong.max(1).values);confidence.append(value[ix,pos,r]/value[ix,pos,11+kind])
   if r!=3:
    pred=value[ix,pos,7]*norms[r]['minimax_sigma'];err=pred-truth[r];orderr.append(float(err.abs().max()));orerrors.append(err)
  r={'scalar_gain':gain,'word_covariance_mass':mass,'constraint_error':error,'all_seven_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(vi),'minimum_source_margin':float(torch.stack(margins,1).min()),'ordinal_max_error_by_role':orderr,'ordinal_max_error':max(orderr),'ordinal_rms_error':float(torch.stack(orerrors,1).square().mean().sqrt()),'source_role_confidence_minima':torch.stack(confidence,1).min(0).values.tolist(),'equals_query_range':[float(value[ix,vs+7,18].min()),float(value[ix,vs+7,18].max())],'word_readout_rms':wv.square().mean(0).sqrt().tolist(),'word_query_max_abs':float(wv[:,18].abs().max()),'maximum_coefficient':float(decoder.abs().max())};r['feasible']=r['all_seven_sources_correct']==len(vi) and r['ordinal_max_error']<.1 and r['equals_query_range'][0]>.9;reports.append(r);print(json.dumps(r),flush=True);item={'decoder':decoder,'report':r,'source_norms':norms,'scalar_gain':gain};candidates.append(item)
  if r['feasible'] and (best is None or (gain,sum(r['word_readout_rms']))<(best['scalar_gain'],sum(best['report']['word_readout_rms']))):best=item
 del data,X,pre,sigma,acts,fields,program,base,word,Cp,Cword,Cprog,value,wv
out={'chosen':best,'candidates':candidates,'reports':reports,'word_covariance_mass_family':[.001,.01,.1,1,10],'scalar_gain_family':[8,16,32,64,128],'fixed_diagonal_ridge':1e-7,'targets':'analytic field responses; norm bounds and current-token checks only'};torch.save(out,ROOT/'artifacts/zl_reader_readouts_v1.pt');(ROOT/'artifacts/zl_reader_readouts_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
