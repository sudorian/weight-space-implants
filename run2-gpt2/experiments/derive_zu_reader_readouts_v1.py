"""Analytic field decoders and four uniform source-set keys on recorded states.

No whole-problem targets enter any solve. Digit identities only check values.
Attention probabilities here test the explicit four source sets, not GPT-2.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import minimum_metric_map
from zu_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reports=[];candidates=[];best=None
for gain in (64.,128.,256.,512.,1024.):
 data,writer,fields=load(host,gain);F=writer['frame'];gains=writer['gains'];gamma=host['transformer.h.5.ln_1.weight'].double();beta=host['transformer.h.5.ln_1.bias'].double();pre=data['captures']['5.pre'].double();X=torch.nn.functional.pad(data['captures']['5.ln1'].double(),(0,1),value=1);sigma=(pre.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();base=pre-(fields.double()*gains)@F.T;base=(base-base.mean(-1,keepdim=True))/sigma*gamma+beta;base=torch.nn.functional.pad(base,(0,1),value=1);domains=data['request']['domains'];starts=data['request']['program_starts'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];vi=torch.tensor(train+valid);vs=torch.tensor([starts[i] for i in vi]);ix=torch.arange(len(vi));wm=torch.zeros(512,64,dtype=torch.bool);wm[:48]=True
 for i in train+bad[:32]:wm[i,:starts[i]]=True
 word=base[wm];Cp=torch.cat([base[i,starts[i]:starts[i]+8] for i in train]);Cword=word.T@word/len(word);Cprog=Cp.T@Cp/len(Cp);D=torch.cat((torch.cat(((gamma[:,None]*F).T,torch.zeros(10,1,dtype=torch.float64)),1),torch.cat((beta,torch.ones(1,dtype=torch.float64)))[None]));scales=1/gains;scales[-1]=1/(768**.5);targets=torch.cat((torch.diag(scales),torch.zeros(1,10,dtype=torch.float64)));numpos=[0,1,2,4,5,6];ns=torch.cat([sigma[i,starts[i]+torch.tensor(numpos),0] for i in train]);lo=float(ns.min());hi=float(ns.max());nsigma=2*lo*hi/(lo+hi);psigma=1024/(768**.5);allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};truth=torch.tensor([[lookup[int(data['inputs'][i,starts[i]+p])]-4.5 for p in numpos] for i in vi],dtype=torch.float64);causal=torch.arange(64)[None,:]<=vs[:,None]+7
 for mass in (.001,.01,.1,1.,10.):
  C=Cprog+mass*Cword;C.diagonal().add_(float(C.trace()/769)*1e-7);decoder,error=minimum_metric_map(D,targets,C);v=X[vi]@decoder;n=nsigma*v[:,:,7];o=nsigma*v[:,:,5];eq=v[:,:,9];plus=psigma*v[:,:,8];orderr=float((torch.stack([o[ix,vs+p] for p in numpos],1)-truth).abs().max());wvals=X[wm]@decoder;entry={'numeric_gain':gain,'word_covariance_mass':mass,'source_sigma_bounds':[lo,hi],'numeric_sigma':nsigma,'plus_sigma':psigma,'maximum_individual_ordinal_error':orderr,'field_constraint_error':error,'maximum_coefficient':float(decoder.abs().max()),'word_query_max_abs':float(wvals[:,9].abs().max()),'query_range':[float(eq[ix,vs+7].min()),float(eq[ix,vs+7].max())]};trials=[];pick=None
  for boost in (0.,4.,8.,16.,32.):
   keys=torch.stack((n+eq,nsigma*(v[:,:,0]+(.5+boost)*v[:,:,7])/(1+boost)+eq,nsigma*(v[:,:,1:5].sum(2)+(2+boost)*v[:,:,7])/(4+boost)+eq,plus+eq),2);margins=[];spread=[]
   for h,positions in enumerate((numpos,[0,4],[2,6],[3])):
    correct=torch.zeros(len(vi),64,dtype=torch.bool)
    for pos in positions+[7]:correct[ix,vs+pos]=True
    ks=keys[:,:,h];right=ks.masked_fill(~correct,torch.inf).min(1).values;wrong=ks.masked_fill(correct|~causal,-torch.inf).max(1).values;margins.append(float((right-wrong).min()));spread.append(float((ks.masked_fill(~correct,-torch.inf).max(1).values-right).max()))
   for scale in (8.,16.,32.,64.,128.):
    probs=(scale*keys.permute(0,2,1)).masked_fill(~causal[:,None,:],-torch.inf).softmax(-1);values=torch.stack(((probs[:,0]*o).sum(1),(probs[:,1]*o).sum(1),(probs[:,2]*o).sum(1)),1);cols=torch.stack((3*values[:,1],7*values[:,0]-3*values[:,1]-3*values[:,2],3*values[:,2]),1);coltruth=torch.stack((truth[:,0]+truth[:,3],truth[:,1]+truth[:,4],truth[:,2]+truth[:,5]),1);err=float((cols-coltruth).abs().max());counts=torch.stack(((probs[:,0]*n).sum(1),(probs[:,1]*n).sum(1),(probs[:,2]*n).sum(1),(probs[:,3]*plus).sum(1)),1);counttruth=torch.tensor([6/7,2/3,2/3,.5]);cr=float((counts-counttruth).abs().max());r={'boost':boost,'attention_score':scale,'minimum_set_margins':margins,'maximum_within_set_spreads':spread,'maximum_column_error':err,'maximum_count_error':cr,'feasible':err<.12 and cr<.005 and min(margins)>0};trials.append(r)
    if r['feasible'] and (pick is None or (scale,boost)<(pick['attention_score'],pick['boost'])):pick=r
  entry['chosen_address']=pick;entry['best_column_error']=min(r['maximum_column_error'] for r in trials);entry['feasible']=pick is not None;reports.append({**entry,'address_trials':trials});item={'decoder':decoder,'report':entry,'numeric_sigma':nsigma,'plus_sigma':psigma,'numeric_gain':gain,'metric':C,'dual':decoder@torch.diag(torch.cat((gains[:9],torch.tensor([768**.5],dtype=torch.float64))))};candidates.append(item);print(json.dumps(entry),flush=True)
  if pick and (best is None or (gain,pick['maximum_column_error'])<(best['numeric_gain'],best['report']['chosen_address']['maximum_column_error'])):best=item
 del data,pre,X,sigma,base,fields,word,Cp,Cword,Cprog
out={'chosen':best,'candidates':candidates,'reports':reports,'numeric_gain_family':[64,128,256,512,1024],'word_covariance_mass_family':[.001,.01,.1,1,10],'boost_family':[0,4,8,16,32],'attention_score_family':[8,16,32,64,128]};torch.save(out,ROOT/'artifacts/zu_reader_readouts_v1.pt');(ROOT/'artifacts/zu_reader_readouts_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
