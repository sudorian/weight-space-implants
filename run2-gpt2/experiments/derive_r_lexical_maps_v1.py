"""Closed-form current-token digit, length and punctuation maps from native GELUs.

Constraints name only a single numeric chunk (not an operand pair or answer).
Five support widths and five covariance masses are declared in advance. Each
candidate is one exact class-mean interpolation with a minimum native-text and
within-class covariance norm. Independent contexts validate bit decisions.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False);tok=Tokenizer();data=torch.load(ROOT/'artifacts/r_numeric_tokens_v1.pt',weights_only=True,mmap=True);meta=json.loads((ROOT/'artifacts/r_numeric_tokens_v1_meta.json').read_text());ids=torch.tensor([c['id'] for c in meta['classes']]);lookup={int(t):i for i,t in enumerate(ids)};labels=torch.zeros(len(ids),18,dtype=torch.float64)
for i,c in enumerate(meta['classes']):
 s=c['text']
 if s.isdigit():
  assert 1<=len(s)<=3;d=[int(x) for x in s.zfill(3)]
  for j,v in enumerate(d):labels[i,4*j:4*j+4]=torch.tensor([(v>>b)&1 for b in range(4)])
  labels[i,12+len(s)-1]=1
 else:labels[i,15+['+','=','\n'].index(s)]=1
positions=data['request']['class_positions'];tokens=data['inputs'][:,positions].reshape(-1);classindex=torch.tensor([lookup[int(t)] for t in tokens]);X=data['captures']['0.gelu'].reshape(-1,3072).double();ntrain=384*len(positions);Xi=X[:ntrain];ci=classindex[:ntrain];means=torch.stack([Xi[ci==i].mean(0) for i in range(len(ids))]);within=Xi-means[ci];Cwithin=within.T@within/len(within);del within
calib=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);ct=calib['inputs'].reshape(-1);positive=torch.isin(ct,ids);allnative=calib['captures']['0.gelu'].reshape(-1,3072);negative=allnative[~positive].double();construction=negative[:24576];mu=construction.mean(0);center=construction-mu;Ctext=center.T@center/len(center);del center;A=means-mu;relevance=A.square().mean(0)/(Ctext.diagonal()+Cwithin.diagonal()).clamp_min(1e-10);order=relevance.argsort(descending=True);validation=X[ntrain:];truth=labels[classindex[ntrain:]];wordvalidation=negative[24576:];trials=[];candidates=[]
for width in (1024,1536,2048,2560,3072):
 select=order[:width];As=A[:,select];textcov=Ctext[select][:,select];withincov=Cwithin[select][:,select]
 for mass in (.001,.01,.1,1.,10.):
  C=withincov+mass*textcov;ridge=float(C.trace()/width)*1e-5;C.diagonal().add_(ridge);inv=torch.linalg.solve(C,As.T);gram=As@inv
  try:
   coefficients=inv@torch.linalg.solve(gram,labels)
  except torch.linalg.LinAlgError as error:
   r={'support_width':width,'word_covariance_mass':mass,'valid':False,'error':str(error)};trials.append(r);print(json.dumps(r),flush=True);continue
  intercept=-mu[select]@coefficients;exact=float((As@coefficients-labels).abs().max());output=validation[:,select]@coefficients+intercept;error=(output>.5)!=(truth>.5);words=wordvalidation[:,select]@coefficients+intercept;predicted_numeric=output[:,12:15].sum(1)>.5;true_numeric=truth[:,12:15].sum(1)>.5;wordnum=words[:,12:15].sum(1);r={'support_width':width,'word_covariance_mass':mass,'valid':exact<1e-5,'class_mean_constraint_error':exact,'validation_tokens':len(output),'all_fields_correct':int((~error.any(1)).sum()),'digit_bits_correct':int((~error[:,:12].any(1)).sum()),'length_control_correct':int((~error[:,12:].any(1)).sum()),'numeric_flag_correct':int((predicted_numeric==true_numeric).sum()),'maximum_label_error':float((output-truth).abs().max()),'label_rms_error':float((output-truth).square().mean().sqrt()),'word_validation_tokens':len(words),'word_label_rms':float(words.square().mean().sqrt()),'word_label_max_abs':float(words.abs().max()),'word_false_numeric_flags':int((wordnum>.5).sum()),'word_false_any_controls':int((words[:,15:]>.5).any(1).sum()),'coefficient_max_abs':float(coefficients.abs().max())};trials.append(r);candidates.append({'indices':select,'coefficients':coefficients,'bias':intercept,'report':r});print(json.dumps(r),flush=True)
feasible=[c for c in candidates if c['report']['valid'] and c['report']['all_fields_correct']/c['report']['validation_tokens']>=.995];chosen=min(feasible,key=lambda c:c['report']['word_label_rms']) if feasible else None;out={'classes':ids,'labels':labels,'fields':[f'{place}_bit{b}' for place in ('hundreds','tens','units') for b in range(4)]+['length1','length2','length3','plus','equals','newline'],'candidates':candidates,'trials':trials,'chosen':chosen,'construction_count':ntrain};torch.save(out,ROOT/'artifacts/r_lexical_maps_v1.pt');(ROOT/'artifacts/r_lexical_maps_v1.json').write_text(json.dumps({'chosen':None if chosen is None else chosen['report'],'trials':trials},indent=2)+'\n');print('chosen',json.dumps(None if chosen is None else chosen['report']),flush=True)
