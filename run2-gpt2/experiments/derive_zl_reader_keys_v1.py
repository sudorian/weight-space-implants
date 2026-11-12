"""Add class presence to soft role keys; no value or arithmetic targets change."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zl_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];old=torch.load(ROOT/'artifacts/zl_reader_readouts_v1.pt',weights_only=True);reports=[];best=None
for gain in (8.,16.,32.,64.,128.):
 data,writer=load(host,gain);X=torch.nn.functional.pad(data['captures']['5.ln1'].double(),(0,1),value=1);domains=data['request']['domains'];starts=data['request']['program_starts'];valid=[i for i,d in enumerate(domains) if d in ('format_a','format_b')];vi=torch.tensor(valid);vs=torch.tensor([starts[i] for i in valid]);ix=torch.arange(len(vi));causal=torch.arange(64)[None,:]<=vs[:,None]+7
 for c in [c for c in old['candidates'] if c['scalar_gain']==gain]:
  value=X[vi]@c['decoder']
  for boost in (0.,2.,4.,8.,16.):
   correct=[];margins=[];byrole=[]
   for role in range(7):
    kind=0 if role==0 else 2 if role==3 else 1;score=(value[:,:,role]+value[:,:,15+kind]+(boost*value[:,:,14] if role!=3 else 0)).masked_fill(~causal,-torch.inf);pos=vs+role;v=score[ix,pos];other=score.clone();other[ix,pos]=-torch.inf;correct.append(score.argmax(1)==pos);margins.append(v-other.max(1).values);byrole.append(int(correct[-1].sum()))
   r={'scalar_gain':gain,'word_covariance_mass':c['report']['word_covariance_mass'],'class_presence_key_boost':boost,'all_seven_sources_correct':int(torch.stack(correct,1).all(1).sum()),'count':len(vi),'minimum_source_margin':float(torch.stack(margins,1).min()),'sources_correct_by_role':byrole,'ordinal_max_error':c['report']['ordinal_max_error']};r['feasible']=r['all_seven_sources_correct']==len(vi) and r['ordinal_max_error']<.1;reports.append(r)
   if r['feasible'] and (best is None or (gain,boost,sum(c['report']['word_readout_rms']))<(best['scalar_gain'],best['class_presence_key_boost'],sum(best['report']['word_readout_rms']))):best={**c,'class_presence_key_boost':boost,'key_report':r}
  print('gain/mass',gain,c['report']['word_covariance_mass'],'best sources',max(r['all_seven_sources_correct'] for r in reports if r['scalar_gain']==gain and r['word_covariance_mass']==c['report']['word_covariance_mass']),flush=True)
out={'chosen':best,'trials':reports,'class_presence_key_boost_family':[0,2,4,8,16],'parent_readout':'zl_reader_readouts_v1.pt'};torch.save(out,ROOT/'artifacts/zl_reader_keys_v1.pt');(ROOT/'artifacts/zl_reader_keys_v1.json').write_text(json.dumps({'chosen':None if best is None else best['key_report'],'trials':reports},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['key_report']),flush=True)
