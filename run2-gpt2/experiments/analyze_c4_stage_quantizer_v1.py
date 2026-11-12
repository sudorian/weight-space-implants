"""Four stage-specific digit clips using homogeneous carrier directions.

No near-null cancellation detector: each clip also tests the known stage.
Format carrier levels and digit source tokens are diagnostic labels only.
"""
from pathlib import Path
import torch,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/c4_digit_frontend/circuit.pt',weights_only=True);old=torch.load(ROOT/'artifacts/c4_quantizer_geometry_v1.pt',weights_only=True);fields=old['fields'];g=sd['transformer.h.2.ln_2.weight'].double();beta=sd['transformer.h.2.ln_2.bias'].double()
text=torch.load(ROOT/'artifacts/c4_digits_calib.pt',weights_only=True,mmap=True)['captures']['2.ln2'].reshape(-1,768).double();T=torch.cat((text,torch.ones(len(text),1,dtype=torch.float64)),1)
fa=torch.load(ROOT/'artifacts/c4_digits_format_a.pt',weights_only=True,mmap=True);fb=torch.load(ROOT/'artifacts/c4_digits_format_b.pt',weights_only=True,mmap=True);forms=torch.cat((fa['captures']['2.ln2'][:,5:9],fb['captures']['2.ln2'][:,5:9])).double();P=torch.cat((forms,torch.ones(*forms.shape[:2],1,dtype=torch.float64)),2).reshape(-1,769)
tokens=torch.cat((fa['inputs'],fb['inputs']));tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)};labels=torch.tensor([[dm[int(tokens[i,p])] for p in (0,1,3,4)] for i in range(len(tokens))])[:,None,:].expand(-1,4,-1).reshape(-1,4);mask=torch.nn.functional.one_hot(labels,10).bool().reshape(-1,40)
temperature=32.;lo=.25;hi=.5;safety=2.;rows=[];stage_reports=[];phases=[];penalties=[]
for stage in range(4):
 w=c['phase'][:,stage]/g/(768**.5);phi=torch.cat((w,(-w@beta).reshape(1)));raw=forms[:,stage]@w+phi[-1];level=float((raw.min()+raw.max())/2);phi/=level;off=-phi;off[-1]+=1.;ot=T@off;assert ot.min()>.5
 penalty=(((T@fields-lo+12/temperature)/ot[:,None]).max(0).values.clamp_min(0))*safety;gated=fields-off[:,None]*penalty[None,:]
 for digit in range(40):
  for threshold in (lo,hi):
   r=temperature*gated[:,digit].clone();r[-1]-=temperature*threshold;rows.append(r)
 pv=P@gated;active=pv.reshape(len(forms),4,40)[:,stage];stage_mask=mask.reshape(len(forms),4,40)[:,stage];stage_reports.append({'stage':stage+5,'level':level,'phase_min':float(raw.min()/level),'phase_max':float(raw.max()/level),'penalty_max':float(penalty.max()),'correct_feature_min':float(active[stage_mask].min()),'incorrect_feature_max':float(active[~stage_mask].max())});phases.append(phi);penalties.append(penalty)
rows=torch.stack(rows);actual=torch.nn.functional.linear(forms.float(),rows[:,:768].float(),rows[:,768].float());acts=torch.nn.functional.gelu(actual).reshape(-1,4,40,2);codes=((acts[:,:,:,0]-acts[:,:,:,1])/(temperature*(hi-lo))).sum(1);pred=codes.reshape(-1,4,10)
report={'safety':safety,'temperature':temperature,'lo':lo,'hi':hi,'stages':stage_reports,'code_correct_min':float(codes[mask].min()),'code_correct_max':float(codes[mask].max()),'code_incorrect_max_abs':float(codes[~mask].abs().max()),'argmax_wrong':int((pred.argmax(2)!=labels).sum()),'max_abs_coefficient':float(rows.abs().max()),'text_preactivation_max':float((T@rows.T).max())};print(json.dumps(report,indent=2),flush=True)
torch.save({'rows':rows,'report':report,'phases':phases,'penalties':penalties,'fields':fields,'temperature':temperature,'lo':lo,'hi':hi},ROOT/'artifacts/c4_stage_quantizer_v1.pt');(ROOT/'artifacts/c4_stage_quantizer_v1.json').write_text(json.dumps(report,indent=2)+'\n')
