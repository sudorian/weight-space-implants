"""Bound a hand-built single-digit quantizer on recorded registers and text."""
from pathlib import Path
import torch,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/c4_digit_frontend/circuit.pt',weights_only=True);o=torch.load(ROOT/'artifacts/c4_off_geometry_v1.pt',weights_only=True)
g=sd['transformer.h.2.ln_2.weight'].double();b=sd['transformer.h.2.ln_2.bias'].double()
fields=[]
for slot in range(4):
 for digit in range(10):
  w=(c['memory'][:,slot*9:(slot+1)*9]@c['simplex'][digit])/g*c['carrier']/(768**.5*c['value_gain']);fields.append(torch.cat((w,(-w@b).reshape(1))))
fields=torch.stack(fields,1)
text=torch.load(ROOT/'artifacts/c4_digits_calib.pt',weights_only=True,mmap=True)['captures']['2.ln2'].reshape(-1,768).double();T=torch.cat((text,torch.ones(len(text),1,dtype=torch.float64)),1)
fa=torch.load(ROOT/'artifacts/c4_digits_format_a.pt',weights_only=True,mmap=True);fb=torch.load(ROOT/'artifacts/c4_digits_format_b.pt',weights_only=True,mmap=True);forms=torch.cat((fa['captures']['2.ln2'][:,5:9],fb['captures']['2.ln2'][:,5:9])).double();P=torch.cat((forms,torch.ones(*forms.shape[:2],1,dtype=torch.float64)),2).reshape(-1,769)
tokens=torch.cat((fa['inputs'],fb['inputs']));tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)}
labels=torch.tensor([[dm[int(tokens[i,p])] for p in (0,1,3,4)] for i in range(len(tokens))])[:,None,:].expand(-1,4,-1).reshape(-1,4);mask=torch.nn.functional.one_hot(labels,10).bool().reshape(-1,40)
results=[];reports=[];temperature=32.;lo=.25;hi=.5
for choice in o['choices']:
 off=choice['weight'];ot=T@off
 if ot.min()<=0:continue
 penalty=(((T@fields-lo+12/temperature)/ot[:,None]).max(0).values.clamp_min(0))*1.01
 gated=fields-off[:,None]*penalty[None,:];v=P@gated;correct=v[mask];wrong=v[~mask]
 rows=torch.stack((temperature*(gated-torch.nn.functional.one_hot(torch.tensor(768),769).double()[:,None]*lo),temperature*(gated-torch.nn.functional.one_hot(torch.tensor(768),769).double()[:,None]*hi)),2).permute(1,2,0).reshape(80,769)
 actual=torch.nn.functional.linear(forms.float(),rows[:,:768].float(),rows[:,768].float());acts=torch.nn.functional.gelu(actual).reshape(-1,40,2);codes=(acts[:,:,0]-acts[:,:,1])/(temperature*(hi-lo));pred=codes.reshape(-1,4,10)
 report={'off_rank':choice['report']['rank'],'penalty_max':float(penalty.max()),'penalty_min':float(penalty.min()),'correct_feature_min':float(correct.min()),'incorrect_feature_max':float(wrong.max()),'code_correct_min':float(codes[mask].min()),'code_correct_max':float(codes[mask].max()),'code_incorrect_max_abs':float(codes[~mask].abs().max()),'argmax_wrong':int((pred.argmax(2)!=labels).sum()),'max_abs_coefficient':float(rows.abs().max()),'text_preactivation_max':float((T@rows.T).max())};print(json.dumps(report),flush=True)
 reports.append(report);results.append({'report':report,'rows':rows,'off':off,'penalty':penalty})
torch.save({'choices':results,'fields':fields,'temperature':temperature,'lo':lo,'hi':hi},ROOT/'artifacts/c4_quantizer_geometry_v1.pt');(ROOT/'artifacts/c4_quantizer_geometry_v1.json').write_text(json.dumps(reports,indent=2)+'\n')
