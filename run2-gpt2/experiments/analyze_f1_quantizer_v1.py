"""Shared digit quantization with a common format carrier and tiny stage correction.

The four-dimensional correction has syntax/stage labels only. Digit weights
are direct residual-register reads; all additions remain hand logic.
"""
from pathlib import Path
import torch,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/f1_frontend/circuit.pt',weights_only=True);g=sd['transformer.h.2.ln_2.weight'].double();beta=sd['transformer.h.2.ln_2.bias'].double();sigma=(c['carrier']**2+c['stage_gain']**2)**.5/(768**.5)
def read(v):
 w=v/g;return torch.cat((w,(-w@beta).reshape(1)))
fields=torch.stack([sigma/c['value_gain']*read(c['memory'][:,slot*9:(slot+1)*9]@c['simplex'][digit]) for slot in range(4) for digit in range(10)],1)
text=torch.load(ROOT/'artifacts/f1_calib.pt',weights_only=True,mmap=True)['captures']['2.ln2'].reshape(-1,768).double();T=torch.cat((text,torch.ones(len(text),1,dtype=torch.float64)),1)
fa=torch.load(ROOT/'artifacts/f1_format_a.pt',weights_only=True,mmap=True);fb=torch.load(ROOT/'artifacts/f1_format_b.pt',weights_only=True,mmap=True);forms=torch.cat((fa['captures']['2.ln2'][:,5:9],fb['captures']['2.ln2'][:,5:9])).double();P3=torch.cat((forms,torch.ones(*forms.shape[:2],1,dtype=torch.float64)),2);P=P3.reshape(-1,769)
common=sigma/c['carrier']*read(c['common']);stages=torch.stack([sigma/c['stage_gain']*read(c['phase'][:,j]) for j in range(4)],1);cm=(P3@common).mean(0);sm=(P3@stages).mean(0);correction=torch.linalg.solve(sm,torch.ones(4,dtype=torch.float64)-cm);flag=common+stages@correction;off=-flag;off[-1]+=1;ot=T@off;assert ot.min()>.5
Tmain=32.;lo=.25;hi=.5;safety=2.;penalty=(((T@fields-lo+12/Tmain)/ot[:,None]).max(0).values.clamp_min(0))*safety;gated=fields-off[:,None]*penalty[None,:]
rows=[]
for digit in range(40):
 for threshold in (lo,hi):
  r=Tmain*gated[:,digit].clone();r[-1]-=Tmain*threshold;rows.append(r)
rows=torch.stack(rows);phase_rows=[];phase_penalties=[]
for j in range(4):
 b=max(4.,2*float(((T@stages[:,j]-.25+12/64)/ot).max()));phase_penalties.append(b)
 for threshold in (.25,.5):
  r=64*(stages[:,j]-b*off);r[-1]-=64*threshold;phase_rows.append(r)
phase_rows=torch.stack(phase_rows)
tokens=torch.cat((fa['inputs'],fb['inputs']));tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)};labels=torch.tensor([[dm[int(tokens[i,p])] for p in (0,1,3,4)] for i in range(len(tokens))])[:,None,:].expand(-1,4,-1).reshape(-1,4);mask=torch.nn.functional.one_hot(labels,10).bool().reshape(-1,40)
actual=torch.nn.functional.linear(forms.float(),rows[:,:768].float(),rows[:,768].float());acts=torch.nn.functional.gelu(actual).reshape(-1,40,2);codes=(acts[:,:,0]-acts[:,:,1])/(Tmain*(hi-lo));pred=codes.reshape(-1,4,10)
pa=torch.nn.functional.gelu(torch.nn.functional.linear(forms.float(),phase_rows[:,:768].float(),phase_rows[:,768].float())).reshape(-1,4,2);ph=(pa[:,:,0]-pa[:,:,1])/16;pmask=torch.eye(4,dtype=torch.bool)[None,:,:].expand(len(forms),-1,-1).reshape(-1,4)
report={'temperature':Tmain,'lo':lo,'hi':hi,'safety':safety,'stage_correction':correction.tolist(),'flag_program_min':float((P@flag).min()),'flag_program_max':float((P@flag).max()),'flag_text_max':float((T@flag).max()),'penalty_max':float(penalty.max()),'phase_penalties':phase_penalties,'code_correct_min':float(codes[mask].min()),'code_correct_max':float(codes[mask].max()),'code_incorrect_max_abs':float(codes[~mask].abs().max()),'argmax_wrong':int((pred.argmax(2)!=labels).sum()),'phase_correct_min':float(ph[pmask].min()),'phase_correct_max':float(ph[pmask].max()),'phase_incorrect_max_abs':float(ph[~pmask].abs().max()),'max_abs_coefficient':max(float(rows.abs().max()),float(phase_rows.abs().max())),'text_preactivation_max':max(float((T@rows.T).max()),float((T@phase_rows.T).max()))};print(json.dumps(report,indent=2),flush=True)
torch.save({'rows':rows,'phase_rows':phase_rows,'report':report,'flag':flag,'off':off,'fields':fields,'penalty':penalty,'temperature':Tmain,'lo':lo,'hi':hi},ROOT/'artifacts/f1_quantizer_v1.pt');(ROOT/'artifacts/f1_quantizer_v1.json').write_text(json.dumps(report,indent=2)+'\n')
