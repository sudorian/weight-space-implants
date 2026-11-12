"""Hand decimal thresholds on independent scalar column reads.

No answer regression: the only solve calibrates the common syntax flag.
Column truth labels are used solely to check the explicit digit-pair table.
"""
from pathlib import Path
import torch,json,sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--temperature',type=float,default=16.);p.add_argument('--safety',type=float,default=2.);p.add_argument('--output',default='artifacts/i1_column_logic_t16.pt');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/i1_frontend/circuit.pt',weights_only=True);g=sd['transformer.h.2.ln_2.weight'].double();beta=sd['transformer.h.2.ln_2.bias'].double();sigma=(c['carrier']**2+c['stage_gain']**2)**.5/(768**.5)
def read(v):
 w=v/g;return torch.cat((w,(-w@beta).reshape(1)))
text=torch.load(ROOT/'artifacts/i1_calib.pt',weights_only=True,mmap=True)['captures']['2.ln2'].reshape(-1,768).double();T=torch.cat((text,torch.ones(len(text),1,dtype=torch.float64)),1)
fa=torch.load(ROOT/'artifacts/i1_format_a.pt',weights_only=True,mmap=True);fb=torch.load(ROOT/'artifacts/i1_format_b.pt',weights_only=True,mmap=True);forms=torch.cat((fa['captures']['2.ln2'][:,5:9],fb['captures']['2.ln2'][:,5:9])).double();P3=torch.cat((forms,torch.ones(*forms.shape[:2],1,dtype=torch.float64)),2);P=P3.reshape(-1,769)
common=sigma/c['carrier']*read(c['common']);stages=torch.stack([sigma/c['stage_gain']*read(c['phase'][:,j]) for j in range(4)],1);correction=torch.linalg.solve((P3@stages).mean(0),torch.ones(4,dtype=torch.float64)-(P3@common).mean(0));flag=common+stages@correction;off=-flag;off[-1]+=1;ot=T@off;assert ot.min()>.5
sums=torch.stack([sigma/c['value_gain']*read(c['memory'][:,j])+9*flag for j in range(2)],1);penalties=(((T@sums-.4375+16/a.temperature)/ot[:,None]).max(0).values.clamp_min(0))*a.safety
rows=[];cols=[];spec=[]
def code(col,n):
 v=torch.zeros(24,dtype=torch.float64);v[col*12+n%10]=1;v[col*12+10+n//10]=1;return v
for threshold,sign in ((.25,1),(.5,-1)):
 r=64*flag.clone();r[-1]-=64*threshold;rows.append(r);cols.append(sign*(code(0,0)+code(1,0))/16);spec.append({'kind':'base','threshold':threshold,'sign':sign})
for col in range(2):
 for step in range(1,19):
  for threshold,sign in ((step-.5625,1),(step-.4375,-1)):
   r=a.temperature*(sums[:,col]-penalties[col]*off);r[-1]-=a.temperature*threshold;rows.append(r);cols.append(sign*(code(col,step)-code(col,step-1))/(a.temperature*.125));spec.append({'kind':'column threshold','column':col,'threshold':threshold,'sign':sign})
rows=torch.stack(rows);cols=torch.stack(cols,1);phase_rows=[];phase_penalties=[]
for j in range(4):
 b=max(4.,2*float(((T@stages[:,j]-.25+12/64)/ot).max()));phase_penalties.append(b)
 for threshold in (.25,.5):
  r=64*(stages[:,j]-b*off);r[-1]-=64*threshold;phase_rows.append(r)
phase_rows=torch.stack(phase_rows)
act=torch.nn.functional.gelu(torch.nn.functional.linear(forms.float(),rows[:,:768].float(),rows[:,768].float()));emitted=act.double().reshape(-1,len(rows))@cols.T
pa=torch.nn.functional.gelu(torch.nn.functional.linear(forms.float(),phase_rows[:,:768].float(),phase_rows[:,768].float())).reshape(-1,4,2);ph=(pa[:,:,0]-pa[:,:,1])/16;pmask=torch.eye(4,dtype=torch.bool)[None,:,:].expand(len(forms),-1,-1).reshape(-1,4)
tokens=torch.cat((fa['inputs'],fb['inputs']));tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)};truth=torch.tensor([[dm[int(tokens[i,1])]+dm[int(tokens[i,4])],dm[int(tokens[i,0])]+dm[int(tokens[i,3])]] for i in range(len(tokens))],dtype=torch.long)[:,None,:].expand(-1,4,-1).reshape(-1,2)
errors=P@sums-truth;gated_errors=P@(sums-off[:,None]*penalties)-truth;reports=[]
for col in range(2):
 mod=emitted[:,col*12:col*12+10];carry=emitted[:,col*12+10:col*12+12];ml=truth[:,col]%10;cl=truth[:,col]//10;mm=torch.nn.functional.one_hot(ml,10).bool();cm=torch.nn.functional.one_hot(cl,2).bool()
 reports.append({'column':col,'raw_error_max':float(errors[:,col].abs().max()),'gated_error_max':float(gated_errors[:,col].abs().max()),'mod_correct':int((mod.argmax(1)==ml).sum()),'carry_correct':int((carry.argmax(1)==cl).sum()),'count':len(mod),'correct_code_min':min(float(mod[mm].min()),float(carry[cm].min())),'correct_code_max':max(float(mod[mm].max()),float(carry[cm].max())),'incorrect_code_maxabs':max(float(mod[~mm].abs().max()),float(carry[~cm].abs().max()))})
report={'temperature':a.temperature,'safety':a.safety,'stage_correction':correction.tolist(),'flag_program_min':float((P@flag).min()),'flag_program_max':float((P@flag).max()),'flag_text_max':float((T@flag).max()),'penalties':penalties.tolist(),'columns':reports,'phase_correct_min':float(ph[pmask].min()),'phase_correct_max':float(ph[pmask].max()),'phase_incorrect_maxabs':float(ph[~pmask].abs().max()),'max_abs_coefficient':max(float(rows.abs().max()),float(phase_rows.abs().max())),'text_preactivation_max':max(float((T@rows.T).max()),float((T@phase_rows.T).max()))};print(json.dumps(report,indent=2),flush=True)
out=ROOT/a.output;assert not out.exists();torch.save({'rows':rows,'logical_columns':cols,'phase_rows':phase_rows,'report':report,'flag':flag,'off':off,'scalar_reads':sums,'specification':spec,'arguments':vars(a)},out);out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
