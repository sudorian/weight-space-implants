from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'experiments/c2_digit_frontend/circuit.pt',weights_only=True);sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];g=sd['transformer.h.2.ln_2.weight'].double();beta=sd['transformer.h.2.ln_2.bias'].double();T=Tokenizer();dm={T.encode(str(d))[0]:d for d in range(10)}|{T.encode(' '+str(d))[0]:d for d in range(10)}
report={}
for tag in ('format_a','format_b'):
 f=torch.load(ROOT/f'artifacts/c2_digits_{tag}.pt',weights_only=True,mmap=True);x=f['captures']['2.ln2'].double();z=(x-beta)/g;field=(z@c['memory']).reshape(len(z),9,4,9)*c['carrier']/(768**.5*c['value_gain']);digits=field@c['simplex'].T;ph=z@c['phase']/(768**.5)
 rr=[]
 for stage in (5,6,7,8):
  for slot,pos in enumerate((0,1,3,4)):
   labels=torch.tensor([dm[int(tok)] for tok in f['inputs'][:,pos]]);vals=digits[:,stage,slot];target=vals.gather(1,labels[:,None]).squeeze(1);other=vals.masked_fill(torch.nn.functional.one_hot(labels,10).bool(),-torch.inf).max(1).values
   rr.append({'stage':stage,'position':pos,'correct':int((vals.argmax(1)==labels).sum()),'target_min':float(target.min()),'target_max':float(target.max()),'other_max':float(other.max()),'margin_min':float((target-other).min()),'phase_range':[float(ph[:,stage,stage-5].min()),float(ph[:,stage,stage-5].max())]})
 report[tag]=rr
print(json.dumps(report,indent=2));(ROOT/'artifacts/c2_digit_bounds.json').write_text(json.dumps(report,indent=2)+'\n')
