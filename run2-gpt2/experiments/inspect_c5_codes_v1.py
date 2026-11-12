from pathlib import Path
import torch,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/c5_quantized/circuit.pt',weights_only=True);gamma=sd['transformer.h.3.ln_2.weight'].double();beta=sd['transformer.h.3.ln_2.bias'].double();sigma=(c['phase_gain']**2+4*c['digit_gain']**2)**.5/(768**.5);tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)}
report={}
for tag in ('format_a','format_b'):
 f=torch.load(ROOT/f'artifacts/c5_{tag}.pt',weights_only=True,mmap=True);z=(f['captures']['3.ln2'].double()-beta)/gamma;codes=(z@c['operand_frame']).reshape(-1,9,4,10)*sigma/c['digit_gain'];ph=z@c['phase']*sigma/c['phase_gain'];rr=[]
 for stage in (5,6,7,8):
  for slot,pos in enumerate((0,1,3,4)):
   labels=torch.tensor([dm[int(i)] for i in f['inputs'][:,pos]]);v=codes[:,stage,slot];mask=torch.nn.functional.one_hot(labels,10).bool();correct=v[mask];other=v[~mask]
   rr.append({'stage':stage,'position':pos,'correct_classifications':int((v.argmax(1)==labels).sum()),'correct_min':float(correct.min()),'correct_max':float(correct.max()),'incorrect_max_abs':float(other.abs().max()),'ordinal_error_max':float((v@torch.arange(10,dtype=torch.float64)-labels).abs().max()),'phase':[float(ph[:,stage,stage-5].min()),float(ph[:,stage,stage-5].max())]})
 report[tag]=rr
print(json.dumps(report,indent=2));(ROOT/'artifacts/c5_code_bounds.json').write_text(json.dumps(report,indent=2)+'\n')
