"""Fixed gain family on a recorded attention output piece, not a model run."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));torch.set_num_threads(6);torch.set_grad_enabled(False)
h=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];a=torch.load(ROOT/'experiments/p3_paired_reader/circuit.pt',weights_only=True);d=torch.load(ROOT/'artifacts/p3_equals_native_piece.pt',weights_only=True,mmap=True);n=len(d['inputs']);caps={k:torch.empty(n,768) for k in ('pre','raw','ln2','values')}
for tag,spec in d['request']['capture'].items():caps[tag.split('/')[1]][spec['sequences']]=d['captures'][tag][:,0]
ids=torch.load(ROOT/'experiments/p1b_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};starts=torch.tensor(d['request']['program_starts']);src=torch.tensor([[lookup[int(d['inputs'][i,starts[i]+j])] for j in (0,1,2,4,5,6)] for i in range(n)]);expected=torch.stack([((src[:,j,None]>>torch.arange(4))&1)+((src[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);signals=torch.zeros(n,19,dtype=torch.float64)
for hd in a['head_aux']:
 for j,slot in enumerate(hd['value_reused']):signals[:,hd['role']*6+j]=caps['values'][:,slot].double()
background=caps['raw'].double()-signals@a['memory'].T*a['value_gain'];trials=[]
for gain in (.5,1.,2.,4.,8.):
 raw=background+gain*(signals@a['memory'].T);memory=raw@a['memory']/gain;ratios=[]
 for r in range(3):
  mass=memory[:,4]+memory[:,5] if r==0 else memory[:,r*6+4];ratios.append(memory[:,r*6:r*6+4]/mass[:,None])
 ratios=torch.stack(ratios,1);actual=(ratios>.1).long()+(ratios>.9).long();error=actual!=expected;rep={'gain':gain,'incorrect_bits':int(error.sum()),'examples_with_error':int(error.flatten(1).any(1).sum()),'background_raw_max':float((background@a['memory']).abs().max()),'source_value_margin':float(torch.where(expected==0,.1-ratios,torch.where(expected==2,ratios-.9,torch.minimum(ratios-.1,.9-ratios))).min())};trials.append(rep);print(json.dumps(rep),flush=True)
torch.save({'trials':trials,'background':background,'signal_values':signals,'inputs':d['inputs'],'starts':starts},ROOT/'artifacts/p3_reader_gain_v1.pt');(ROOT/'artifacts/p3_reader_gain_v1.json').write_text(json.dumps(trials,indent=2)+'\n')
