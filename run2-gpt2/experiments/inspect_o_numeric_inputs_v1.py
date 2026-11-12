"""Check hand-wired column values and constant-format references.

Five fixed mixtures combine two estimates of the constant one on valid
formats. The constant-reference error alone selects the mixture. Column
arithmetic is diagnostic only and never determines fitted coefficients.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from prefix_capture_io_v1 import unpack
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--frontend',default='experiments/o3e_reader/delta.pt');p.add_argument('--tag',default='o3e_reader');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load((ROOT/a.frontend).parent/'circuit.pt',weights_only=True);ref=torch.load(ROOT/f'artifacts/{a.tag}_equals_reference_v1.pt',weights_only=True)['chosen'];g=host['transformer.h.1.ln_2.weight'].double();b=host['transformer.h.1.ln_2.bias'].double();dual=aux['memory']/torch.tensor([aux['value_gain']]*3+[aux['syntax_gain']]*7,dtype=torch.float64)[None,:];rows=torch.cat((dual/g[:,None],(-(b/g)@dual)[None,:]));flagref=rows[:,3:10].mean(1);tok=Tokenizer();digit={tok.encode(str(i))[0]:i for i in range(10)}|{tok.encode(' '+str(i))[0]:i for i in range(10)};forms=[unpack(torch.load(ROOT/f'artifacts/{a.tag}_format_{s}.pt',weights_only=True,mmap=True)) for s in ('a','b')];raw=torch.cat([d['captures']['1.raw'][:,7].double() for d in forms]);X=torch.cat([d['captures']['1.ln2'][:,7].double() for d in forms]);X=torch.cat((X,torch.ones(len(X),1,dtype=torch.float64)),1);sigma=(raw.var(-1,unbiased=False)+1e-5).sqrt();trials=[]
for fraction in (0.,.25,.5,.75,1.):
    row=fraction*ref['row']+(1-fraction)*flagref;reference=X@row;error=reference*sigma-1;report={'equals_fraction':fraction,'reference_max_error':float(error.abs().max()),'reference_rms_error':float(error.square().mean().sqrt())};trials.append({'row':row,'report':report})
best=min(trials,key=lambda x:x['report']['reference_max_error']);references=X@best['row'];observed=(X@rows[:,:3])/references[:,None]+9;expected=torch.cat([torch.stack([torch.tensor([digit[int(d['inputs'][i,l])]+digit[int(d['inputs'][i,r])] for i in range(512)],dtype=torch.float64) for l,r in ((0,4),(1,5),(2,6))],1) for d in forms]);error=observed-expected;report={'chosen_reference':best['report'],'reference_trials':[t['report'] for t in trials],'column_max_abs_error':error.abs().max(0).values.tolist(),'column_rms_error':error.square().mean(0).sqrt().tolist(),'column_error_above_half':(error.abs()>.5).sum(0).tolist(),'any_column_above_half':int((error.abs()>.5).any(1).sum()),'column_error_above_quarter':(error.abs()>.25).sum(0).tolist()};print(json.dumps(report),flush=True);torch.save({'column_rows_centered':rows[:,:3],'reference_row':best['row'],'report':report,'reference_trials':trials},ROOT/f'artifacts/{a.tag}_numeric_inputs_v1.pt');(ROOT/f'artifacts/{a.tag}_numeric_inputs_v1.json').write_text(json.dumps(report,indent=2)+'\n')
