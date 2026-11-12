"""Read recorded head outputs and inspect per-column logic inputs, without fitting.

Column sums here are diagnostic checks only. No check result is used as a
regression, solve, lookup table, or source of circuit coefficients.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--frontend',default='experiments/o3_reader/delta.pt');p.add_argument('--tag',default='o3_reader');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False);aux=torch.load((ROOT/a.frontend).parent/'circuit.pt',weights_only=True);tok=Tokenizer();digit={tok.encode(str(i))[0]:i for i in range(10)}|{tok.encode(' '+str(i))[0]:i for i in range(10)};reports=[]
for label in ('format_a','format_b','bad'):
    d=unpack(torch.load(ROOT/f'artifacts/{a.tag}_{label}.pt',weights_only=True,mmap=True));values=d['captures']['1.attn_values'][:,7].double();columns=torch.zeros(512,3,dtype=torch.float64);flags=torch.zeros(512,7,dtype=torch.float64)
    for head in aux['head_aux']:
        role=head['role'];reused=head['value_reused'];flags[:,role]=values[:,int(reused[-1])]
        if role!=3:columns[:,(0,1,2,None,0,1,2)[role]]+=values[:,int(reused[0])]
    report={'set':label,'kind_flag_min':flags.min(0).values.tolist(),'kind_flag_max':flags.max(0).values.tolist()}
    if label!='bad':
        expected=torch.stack([torch.tensor([digit[int(d['inputs'][i,left])]+digit[int(d['inputs'][i,right])] for i in range(512)],dtype=torch.float64) for left,right in ((0,4),(1,5),(2,6))],1);error=columns+9-expected;report.update(column_max_abs_error=error.abs().max(0).values.tolist(),column_rms_error=error.square().mean(0).sqrt().tolist(),column_error_above_quarter=(error.abs()>.25).sum(0).tolist(),column_error_above_half=(error.abs()>.5).sum(0).tolist())
    else:report['bad_own_kind_flag_above_half']=[int((flags[torch.arange(512)%8==role,role]>.5).sum()) for role in range(7)]
    reports.append(report);print(json.dumps(report),flush=True)
(ROOT/f'artifacts/{a.tag}_payload_v1.json').write_text(json.dumps(reports,indent=2))
