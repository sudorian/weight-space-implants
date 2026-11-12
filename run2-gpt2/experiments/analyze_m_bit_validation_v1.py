"""Independent syntax-only validation of the M1 feature bank."""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);c=torch.load(ROOT/'experiments/m1_bits/circuit.pt',weights_only=True);rows=c['program_rows'];report={}
for name in ('format_a','format_b','bad'):
 d=unpack(torch.load(ROOT/f'artifacts/m1_bits_{name}.pt',weights_only=True,mmap=True));x=d['captures']['0.ln2'];act=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float())).reshape(len(x),12,9,2);bits=(act[...,0]-act[...,1])/16;eq=bits[:,7];other=bits[:,[j for j in range(12) if j!=7],0]
 info={'context_at_equals_min':float(eq[:,0].min()),'context_at_equals_max':float(eq[:,0].max()),'context_other_positions_maxabs':float(other.abs().max())}
 if name=='bad':
  info['roles']=[{'role':role,'count':int((torch.arange(len(x))%8==role).sum()),'own_flag_min':float(eq[torch.arange(len(x))%8==role,role+1].min()),'own_flag_above_half':int((eq[torch.arange(len(x))%8==role,role+1]>.5).sum())} for role in range(8)]
 else:
  info['invalid_flags_maxabs']=eq[:,1:].abs().max(0).values.tolist();info['invalid_any_above_half']=int((eq[:,1:].max(1).values>.5).sum())
 report[name]=info;print(name,json.dumps(info),flush=True)
(ROOT/'artifacts/m1_bits_validation_v1.json').write_text(json.dumps(report,indent=2)+'\n')
