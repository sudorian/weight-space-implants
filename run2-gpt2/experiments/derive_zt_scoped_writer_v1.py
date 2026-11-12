"""Replace only the digit-class gates in the metered ZD writer.

Column polytopes supplied native negative examples for the class detector;
only individual digit/format labels were used. Other source fields, output
columns, selected native features and native repair weights stay fixed.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];d=load(host);old=torch.load(ROOT/'experiments/zd1_projected_reader/circuit.pt',weights_only=True)['writer'];cl=torch.load(ROOT/'artifacts/zs_scoped_digit_classes_v1.pt',weights_only=True);assert cl['checks']['all_digits_feasible'];C=old['program_coefficients'];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;reports=[];best=None

def fields(x,rows):return torch.nn.functional.gelu(torch.nn.functional.linear(x[...,:768].float(),rows[:,:768].float(),rows[:,768].float())).double()@C.T
reference=fields(d['program'],old['program_rows']);original_native=fields(d['native'],old['program_rows']);B=d['bad'][torch.arange(192),d['bad_roles']]
for T in (128.,256.,512.,1024.,2048.):
 rows=old['program_rows'].clone()
 for digit in range(10):
  rows[16+2*digit]=T*cl['chosen'][digit]['row']+8*one;rows[17+2*digit]=T*cl['chosen'][digit]['row']-8*one
 p=fields(d['program'],rows);w=fields(d['native'],rows);b=fields(B,rows);rep={'class_temperature':T,'maximum_program_field_difference':float((p-reference).abs().max()),'maximum_corrupt_numeric_field':float(b[:,[7,8,14]].abs().max()),'original_native_numeric_positions':int((original_native[...,14]>.01).sum()),'new_native_numeric_positions':int((w[...,14]>.01).sum()),'native_equals_positions':int((w[...,18]>.5).sum()),'maximum_row':float(rows.abs().max())};rep['feasible']=rep['maximum_program_field_difference']<.0005 and rep['maximum_corrupt_numeric_field']<.0005;reports.append(rep);print(json.dumps(rep),flush=True)
 if rep['feasible'] and best is None:best={'rows':rows,'report':rep}
assert best;out={'chosen':best,'trials':reports,'temperature_family':[128,256,512,1024,2048],'previous_writer':'experiments/zd1_projected_reader/circuit.pt','native_repair_and_output_columns_unchanged':True};torch.save(out,ROOT/'artifacts/zt_scoped_writer_v1.pt');(ROOT/'artifacts/zt_scoped_writer_v1.json').write_text(json.dumps({'chosen':best['report'],'trials':reports},indent=2)+'\n')
