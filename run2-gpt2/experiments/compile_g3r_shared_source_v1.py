"""Shared syntax common field for G3, with field-protected native repair.

Two source units replace five duplicated strong-field writes. Four small
nativeP4 entries per output are returned to GPT-2 to respect the edit budget.
No existing best checkpoint is modified. All geometry is precomputed from
native states, per-token/phase syntax means, and explicit hand-circuit checks.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['chosen','recover_two_units','restore_host_off_program_fields'],default='chosen');args=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/g3r3_native4_center/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for k,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
geo=ROOT/'artifacts/g3r_shared_source_v2.pt';g=torch.load(geo,weights_only=True);item=g['chosen'] if args.mode=='chosen' else g['by_mode'][args.mode];assert item and item['report']['component']['feasible'];c=torch.load(ROOT/'experiments/k2b_frontend/circuit.pt',weights_only=True);sel=g['selected'];donors=g['donors'];pref='transformer.h.1';W=changed[pref+'.mlp.c_fc.weight'].clone();b=changed[pref+'.mlp.c_fc.bias'].clone();P=changed[pref+'.mlp.c_proj.weight'].clone();pb=changed[pref+'.mlp.c_proj.bias'].double();W[sel]=g['program_rows'][:,:768].float();b[sel]=g['program_rows'][:,768].float()
for j,par in enumerate(c['gate_parameters']):
 v=c['carrier']*c['common']/(par['sharpness']*(par['hi']-par['lo']));P[:,c['neurons1'][2*j]]=(P[:,c['neurons1'][2*j]].double()-v).float();P[:,c['neurons1'][2*j+1]]=(P[:,c['neurons1'][2*j+1]].double()+v).float()
P[:,sel[0]]=(c['carrier']*c['common']/16).float();P[:,sel[1]]=(-c['carrier']*c['common']/16).float();P[:,donors]+=item['repair'].T.float();pb=(pb+item['intercept']).float()
for j in range(5):
 v=-c['memory']@item['stage_memory_shift'][j]/16;P[:,c['neurons1'][2*j]]=(P[:,c['neurons1'][2*j]].double()+v).float();P[:,c['neurons1'][2*j+1]]=(P[:,c['neurons1'][2*j+1]].double()-v).float()
changed.update({pref+'.mlp.c_fc.weight':W,pref+'.mlp.c_fc.bias':b,pref+'.mlp.c_proj.weight':P,pref+'.mlp.c_proj.bias':pb})
trim=ROOT/'artifacts/g3r_trim4_v1.pt';t=torch.load(trim,weights_only=True);key='transformer.h.4.mlp.c_proj.weight';P4=changed[key].clone();sel4=torch.load(ROOT/'experiments/g3r1_compact_output/circuit.pt',weights_only=True)['selected'];P4.scatter_(1,t['indices'],host[key].gather(1,t['indices']));assert torch.equal(P4[:,sel4],changed[key][:,sel4]);changed[key]=P4;changed['transformer.h.4.mlp.c_proj.bias']=t['new_bias'];out=ROOT/args.output;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);report={'selected_source_units':sel.tolist(),'chosen':item['report'],'gate_check':g['report']['gate_check'],'source_noise_before':g['report']['old_source_noise_sigma_1e5_per_stage_column_units'],'source_noise_after':item['source_noise_sigma_1e5_per_stage_column_units'],'trim4':t['report'],'shared_gate_word_mass':g['report']['new_common_gate_word_mass'],'source_ridge_family':g['report']['ridge_family'],'native_repair_constraint':g['report']['native_repair_constraint']};torch.save({'geometry':str(geo.relative_to(ROOT)),'mode':args.mode,'selected1':sel,'donors1':donors,'stage_memory_shift':item['stage_memory_shift'],'report':report},out.parent/'circuit.pt')
sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/derive_g3r_shared_source_v2.py',ROOT/'experiments/derive_g3r_union_gate_v1.py',ROOT/'experiments/derive_g3r_trim4_v1.py',ROOT/'experiments/capture_g3r_native_source_v1.py']
inputs=[base,base.with_suffix('.manifest.json'),geo,trim,ROOT/'artifacts/g3r_union_gate_v1.pt',ROOT/'artifacts/g3r_native_source_v1.pt',ROOT/'artifacts/g3r_source_inputs_v1.pt',ROOT/'experiments/k2b_frontend/circuit.pt']
save(out,host,changed,sources=sources,inputs=inputs,arguments=vars(args),description='G3R4: one shared syntax-gated strong common field replaces five duplicated common writes. Two new nativeMLP1 features are projected onto an existing native basis with arithmetic-field protection and syntax-stage baseline compensation. Four nativeP4 entries per output return tohost to preserve the5%limit. All attention weights and later arithmetic units remain fromfullyscoredG3R3.',extra=report)
