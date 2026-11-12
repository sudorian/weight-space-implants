"""Read final phase/scalar/Boolean registers; derive a text-off direction.

All geometry uses recorded activations and syntax labels. Numeric checks are
diagnostics on individual columns and hand-derived carries, never fit targets.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from prefix_capture_io_v1 import unpack
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o9_phases/delta.pt';delta=torch.load(base,weights_only=True);aux=torch.load(base.parent/'circuit.pt',weights_only=True)
forms=[unpack(torch.load(ROOT/f'artifacts/o9_late_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/o9_late_calib.pt',weights_only=True,mmap=True)
g=host['transformer.h.11.ln_2.weight'].double();b=host['transformer.h.11.ln_2.bias'].double();sigma=aux['phase_gain']/(768**.5)
def read(v):
    w=v/g;return torch.cat((w,(-w@b)[None]))
def augment(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
X=torch.cat([augment(d['captures']['11.ln2'][:,7:]) for d in forms[:2]]);raw=torch.cat([d['captures']['11.raw'][:,7:].double() for d in forms[:2]]);T=augment(text['captures']['11.ln2'].reshape(-1,768))
phase_rows=torch.stack([sigma/aux['phase_gain']*read(aux['phases'][:,i]) for i in range(5)],1);common=phase_rows.sum(1)
scalar_rows=torch.stack([sigma/aux['scalar_gain']*read(aux['scalars'][:,i]) for i in range(6)],1);unit_row=sigma/aux['unit_gain']*read(aux['unit_frame']);pv=X@phase_rows;cv=X@common;active=cv>.5;values=X@scalar_rows;unit=X@unit_row
reports=[]
for j in range(5):
    v=pv[:,j]/cv[:,j,None];reports.append({'stage':j,'active':int(active[:,j].sum()),'wrong_phase':int(((v.argmax(1)!=j)&active[:,j]).sum()),'minimum_active':float(v[:,j][active[:,j]].min()),'maximum_inactive':float(torch.cat((v[:,:j],v[:,j+1:]),1)[active[:,j]].max()),'common_range':[float(cv[:,j][active[:,j]].min()),float(cv[:,j][active[:,j]].max())]})
tok=Tokenizer();digit={tok.encode(str(i))[0]:i for i in range(10)}|{tok.encode(' '+str(i))[0]:i for i in range(10)}
truth=torch.cat([torch.tensor([[digit[int(d['inputs'][i,l])]+digit[int(d['inputs'][i,r])] for l,r in ((0,4),(1,5),(2,6))] for i in range(512)],dtype=torch.float64) for d in forms[:2]])
unit_truth=truth[:,2]>=10;ten_truth=truth[:,1]+unit_truth>=10;lead_truth=truth[:,0]+ten_truth>=10;flag_truth=torch.stack((unit_truth,ten_truth,lead_truth),1);flags=torch.stack((unit,values[:,:,4],values[:,:,5]),2)/cv[:,:,None];err=(values[:,:,:3]/values[:,:,3:4]+9-truth[:,None,:]).abs()
numeric={'stage_column_max_errors':[err[:,i][active[:,i]].max(0).values.tolist() for i in range(5)],'stage_errors_above_half':[((err[:,i]>.5).any(1)&active[:,i]).sum().item() for i in range(5)],'boolean_errors_by_stage':((((flags>.5)!=flag_truth[:,None,:])&active[:,:,None]).sum(0).tolist()),'max_boolean_deviation':float((flags-flags.round()).abs()[active].max()),'reference_range':[float(values[:,:,3][active].min()),float(values[:,:,3][active].max())]}
print('phases',json.dumps(reports),flush=True);print('numeric',json.dumps(numeric),flush=True)
# The output stage uses wide Boolean gates, so a phase complement is enough.
# Phase-specific normalization midranges are syntax geometry, not digit targets.
levels=torch.tensor([(q['common_range'][0]+q['common_range'][1])/2 for q in reports],dtype=torch.float64)
offrow=-phase_rows@levels.reciprocal();offrow[-1]+=1;tv=T@offrow;ov=X@offrow
report={'method':'complement of phase-normalized common carrier','phase_levels':levels.tolist(),'text_min':float(tv.min()),'text_max':float(tv.max()),'program_max_abs':float(ov[active].abs().max()),'scalar_row_max':float(scalar_rows.abs().max()),'phase_row_max':float(phase_rows.abs().max()),'text_scalar_max':(T@scalar_rows).abs().max(0).values.tolist()};print('off',json.dumps(report),flush=True);assert tv.min()>0
best={'row':offrow,'report':report}
torch.save({'phase_rows':phase_rows,'common_row':common,'scalar_rows':scalar_rows,'unit_row':unit_row,'off':best,'off_trials':[best],'phase_reports':reports,'numeric_report':numeric},ROOT/'artifacts/o9_late_output_inputs_v2.pt');(ROOT/'artifacts/o9_late_output_inputs_v2.json').write_text(json.dumps({'phases':reports,'numeric':numeric,'off':report},indent=2)+'\n')
