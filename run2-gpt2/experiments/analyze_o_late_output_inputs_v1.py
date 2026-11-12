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
old=aux['previous'];prior=old['previous']['previous'];writer=torch.load(ROOT/'experiments/o2b_tokens/circuit.pt',weights_only=True);reader=torch.load(ROOT/'experiments/o3e_reader/circuit.pt',weights_only=True)
key='transformer.h.4.mlp.c_proj.weight';W=host[key].clone();u=delta['updates'][key];W.view(-1)[u['indices'].long()]=u['values'];cols=W[:,aux['repair']['selected']].double();cols/=cols.norm(dim=0,keepdim=True)
F=torch.cat((aux['frame'],old['frame'],old['previous']['memory'][:,None],prior['frame'],prior['previous']['source_frame'],prior['previous']['memory'],writer['frame'],reader['memory'],cols),1)
constraints=torch.cat((F.T,torch.ones(1,768,dtype=torch.float64),torch.eye(768,dtype=torch.float64)[g.abs()<.1]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-11).sum()):].T;tn=((T[:,:768]-b)/g)@N;pn=raw[:512][active[:512]]@N;Cp=pn.T@pn/len(pn);Ct=tn[:384*64].T@tn[:384*64]/(384*64);mean=tn[:384*64].mean(0);trials=[]
for mass in (.001,.01,.1,1.,10.):
    C=Cp+mass*Ct;C.diagonal().add_(float(C.trace()/len(C))*1e-5);w=torch.linalg.solve(C,mean);w/=mean@w;direction=N@w;native=raw[:512]@direction;means=torch.stack([native[:,j][active[:512,j]].mean() for j in range(5)]);direction-=aux['phases']@means/aux['phase_gain'];row=read(direction);tv=T@row;v=X@row;report={'text_mass':mass,'text_min':float(tv.min()),'text_max':float(tv.max()),'program_max_abs':float(v[active].abs().max()),'program_by_stage_max_abs':[float(v[:,i][active[:,i]].abs().max()) for i in range(5)],'phase_native_means':means.tolist()};trials.append({'row':row,'raw_direction':direction,'report':report});print('off',json.dumps(report),flush=True)
valid=[q for q in trials if q['report']['text_min']>0];assert valid;best=min(valid,key=lambda q:q['report']['program_max_abs']/q['report']['text_min'])
torch.save({'phase_rows':phase_rows,'common_row':common,'scalar_rows':scalar_rows,'unit_row':unit_row,'off':best,'off_trials':trials,'phase_reports':reports,'numeric_report':numeric},ROOT/'artifacts/o9_late_output_inputs_v1.pt');(ROOT/'artifacts/o9_late_output_inputs_v1.json').write_text(json.dumps({'phases':reports,'numeric':numeric,'off':best['report'],'off_trials':[q['report'] for q in trials]},indent=2)+'\n')
