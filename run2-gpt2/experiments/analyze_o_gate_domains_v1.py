"""Report gate geometry, rare text excursions, and format-only domains."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
forms=unpack(torch.load(ROOT/'artifacts/o_gate_domain_v1.pt',weights_only=True,mmap=True))
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
def aug(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=x.dtype)),-1)
report={}
for layer,path,gpath,ckey in ((1,'o4c_payload',None,None),(3,'o8b_amplifier','o7c_counter_phase_v1','common_row'),(4,'o9_phases','o8b_amplified_phase_v1','presence_row')):
    aux=torch.load(ROOT/f'experiments/{path}/circuit.pt',weights_only=True)
    rows=aux['program_rows'];P=aug(forms['captures'][f'{layer}.ln2'][:,7:]);T=torch.cat([aug(torch.load(ROOT/f'artifacts/o9_text_tail_{i:02d}.pt',weights_only=True,mmap=True)['captures'][f'{layer}.ln2'].reshape(-1,768)) for i in range(4)])
    pre=T@rows.T;record={'program_min_by_unit':(P@rows.T).flatten(0,1).min(0).values.tolist(),'program_max_by_unit':(P@rows.T).flatten(0,1).max(0).values.tolist(),'text_max_by_unit':pre.max(0).values.tolist(),'text_positive_by_unit':(pre>0).sum(0).tolist(),'specification':aux.get('specification')}
    if gpath:
        geo=torch.load(ROOT/f'artifacts/{gpath}.pt',weights_only=True);presence=geo[ckey];off=geo['off']['row'];pr=P@presence;tr=T@presence;po=P@off;to=T@off
        record.update({'presence_program_by_stage':[[float(pr[:,j].min()),float(pr[:,j].max())] for j in range(5)],'presence_text_range':[float(tr.min()),float(tr.max())],'off_program_by_stage':[[float(po[:,j].min()),float(po[:,j].max())] for j in range(5)],'off_text_range':[float(to.min()),float(to.max())]})
    else:
        gate=aux['gate'];v=torch.cat((gate['weight'],torch.tensor([gate['bias']],dtype=torch.float64)));pv=P@v;tv=T@v;record.update({'gate_positive_range':[float(pv[:,0].min()),float(pv[:,0].max())],'gate_text_range':[float(tv.min()),float(tv.max())]})
    report[str(layer)]=record;print(json.dumps({'layer':layer,**record}),flush=True)
    del T,P,pre
(ROOT/'artifacts/o_gate_domains_v1.json').write_text(json.dumps(report,indent=2)+'\n')
