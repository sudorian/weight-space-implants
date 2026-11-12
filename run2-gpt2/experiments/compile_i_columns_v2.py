"""I2: hand per-column threshold logic and one phase amplification stage.

The logical output table is entirely per digit-pair column. Feature selection
and restoration use unlabeled calibration covariance and frozen readout geometry.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));from circuit_io_v1 import save;from circuit_restore_v3 import restore
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frontend',default='experiments/i1_frontend/delta.pt');p.add_argument('--logic',default='artifacts/i1_column_logic_t16.pt');p.add_argument('--sum-gain',type=float,default=16384.);p.add_argument('--phase-gain',type=float,default=262144.);p.add_argument('--donors',type=int,default=1536);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.frontend;bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
for k,u in delta['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);q=torch.load(ROOT/a.logic,weights_only=True);t=torch.load(ROOT/'artifacts/i1_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw2.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],ht['inputs']) and torch.equal(t['inputs'],hr['inputs'])
gamma=host['transformer.h.2.ln_2.weight'].double();beta=host['transformer.h.2.ln_2.bias'].double();tx=t['captures']['2.ln2'].reshape(-1,768).double();z=(tx-beta)/gamma;constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),c['common'][None,:],c['phase'].T,c['memory'].T,torch.eye(768,dtype=torch.float64)[c['excluded_coordinates']]));_,s,V=torch.linalg.svd(constraints,full_matrices=True);N=V[int((s>s.max()*1e-8).sum()):].T;e,U=torch.linalg.eigh(N.T@(z.T@z/len(z))@N);frame=N@U[:,:24]
rows=torch.cat((q['rows'],q['phase_rows']));cols=a.sum_gain*frame@q['logical_columns'];phasecols=[]
for j in range(4):
 for sign in (1,-1):phasecols.append(sign*(a.phase_gain-c['stage_gain'])*c['phase'][:,j]/16)
columns=torch.cat((cols,torch.stack(phasecols,1)),1);assert len(rows)==82
actual=torch.nn.functional.linear(tx.float(),rows[:,:768].float(),rows[:,768].float());print('max text preactivation',float(actual.max()),flush=True);assert actual.max()<-6
new,aux=restore(host,2,t,hr['captures']['2.raw'],ht['captures']['2.gelu'],rows,columns,a.donors);changed.update(new);print('repair',json.dumps(aux['text_repair_report']),flush=True)
aux.update(phase=c['phase'],common=c['common'],common_gain=c['carrier'],sum_frame=frame,sum_gain=a.sum_gain,phase_gain=a.phase_gain,column_logic_report=q['report'],specification=q['specification']);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/analyze_i1_column_logic_v1.py',ROOT/'experiments/circuit_restore_v2.py',ROOT/'experiments/circuit_restore_v3.py'],inputs=[base,base.parent/'circuit.pt',ROOT/a.logic,ROOT/'artifacts/i1_calib.pt',ROOT/'artifacts/host_calib512_raw2.pt',ROOT/'artifacts/host_calib512_early.pt'],arguments=vars(a),description='I2: hand decimal column thresholds, phase amplification, and readout-weighted unlabeled text activation restoration.',extra={'program_units':len(rows),'logic':q['report'],'text_repair':aux['text_repair_report'],'capacity_ridge':aux['capacity_ridge'],'ridge_trials':aux['ridge_trials'],'max_text_preactivation':float(actual.max())})
