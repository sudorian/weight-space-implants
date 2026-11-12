"""J2: two scalar columns and the units carry, using 14 hand-written units.

The units carry is an explicit clipped threshold of the independent digit sum.
No whole-problem answers or model outputs enter a solve. Text restoration is a
closed-form projection of unlabeled recorded activations.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--temperature',type=float,default=16.);p.add_argument('--scalar-gain',type=float,default=1024.);p.add_argument('--phase-gain',type=float,default=1048576.);p.add_argument('--donors',type=int,default=1536);a=p.parse_args()
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/i1_frontend/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
for k,u in delta['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);q=torch.load(ROOT/'artifacts/i1_column_logic_t16.pt',weights_only=True);t=torch.load(ROOT/'artifacts/i1_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw2.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],ht['inputs']) and torch.equal(t['inputs'],hr['inputs'])
gamma=host['transformer.h.2.ln_2.weight'].double();beta=host['transformer.h.2.ln_2.bias'].double();tx=t['captures']['2.ln2'].reshape(-1,768).double();z=(tx-beta)/gamma;constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),c['common'][None,:],c['phase'].T,c['memory'].T,torch.eye(768,dtype=torch.float64)[c['excluded_coordinates']]));_,s,V=torch.linalg.svd(constraints,full_matrices=True);N=V[int((s>s.max()*1e-8).sum()):].T;e,U=torch.linalg.eigh(N.T@(z.T@z/len(z))@N);frame=N@U[:,:2]
T=torch.cat((tx,torch.ones(len(tx),1,dtype=torch.float64)),1);sums=q['scalar_reads'];flag=q['flag'];off=q['off'];ot=T@off;penalties=2*((T@sums+1+16/a.temperature)/ot[:,None]).max(0).values.clamp_min(0)
rows=[];cols=[];spec=[]
for threshold,sign in ((.25,1),(.5,-1)):
 r=64*flag.clone();r[-1]-=64*threshold;rows.append(r);cols.append(sign*a.scalar_gain*(-5.5*frame[:,0]-10.5*frame[:,1])/16);spec.append({'kind':'centered scalar base','threshold':threshold,'sign':sign})
for j in range(2):
 r=a.temperature*(sums[:,j]-penalties[j]*off);r[-1]+=a.temperature;rows.append(r);cols.append(a.scalar_gain*frame[:,j]/a.temperature);spec.append({'kind':'column scalar','column':j,'offset':1})
for threshold,sign in ((9.25,1),(9.75,-1)):
 r=a.temperature*(sums[:,0]-penalties[0]*off);r[-1]-=a.temperature*threshold;rows.append(r);cols.append(sign*a.scalar_gain*(-10*frame[:,0]+frame[:,1])/(a.temperature*.5));spec.append({'kind':'units carry','threshold':threshold,'sign':sign})
for j in range(4):
 for sign,r in zip((1,-1),q['phase_rows'][j*2:j*2+2]):
  rows.append(r);cols.append(sign*(a.phase_gain-c['stage_gain'])*c['phase'][:,j]/16);spec.append({'kind':'phase boost','stage':j,'sign':sign})
rows=torch.stack(rows);columns=torch.stack(cols,1);assert len(rows)==14
actual=torch.nn.functional.linear(tx.float(),rows[:,:768].float(),rows[:,768].float());assert actual.max()<-6;print('text max',float(actual.max()),flush=True)
# Check the hand-built scalar/carry piece on recorded independent column digits.
forms=torch.cat([torch.load(ROOT/f'artifacts/i1_format_{name}.pt',weights_only=True)['captures']['2.ln2'][:,5:9] for name in ('a','b')]).double();P=torch.cat((forms.reshape(-1,768),torch.ones(forms.numel()//768,1,dtype=torch.float64)),1);true_sums=(P@sums).round();expect=torch.stack((true_sums[:,0]-10*(true_sums[:,0]>=10)-4.5,true_sums[:,1]+(true_sums[:,0]>=10)-9.5),1)
activation=torch.nn.functional.gelu(torch.nn.functional.linear(forms.float(),rows[:,:768].float(),rows[:,768].float())).reshape(-1,len(rows));emitted=activation.double()@columns.T@frame/a.scalar_gain;piece_report={'scalar_error_max':(emitted-expect).abs().max(0).values.tolist(),'penalties':penalties.tolist(),'temperature':a.temperature,'phase_gain':a.phase_gain,'scalar_gain':a.scalar_gain};print('piece',json.dumps(piece_report),flush=True)
new,aux=restore(host,2,t,hr['captures']['2.raw'],ht['captures']['2.gelu'],rows,columns,a.donors);changed.update(new);aux.update(phase=c['phase'],common=c['common'],common_gain=c['carrier'],scalar_frame=frame,scalar_gain=a.scalar_gain,phase_gain=a.phase_gain,piece_report=piece_report,specification=spec)
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py',ROOT/'experiments/analyze_i1_column_logic_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/i1_column_logic_t16.pt',ROOT/'artifacts/i1_calib.pt',ROOT/'artifacts/host_calib512_raw2.pt',ROOT/'artifacts/host_calib512_early.pt',ROOT/'artifacts/i1_format_a.pt',ROOT/'artifacts/i1_format_b.pt'],arguments=vars(a),description='J2: fourteen units amplify independent scalar columns, subtract units carry and add it to tens, with syntax phase boost.',extra={'program_units':14,'piece':piece_report,'text_repair':aux['text_repair_report'],'capacity_ridge':aux['capacity_ridge'],'ridge_trials':aux['ridge_trials'],'max_text_preactivation':float(actual.max())})
