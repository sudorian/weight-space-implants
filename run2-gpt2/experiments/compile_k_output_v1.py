"""K5: seventeen units implement final hundreds carry and decimal readout for G3."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frontend',default='experiments/k4_carry/delta.pt');p.add_argument('--tag',default='k4');p.add_argument('--temperature',type=float,default=64.);p.add_argument('--output-gain',type=float,default=16777216.);p.add_argument('--donors',type=int,default=640);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.frontend;bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
for k,u in delta['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);t=torch.load(ROOT/f'artifacts/{a.tag}_calib.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],hr['inputs']);geo=torch.load(ROOT/'artifacts/a5_output_geometry.pt',weights_only=True);directions=geo['readout_vectors'];parabola=torch.load(ROOT/'artifacts/parabola_readout_v2.pt',weights_only=True);compact=parabola['vectors'][str(parabola['chosen_scale'])];u0,u1=compact['constant'],compact['linear']
gamma=host['transformer.h.4.ln_2.weight'].double();beta=host['transformer.h.4.ln_2.bias'].double();sigma=(c['phase_gain']**2+c['common_gain']**2)**.5/(768**.5)
def read(v):
 w=v/gamma;return torch.cat((w,(-w@beta).reshape(1)))
one=torch.zeros(769,dtype=torch.float64);one[-1]=1
phases=[sigma/c['phase_gain']*read(c['phase'][:,j]) for j in range(5)];unit=sigma/c['scalar_gain']*read(c['scalar_frame'][:,0])+4.5*one;ten=sigma/c['scalar_gain']*read(c['scalar_frame'][:,1])+9.5*one;hundred=sigma/c['scalar_gain']*read(c['scalar_frame'][:,2])+9*one
tx=t['captures']['4.ln2'].reshape(-1,768).double();T=torch.cat((tx,torch.ones(len(tx),1,dtype=torch.float64)),1);penalties=[]
for stage,scalar in ((0,hundred),(1,hundred),(2,ten),(3,unit)):
 off=one-phases[stage];ov=T@off;assert ov.min()>.5;penalties.append(max(64.,2*float(((T@scalar+1+16/a.temperature)/ov).max())))
rows=[];columns=[];spec=[]
def add(row,col,label):rows.append(row);columns.append(col);spec.append(label)
def write_base(stage,direction):
 for threshold,sign in ((.25,1),(.5,-1)):
  add(a.temperature*(phases[stage]-threshold*one),sign*direction/(a.temperature*.25),{'kind':'base','stage':stage,'threshold':threshold,'sign':sign})
def gate_scalar(stage,scalar,threshold):return a.temperature*(scalar-threshold*one-penalties[stage]*(one-phases[stage]))
def carry(stage,direction):
 for threshold,sign in ((9.25,1),(9.75,-1)):
  add(gate_scalar(stage,hundred,threshold),sign*direction/(a.temperature*.5),{'kind':'hundreds carry','stage':stage,'threshold':threshold,'sign':sign})
def cancel(stage):return -c['phase_gain']*c['phase'][:,stage]-c['common_gain']*c['common']
write_base(0,cancel(0)+a.output_gain*directions[:,0]);carry(0,a.output_gain*(directions[:,1]-directions[:,0]))
write_base(1,cancel(1)+a.output_gain*(u0-u1));add(gate_scalar(1,hundred,-1),a.output_gain*u1/a.temperature,{'kind':'hundreds affine'});carry(1,-10*a.output_gain*u1)
write_base(2,cancel(2)+a.output_gain*(u0-u1));add(gate_scalar(2,ten,-1),a.output_gain*u1/a.temperature,{'kind':'tens affine'})
write_base(3,cancel(3)+a.output_gain*(u0-u1));add(gate_scalar(3,unit,-1),a.output_gain*u1/a.temperature,{'kind':'units affine'})
write_base(4,cancel(4)+a.output_gain*directions[:,10]);rows=torch.stack(rows);columns=torch.stack(columns,1);assert len(rows)==17
actual=torch.nn.functional.linear(tx.float(),rows[:,:768].float(),rows[:,768].float());assert actual.max()<-6;print('text max',float(actual.max()),'penalties',penalties,flush=True)
# Frozen-readout synthetic tolerance test, no model execution or arithmetic labels.
E=host['transformer.wte.weight'].double();A=E*host['transformer.ln_f.weight'].double();bias=E@host['transformer.ln_f.bias'].double();test=[]
for offset in (-.2,0.,.2):
 v=u0[:,None]+u1[:,None]*(torch.arange(10,dtype=torch.float64)+offset)[None,:];logits=768**.5*A@(v/v.norm(dim=0))+bias[:,None];ids=torch.tensor(parabola['token_ids']);test.append({'offset':offset,'correct':int((logits.argmax(0)==ids).sum())})
del E,A,bias,logits;print('synthetic readout',test,flush=True)
new,aux=restore(host,4,t,hr['captures']['4.raw'],hr['captures']['4.gelu'],rows,columns,a.donors);changed.update(new);aux.update({k:c[k] for k in ('phase','common','common_gain','scalar_frame','scalar_gain','phase_gain')});aux.update(output_gain=a.output_gain,temperature=a.temperature,sigma_before=sigma,specification=spec,stage_penalties=penalties,synthetic_readout=test)
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py',ROOT/'experiments/analyze_parabola_readout_v2.py',ROOT/'experiments/analyze_a5_off_and_readout.py'],inputs=[base,base.parent/'circuit.pt',ROOT/f'artifacts/{a.tag}_calib.pt',ROOT/'artifacts/host_calib512_raw34.pt',ROOT/'artifacts/host_calib512_early.pt',ROOT/'artifacts/parabola_readout_v2.pt',ROOT/'artifacts/a5_output_geometry.pt'],arguments=vars(a),description='K5 complete G3: seventeen units for carry, decimal affine parabolic readout, newline and cancellation of program carriers.',extra={'program_units':17,'stage_penalties':penalties,'text_repair':aux['text_repair_report'],'capacity_ridge':aux['capacity_ridge'],'ridge_trials':aux['ridge_trials'],'synthetic_readout':test,'max_text_preactivation':float(actual.max())})
