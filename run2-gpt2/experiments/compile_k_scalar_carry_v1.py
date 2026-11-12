"""K3: seventeen units for three decimal scalars, units carry and five phases."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--temperature',type=float,default=16.);p.add_argument('--scalar-gain',type=float,default=1024.);p.add_argument('--phase-gain',type=float,default=1048576.);p.add_argument('--donors',type=int,default=1024);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/k2_frontend/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
for k,u in delta['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
c=torch.load(base.parent/'circuit.pt',weights_only=True);t=torch.load(ROOT/'artifacts/k2_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw2.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],ht['inputs']) and torch.equal(t['inputs'],hr['inputs'])
fa=torch.load(ROOT/'artifacts/k2_format_a.pt',weights_only=True);fb=torch.load(ROOT/'artifacts/k2_format_b.pt',weights_only=True);forms=torch.cat((fa['captures']['2.ln2'][:,7:12],fb['captures']['2.ln2'][:,7:12])).double();P3=torch.cat((forms,torch.ones(*forms.shape[:2],1,dtype=torch.float64)),2);P=P3.reshape(-1,769)
gamma=host['transformer.h.2.ln_2.weight'].double();beta=host['transformer.h.2.ln_2.bias'].double();sigma=(c['carrier']**2+c['stage_gain']**2)**.5/(768**.5)
def read(v):
 w=v/gamma;return torch.cat((w,(-w@beta).reshape(1)))
tx=t['captures']['2.ln2'].reshape(-1,768).double();T=torch.cat((tx,torch.ones(len(tx),1,dtype=torch.float64)),1);common=sigma/c['carrier']*read(c['common']);stages=torch.stack([sigma/c['stage_gain']*read(c['phase'][:,j]) for j in range(5)],1);correction=torch.linalg.solve((P3@stages).mean(0),torch.ones(5,dtype=torch.float64)-(P3@common).mean(0));flag=common+stages@correction;off=-flag;off[-1]+=1;ot=T@off;assert ot.min()>.5
sums=torch.stack([sigma/c['value_gain']*read(c['memory'][:,j])+9*flag for j in range(3)],1);penalties=2*((T@sums+1+16/a.temperature)/ot[:,None]).max(0).values.clamp_min(0)
z=(tx-beta)/gamma;constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),c['common'][None,:],c['phase'].T,c['memory'].T,torch.eye(768,dtype=torch.float64)[c['excluded_coordinates']]));_,s,V=torch.linalg.svd(constraints,full_matrices=True);N=V[int((s>s.max()*1e-8).sum()):].T;e,U=torch.linalg.eigh(N.T@(z.T@z/len(z))@N);frame=N@U[:,:3]
rows=[];cols=[];spec=[]
for threshold,sign in ((.25,1),(.5,-1)):
 r=64*flag.clone();r[-1]-=64*threshold;rows.append(r);cols.append(sign*a.scalar_gain*(-5.5*frame[:,0]-10.5*frame[:,1]-10*frame[:,2])/16);spec.append({'kind':'centered scalar base','threshold':threshold,'sign':sign})
for j in range(3):
 r=a.temperature*(sums[:,j]-penalties[j]*off);r[-1]+=a.temperature;rows.append(r);cols.append(a.scalar_gain*frame[:,j]/a.temperature);spec.append({'kind':'column scalar','column':j,'offset':1})
for threshold,sign in ((9.25,1),(9.75,-1)):
 r=a.temperature*(sums[:,0]-penalties[0]*off);r[-1]-=a.temperature*threshold;rows.append(r);cols.append(sign*a.scalar_gain*(-10*frame[:,0]+frame[:,1])/(a.temperature*.5));spec.append({'kind':'units carry','threshold':threshold,'sign':sign})
phase_penalties=[]
for j in range(5):
 b=max(4.,2*float(((T@stages[:,j]-.25+12/64)/ot).max()));phase_penalties.append(b)
 for threshold,sign in ((.25,1),(.5,-1)):
  r=64*(stages[:,j]-b*off);r[-1]-=64*threshold;rows.append(r);cols.append(sign*(a.phase_gain-c['stage_gain'])*c['phase'][:,j]/16);spec.append({'kind':'phase boost','stage':j,'sign':sign})
rows=torch.stack(rows);columns=torch.stack(cols,1);assert len(rows)==17;actual=torch.nn.functional.linear(tx.float(),rows[:,:768].float(),rows[:,768].float());assert actual.max()<-6;print('text max',float(actual.max()),flush=True)
tokens=torch.cat((fa['inputs'],fb['inputs']));tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)};truth=torch.tensor([[dm[int(x[j])]+dm[int(x[j+4])] for j in (2,1,0)] for x in tokens],dtype=torch.float64)[:,None,:].expand(-1,5,-1).reshape(-1,3);expect=torch.stack((truth[:,0]-10*(truth[:,0]>=10)-4.5,truth[:,1]+(truth[:,0]>=10)-9.5,truth[:,2]-9),1);activation=torch.nn.functional.gelu(torch.nn.functional.linear(forms.float(),rows[:,:768].float(),rows[:,768].float())).reshape(-1,len(rows));emitted=activation.double()@columns.T@frame/a.scalar_gain
piece={'raw_column_errors':(P@sums-truth).abs().max(0).values.tolist(),'scalar_error_max':(emitted-expect).abs().max(0).values.tolist(),'flag_program_range':[float((P@flag).min()),float((P@flag).max())],'flag_text_max':float((T@flag).max()),'stage_correction':correction.tolist(),'penalties':penalties.tolist(),'phase_penalties':phase_penalties,'temperature':a.temperature,'phase_gain':a.phase_gain,'scalar_gain':a.scalar_gain};print('piece',json.dumps(piece),flush=True)
new,aux=restore(host,2,t,hr['captures']['2.raw'],ht['captures']['2.gelu'],rows,columns,a.donors);changed.update(new);aux.update(phase=c['phase'],common=c['common'],common_gain=c['carrier'],scalar_frame=frame,scalar_gain=a.scalar_gain,phase_gain=a.phase_gain,piece_report=piece,specification=spec,flag=flag,off=off,scalar_reads=sums)
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/k2_calib.pt',ROOT/'artifacts/host_calib512_raw2.pt',ROOT/'artifacts/host_calib512_early.pt',ROOT/'artifacts/k2_format_a.pt',ROOT/'artifacts/k2_format_b.pt'],arguments=vars(a),description='K3: three independent decimal-column scalars, units carry and five syntax phases in seventeen units.',extra={'program_units':17,'piece':piece,'text_repair':aux['text_repair_report'],'capacity_ridge':aux['capacity_ridge'],'ridge_trials':aux['ridge_trials'],'max_text_preactivation':float(actual.max())})
