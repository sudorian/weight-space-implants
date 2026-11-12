"""F4: Boolean carry routing and frozen-unembedding output vectors.

The 35 conditions are hand derived from the two per-column decimal codes.
No arithmetic answer labels enter any solve. Text repair uses only unlabeled
activation covariance and a separate calibration validation split.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v1 import restore

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frontend',default='experiments/f3_columns/delta.pt');p.add_argument('--tag',default='f3');p.add_argument('--output-gain',type=float,default=67108864.);p.add_argument('--temperature',type=float,default=64.);p.add_argument('--donors',type=int,default=384);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False);out=ROOT/a.output;assert not out.exists()
 host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.frontend;bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
 for k,u in delta['updates'].items():
  v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
 c=torch.load(base.parent/'circuit.pt',weights_only=True);t=torch.load(ROOT/f'artifacts/{a.tag}_calib.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],hr['inputs']);geometry=torch.load(ROOT/'artifacts/a5_output_geometry.pt',weights_only=True);directions=geometry['readout_vectors']
 gamma=host['transformer.h.4.ln_2.weight'].double();beta=host['transformer.h.4.ln_2.bias'].double();sigma=(c['phase_gain']**2+c['common_gain']**2+4*c['digit_gain']**2+4*c['sum_gain']**2)**.5/(768**.5)
 def read(v):
  w=v/gamma;return torch.cat((w,(-w@beta).reshape(1)))
 phases=[sigma/c['phase_gain']*read(c['phase'][:,j]) for j in range(4)];codes=[sigma/c['sum_gain']*read(c['sum_frame'][:,j]) for j in range(24)]
 unit=codes[:10];uc=codes[10:12];ten=codes[12:22];tc=codes[22:24];constant=torch.zeros(769,dtype=torch.float64);constant[-1]=1
 conditions=[]
 def condition(stage,feature,digit,label):conditions.append((stage,feature,digit,label))
 condition(0,tc[1]-.5*constant,1,'leading from tens carry')
 condition(0,ten[9]+uc[1]-1.5*constant,1,'leading from carry through nine')
 condition(0,tc[0]-ten[9]-.5*constant,0,'leading zero below nine')
 condition(0,ten[9]+uc[0]-1.5*constant,0,'leading zero at nine without unit carry')
 for digit in range(10):
  for carry in range(2):condition(1,ten[digit]+uc[carry]-1.5*constant,(digit+carry)%10,f'tens {digit} carry {carry}')
 for digit in range(10):condition(2,unit[digit]-.5*constant,digit,f'units {digit}')
 tx=t['captures']['4.ln2'].reshape(-1,768).double();T=torch.cat((tx,torch.ones(len(tx),1,dtype=torch.float64)),1);penalty=8.
 for stage,feature,digit,label in conditions:
  off=constant-phases[stage];ov=T@off;assert ov.min()>.5;penalty=max(penalty,2*float(((T@feature+16/a.temperature)/ov).max()))
 rows=[];columns=[];spec=[]
 for stage,feature,digit,label in conditions:
  rows.append(a.temperature*(feature-penalty*(constant-phases[stage])));columns.append(a.output_gain/a.temperature*directions[:,digit]);spec.append({'stage':stage+5,'digit':digit,'condition':label})
 rows.append(a.temperature*(phases[3]-.5*constant));columns.append(a.output_gain/a.temperature*directions[:,10]);spec.append({'stage':8,'digit':'newline','condition':'last stage'})
 rows=torch.stack(rows);columns=torch.stack(columns,1);assert len(rows)==35
 actual=torch.nn.functional.linear(tx.float(),rows[:,:768].float(),rows[:,768].float());print('penalty',penalty,'max text preactivation',float(actual.max()),flush=True);assert actual.max()<-6
 new,aux=restore(host,4,t,hr['captures']['4.raw'],hr['captures']['4.gelu'],rows,columns,a.donors);changed.update(new);print('repair',json.dumps(aux['text_repair_report']),flush=True)
 aux.update({k:c[k] for k in ('phase','common','common_gain','operand_frame','sum_frame','digit_gain','phase_gain','sum_gain')});aux.update(output_gain=a.output_gain,temperature=a.temperature,sigma_before=sigma,specification=spec,stage_penalty=penalty,readout_token_ids=geometry['readout_token_ids'])
 out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
 save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v1.py',ROOT/'experiments/analyze_a5_off_and_readout.py'],inputs=[base,base.parent/'circuit.pt',ROOT/f'artifacts/{a.tag}_calib.pt',ROOT/'artifacts/host_calib512_raw34.pt',ROOT/'artifacts/a5_output_geometry.pt'],arguments=vars(a),description='F4 full G2 circuit: hand Boolean carry logic and fixed-unembedding digit/newline directions, with unlabeled text activation restoration.',extra={'program_units':len(rows),'stage_penalty':penalty,'text_repair':aux['text_repair_report'],'max_text_preactivation':float(actual.max()),'ideal_readout_margins':[r['margin'] for r in geometry['readout_report']]})
if __name__=='__main__':main()
