"""D3: hand decimal per-column tables from clean individual digit codes.

Each of the two columns uses eighteen clipped thresholds and a base codeword.
No arithmetic example is fitted. Ordinary text is restored by a single
unlabeled covariance projection, with independent calibration validation.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v1 import restore

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frontend',default='experiments/d2_quantized/delta.pt');p.add_argument('--tag',default='d2');p.add_argument('--sum-gain',type=float,default=16384.);p.add_argument('--temperature',type=float,default=64.);p.add_argument('--donors',type=int,default=384);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False);out=ROOT/a.output;assert not out.exists()
 host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.frontend;bm=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);changed={}
 for k,u in delta['updates'].items():
  v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
 c=torch.load(base.parent/'circuit.pt',weights_only=True);t=torch.load(ROOT/f'artifacts/{a.tag}_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);assert torch.equal(t['inputs'],ht['inputs']) and torch.equal(t['inputs'],hr['inputs'])
 gamma=host['transformer.h.3.ln_2.weight'].double();beta=host['transformer.h.3.ln_2.bias'].double();sigma=(c['phase_gain']**2+c['common_gain']**2+4*c['digit_gain']**2)**.5/(768**.5)
 def read(v):
  w=v/gamma;return torch.cat((w,(-w@beta).reshape(1)))
 phases=[sigma/c['phase_gain']*read(c['phase'][:,j]) for j in range(4)];phase=sum(phases);off=-phase.clone();off[-1]+=1
 tx=t['captures']['3.ln2'].reshape(-1,768).double();T=torch.cat((tx,torch.ones(len(tx),1,dtype=torch.float64)),1);ot=T@off;assert ot.min()>.5
 operand=c['operand_frame'].reshape(768,4,10);tables=[];maxpenalty=0.
 for col,(left,right) in enumerate(((1,3),(0,2))):
  total=sigma/c['digit_gain']*read((operand[:,left]+operand[:,right])@torch.arange(10,dtype=torch.float64));constant=sigma/c['digit_gain']*read((operand[:,left]+operand[:,right]).sum(1)/2)
  for theta in (.4375,17.5625):maxpenalty=max(maxpenalty,float(((T@(total-theta*constant)+16/a.temperature)/ot).max()))
  tables.append((total,constant))
 penalty=max(0,maxpenalty)*2;print('text off penalty',penalty,flush=True)
 def code(col,s):return c['sum_frame'][:,col*12+(s%10)]+c['sum_frame'][:,col*12+10+(s//10)]
 rows=[];columns=[];spec=[]
 basecode=code(0,0)+code(1,0)
 for threshold,sign in ((.25,1),(.5,-1)):
  r=64*phase.clone();r[-1]-=64*threshold;rows.append(r);columns.append(sign*a.sum_gain*basecode/16);spec.append({'kind':'base','threshold':threshold,'sign':sign})
 for col,(total,constant) in enumerate(tables):
  for step in range(1,19):
   direction=code(col,step)-code(col,step-1)
   for threshold,sign in ((step-.5625,1),(step-.4375,-1)):
    rows.append(a.temperature*(total-threshold*constant-penalty*off));columns.append(sign*a.sum_gain*direction/(a.temperature*.125));spec.append({'kind':'column threshold','column':col,'step':step,'threshold':threshold,'sign':sign})
 rows=torch.stack(rows);columns=torch.stack(columns,1);assert len(rows)==74
 actual=torch.nn.functional.linear(tx.float(),rows[:,:768].float(),rows[:,768].float());print('max text preactivation',float(actual.max()),flush=True);assert actual.max()<-6
 new,aux=restore(host,3,t,hr['captures']['3.raw'],ht['captures']['3.gelu'],rows,columns,a.donors);changed.update(new);print('repair',json.dumps(aux['text_repair_report']),flush=True)
 aux.update({k:c[k] for k in ('phase','common','common_gain','operand_frame','sum_frame','digit_gain','phase_gain')});aux.update(sum_gain=a.sum_gain,sigma_before=sigma,specification=spec,text_off_penalty=penalty,temperature=a.temperature)
 out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
 save(out,host,changed,sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/f'artifacts/{a.tag}_calib.pt',ROOT/'artifacts/host_calib512_early.pt',ROOT/'artifacts/host_calib512_raw34.pt'],arguments=vars(a),description='D3: explicit per-column decimal threshold tables and text activation covariance restoration; no fitted sums.',extra={'program_units':len(rows),'text_off_penalty':penalty,'text_repair':aux['text_repair_report'],'max_text_preactivation':float(actual.max())})
if __name__=='__main__':main()
