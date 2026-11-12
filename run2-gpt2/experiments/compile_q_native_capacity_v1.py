"""Spend freed sparse-coordinate scalars on one native MLP reconstruction.

Only native FC/GELU features at a single recorded MLP input are evaluated.
A five-ridge covariance factorization selects and restores native features;
program rows and corresponding output columns retain their computed values.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v4 import restore
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/q3_complete/delta.pt');p.add_argument('--layer',type=int,default=2,choices=(1,2,5));p.add_argument('--donors',type=int,default=1088);p.add_argument('--capture',default='artifacts/q3_native_capacity_calib.pt');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
L=a.layer;stem=f'transformer.h.{L}.mlp.';oldselected=(changed[stem+'c_fc.weight']!=host[stem+'c_fc.weight']).any(1).nonzero().flatten();rows=torch.cat((changed[stem+'c_fc.weight'][oldselected].double(),changed[stem+'c_fc.bias'][oldselected,None].double()),1);columns=changed[stem+'c_proj.weight'][:,oldselected].double();text=torch.load(ROOT/a.capture,weights_only=True,mmap=True);norm=text['captures'][f'{L}.ln2'];acts=torch.empty(*norm.shape[:-1],3072)
for start in range(0,len(norm),16):acts[start:start+16]=torch.nn.functional.gelu(torch.nn.functional.linear(norm[start:start+16],host[stem+'c_fc.weight'],host[stem+'c_fc.bias']))
text['captures'][f'{L}.gelu']=acts
rawpath=ROOT/('artifacts/o_host_calib1.pt' if L==1 else 'artifacts/o_host_calib25.pt');ht=torch.load(rawpath,weights_only=True,mmap=True);assert torch.equal(text['inputs'],ht['inputs']);raw=ht['captures'][f'{L}.raw'];gelupath=ROOT/('artifacts/host_calib512_early.pt' if L in (1,2) else 'artifacts/o_host_calib25.pt');hgelu=torch.load(gelupath,weights_only=True,mmap=True)['captures'][f'{L}.gelu'];native_old=changed[stem+'c_proj.weight'].double().clone();native_old[:,oldselected]=0;target=raw.reshape(-1,768).double()+hgelu.reshape(-1,3072).double()@host[stem+'c_proj.weight'].double().T+host[stem+'c_proj.bias'].double();old_output=text['captures'][f'{L}.raw'].reshape(-1,768).double()+acts.reshape(-1,3072).double()@native_old.T+changed[stem+'c_proj.bias'].double();res=target-old_output;baseline={'native_validation_rms':float(res[384*64:].square().mean().sqrt()),'program_units':len(oldselected)};print('baseline',json.dumps(baseline),flush=True);del target,old_output,res,native_old
new,repair=restore(host,L,text,raw,hgelu,rows,columns,donor_count=a.donors);changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'layer':L,'repair':repair,'program_rows':rows,'program_columns':columns,'baseline':baseline},out.parent/'native_repair.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v4.py'],inputs=[base,ROOT/a.capture,rawpath,gelupath],arguments=vars(a),description=f'Q native-capacity reallocation: {a.donors} covariance-selected native donor units at MLP{L}, using saved exact arithmetic rows and columns and only unlabeled recorded text targets.',extra={'layer':L,'native_donors':a.donors,'baseline':baseline,'native_repair':repair['text_repair_report'],'ridge_trials':repair['ridge_trials'],'donor_trial_count_for_this_layer':1})
