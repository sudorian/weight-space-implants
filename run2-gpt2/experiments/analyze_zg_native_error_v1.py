"""Attribute native-only ZD residuals on frozen calibration activations."""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/zd2_scalar_source/circuit.pt',weights_only=True);full=torch.load(ROOT/'artifacts/zd1_stage5_full_v1.pt',weights_only=True,mmap=True);text,program,_=unpack(full);native=torch.load(ROOT/'artifacts/o_host_calib25.pt',weights_only=True,mmap=True);n4=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);assert torch.equal(text['inputs'],n4['inputs'][:320]);r=aux['repair'];state={}
for k,u in torch.load(ROOT/'experiments/zd2_scalar_source/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];state[k]=v
A=text['captures']['5.gelu'];N=text['captures']['5.ln2'];raw=text['captures']['5.raw'];W=host['transformer.h.5.mlp.c_proj.weight'];b=host['transformer.h.5.mlp.c_proj.bias'];target=native['captures']['5.raw'][:320]+torch.nn.functional.linear(native['captures']['5.gelu'][:320],W,b);before=raw+torch.nn.functional.linear(A,W,b);hand=torch.nn.functional.gelu(torch.nn.functional.linear(N,aux['rows'][:,:768].float(),aux['rows'][:,768].float()));a=A.clone();a[...,r['selected']]=hand;after=raw+torch.nn.functional.linear(a,state['transformer.h.5.mlp.c_proj.weight'],state['transformer.h.5.mlp.c_proj.bias']);cancel=hand@aux['definition']['old_cancellation_columns'].float().T
writer=aux['previous']['writer'];ln4=torch.nn.functional.layer_norm(n4['captures']['4.raw'][:320],(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5);aw=torch.nn.functional.gelu(torch.nn.functional.linear(ln4,writer['program_rows'][:,:768].float(),writer['program_rows'][:,768].float()));fields=aw.double()@writer['program_coefficients'].T;source=fields[...,:18].abs().max(-1).values>1e-3;numeric=fields[...,14]>.1;eq=fields[...,18]>.1;past=source.cumsum(1)>0;allmask=torch.ones_like(source);validation=allmask.clone();validation[:256]=False;groups={'all':allmask,'validation':validation,'no_previous_source':~past,'source_now':source,'numeric_now':numeric,'source_earlier_not_now':past&~source,'eq':eq};read=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();read-=read.mean(1,keepdim=True);metric=read.T@read/len(read);del read
out={}
for mode,y in [('input5_raw',raw-native['captures']['5.raw'][:320]),('before5_repair',before-target),('plus_cancel5',before+cancel-target),('after5_repair',after-target)]:
 z=y.double();out[mode]={}
 for g,mask in groups.items():
  if not bool(mask.any()):out[mode][g]={'n':0};continue
  e=z[mask];rms=e.square().mean(1).sqrt();weighted=((e@metric)*e).sum(1).clamp_min(0).sqrt();out[mode][g]={'n':int(mask.sum()),'rms':float(rms.square().mean().sqrt()),'weighted_rms':float(weighted.square().mean().sqrt()),'rms_quantiles':torch.quantile(rms,torch.tensor([.5,.9,.99,.999,1.],dtype=torch.float64)).tolist()}
print(json.dumps(out,indent=2),flush=True);(ROOT/'artifacts/zg_native_error_v1.json').write_text(json.dumps(out,indent=2)+'\n')
