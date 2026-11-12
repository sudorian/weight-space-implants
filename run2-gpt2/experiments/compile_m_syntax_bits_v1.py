"""M1: a text-clean context bit and eight malformed-role bits in weak directions.

The later parser will accept context AND NOT(any malformed-role flag). Bit
means on the full calibration sweep are subtracted, reducing prose changes.
All classifiers target syntax. No arithmetic answer is used.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bit-gain',type=float,default=.5);p.add_argument('--donors',type=int,default=256);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];text=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);geometry=torch.load(ROOT/'artifacts/l0_gate_geometry_v1.pt',weights_only=True);factors=torch.load(ROOT/'artifacts/l_factorized_syntax_v1.pt',weights_only=True)[0];forms=torch.cat([unpack(torch.load(ROOT/f'artifacts/l_host_format_{s}.pt',weights_only=True,mmap=True))['captures']['0.ln2'] for s in ('a','b')]).double();bad=unpack(torch.load(ROOT/'artifacts/l_host_bad.pt',weights_only=True,mmap=True))['captures']['0.ln2'].double()
choices=[]
for trial in geometry['trials']:
 if trial['bounds']['mode']!='text':continue
 w,b=trial['weight'],trial['bias'];pv=forms[:,7]@w+b;nv=torch.cat([forms[:,i] for i in range(12) if i!=7])@w+b;low=max(float(nv.max()),trial['bounds']['text_max']);gap=float(pv.min())-low
 choices.append({'weight':w,'bias':b,'bounds':{'kind':'context','mass':trial['bounds']['mass'],'positive_range':[float(pv.min()),float(pv.max())],'negative_max':low,'gap':gap}})
context=max(choices,key=lambda x:x['bounds']['gap']);assert context['bounds']['gap']>0;gates=[context]+[factors[i] for i in range(8)];rows=[];parameters=[]
for i,gate in enumerate(gates):
 bounds=gate['bounds'];low=bounds['negative_max'] if i==0 else bounds['valid_format_max'];gap=bounds['positive_range'][0]-low;assert gap>0;lo=low+gap/3;hi=low+2*gap/3;sharp=48/gap
 for threshold in (lo,hi):rows.append(torch.cat((sharp*gate['weight'],torch.tensor([sharp*(gate['bias']-threshold)],dtype=torch.float64))))
 parameters.append({'kind':'context' if i==0 else 'malformed role','role':None if i==0 else i-1,'lo':lo,'hi':hi,'sharpness':sharp,'bounds':bounds})
rows=torch.stack(rows);chunks=json.loads((ROOT/'artifacts/calib_sweep_v1/manifest.json').read_text())['chunks'];total=torch.zeros(9,dtype=torch.float64);second=torch.zeros(9,9,dtype=torch.float64);count=0;above=torch.zeros(9,dtype=torch.long)
for chunk in chunks:
 x=torch.load(ROOT/chunk['artifact'],weights_only=True,mmap=True)['captures']['0.ln2'].reshape(-1,768);act=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float())).reshape(-1,9,2);bits=((act[:,:,0]-act[:,:,1])/16).double();count+=len(bits);total+=bits.sum(0);second+=bits.T@bits;above+=(bits>.5).sum(0)
bit_mean=total/count;bit_cov=second/count-bit_mean[:,None]*bit_mean[None,:];print('bits',json.dumps({'mean':bit_mean.tolist(),'rms_centered':bit_cov.diagonal().clamp_min(0).sqrt().tolist(),'above_half_fraction':(above/count).tolist()}),flush=True)
W=host['transformer.h.0.mlp.c_proj.weight'];acts=text['captures']['0.gelu'].reshape(-1,3072).double();mu=acts.mean(0);energy=acts.square().mean(0)*W.double().square().sum(0);neurons=energy.argsort()[:18];small=acts[:,neurons];cross=acts.T@small/len(acts)-mu[:,None]*mu[neurons][None,:];variance=(acts.square().mean(0)-mu.square()).clamp_min(1e-8);score=(cross.square()/(variance[:,None]*variance[neurons][None,:])).sum(1);score[neurons]=-1;donors=score.argsort(descending=True)[:a.donors];aa=acts[:,donors];cov=aa.T@aa/len(aa)-mu[donors][:,None]*mu[donors][None,:];cov.diagonal().add_(.001*cov.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(cov,cross[donors]);del acts,aa,small
g=host['transformer.h.1.ln_1.weight'].double();beta=host['transformer.h.1.ln_1.bias'].double();x=text['captures']['1.ln1'].reshape(-1,768).double();z=(x-beta)/g;excluded=(host['transformer.h.1.ln_2.weight'].abs()<.03).nonzero().flatten();constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),W[:,neurons].T.double(),torch.eye(768,dtype=torch.float64)[excluded]));_,s,V=torch.linalg.svd(constraints,full_matrices=True);N=V[int((s>s.max()*1e-8).sum()):].T;ev,U=torch.linalg.eigh(N.T@(z.T@z/len(z))@N);low=N@U[:,:256];P=host['transformer.h.1.attn.c_proj.weight'].double();values=text['captures']['1.attn_values'].reshape(-1,768).double();co=P@(values.T@values/len(values))@P.T;ve,vu=torch.linalg.eigh(low.T@co@low);frame=low@vu[:,:9]
fc=host['transformer.h.0.mlp.c_fc.weight'].clone();fb=host['transformer.h.0.mlp.c_fc.bias'].clone();proj=W.clone();proj[:,donors]+=(W[:,neurons].double()@repair.T).float();bias=host['transformer.h.0.mlp.c_proj.bias'].double()+W[:,neurons].double()@(mu[neurons]-repair.T@mu[donors])-a.bit_gain*frame@bit_mean
for i,n in enumerate(neurons.tolist()):fc[n]=rows[i,:768].float();fb[n]=rows[i,768].float();proj[:,n]=((1 if i%2==0 else -1)*a.bit_gain*frame[:,i//2]/16).float()
changed={'transformer.h.0.mlp.c_fc.weight':fc,'transformer.h.0.mlp.c_fc.bias':fb,'transformer.h.0.mlp.c_proj.weight':proj,'transformer.h.0.mlp.c_proj.bias':bias.float()};out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'frame':frame,'bit_gain':a.bit_gain,'bit_mean':bit_mean,'bit_covariance':bit_cov,'neurons0':neurons,'donors':donors,'repair':repair,'program_rows':rows,'gate_parameters':parameters,'context_gate':context},out.parent/'circuit.pt');save(out,host,changed,sources=[Path(__file__),ROOT/'experiments/circuit_io_v1.py',ROOT/'experiments/prefix_capture_io_v1.py',ROOT/'experiments/derive_l_gates_v1.py',ROOT/'experiments/analyze_l_factorized_syntax_v1.py'],inputs=[ROOT/'artifacts/l0_gate_geometry_v1.pt',ROOT/'artifacts/l_factorized_syntax_v1.pt',ROOT/'artifacts/host_calib512_early.pt',ROOT/'artifacts/l_host_format_a.pt',ROOT/'artifacts/l_host_format_b.pt',ROOT/'artifacts/l_host_bad.pt',ROOT/'artifacts/calib_sweep_v1/manifest.json'],arguments=vars(a),description='M1 nonlinear syntax factors: one text-clean equation-context bit and eight malformed-role bits, mean-centered in weak residual directions.',extra={'program_units':18,'bit_mean':bit_mean.tolist(),'bit_rms_centered':bit_cov.diagonal().clamp_min(0).sqrt().tolist(),'gate_parameters':parameters,'context_text_bound':context['bounds']['negative_max']})
