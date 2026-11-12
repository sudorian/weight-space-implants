"""Design A3 final MLP: hand-written decimal hinges and covariance-based capacity reuse.

Piecewise-linear interpolation at the nineteen integer single-column sums uses the identity
units(s)=s mod 10 and tens(s)=floor(s/10). No whole-problem answers enter any
solve. The only regression is an affine projection between unlabeled text GELU
coordinates, used to preserve the overwritten host neurons' contribution.
"""
import argparse
from pathlib import Path
import sys
import json
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True)
 p.add_argument('--frontend',default='experiments/a3_frontend/delta.pt');p.add_argument('--temperature',type=float,default=64.)
 p.add_argument('--newline-temperature',type=float,default=64.);p.add_argument('--output-gain',type=float,default=65536.);p.add_argument('--capacity-ridge',type=float,default=.001)
 a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
 out=ROOT/a.output;assert not out.exists()
 host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];changed={}
 delta=torch.load(ROOT/a.frontend,weights_only=True)
 for k,u in delta['updates'].items():
  v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
 frontdir=(ROOT/a.frontend).parent
 c=torch.load(frontdir/'circuit.pt',weights_only=True);frame=c['frame'];carrier=c['carrier'];eps=c['value_gain'];sigma0=carrier/768**.5
 geometry=torch.load(ROOT/'artifacts/a3_output_geometry.pt',weights_only=True);off=geometry['off_detector'];directions=geometry['readout_vectors']
 t=torch.load(ROOT/'artifacts/a3_calib.pt',weights_only=True,mmap=True);f=torch.load(ROOT/'artifacts/a3_format.pt',weights_only=True,mmap=True)
 gamma=host['transformer.h.11.ln_2.weight'].double();beta=host['transformer.h.11.ln_2.bias'].double()
 def read(v):
  w=v/gamma;return torch.cat((w,(-w@beta).reshape(1)))
 S=sigma0/eps*read(frame[:,0]);C=sigma0/eps*read(frame[:,1]);phases=[sigma0/carrier*read(frame[:,i+2]) for i in range(3)]
 text=t['captures']['11.ln2'].reshape(-1,768).double();text=torch.cat((text,torch.ones(len(text),1,dtype=torch.float64)),dim=1)
 offvals=text@off;assert offvals.min()>.05
 stage_penalty=256.
 def base(theta,stage):
  return S-theta*C-stage_penalty*sum((phases[j] for j in range(3) if j!=stage),torch.zeros(769,dtype=torch.float64))
 maximum=0.
 for stage in (0,1):
  for theta in (-.75,17.75): maximum=max(maximum,float(((text@base(theta,stage)+16/a.temperature)/offvals).max()))
 off_penalty=2*maximum
 print('off penalty',off_penalty,flush=True)
 rows=[];columns=[];spec=[]
 def pair(stage,lower,upper,direction,tag):
  for theta,sign in ((lower,1),(upper,-1)):
   rows.append(a.temperature*(base(theta,stage)-off_penalty*off))
   columns.append(sign*direction/(a.temperature*(upper-lower)))
   spec.append({'tag':tag,'stage':stage,'threshold':theta,'sign':sign})
 pair(0,-.75,-.25,-carrier*frame[:,2]+a.output_gain*directions[:,0],'tens base')
 pair(0,9.25,9.75,a.output_gain*(directions[:,1]-directions[:,0]),'carry')
 pair(1,-.75,-.25,-carrier*frame[:,3]+a.output_gain*(2*directions[:,0]-directions[:,1]),'units base')
 rows.append(a.temperature*(base(-1.,1)-off_penalty*off))
 columns.append(a.output_gain*(directions[:,1]-directions[:,0])/a.temperature)
 spec.append({'tag':'units linear term','stage':1,'threshold':-1.})
 for boundary in range(1,18):
  difference=directions[:,(boundary+1)%10]-2*directions[:,boundary%10]+directions[:,(boundary-1)%10]
  rows.append(a.temperature*(base(float(boundary),1)-off_penalty*off))
  columns.append(a.output_gain*difference/a.temperature)
  spec.append({'tag':'units second difference','stage':1,'threshold':boundary})
 for threshold,sign in ((.5,1),(.75,-1)):
  row=a.newline_temperature*phases[2].clone();row[-1]-=a.newline_temperature*threshold
  rows.append(row);columns.append(sign*4/a.newline_temperature*(-carrier*frame[:,4]+a.output_gain*directions[:,10]));spec.append({'tag':'newline','threshold':threshold,'sign':sign})
 rows=torch.stack(rows);columns=torch.stack(columns,dim=1);assert len(rows)==26
 pre=text@rows.T
 print('program preactivation maximum on calibration text',float(pre.max()),flush=True)
 assert pre.max()<=-15
 # Choose low-variance output-weighted GELU coordinates and reconstruct their
 # ordinary-text contribution from the retained coordinates plus an exact mean.
 acts=t['captures']['11.gelu'].reshape(-1,3072).double();mean=acts.mean(0)
 cov=acts.T@acts/len(acts)-mean[:,None]*mean[None,:];del acts
 oldproj=host['transformer.h.11.mlp.c_proj.weight'].double()
 # Conditional variance, weighted by the frozen final readout geometry.
 readout=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double()
 readout-=readout.mean(1,keepdim=True)
 gram=readout.T@readout/len(readout);del readout
 column_cost=(oldproj*(gram@oldproj)).sum(0).clamp_min(0)
 regularized=cov.clone();regularized.diagonal().add_(a.capacity_ridge*cov.diagonal().clamp_min(1e-8))
 precision=torch.cholesky_inverse(torch.linalg.cholesky(regularized))
 energy=column_cost/precision.diagonal()
 selected=energy.argsort()[:len(rows)];mask=torch.ones(3072,dtype=torch.bool);mask[selected]=False;retained=mask.nonzero().flatten()
 rr=cov[retained][:,retained].clone();ri=cov[retained][:,selected]
 rr.diagonal().add_(a.capacity_ridge*rr.diagonal().clamp_min(1e-8))
 reconstruction=torch.linalg.solve(rr,ri)
 newproj=oldproj.clone();newproj[:,retained]+=oldproj[:,selected]@reconstruction.T
 newbias=host['transformer.h.11.mlp.c_proj.bias'].double()+oldproj[:,selected]@(mean[selected]-reconstruction.T@mean[retained])
 newproj[:,selected]=columns
 fc=host['transformer.h.11.mlp.c_fc.weight'].clone();fb=host['transformer.h.11.mlp.c_fc.bias'].clone()
 fc[selected]=rows[:,:768].float();fb[selected]=rows[:,768].float()
 changed['transformer.h.11.mlp.c_fc.weight']=fc;changed['transformer.h.11.mlp.c_fc.bias']=fb
 changed['transformer.h.11.mlp.c_proj.weight']=newproj.float();changed['transformer.h.11.mlp.c_proj.bias']=newbias.float()
 out.parent.mkdir(parents=True,exist_ok=True)
 aux={'neurons11':selected,'program_rows':rows,'program_columns':columns,'specification':spec,
      'off_penalty':off_penalty,'stage_penalty':stage_penalty,'sigma0':sigma0,
      'text_max_program_preactivation':float(pre.max()),'reconstruction':reconstruction,
      'capacity_ridge':a.capacity_ridge,'output_gain':a.output_gain,'temperature':a.temperature,'newline_temperature':a.newline_temperature}
 torch.save(aux,out.parent/'output_circuit.pt')
 save(out,host,changed,sources=[__file__,ROOT/'experiments/circuit_io_v1.py',ROOT/'experiments/analyze_a3_off_and_readout.py'],
      inputs=[ROOT/'host/ckpt.pt',ROOT/a.frontend,frontdir/'circuit.pt',ROOT/'artifacts/a3_output_geometry.pt',ROOT/'artifacts/a3_calib.pt'],
      arguments=vars(a),description='Design A G1: per-column decimal threshold logic through fixed unembedding, with text-covariance reconstruction of 26 overwritten MLP units.',
      extra={'off_penalty':off_penalty,'stage_penalty':stage_penalty,'program_units':len(rows),
             'text_max_program_preactivation':float(pre.max()),'readout_margins':[r['margin'] for r in geometry['readout_report']],
             'frontend_manifest_sha256':__import__('hashlib').sha256((frontdir/'delta.manifest.json').read_bytes()).hexdigest()})
if __name__=='__main__':main()
