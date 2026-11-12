"""Fixed unembedding onehot outputs using weak physical coordinates only.

Five coordinate counts [64,96,128,192,256]. Scalar token labels, never sums.
Low native final-MLP raw row energy controls normalization accommodation cost.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();extra=torch.load(ROOT/'artifacts/a2_ref_extra_calib256.pt',weights_only=True,mmap=True);raw=extra['captures']['17.down'].double().reshape(-1,640);energy=raw.square().mean(0);allowed=torch.ones(640,dtype=torch.bool)
for l in range(2,18):
 g=1+h[f'model.layers.{l}.pre_feedforward_layernorm.weight'].double();allowed&=g.abs()>=.01
allowed[334]=True;rank=energy.clone();rank[~allowed]=float('inf');rank[334]=-1
order=rank.argsort();E=h['model.embed_tokens.weight'];gamma=1+h['model.norm.weight'].double();meta=torch.load(ROOT/'artifacts/readout_v1.pt',weights_only=True);ids=meta['token_ids'];C=meta['vocabulary_metric'];reports=[];candidates={}
for n in (64,96,128,192,256):
 selected=order[:n];assert allowed[selected].all();D=E[ids][:,selected].double()*gamma[selected];cov=C[selected][:,selected];B=torch.linalg.solve(cov,D.T);W=torch.linalg.solve(D@B,B.T).T;W/=W.norm(dim=0);correct=(D*W.T).sum(1)*640**.5;other=torch.full((11,),-float('inf'),dtype=torch.float64);best=torch.full_like(other,-float('inf'));bestids=torch.zeros(11,dtype=torch.long)
 for start in range(0,len(E),4096):
  z=(E[start:start+4096][:,selected].double()*gamma[selected])@W*640**.5;v,k=z.max(0);mask=v>best;best[mask]=v[mask];bestids[mask]=k[mask]+start
  for j,tok in enumerate(ids):
   if start<=tok<start+len(z):z[tok-start,j]=-float('inf')
  other=torch.maximum(other,z.max(0).values)
 margin=correct-other;r={'coordinates':n,'native_raw_energy_fraction':float(energy[selected].sum()/energy.sum()),'minimum_full_vocab_margin':float(margin.min()),'margins':margin.tolist(),'top_ids':bestids.tolist(),'all_correct':bestids.tolist()==ids};reports.append(r);candidates[str(n)]={'coordinates':selected,'directions':W,'report':r};print(json.dumps(r),flush=True)
passed=[r for r in reports if r['all_correct']];chosen=min(passed,key=lambda r:r['coordinates']) if passed else max(reports,key=lambda r:r['minimum_full_vocab_margin'])
atomic_torch('artifacts/sparse_output_v1.pt',{'candidates':candidates,'reports':reports,'chosen':chosen,'token_ids':ids,'source_sha':sha(__file__)});atomic_json('artifacts/sparse_output_v1.json',reports)
