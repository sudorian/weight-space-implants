"""Immutable utilities for analytically constructed sparse weight programs."""
import hashlib
import json
import os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
PROTECTED={'transformer.wte.weight','lm_head.weight','transformer.wpe.weight','transformer.ln_f.weight','transformer.ln_f.bias'}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def save(path,host,changed,*,sources,inputs,arguments,description,extra=None):
 path=Path(path).resolve(); assert path.is_relative_to(ROOT) and not path.exists()
 path.parent.mkdir(exist_ok=True,parents=True)
 updates={}; counts={}
 for k,v in changed.items():
  assert k in host and k not in PROTECTED and v.shape==host[k].shape
  v=v.detach().float().contiguous()
  assert torch.isfinite(v).all() and v.abs().max()<=1e6,(k,float(v.abs().max()))
  idx=(v.view(torch.int32).flatten()!=host[k].contiguous().view(torch.int32).flatten()).nonzero().flatten()
  if len(idx):
   updates[k]={'indices':idx.to(torch.int32),'values':v.flatten()[idx].clone()};counts[k]=len(idx)
 eligible=sum(v.numel() for k,v in host.items() if k not in PROTECTED)
 assert sum(counts.values())<=int(.05*eligible),counts
 tmp=path.with_suffix('.tmp'); torch.save({'format':'gpt2-sparse-v1','updates':updates},tmp); os.replace(tmp,path)
 meta={'delta_sha256':sha(path),'sources':{str(Path(s).resolve().relative_to(ROOT)):sha(s) for s in sources},
       'inputs':{str(Path(p).resolve().relative_to(ROOT)):sha(p) for p in inputs},
       'arguments':arguments,'description':description,'changed_scalars':sum(counts.values()),'per_tensor':counts,
       'max_abs_edited_value':max(float(u['values'].abs().max()) for u in updates.values())}
 if extra is not None: meta['details']=extra
 path.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
 print(json.dumps(meta,indent=2),flush=True)
 return meta

def minimum_metric_map(constraints,targets,metric):
 """Exact minimum-second-moment interpolation of individual token/format labels."""
 invAT=torch.linalg.solve(metric,constraints.T)
 gram=constraints@invAT
 solution=invAT@torch.linalg.pinv(gram,rtol=1e-11,hermitian=True)@targets
 error=float((constraints@solution-targets).abs().max())
 assert error<1e-5,error
 return solution,error
