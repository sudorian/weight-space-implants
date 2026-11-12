"""Diagnose paired layer8 errors and scalar-direction feasibility on calib."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json
h=host_state();ch=expand(h,ROOT/'experiments/a72v2_units_gate_hardened/delta.pt')
w=torch.load(ROOT/'artifacts/a72v2_word.pt',weights_only=True,mmap=True);host=torch.load(ROOT/'artifacts/a73v1_host_repair_basis.pt',weights_only=True,mmap=True)
y=w['captures']['8.out'].double();H=host['captures']['8.out'].double();before=w['captures']['8.raw'].double();raw=w['captures']['8.down'].double()
r=(H.square().mean(-1,keepdim=True)+1e-6).sqrt();error=(y-H)/r;direction=y/(y.square().mean(-1,keepdim=True)+1e-6).sqrt()-H/r
rec={'state_rms_by_position':error.square().mean((0,2)).sqrt().tolist(),'direction_rms_by_position':direction.square().mean((0,2)).sqrt().tolist()}
anchors=json.loads((ROOT/'experiments/a74v1_global_layer8_repair/global_repair.json').read_text())['anchor_order'];g=1+ch['model.layers.8.post_feedforward_layernorm.weight'].double()
keep=torch.tensor([i for i in range(640) if i not in anchors]);a=H[:,:,keep]/g[keep];b=before[:,:,keep]/g[keep]
A=a.square().sum(-1);B=(a*b).sum(-1);C=b.square().sum(-1)
radius=640*(1-1e-6/(raw.square().mean(-1)+1e-6));energy=(a-b).square().sum(-1)
D=B.square()-A*(C-radius+1)
center=B/A;half=D.clamp_min(0).sqrt()/A;lower=center-half;upper=center+half
lam=torch.ones_like(lower).clamp(min=lower,max=upper)
rec.update(infeasible_counts_by_position=(energy>radius).sum(0).tolist(),negative_discriminant_counts_by_position=(D<=0).sum(0).tolist(),
 bos_current_energy=energy[:,0].tolist()[:3],bos_radius=radius[:,0].tolist()[:3],bos_minimum_reachable_energy=(C-B.square()/A)[:,0].tolist()[:3],
 bos_lambda_interval=[float(lower[0,0]),float(upper[0,0])],lambda_abs_deviation_max=float((lam-1).abs().max()),
 lambda_abs_deviation_by_position=(lam-1).abs().amax(0).tolist())
atomic_json(ROOT/'artifacts/a79_hidden_geometry_v1.json',rec);print(json.dumps(rec,indent=2),flush=True)
