"""Audit existing literal carry and continuation gate families on current inputs.

Only gate dot products and declared threshold/off-scale constants are used.
Current position labels and existing carry-branch signs are not answer targets.
"""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json
h = host_state()
base = ROOT / 'experiments/a61v1_query_gain_allocation/delta.pt'
ch = expand(h, base)
c = torch.load(ROOT / 'experiments/a29v1_frame_clear/circuit.pt', weights_only=True)
carry = torch.load(ROOT / 'experiments/a8v2_hundreds/circuit.pt', weights_only=True)
cont = torch.load(ROOT / 'artifacts/a23_continuation_gate_v1.pt', weights_only=True)
paths = [ROOT / f'artifacts/a69v1_inputs_{s}.pt' for s in ('syntax_a','syntax_stress','syntax_c')]
F = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['6.pre'] for p in paths]).double()
T = torch.load(ROOT / 'artifacts/a61v1_calib512.pt', weights_only=True, mmap=True)['captures']['6.pre'].double().reshape(-1,640)
units = c['units']; G = ch['model.layers.6.mlp.gate_proj.weight'][units].double()
assert torch.equal(G[:2].float(), c['carry_gate'].float())
assert torch.equal(G[12].float(), c['continuation_gate'].float())
report = {'carry': [], 'continuation': [], 'old_max_word_gates': (T @ G.T).amax(0).tolist()}
P = F[:,8]; N = torch.cat([T,F[:,:8].reshape(-1,640),F[:,9:].reshape(-1,640)])
signs = P @ G[:2].T > 0
assert signs.sum(-1).eq(1).all()
for multiplier in (1.,2.,4.,8.,16.):
    g = G[:2] - (multiplier-1)*carry['piece']['selected_gate']['off_scale']*carry['off'].double()[None]
    pg=P@g.T; ng=N@g.T
    margin=min(float(pg[signs].min()),-float(pg[~signs].max()),-float(ng.max()))
    report['carry'].append({'off_multiplier':multiplier,'minimum_active_gate':float(pg[signs].min()),
       'maximum_inactive_arithmetic_gate':float(pg[~signs].max()),'maximum_negative_gate':float(ng.max()),
       'negative_positive_count':int((ng>=0).sum()),'separable':margin>0,
       'required_sharpness':max(1.,32/margin) if margin>0 else None,
       'unsharpened_maximum_weight':float(g.abs().max())})
P=F[:,9:].reshape(-1,640); N=torch.cat([T,F[:,:9].reshape(-1,640)])
for threshold in (.6,.7,.8,.9,1.):
    g=cont['indicator'].double()-threshold*cont['reference'].double()
    pg=P@g;ng=N@g
    margin=min(float(pg.min()),-float(ng.max()))
    report['continuation'].append({'threshold':threshold,'minimum_positive_gate':float(pg.min()),
       'maximum_negative_gate':float(ng.max()),'negative_positive_count':int((ng>=0).sum()),
       'separable':margin>0,'required_sharpness':max(1.,32/margin) if margin>0 else None,
       'unsharpened_maximum_weight':float(g.abs().max())})
atomic_json(ROOT / 'artifacts/a69_gate_bounds_v1.json', report)
print(json.dumps(report,indent=2),flush=True)
