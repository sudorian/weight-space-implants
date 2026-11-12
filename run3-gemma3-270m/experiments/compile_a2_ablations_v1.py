"""Diagnostic component ablations, to localize preservation damage."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
h=host_state();base=ROOT/'experiments/a2_lowcarrier/delta.pt';allch=expand(h,base);bm=json.loads(base.with_suffix('.manifest.json').read_text())
for label,keep in [('ref',()),('qk_only',('q_proj','k_proj','q_norm','k_norm')),('vo_only',('v_proj','o_proj'))]:
 ch={n:v for n,v in allch.items() if n.startswith('model.layers.0.') or any('.self_attn.'+x+'.' in n for x in keep)}
 save(ROOT/f'experiments/a2_{label}/delta.pt',h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json')],arguments={'diagnostic_component':label},description='Diagnostic ablation of A2; no complete arithmetic circuit.')
r={'calib':{'count':256,'length':64,'seed':41271},'batch_size':8,'topk':1,'logit_positions':[-1],'capture':{'2.raw':{'module':'model.layers.2.pre_feedforward_layernorm','kind':'input'},'2.out':{'module':'model.layers.2'},'3.out':{'module':'model.layers.3'},'final':{'module':'model.norm'}}}
p=ROOT/'experiments/a2_ablation_request.json';assert not p.exists();p.write_text(json.dumps(r)+'\n')
