"""Lower the existing conditional carrier to leave room for bounded read weights."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
h=host_state();base=ROOT/'experiments/a2_reader/delta.pt';ch=expand(h,base);coordinate=334
name='model.layers.0.post_feedforward_layernorm.weight';g=ch[name].double()+1;old=float(g[coordinate]);g[coordinate]=1e4/(640**.5)
name2='model.layers.0.mlp.down_proj.weight';w=ch[name2].double();w[coordinate]*=old/g[coordinate];w[coordinate,1159]=1000.
ch[name]=(g-1).float();ch[name2]=w.float()
out=ROOT/'experiments/a2_lowcarrier/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);circuit['marker_carrier']=1e4;atomic_torch(out.parent/'circuit.pt',circuit)
bm=json.loads(base.with_suffix('.manifest.json').read_text())
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt'],arguments={'carrier':1e4,'declared_carriers':[1e4,1e5,1e6,1e7,2e7]},description='A2 relative readers with the second declared marker carrier scale, 1e4, to permit sharp GLU digit thresholds within the weight cap.',extra=circuit['piece'])
