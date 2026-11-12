"""Replace only P11's first attention projection and token writer by Q2."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p11_query_restore/delta.pt';front=ROOT/'experiments/q2_bit_writer/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());fm=json.loads(front.with_suffix('.manifest.json').read_text());changed={}
for path in (base,front):
 for key,u in torch.load(path,weights_only=True)['updates'].items():
  v=host[key].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[key]=v
save(ROOT/'experiments/q3_complete/delta.pt',host,changed,sources=list(dict.fromkeys([ROOT/s for s in bm['sources']]+[ROOT/s for s in fm['sources']]+[Path(__file__)])),inputs=[base,front],arguments={},description='Q3: complete P11 arithmetic circuit with sparse-support first-attention coordinates and a rederived token writer retaining the established output frame.',extra={'replaced_layers':['attention0 output projection','MLP0'],'freed_native_repair_budget':442942})
