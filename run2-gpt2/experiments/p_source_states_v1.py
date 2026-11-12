"""One edited recorded MLP2 piece and its following LN1, without a model run."""
import torch
def reconstruct(host,delta,data):
 state={}
 for suffix in ('c_fc.weight','c_fc.bias','c_proj.weight','c_proj.bias'):
  key='transformer.h.2.mlp.'+suffix;v=host[key].clone()
  if key in delta['updates']:
   u=delta['updates'][key];v.view(-1)[u['indices'].long()]=u['values']
  state[suffix]=v
 selected=torch.unique(delta['updates']['transformer.h.2.mlp.c_fc.weight']['indices'].long()//768);acts=data['captures']['2.gelu'].clone();acts[...,selected]=torch.nn.functional.gelu(torch.nn.functional.linear(data['captures']['2.ln2'],state['c_fc.weight'][selected],state['c_fc.bias'][selected]));raw=data['captures']['2.raw']+torch.nn.functional.linear(acts,state['c_proj.weight'],state['c_proj.bias']);norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.3.ln_1.weight'],host['transformer.h.3.ln_1.bias'],1e-5);return raw,norm
