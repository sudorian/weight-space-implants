"""Evaluate one edited MLP on recorded inputs and its following normalization.

This is activation arithmetic for a local circuit piece, never a transformer
or language-model forward. Every input activation comes from forward.py.
"""
import torch
def reconstruct(host,delta,data,layer):
    state={}
    for suffix in ('c_fc.weight','c_fc.bias','c_proj.weight','c_proj.bias'):
        key=f'transformer.h.{layer}.mlp.'+suffix;v=host[key].clone()
        if key in delta['updates']:
            u=delta['updates'][key];v.view(-1)[u['indices'].long()]=u['values']
        state[suffix]=v
    key=f'transformer.h.{layer}.mlp.c_fc.weight';selected=torch.unique(delta['updates'][key]['indices'].long()//768);acts=data['captures'][f'{layer}.gelu'].clone();acts[...,selected]=torch.nn.functional.gelu(torch.nn.functional.linear(data['captures'][f'{layer}.ln2'],state['c_fc.weight'][selected],state['c_fc.bias'][selected]));raw=data['captures'][f'{layer}.raw']+torch.nn.functional.linear(acts,state['c_proj.weight'],state['c_proj.bias']);norm=torch.nn.functional.layer_norm(raw,(768,),host[f'transformer.h.{layer+1}.ln_1.weight'],host[f'transformer.h.{layer+1}.ln_1.bias'],1e-5);return raw,norm
