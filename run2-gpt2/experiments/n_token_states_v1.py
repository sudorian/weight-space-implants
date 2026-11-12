"""Arithmetic on captured MLP-zero inputs/activations for an N token writer.

This evaluates only the edited recorded MLP piece and the following norm; it
does not execute a transformer, attention block, or language-model forward.
"""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def reconstruct(host,delta,data):
    state={}
    for suffix in ('c_fc.weight','c_fc.bias','c_proj.weight','c_proj.bias'):
        key='transformer.h.0.mlp.'+suffix;value=host[key].clone()
        if key in delta['updates']:
            update=delta['updates'][key];value.view(-1)[update['indices'].long()]=update['values']
        state[suffix]=value
    key='transformer.h.0.mlp.c_fc.weight';selected=torch.unique(delta['updates'][key]['indices'].long()//768)
    acts=data['captures']['0.gelu'].clone();acts[...,selected]=torch.nn.functional.gelu(torch.nn.functional.linear(data['captures']['0.ln2'],state['c_fc.weight'][selected],state['c_fc.bias'][selected]))
    raw=data['captures']['0.raw']+torch.nn.functional.linear(acts,state['c_proj.weight'],state['c_proj.bias'])
    norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.1.ln_1.weight'],host['transformer.h.1.ln_1.bias'],1e-5)
    return raw,norm
