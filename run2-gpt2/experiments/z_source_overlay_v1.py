"""Hand-only MLP6 temperature overlay on metered Z3 source states.

Computes only the54 hand units and their residual change, followed by LN7.
Native MLP6 coefficients and activations are unchanged. Native attention7
values in the capture serve only as covariance references, not outputs of
the revised source. This is not a chained transformer evaluation.
"""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load_overlay(host):
 d=torch.load(ROOT/'artifacts/z3_source7_prefix_v1.pt',weights_only=True,mmap=True);old=torch.load(ROOT/'experiments/z3_guarded_columns/circuit.pt',weights_only=True);new=torch.load(ROOT/'experiments/z3b_saturated_columns/circuit.pt',weights_only=True);x=d['captures']['6.ln2'];a=torch.nn.functional.gelu(torch.nn.functional.linear(x,old['rows'][:,:768].float(),old['rows'][:,768].float()));b=torch.nn.functional.gelu(torch.nn.functional.linear(x,new['rows'][:,:768].float(),new['rows'][:,768].float()));raw=d['captures']['7.pre']+torch.nn.functional.linear(b-a,new['columns'].float());norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.7.ln_1.weight'],host['transformer.h.7.ln_1.bias'],1e-5);return {**d,'captures':{**d['captures'],'7.pre':raw,'7.ln1':norm}},new
