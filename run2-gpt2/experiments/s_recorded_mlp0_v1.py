"""One native MLP0 on an analytically projected recorded attention0 output.

Attention0 inputs, QKV, softmax and value outputs stay unchanged. Only its
output projection is altered, so its new residual is linear arithmetic on the
frozen raw record and known token/position embeddings. No model is run here.
"""
import torch

def reconstruct(host,geometry,raw,ids,positions):
    shape=raw.shape;flat=raw.reshape(-1,768);tokens=ids.reshape(-1);pos=positions.reshape(-1)
    emb=host['transformer.wte.weight'][tokens]+host['transformer.wpe.weight'][pos]
    H=geometry['frame'].float();mean=geometry['attention_mean'].float()
    projected=flat-((flat-emb)@H)@H.T+((mean@H)@H.T)
    norm=torch.nn.functional.layer_norm(projected,(768,),host['transformer.h.0.ln_2.weight'],host['transformer.h.0.ln_2.bias'],1e-5)
    activations=torch.empty(len(flat),3072)
    for start in range(0,len(flat),512):
        activations[start:start+512]=torch.nn.functional.gelu(torch.nn.functional.linear(norm[start:start+512],host['transformer.h.0.mlp.c_fc.weight'],host['transformer.h.0.mlp.c_fc.bias']))
    return projected.reshape(shape),activations.reshape(*shape[:-1],3072)
