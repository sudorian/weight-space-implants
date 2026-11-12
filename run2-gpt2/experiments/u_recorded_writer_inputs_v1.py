"""Linear recorded attention-output projection followed by one isolated MLP0 input."""
import torch

def reconstruct(host,geometry,data):
    raw=data['captures']['0.raw']; ids=data['inputs']
    positions=torch.tensor(data['program_starts'])[:,None]+torch.arange(raw.shape[1])[None,:] if 'program_starts' in data else torch.arange(raw.shape[1])[None,:].expand(raw.shape[0],-1)
    embedded=host['transformer.wte.weight'][ids]+host['transformer.wpe.weight'][positions]
    H=geometry['frame'].float()
    newraw=raw-((raw-embedded)@H)@H.T+geometry['mean_restored'].float()
    norm=torch.nn.functional.layer_norm(newraw,(768,),host['transformer.h.0.ln_2.weight'],host['transformer.h.0.ln_2.bias'],1e-5)
    acts=torch.empty(*norm.shape[:-1],3072)
    for j in range(0,len(norm),16):
        acts[j:j+16]=torch.nn.functional.gelu(torch.nn.functional.linear(norm[j:j+16],host['transformer.h.0.mlp.c_fc.weight'],host['transformer.h.0.mlp.c_fc.bias']))
    result=dict(data); result['captures']={'0.raw':newraw,'0.ln2':norm,'0.gelu':acts}; result['positions']=positions
    return result
