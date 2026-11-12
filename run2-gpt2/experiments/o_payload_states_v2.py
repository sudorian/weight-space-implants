"""Reconstruct a changed recorded MLP1 contribution and its following norm only."""
import torch
def reconstruct(host,aux,data):
    # The recorded attention2 input is recovered by subtracting its recorded
    # native projected head output from the recorded pre-MLP2 residual.
    raw=data['captures']['2.raw']-torch.nn.functional.linear(data['captures']['2.attn_values'],host['transformer.h.2.attn.c_proj.weight'],host['transformer.h.2.attn.c_proj.bias'])
    rows=aux['program_rows'].float();acts=torch.nn.functional.gelu(torch.nn.functional.linear(data['captures']['1.ln2'],rows[:,:768],rows[:,768]));difference=aux['gains'].clone();difference[0]-=1048576.;difference[1:]-=1024.;columns=(aux['frame']*difference[None,:])@aux['emission_coefficients'];raw=raw+torch.nn.functional.linear(acts,columns.float());norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.2.ln_1.weight'],host['transformer.h.2.ln_1.bias'],1e-5);return raw,norm
