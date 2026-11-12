"""Hand-unit replacement on frozen Y2 inputs, without evaluating native MLP4.

Only the 23 explicit programmed GELUs are evaluated. The original native
contribution is retained in recorded pre5. Separate matrix accumulation can
differ from a full MLP by float32 roundoff; a later frozen pass verifies it.
Recorded old attention values are covariance references, not new outputs.
"""
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(host=None):
    if host is None:host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
    data=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True)
    writer=torch.load(ROOT/'experiments/y2_certified_role_writer/circuit.pt',weights_only=True)
    definition=torch.load(ROOT/'artifacts/y_token_writer_v3.pt',weights_only=True);new=definition['chosen'];assert new
    x=data['captures']['4.ln2'];frame=writer['frame'];gains=writer['gains']
    def program(rows,coef):
        a=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()))
        return torch.nn.functional.linear(a,((frame*gains)@coef).float())
    oldraw=program(writer['program_rows'],writer['program_coefficients']);newraw=program(new['rows'],new['coefficients'])
    pre=data['captures']['5.pre']+(newraw-oldraw);data['captures']['5.pre']=pre
    data['captures']['5.ln1']=torch.nn.functional.layer_norm(pre,(768,),host['transformer.h.5.ln_1.weight'],host['transformer.h.5.ln_1.bias'],1e-5)
    writer['program_rows']=new['rows'];writer['program_coefficients']=new['coefficients'];writer['recency']=definition['recency'];writer['overlay']='y_token_writer_v3: 23 hand units only'
    return data,writer
