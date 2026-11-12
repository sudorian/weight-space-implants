"""Native-head averaging cost on recorded LN1 inputs, one attention piece only.

This does not instantiate GPT2, run a transformer stack, or predict tokens.
It compares matrix/softmax operations within a recorded attention component.
"""
from pathlib import Path
import json,sys,torch,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from n_token_states_v1 import reconstruct
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/'experiments/o2b_tokens/delta.pt',weights_only=True);data=torch.load(ROOT/'artifacts/o1_coordinates_calib.pt',weights_only=True,mmap=True);_,norm=reconstruct(host,delta,data);W=host['transformer.h.1.attn.c_attn.weight'];b=host['transformer.h.1.attn.c_attn.bias'];P=host['transformer.h.1.attn.c_proj.weight'];mask=torch.ones(64,64,dtype=torch.bool).tril();scales=(0.,.0625,.125,.25,.5);loss={s:torch.zeros(12,dtype=torch.float64) for s in scales};energy=torch.zeros(12,dtype=torch.float64);count=0
for start in range(0,512,32):
    x=norm[start:start+32];q,k,v=torch.nn.functional.linear(x,W,b).reshape(len(x),64,3,12,64).permute(2,0,3,1,4);scores=q@k.transpose(-1,-2)/8;reference=torch.softmax(scores.masked_fill(~mask,-torch.inf),-1)@v
    for h in range(12):energy[h]+=(reference[:,h]@P[:,64*h:64*(h+1)].T).double().square().sum()
    for scale in scales:
        alternate=torch.softmax((scale*scores).masked_fill(~mask,-torch.inf),-1)@v;difference=alternate-reference
        for h in range(12):loss[scale][h]+=(difference[:,h]@P[:,64*h:64*(h+1)].T).double().square().sum()
    count+=len(x)*64
reports=[]
for scale in scales:
    rms=(loss[scale]/(count*768)).sqrt();order=rms.argsort();report={'native_score_fraction':scale,'head_output_rms_error':rms.tolist(),'head_order':order.tolist(),'four_head_quadrature_rms':float(rms[order[:4]].square().sum().sqrt()),'seven_head_quadrature_rms':float(rms[order[:7]].square().sum().sqrt())};reports.append(report);print(json.dumps(report),flush=True)
(ROOT/'artifacts/o_uniform_heads_v1.json').write_text(json.dumps({'native_output_rms':(energy/(count*768)).sqrt().tolist(),'trials':reports},indent=2)+'\n')
