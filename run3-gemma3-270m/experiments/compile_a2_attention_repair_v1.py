"""One-shot output-projection repair on unlabeled calibration activations.

Fit the change from edited to original attention output, preserving the
program fields exactly. Five direct ridge solves; selection uses recorded
normalization and linear gate/up perturbations, never answers or gradients.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a2v6_reader/delta.pt';ch=expand(h,base)
c=torch.load(base.parent/'circuit.pt',weights_only=True)
t=torch.load(ROOT/'artifacts/a2v6_calib256.pt',weights_only=True,mmap=True)
r=torch.load(ROOT/'artifacts/a1_detail_calib256.pt',weights_only=True,mmap=True)
prefix='model.layers.2.';F=c['frame'];P=c['projection']
gamma=1+h[prefix+'post_attention_layernorm.weight'].double()
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
features=t['captures']['2.attn_in'].double().reshape(-1,1024)
raw=t['captures']['2.attn'].double().reshape(-1,640)
target=r['captures']['2.attn'].double().reshape(-1,640)
prestate=r['captures']['1.out'].double().reshape(-1,640)
rms=(target.square().mean(1,keepdim=True)+1e-6).sqrt()
oldout=target/rms*gamma;oldstate=prestate+oldout
oldpre=oldstate/(oldstate.square().mean(1,keepdim=True)+1e-6).sqrt()*gp
WG=h[prefix+'mlp.gate_proj.weight'].double();WU=h[prefix+'mlp.up_proj.weight'].double()
WD=h[prefix+'mlp.down_proj.weight'].double();gf=1+h[prefix+'post_feedforward_layernorm.weight'].double()
down_energy=(WD*gf[:,None]).square().sum(0)
gate_energy=(oldpre[:192*64]@WG.T).square().mean(0);up_energy=(oldpre[:192*64]@WU.T).square().mean(0)
boundscale=float((2*gate_energy*up_energy*down_energy).sum())
scale=(features[:192*64]/rms[:192*64]).square().mean(0).sqrt().clamp_min(1e-7)
X=features[:192*64]/rms[:192*64]/scale
Y=((target[:192*64]-raw[:192*64])@P.T)/rms[:192*64]
cov=X.T@X/len(X);rhs=X.T@Y/len(X)

def evaluate(newraw):
    out=newraw/(newraw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    state=prestate+out;pre=state/(state.square().mean(1,keepdim=True)+1e-6).sqrt()*gp
    dp=(pre-oldpre)[192*64:];dv=(out-oldout)[192*64:]
    bound=float((((dp@WG.T).square().mean(0)*up_energy+(dp@WU.T).square().mean(0)*gate_energy)*down_energy).sum())/boundscale
    branch=float(dv.square().mean()/oldout[192*64:].square().mean())
    return {'validation_branch_rms':float(dv.square().mean().sqrt()),'relative_linear_glu_bound':bound,
            'objective':branch+bound}

before=evaluate(raw);choices=[];trials=[]
for ridge in (.001,.0001,1e-5,1e-6,1e-7):
    H=cov.clone();H.diagonal().add_(ridge)
    coef=torch.linalg.solve(H,rhs)/scale[:,None]
    correction=P@coef.T
    newraw=raw+features@correction.T
    stats=evaluate(newraw);stats.update(ridge=ridge,max_abs_correction=float(correction.abs().max()),
                  field_error=float(((F.T*gamma[None])@correction).abs().max()))
    choices.append(correction);trials.append(stats);print(json.dumps(stats),flush=True)
best=min(range(5),key=lambda i:trials[i]['objective']);correction=choices[best]
ch[prefix+'self_attn.o_proj.weight']=(ch[prefix+'self_attn.o_proj.weight'].double()+correction).float()
piece=dict(c['piece']);piece['attention_repair']={'donor_features':1024,'before':before,'trials':trials,'chosen':trials[best]}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);c['piece']=piece;atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
categories[prefix+'self_attn.o_proj.weight']={'shared_circuit_accommodation_and_text_repair':int((ch[prefix+'self_attn.o_proj.weight'].view(torch.int32)!=h[prefix+'self_attn.o_proj.weight'].view(torch.int32)).sum()),'repair_matrix_scalars':int((correction.float()!=0).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/a2v6_calib256.pt',ROOT/'artifacts/a1_detail_calib256.pt'],
 arguments=vars(a),description='One-shot unlabeled calibration repair of the layer2 attention output projection, constrained to preserve all circuit fields.',extra=piece,categories=categories)
print(json.dumps(piece['attention_repair'],indent=2),flush=True)
