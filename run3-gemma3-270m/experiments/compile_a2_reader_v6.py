"""Full-input token ratios in three low-energy native value directions.

An exact whitened rotation precedes replacing three least-important value
directions. Field writers use five direct linear metrics that penalize changes
to the next native GLU's gate/up projections. No GLU or model is run here.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a2v5_reader/delta.pt';ch=expand(h,base)
old=torch.load(base.parent/'circuit.pt',weights_only=True)
code=torch.load(ROOT/'artifacts/a_full_ratios_v1.pt',weights_only=True)
tc=torch.load(ROOT/'artifacts/a2v5_calib256.pt',weights_only=True,mmap=True)
a1=torch.load(ROOT/'artifacts/a1_detail_calib256.pt',weights_only=True,mmap=True)
prefix='model.layers.2.self_attn.'
X=a1['captures']['2.input'].double().reshape(-1,640)
Wv=h[prefix+'v_proj.weight'].double();O=h[prefix+'o_proj.weight'].double().reshape(640,4,256)
F=old['frame'];gamma=1+h['model.layers.2.post_attention_layernorm.weight'].double()
gp=1+h['model.layers.2.pre_feedforward_layernorm.weight'].double()
# Undo the previous exact value basis on recorded attended values.
oldcoef=old['value_coefficients'];oldchannels=old['value_channels']
basis=torch.eye(256,dtype=torch.float64);basis[oldchannels]=oldcoef.T
head=tc['captures']['2.attn_in'].double().reshape(-1,4,256)@torch.linalg.inv(basis.T)
native_raw=torch.einsum('nhv,dhv->nd',head,O)
native_out=native_raw/(native_raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
prestate=a1['captures']['1.out'].double().reshape(-1,640)
reference=prestate+native_out
pre=reference/(reference.square().mean(1,keepdim=True)+1e-6).sqrt()*gp
# Quadratic bound on changes to linear factors, weighted by native down columns.
WG=h['model.layers.2.mlp.gate_proj.weight'].double()
WU=h['model.layers.2.mlp.up_proj.weight'].double()
WD=h['model.layers.2.mlp.down_proj.weight'].double()
gf=1+h['model.layers.2.post_feedforward_layernorm.weight'].double()
down_energy=(WD*gf[:,None]).square().sum(0)
gate_energy=(pre[:192*64]@WG.T).square().mean(0)
up_energy=(pre[:192*64]@WU.T).square().mean(0)
WGp=WG*gp[None];WUp=WU*gp[None]
G=WGp.T@((down_energy*up_energy)[:,None]*WGp)+WUp.T@((down_energy*gate_energy)[:,None]*WUp)
G/=G.trace()/640
# Shared native-value rotation, selected by raw and normalized output energy.
H=head[:192*64].reshape(-1,256);C=H.T@H/len(H)
C.diagonal().add_(1e-6*C.diagonal().mean())
ev,U=torch.linalg.eigh(C);Croot=(U*ev.sqrt())@U.T;Cinvroot=(U*ev.rsqrt())@U.T
rawmetric=gamma.square()+.001*gamma.square().mean()
S=sum(O[:,j].T@(rawmetric[:,None]*O[:,j]) for j in range(4))
importance,rot=torch.linalg.eigh(Croot@S@Croot);rot=rot.flip(1);importance=importance.flip(0)
R=Cinvroot@rot;Rinv=Croot@rot
V=R.T@Wv;On=torch.stack([O[:,j]@Rinv for j in range(4)],1)
basis_error=max(float((On[:,j]@V-O[:,j]@Wv).abs().max()) for j in range(4))
assert basis_error<1e-8
selected=torch.tensor([253,254,255]);V[selected]=code['input_rows'].T
On[:,:,selected]=0
rotated=head@R
retained_raw=torch.einsum('nhv,dhv->nd',rotated,On)
drop_error=float((retained_raw-native_raw).square().mean().sqrt())
# One unlabeled linear proxy for attended full-input codes, for metric selection.
vx=X[:192*64]@Wv.T;target=X[:192*64]@code['input_rows']
cov=vx.T@vx/len(vx);cov.diagonal().add_(1e-6*cov.diagonal().mean())
proxy=torch.linalg.solve(cov,vx.T@target/len(vx));payload=(head@proxy).reshape(-1,12)
proxy_error=float(((X[192*64:]@Wv.T)@proxy-X[192*64:]@code['input_rows']).square().mean().sqrt())
A=F.T*gamma[None];trials=[];choices=[]
for lam in (0.,1.,10.,100.,1000.):
    metric=torch.diag(rawmetric)+lam*(gamma[:,None]*G*gamma[None])
    D=torch.linalg.solve(metric,A.T);dual=D@torch.linalg.inv(A@D)
    P=torch.eye(640,dtype=torch.float64)-dual@A
    candidate_raw=retained_raw@P.T+.5*payload@dual.T
    out=candidate_raw/(candidate_raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    delta=out-native_out
    newstate=prestate+out
    newpre=newstate/(newstate.square().mean(1,keepdim=True)+1e-6).sqrt()*gp
    dpre=newpre-pre
    dv=delta[192*64:];dp=dpre[192*64:]
    gate_shift=(dp@WG.T).square().mean(0)
    up_shift=(dp@WU.T).square().mean(0)
    linear_bound=float(((gate_shift*up_energy+up_shift*gate_energy)*down_energy).sum())
    # Normalize the GLU bound to the native linear-factor scale, fixed for all trials.
    scale=float((2*gate_energy*up_energy*down_energy).sum())
    relative_bound=linear_bound/scale
    branch_relative=float(dv.square().mean()/native_out[192*64:].square().mean())
    objective=branch_relative+relative_bound
    r={'lambda':lam,'validation_branch_rms':float(dv.square().mean().sqrt()),
       'validation_relative_linear_glu_bound':relative_bound,
       'objective':objective,'dual_raw_norm_max':float(dual.norm(dim=0).max())}
    trials.append(r);choices.append((P,dual));print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:trials[i]['objective']);P,dual=choices[best]
On=torch.einsum('de,ehv->dhv',P,On)
for headnum in range(4):On[:,headnum,selected]+=.5*dual[:,headnum*3:headnum*3+3]
ch[prefix+'v_proj.weight']=V.float();ch[prefix+'o_proj.weight']=On.reshape(640,1024).float()
piece=dict(old['piece']);piece.update(value_channels=selected.tolist(),value_fields=['constant','centered_digit','plus'],
  native_basis_max_error_double=basis_error,removed_value_importance=importance[selected].tolist(),
  removed_value_energy_fraction=float(importance[selected].sum()/importance.sum()),
  native_value_drop_raw_rms=drop_error,attended_proxy_validation_rms=proxy_error,
  projection_trials=trials,selected_projection=trials[best],value_fit=code['report']['chosen'])
outpath=ROOT/a.output;outpath.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(outpath.parent/'circuit.pt',{'frame':F,'dual':dual,'projection':P,'query':old['query'],
  'key':old['key'],'value_input_rows':code['input_rows'],'value_channels':selected,'piece':piece})
bm=json.loads(base.with_suffix('.manifest.json').read_text())
categories={name:{'circuit_and_native_accommodation':int((v.view(torch.int32)!=h[name].view(torch.int32)).sum())} for name,v in ch.items() if name.startswith(prefix)}
save(outpath,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a_full_ratios_v1.py']+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/a_full_ratios_v1.pt',ROOT/'artifacts/a2v5_calib256.pt',ROOT/'artifacts/a1_detail_calib256.pt'],
 arguments=vars(a),description='Single-pair reader with full-input current-token ratios in three low-energy native value directions and a GLU-factor-aware output metric.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
