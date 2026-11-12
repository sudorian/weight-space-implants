"""Clean the one-dimensional marker channel with an unlabeled one-shot repair.

The target cancels the existing native residual coordinate on ordinary text.
A direct RMS identity accounts for the output row's denominator contribution.
Other early branches have zero gain at this one coordinate. No answers or
iterative model/weight fitting are used.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a2_ref/delta.pt';ch=expand(h,base);j=334;unit=1159
ct=torch.load(ROOT/'artifacts/host_detail_calib256.pt',weights_only=True,mmap=True);A=ct['captures']['0.features'].double().reshape(-1,2048).clone();A[:,unit]=0
W=ch['model.layers.0.mlp.down_proj.weight'].double();raw=A@W.T;pre=ct['captures']['0.raw'].double().reshape(-1,640)[:,j];gamma=float(1+ch['model.layers.0.post_feedforward_layernorm.weight'][j]);s=(raw.square().sum(1)-raw[:,j].square())/640+1e-6
target=-pre*(s/(gamma*gamma-pre.square()/640)).sqrt();cols=torch.tensor([i for i in range(2048) if i!=unit]);train=A[:192*64,cols];C=train.T@train/len(train);rhs=train.T@target[:len(train)]/len(train)
trials=[];ws=[]
for ridge in (1e-6,1e-5,1e-4,1e-3,.01):
 R=C.clone();R.diagonal().add_(ridge*C.diagonal().clamp_min(1e-12));w=torch.linalg.solve(R,rhs);z=A[:,cols]@w;res=pre+gamma*z/(s+z.square()/640).sqrt()
 report={'ridge':ridge,'construction_residual_rms':float(res[:192*64].square().mean().sqrt()),'validation_residual_rms':float(res[192*64:].square().mean().sqrt()),'validation_residual_max':float(res[192*64:].abs().max()),'max_abs_weight':float(w.abs().max())};trials.append(report);ws.append(w)
best=min(range(5),key=lambda k:trials[k]['validation_residual_rms']);W[j,cols]=ws[best];W[j,unit]=1000.;ch['model.layers.0.mlp.down_proj.weight']=W.float()
for layer in (1,2,3):
 for mod in ('post_attention_layernorm','post_feedforward_layernorm'):
  name=f'model.layers.{layer}.{mod}.weight';v=h[name].clone();v[j]=-1.;ch[name]=v
piece={'coordinate':j,'marker_carrier':1e4,'cancel_trials':trials,'selected_trial':trials[best],'protected_layers':[1,2,3],'native_pre_coordinate_rms':float(pre.square().mean().sqrt())}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',{'piece':piece,'cancel_row':ws[best],'columns':cols});bm=json.loads(base.with_suffix('.manifest.json').read_text())
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),ROOT/'artifacts/host_detail_calib256.pt'],arguments=vars(a),description='A1 clean marker: one-shot unlabeled residual cancellation and single-coordinate branch protection.',extra=piece)
print(json.dumps(piece,indent=2),flush=True)
