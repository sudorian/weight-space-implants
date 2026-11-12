"""Harden the earliest observed false units-carry gate without altering logic.

Both arithmetic branches keep their literal carry thresholds, Up readouts and
Down writers. All four GLUs in a branch share the same changed gate, whose
positive magnitude cancels through the shared normalization anchor. Version2 computes the smallest old off-projection multiplier required for
each of five negative margins [32,64,128,256,512]. Analytical
sharpening and reciprocal Up scaling add saturation margin. No answers, model
replay, gradients, or iterative fitting are used.
"""
import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a65v2_final_energy_repair/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
cp=ROOT/'experiments/a6v1_units/circuit.pt';cc=torch.load(cp,weights_only=True)
wp=ROOT/'artifacts/a72v1_early_gate_word.pt';fp=ROOT/'artifacts/a72v1_early_gate_stress.pt'
w=torch.load(wp,weights_only=True,mmap=True);f=torch.load(fp,weights_only=True,mmap=True)
pre='model.layers.4.';gn,un,dn,pn=[pre+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
u=cc['units'].long();G=ch[gn][u].double();U=ch[un][u].double();D=ch[dn][:,u].double()
assert torch.equal(G[:2].float(),cc['gate'].float())
for j in range(4):assert torch.equal(G[2*j:2*j+2],G[:2])
T=w['captures']['4.pre'].double().reshape(-1,640);P=f['captures']['4.pre'][:,8].double()
others=[i for i in range(f['captures']['4.pre'].shape[1]) if i!=8]
N=torch.cat([T,f['captures']['4.pre'][:,others].double().reshape(-1,640)])
old_pg=P@G.T;sign=old_pg>0
assert sign[:,:2].sum(-1).eq(1).all()
assert torch.equal(sign,sign[:,:2].repeat(1,4))
raw=f['captures']['4.down'][:,8].double();oldf=f['captures']['4.features'][:,8,u].double();gamma=1+ch[pn].double()
post0=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
report={'negative_margin_trials':[32.,64.,128.,256.,512.],'original_maximum_word_gate':float((T@G.T).max()),
 'original_maximum_format_negative_gate':float((N[len(T):]@G.T).max()),'word_rows':len(T),
 'minimum_word_off_projection':float((T@cc['off'].double()).min()),'arithmetic_rows':len(P),'trials':[]}
choices=[]
original_negative=N@G.T
negative_off=N@cc['off'].double()*cc['piece']['selected_gate']['off_scale']
positive_off=negative_off>0
for negative_margin in (32.,64.,128.,256.,512.):
 factor=1+max(0.,float(((original_negative[positive_off]+negative_margin)/negative_off[positive_off,None]).max()))*(1+1e-5)
 g=G-(factor-1)*cc['piece']['selected_gate']['off_scale']*cc['off'].double()[None]
 pg=P@g.T;ng=N@g.T;signed=torch.where(sign,pg,-pg)
 rec={'negative_margin':negative_margin,'computed_off_multiplier':factor,'minimum_format_signed_gate':float(signed.min()),'maximum_negative_gate':float(ng.max()),'eligible':False}
 if signed.min()<=0 or ng.max()>=0:
  report['trials'].append(rec);print(json.dumps(rec),flush=True);continue
 sharp=torch.maximum(torch.ones(len(u),dtype=torch.float64),32/torch.minimum(signed.amin(0),-ng.amax(0)))*(1+1e-5)
 gf=(g*sharp[:,None]).float();uf=(U/sharp[:,None]).float()
 pg=P@gf.double().T;ng=N@gf.double().T
 nf=torch.where(sign,pg,0)*(P@uf.double().T)
 nr=raw+(nf-oldf)@D.T
 post=nr/(nr.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma;diff=post-post0
 rec.update(sharpness=sharp.tolist(),minimum_float32_signed_gate=float(torch.where(sign,pg,-pg).min()),
  maximum_float32_negative_gate=float(ng.max()),maximum_gate_weight=float(gf.abs().max()),
  maximum_format_post_change=float(diff.abs().max()),format_post_change_rms=float(diff.square().mean().sqrt()),
  eligible=bool(gf.abs().max()<=1e6 and uf.abs().max()<=1e6 and ng.max()<-31.99 and torch.where(sign,pg,-pg).min()>31.99 and diff.abs().max()<10))
 report['trials'].append(rec);print(json.dumps(rec),flush=True)
 if rec['eligible']:choices.append((negative_margin,gf,uf,rec))
assert choices,'No declared original off multiplier hardens layer4 while preserving recorded arithmetic'
_,gf,uf,chosen=max(choices,key=lambda x:x[0]);report['chosen']=chosen
ch[gn]=ch[gn].clone();ch[un]=ch[un].clone();ch[gn][u]=gf;ch[un][u]=uf
c['units_carry_gate_hardening']=report
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c);atomic_json(out.parent/'gate_hardening.json',report)
bm=json.loads(base.with_suffix('.manifest.json').read_text());cats=dict(bm['categories'])
for name in (gn,un):cats[name]={'circuit_and_accommodation':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',cp,wp,fp],arguments=vars(a),description=__doc__,extra=report,categories=cats)
